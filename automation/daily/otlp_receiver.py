"""Zero-dependency in-pod OTLP/gRPC MetricsService receiver and wire-format parser.

Listens on `0.0.0.0:4317` (default OTLP/gRPC port) inside the `user-workload`
container on `pathways-head` to capture all 91 canonical `pathways_*` metrics
pushed by `OtlpExporter` (`--enable_pathways_metrics --additional_metrics_regex=.*`,
cl/1018902971, cl/1019381440) from:
  - `pathways-proxy` (`http://127.0.0.1:4317`)
  - `pathways-rm` (`http://127.0.0.1:4317`)
  - `pathways-worker` pods (`http://<head-service>:4317`)
including the final `ForceMetricsCollection()` flush triggered on `SIGTERM` or
process exit (cl/1015142474, cl/1015142480, cl/1015142479).

Uses `grpc.method_handlers_generic_handler` + pure-Python protobuf wire decoding
so no `opentelemetry-proto` Python package installation is required in the
client container.
"""

from __future__ import annotations

import argparse
from concurrent import futures
import datetime
import json
import math
import pathlib
import signal
import struct
import threading
import time
from typing import Any, Iterator, Optional

OTLP_EXPORT_METHOD = (
    "/opentelemetry.proto.collector.metrics.v1.MetricsService/Export"
)
DEFAULT_OTLP_PORT = 4317
DEFAULT_OTLP_OUTPUT = "/tmp/results/otlp_metrics.json"


# ---------------------------------------------------------------------------
# Pure-Python Protobuf Wire-Format Reader (proto3)
# ---------------------------------------------------------------------------
def _read_varint(buf: bytes, pos: int, end: int) -> tuple[int, int]:
  result = 0
  shift = 0
  while pos < end:
    b = buf[pos]
    pos += 1
    result |= (b & 0x7F) << shift
    if not (b & 0x80):
      return result, pos
    shift += 7
    if shift >= 64:
      raise ValueError("Malformed protobuf varint")
  raise ValueError("Truncated protobuf varint")


def _iter_fields(
    buf: bytes, start: int = 0, end: Optional[int] = None
) -> Iterator[tuple[int, int, Any]]:
  """Yields `(field_number, wire_type, value)` from a protobuf message slice."""
  if end is None:
    end = len(buf)
  pos = start
  while pos < end:
    tag, pos = _read_varint(buf, pos, end)
    field_no = tag >> 3
    wire_type = tag & 0x07
    if wire_type == 0:  # varint
      val, pos = _read_varint(buf, pos, end)
      yield field_no, wire_type, val
    elif wire_type == 1:  # 64-bit
      if pos + 8 > end:
        raise ValueError("Truncated 64-bit field")
      yield field_no, wire_type, buf[pos : pos + 8]
      pos += 8
    elif wire_type == 2:  # length-delimited
      length, pos = _read_varint(buf, pos, end)
      if pos + length > end:
        raise ValueError("Truncated length-delimited field")
      yield field_no, wire_type, buf[pos : pos + length]
      pos += length
    elif wire_type == 5:  # 32-bit
      if pos + 4 > end:
        raise ValueError("Truncated 32-bit field")
      yield field_no, wire_type, buf[pos : pos + 4]
      pos += 4
    else:
      raise ValueError(f"Unsupported protobuf wire_type={wire_type}")


def _decode_any_value(buf: bytes) -> Any:
  """Decodes `opentelemetry.proto.common.v1.AnyValue`."""
  for fno, wtype, raw in _iter_fields(buf):
    if fno == 1 and wtype == 2:
      return raw.decode("utf-8", errors="replace")
    if fno == 2 and wtype == 0:
      return bool(raw)
    if fno == 3 and wtype == 0:
      # proto3 int64 is two's-complement varint
      return raw - (1 << 64) if raw >= (1 << 63) else raw
    if fno == 4 and wtype == 1:
      return struct.unpack("<d", raw)[0]
    if fno == 7 and wtype == 2:
      return raw.hex()
  return ""


def _decode_key_value(buf: bytes) -> tuple[str, Any]:
  """Decodes `opentelemetry.proto.common.v1.KeyValue`."""
  key = ""
  val: Any = ""
  for fno, wtype, raw in _iter_fields(buf):
    if fno == 1 and wtype == 2:
      key = raw.decode("utf-8", errors="replace")
    elif fno == 2 and wtype == 2:
      val = _decode_any_value(raw)
  return key, val


def _decode_resource(buf: bytes) -> dict[str, Any]:
  """Decodes `opentelemetry.proto.resource.v1.Resource` attributes."""
  attrs: dict[str, Any] = {}
  for fno, wtype, raw in _iter_fields(buf):
    if fno == 1 and wtype == 2:
      k, v = _decode_key_value(raw)
      if k:
        attrs[k] = v
  return attrs


def _decode_number_data_point(buf: bytes) -> dict[str, Any]:
  """Decodes `opentelemetry.proto.metrics.v1.NumberDataPoint`."""
  attrs: dict[str, Any] = {}
  start_ns = 0
  time_ns = 0
  val: Optional[float] = None
  for fno, wtype, raw in _iter_fields(buf):
    if fno == 7 and wtype == 2:
      k, v = _decode_key_value(raw)
      if k:
        attrs[k] = v
    elif fno == 2 and wtype == 1:
      start_ns = struct.unpack("<Q", raw)[0]
    elif fno == 3 and wtype == 1:
      time_ns = struct.unpack("<Q", raw)[0]
    elif fno == 4 and wtype == 1:
      val = float(struct.unpack("<d", raw)[0])
    elif fno == 6 and wtype == 1:
      val = float(struct.unpack("<q", raw)[0])
  return {
      "attributes": attrs,
      "start_time_unix_nano": start_ns,
      "time_unix_nano": time_ns,
      "value": val if val is not None else 0.0,
  }


def _histogram_quantile(
    q: float, bounds: list[float], counts: list[int]
) -> Optional[float]:
  """Estimates quantile `q` (in [0, 1]) from explicit-bucket histogram counts."""
  total = sum(counts)
  if total <= 0 or not counts:
    return None
  if not bounds:
    return None
  target = q * total
  cum = 0
  for i, cnt in enumerate(counts):
    prev_cum = cum
    cum += cnt
    if cum >= target:
      low = bounds[i - 1] if i > 0 else max(0.0, bounds[0] * 0.5)
      high = bounds[i] if i < len(bounds) else low
      if cnt <= 0 or high <= low:
        return round(float(low), 6)
      frac = (target - prev_cum) / float(cnt)
      return round(float(low + frac * (high - low)), 6)
  return round(float(bounds[-1]), 6)


def _decode_histogram_data_point(buf: bytes) -> dict[str, Any]:
  """Decodes `opentelemetry.proto.metrics.v1.HistogramDataPoint`."""
  attrs: dict[str, Any] = {}
  start_ns = 0
  time_ns = 0
  count = 0
  total_sum = 0.0
  bucket_counts: list[int] = []
  explicit_bounds: list[float] = []
  for fno, wtype, raw in _iter_fields(buf):
    if fno == 9 and wtype == 2:
      k, v = _decode_key_value(raw)
      if k:
        attrs[k] = v
    elif fno == 2 and wtype == 1:
      start_ns = struct.unpack("<Q", raw)[0]
    elif fno == 3 and wtype == 1:
      time_ns = struct.unpack("<Q", raw)[0]
    elif fno == 4 and wtype == 1:
      count = int(struct.unpack("<Q", raw)[0])
    elif fno == 5 and wtype == 1:
      total_sum = float(struct.unpack("<d", raw)[0])
    elif fno == 6:
      if wtype == 2:
        n = len(raw) // 8
        bucket_counts.extend(struct.unpack(f"<{n}Q", raw[: n * 8]))
      elif wtype == 1:
        bucket_counts.append(int(struct.unpack("<Q", raw)[0]))
    elif fno == 7:
      if wtype == 2:
        n = len(raw) // 8
        explicit_bounds.extend(struct.unpack(f"<{n}d", raw[: n * 8]))
      elif wtype == 1:
        explicit_bounds.append(float(struct.unpack("<d", raw)[0]))
  mean = (total_sum / count) if count > 0 else None
  p50 = _histogram_quantile(0.50, explicit_bounds, bucket_counts)
  p90 = _histogram_quantile(0.90, explicit_bounds, bucket_counts)
  if p50 is None and mean is not None:
    p50 = mean
  if p90 is None and mean is not None:
    p90 = mean
  return {
      "attributes": attrs,
      "start_time_unix_nano": start_ns,
      "time_unix_nano": time_ns,
      "count": count,
      "sum": total_sum,
      "mean": round(mean, 6) if mean is not None and math.isfinite(mean) else None,
      "p50": p50,
      "p90": p90,
      "bucket_counts": bucket_counts,
      "explicit_bounds": explicit_bounds,
  }


def _decode_summary_data_point(buf: bytes) -> dict[str, Any]:
  """Decodes `opentelemetry.proto.metrics.v1.SummaryDataPoint`."""
  attrs: dict[str, Any] = {}
  start_ns = 0
  time_ns = 0
  count = 0
  total_sum = 0.0
  quantiles: dict[str, float] = {}
  for fno, wtype, raw in _iter_fields(buf):
    if fno == 7 and wtype == 2:
      k, v = _decode_key_value(raw)
      if k:
        attrs[k] = v
    elif fno == 2 and wtype == 1:
      start_ns = struct.unpack("<Q", raw)[0]
    elif fno == 3 and wtype == 1:
      time_ns = struct.unpack("<Q", raw)[0]
    elif fno == 4 and wtype == 1:
      count = int(struct.unpack("<Q", raw)[0])
    elif fno == 5 and wtype == 1:
      total_sum = float(struct.unpack("<d", raw)[0])
    elif fno == 6 and wtype == 2:
      q_val = 0.0
      v_val = 0.0
      for qfno, qwtype, qraw in _iter_fields(raw):
        if qfno == 1 and qwtype == 1:
          q_val = float(struct.unpack("<d", qraw)[0])
        elif qfno == 2 and qwtype == 1:
          v_val = float(struct.unpack("<d", qraw)[0])
      quantiles[f"{q_val:.2f}"] = v_val
  mean = (total_sum / count) if count > 0 else None
  return {
      "attributes": attrs,
      "start_time_unix_nano": start_ns,
      "time_unix_nano": time_ns,
      "count": count,
      "sum": total_sum,
      "mean": round(mean, 6) if mean is not None and math.isfinite(mean) else None,
      "p50": quantiles.get("0.50", mean),
      "p90": quantiles.get("0.90", mean),
      "quantiles": quantiles,
  }


def _decode_metric(buf: bytes) -> Optional[dict[str, Any]]:
  """Decodes `opentelemetry.proto.metrics.v1.Metric`."""
  name = ""
  description = ""
  unit = ""
  kind = ""
  data_points: list[dict[str, Any]] = []
  for fno, wtype, raw in _iter_fields(buf):
    if fno == 1 and wtype == 2:
      name = raw.decode("utf-8", errors="replace")
    elif fno == 2 and wtype == 2:
      description = raw.decode("utf-8", errors="replace")
    elif fno == 3 and wtype == 2:
      unit = raw.decode("utf-8", errors="replace")
    elif fno == 5 and wtype == 2:  # Gauge
      kind = "gauge"
      for gfno, gwtype, graw in _iter_fields(raw):
        if gfno == 1 and gwtype == 2:
          data_points.append(_decode_number_data_point(graw))
    elif fno == 7 and wtype == 2:  # Sum
      kind = "sum"
      for sfno, swtype, sraw in _iter_fields(raw):
        if sfno == 1 and swtype == 2:
          data_points.append(_decode_number_data_point(sraw))
    elif fno == 9 and wtype == 2:  # Histogram
      kind = "histogram"
      for hfno, hwtype, hraw in _iter_fields(raw):
        if hfno == 1 and hwtype == 2:
          data_points.append(_decode_histogram_data_point(hraw))
    elif fno == 11 and wtype == 2:  # Summary
      kind = "summary"
      for yfno, ywtype, yraw in _iter_fields(raw):
        if yfno == 1 and ywtype == 2:
          data_points.append(_decode_summary_data_point(yraw))
  if not name:
    return None
  return {
      "name": name,
      "description": description,
      "unit": unit,
      "kind": kind,
      "data_points": data_points,
  }


def decode_export_metrics_request(raw_request: bytes) -> list[dict[str, Any]]:
  """Decodes an `ExportMetricsServiceRequest` protobuf payload into Python dicts."""
  batches: list[dict[str, Any]] = []
  for fno, wtype, rm_raw in _iter_fields(raw_request):
    if fno != 1 or wtype != 2:
      continue
    resource_attrs: dict[str, Any] = {}
    metrics_list: list[dict[str, Any]] = []
    for rfno, rwtype, rraw in _iter_fields(rm_raw):
      if rfno == 1 and rwtype == 2:
        resource_attrs = _decode_resource(rraw)
      elif rfno == 2 and rwtype == 2:
        for sfno, swtype, sraw in _iter_fields(rraw):
          if sfno == 2 and swtype == 2:
            m = _decode_metric(sraw)
            if m is not None:
              metrics_list.append(m)
    batches.append({"resource": resource_attrs, "metrics": metrics_list})
  return batches


# ---------------------------------------------------------------------------
# In-Memory Aggregator & gRPC Receiver
# ---------------------------------------------------------------------------
def _label_key(attrs: dict[str, Any]) -> str:
  if not attrs:
    return ""
  pairs = ",".join(f'{k}="{attrs[k]}"' for k in sorted(attrs.keys()))
  return f"{{{pairs}}}"


class OtlpMetricsStore:
  """Thread-safe store merging cumulative OTLP metric exports by resource & name."""

  def __init__(self, output_path: str = DEFAULT_OTLP_OUTPUT) -> None:
    self._lock = threading.Lock()
    self._output_path = pathlib.Path(output_path)
    self._export_calls = 0
    # Keyed by (service_name, instance_id, metric_name) -> metric dict
    self._latest_by_source: dict[tuple[str, str, str], dict[str, Any]] = {}

  def ingest_raw(self, raw_request: bytes) -> None:
    batches = decode_export_metrics_request(raw_request)
    with self._lock:
      self._export_calls += 1
      for batch in batches:
        res = batch.get("resource") or {}
        svc = str(res.get("service.name") or "unknown")
        inst = str(res.get("service.instance.id") or "default")
        for m in batch.get("metrics") or []:
          mname = str(m.get("name") or "")
          if mname:
            self._latest_by_source[(svc, inst, mname)] = {
                "service_name": svc,
                "service_instance_id": inst,
                **m,
            }
      self._write_locked()

  def snapshot(self) -> dict[str, Any]:
    with self._lock:
      return self._build_snapshot_locked()

  def _build_snapshot_locked(self) -> dict[str, Any]:
    flat: dict[str, float] = {}
    string_labels: dict[str, list[dict[str, Any]]] = {}
    metrics_by_name: dict[str, list[dict[str, Any]]] = {}

    for (_, _, mname), m in sorted(self._latest_by_source.items()):
      metrics_by_name.setdefault(mname, []).append(m)
      kind = m.get("kind")
      dps = m.get("data_points") or []
      if not dps:
        continue

      if kind in ("gauge", "sum"):
        total_val = 0.0
        for dp in dps:
          v = float(dp.get("value") or 0.0)
          total_val += v
          attrs = dp.get("attributes") or {}
          if attrs:
            flat[f"{mname}{_label_key(attrs)}"] = v
            string_labels.setdefault(mname, []).append(attrs)
        flat[mname] = flat.get(mname, 0.0) + total_val
      elif kind in ("histogram", "summary"):
        tot_cnt = 0
        tot_sum = 0.0
        p50_vals: list[float] = []
        p90_vals: list[float] = []
        for dp in dps:
          cnt = int(dp.get("count") or 0)
          s = float(dp.get("sum") or 0.0)
          tot_cnt += cnt
          tot_sum += s
          if dp.get("p50") is not None and cnt > 0:
            p50_vals.append(float(dp["p50"]))
          if dp.get("p90") is not None and cnt > 0:
            p90_vals.append(float(dp["p90"]))
          attrs = dp.get("attributes") or {}
          if attrs and cnt > 0:
            lkey = _label_key(attrs)
            flat[f"{mname}_count{lkey}"] = float(cnt)
            flat[f"{mname}_sum{lkey}"] = s
            if dp.get("p50") is not None:
              flat[f"{mname}_p50{lkey}"] = float(dp["p50"])
            string_labels.setdefault(mname, []).append(attrs)
        if tot_cnt > 0:
          prev_cnt = flat.get(f"{mname}_count", 0.0)
          prev_sum = flat.get(f"{mname}_sum", 0.0)
          new_cnt = prev_cnt + tot_cnt
          new_sum = prev_sum + tot_sum
          flat[f"{mname}_count"] = new_cnt
          flat[f"{mname}_sum"] = new_sum
          flat[f"{mname}_mean"] = new_sum / new_cnt
          if p50_vals:
            flat[f"{mname}_p50"] = sorted(p50_vals)[len(p50_vals) // 2]
          if p90_vals:
            flat[f"{mname}_p90"] = max(p90_vals)

    return {
        "updated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "export_rpc_count": self._export_calls,
        "unique_metric_names": sorted(metrics_by_name.keys()),
        "flat_metrics": flat,
        "metric_attributes": string_labels,
        "metrics_by_name": metrics_by_name,
    }

  def _write_locked(self) -> None:
    try:
      self._output_path.parent.mkdir(parents=True, exist_ok=True)
      tmp_path = self._output_path.with_suffix(".tmp")
      tmp_path.write_text(
          json.dumps(self._build_snapshot_locked(), indent=2, sort_keys=True),
          encoding="utf-8",
      )
      tmp_path.replace(self._output_path)
    except OSError:
      pass

  def flush(self) -> None:
    with self._lock:
      self._write_locked()


def start_otlp_grpc_server(
    port: int = DEFAULT_OTLP_PORT,
    output_path: str = DEFAULT_OTLP_OUTPUT,
) -> tuple[Any, OtlpMetricsStore]:
  """Starts a background gRPC server implementing `MetricsService/Export`."""
  import grpc  # pylint: disable=g-import-not-at-top

  store = OtlpMetricsStore(output_path=output_path)

  def _handle_export(raw_request: bytes, context: Any) -> bytes:
    del context
    store.ingest_raw(raw_request)
    # Empty protobuf bytes is a valid ExportMetricsServiceResponse with 0 rejected points.
    return b""

  rpc_handler = grpc.method_handlers_generic_handler(
      "opentelemetry.proto.collector.metrics.v1.MetricsService",
      {
          "Export": grpc.unary_unary_rpc_method_handler(
              _handle_export,
              request_deserializer=lambda b: b,
              response_serializer=lambda b: b,
          )
      },
  )
  server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
  server.add_generic_rpc_handlers((rpc_handler,))
  server.add_insecure_port(f"[::]:{port}")
  server.start()
  return server, store


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--port", type=int, default=DEFAULT_OTLP_PORT)
  parser.add_argument("--output", default=DEFAULT_OTLP_OUTPUT)
  args = parser.parse_args()

  server, store = start_otlp_grpc_server(port=args.port, output_path=args.output)
  stop_event = threading.Event()

  def _on_signal(signum: int, frame: Any) -> None:
    del signum, frame
    store.flush()
    stop_event.set()

  signal.signal(signal.SIGTERM, _on_signal)
  signal.signal(signal.SIGINT, _on_signal)
  store.flush()
  while not stop_event.is_set():
    time.sleep(0.5)
  server.stop(grace=1.0)
  store.flush()


if __name__ == "__main__":
  main()
