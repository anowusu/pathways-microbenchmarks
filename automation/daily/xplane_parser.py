"""Zero-dependency pure-Python XSpace (.xplane.pb) parser and XProf metric extractor.

Works inside python:3.12-slim without tensorflow or compiled protobuf descriptors.
Schema: google3/third_party/tensorflow/tsl/profiler/protobuf/xplane.proto
"""

import collections
import glob
import os
import statistics
import struct
from typing import Any, Optional


def _iter_fields(buf: memoryview):
  """Yields (field_number, wire_type, value) from a serialized protobuf message."""
  i, n = 0, len(buf)
  while i < n:
    tag, shift = 0, 0
    while True:
      b = buf[i]
      i += 1
      tag |= (b & 0x7F) << shift
      if not (b & 0x80):
        break
      shift += 7
    field_num, wire_type = tag >> 3, tag & 0x7
    if wire_type == 0:
      val, shift = 0, 0
      while True:
        b = buf[i]
        i += 1
        val |= (b & 0x7F) << shift
        if not (b & 0x80):
          break
        shift += 7
      yield field_num, wire_type, val
    elif wire_type == 1:
      yield field_num, wire_type, buf[i : i + 8]
      i += 8
    elif wire_type == 2:
      ln, shift = 0, 0
      while True:
        b = buf[i]
        i += 1
        ln |= (b & 0x7F) << shift
        if not (b & 0x80):
          break
        shift += 7
      yield field_num, wire_type, buf[i : i + ln]
      i += ln
    elif wire_type == 5:
      yield field_num, wire_type, buf[i : i + 4]
      i += 4
    else:
      raise ValueError(f"Unsupported protobuf wire_type={wire_type} at offset {i}")


def _to_int64(v: int) -> int:
  return v - (1 << 64) if v & (1 << 63) else v


def _parse_stat(buf: memoryview) -> tuple[int, str, Any]:
  md_id, kind, val = 0, "", None
  for fnum, wtype, v in _iter_fields(buf):
    if fnum == 1:
      md_id = _to_int64(v)
    elif fnum == 2 and wtype == 1:
      kind, val = "double", struct.unpack("<d", v)[0]
    elif fnum == 3:
      kind, val = "uint64", v
    elif fnum == 4:
      kind, val = "int64", _to_int64(v)
    elif fnum == 5:
      kind, val = "str", bytes(v).decode("utf-8", errors="replace")
    elif fnum == 6:
      kind, val = "bytes", bytes(v)
    elif fnum == 7:
      kind, val = "ref", v
  return md_id, kind, val


def _parse_event_metadata(buf: memoryview) -> dict[str, Any]:
  out: dict[str, Any] = {"id": 0, "name": "", "display_name": "", "stats": []}
  for fnum, _, v in _iter_fields(buf):
    if fnum == 1:
      out["id"] = _to_int64(v)
    elif fnum == 2:
      out["name"] = bytes(v).decode("utf-8", errors="replace")
    elif fnum == 4:
      out["display_name"] = bytes(v).decode("utf-8", errors="replace")
    elif fnum == 5:
      out["stats"].append(_parse_stat(v))
  return out


def _parse_stat_metadata(buf: memoryview) -> dict[str, Any]:
  out: dict[str, Any] = {"id": 0, "name": ""}
  for fnum, _, v in _iter_fields(buf):
    if fnum == 1:
      out["id"] = _to_int64(v)
    elif fnum == 2:
      out["name"] = bytes(v).decode("utf-8", errors="replace")
  return out


def _parse_map_entry(buf: memoryview, val_parser):
  k, val = 0, None
  for fnum, _, v in _iter_fields(buf):
    if fnum == 1:
      k = _to_int64(v)
    elif fnum == 2:
      val = val_parser(v)
  return k, val


def _parse_event(buf: memoryview) -> dict[str, Any]:
  ev: dict[str, Any] = {
      "metadata_id": 0,
      "offset_ps": 0,
      "duration_ps": 0,
      "stats": [],
  }
  for fnum, _, v in _iter_fields(buf):
    if fnum == 1:
      ev["metadata_id"] = _to_int64(v)
    elif fnum == 2:
      ev["offset_ps"] = _to_int64(v)
    elif fnum == 3:
      ev["duration_ps"] = _to_int64(v)
    elif fnum == 4:
      ev["stats"].append(_parse_stat(v))
    elif fnum == 5:
      ev["num_occurrences"] = _to_int64(v)
  return ev


def _parse_line(buf: memoryview) -> dict[str, Any]:
  line: dict[str, Any] = {
      "id": 0,
      "name": "",
      "display_name": "",
      "timestamp_ns": 0,
      "duration_ps": 0,
      "events": [],
  }
  for fnum, _, v in _iter_fields(buf):
    if fnum == 1:
      line["id"] = _to_int64(v)
    elif fnum == 2:
      line["name"] = bytes(v).decode("utf-8", errors="replace")
    elif fnum == 3:
      line["timestamp_ns"] = _to_int64(v)
    elif fnum == 4:
      line["events"].append(_parse_event(v))
    elif fnum == 9:
      line["duration_ps"] = _to_int64(v)
    elif fnum == 11:
      line["display_name"] = bytes(v).decode("utf-8", errors="replace")
  return line


def _resolve_stats(raw_stats, stat_md: dict[int, dict[str, Any]]) -> dict[str, Any]:
  out: dict[str, Any] = {}
  for md_id, kind, val in raw_stats:
    name = stat_md.get(md_id, {}).get("name", f"stat_{md_id}")
    out[name] = stat_md.get(val, {}).get("name", val) if kind == "ref" else val
  return out


def parse_xspace(data: bytes) -> dict[str, Any]:
  """Parses serialized XSpace (.xplane.pb) bytes into a dictionary with resolved stats."""
  mv = memoryview(data)
  xspace: dict[str, Any] = {
      "planes": [],
      "hostnames": [],
      "errors": [],
      "warnings": [],
  }
  for fnum, _, v in _iter_fields(mv):
    if fnum == 1:
      plane: dict[str, Any] = {
          "id": 0,
          "name": "",
          "lines": [],
          "event_metadata": {},
          "stat_metadata": {},
          "raw_stats": [],
      }
      for pfnum, _, pv in _iter_fields(v):
        if pfnum == 1:
          plane["id"] = _to_int64(pv)
        elif pfnum == 2:
          plane["name"] = bytes(pv).decode("utf-8", errors="replace")
        elif pfnum == 3:
          plane["lines"].append(_parse_line(pv))
        elif pfnum == 4:
          k, emd = _parse_map_entry(pv, _parse_event_metadata)
          plane["event_metadata"][k] = emd
        elif pfnum == 5:
          k, smd = _parse_map_entry(pv, _parse_stat_metadata)
          plane["stat_metadata"][k] = smd
        elif pfnum == 6:
          plane["raw_stats"].append(_parse_stat(pv))
      plane["stats"] = _resolve_stats(plane["raw_stats"], plane["stat_metadata"])
      for line in plane["lines"]:
        for ev in line["events"]:
          emd = plane["event_metadata"].get(ev["metadata_id"], {})
          ev["name"] = emd.get("name", "")
          ev["display_name"] = emd.get("display_name", "")
          ev["args"] = _resolve_stats(
              emd.get("stats", []) + ev["stats"], plane["stat_metadata"]
          )
      xspace["planes"].append(plane)
    elif fnum == 2:
      xspace["errors"].append(bytes(v).decode("utf-8", errors="replace"))
    elif fnum == 3:
      xspace["warnings"].append(bytes(v).decode("utf-8", errors="replace"))
    elif fnum == 4:
      xspace["hostnames"].append(bytes(v).decode("utf-8", errors="replace"))
  return xspace


def pct_stats(vals: list[float]) -> dict[str, Optional[float]]:
  if not vals:
    return {"p50": None, "p90": None, "mean": None, "count": 0}
  s = sorted(vals)
  n = len(s)
  p50 = s[n // 2] if n % 2 == 1 else 0.5 * (s[n // 2 - 1] + s[n // 2])
  idx90 = min(n - 1, int(round(0.9 * (n - 1))))
  return {
      "p50": round(p50, 4),
      "p90": round(s[idx90], 4),
      "mean": round(statistics.mean(s), 4),
      "count": n,
  }


def extract_cpu_events_aligned(xs: dict[str, Any]) -> list[dict[str, Any]]:
  """Extracts /host:CPU events with absolute UNIX timestamps (us) from Task Environment."""
  start_ns = 0
  for p in xs["planes"]:
    if p["name"] == "Task Environment":
      start_ns = int(p["stats"].get("profile_start_time", 0))
  evs = []
  for p in xs["planes"]:
    if p["name"] == "/host:CPU":
      for l in p["lines"]:
        for e in l["events"]:
          abs_ps = (start_ns + l["timestamp_ns"]) * 1000 + e["offset_ps"]
          evs.append({
              "name": e["name"],
              "line": l["name"],
              "abs_start_us": abs_ps / 1e6,
              "abs_end_us": (abs_ps + e["duration_ps"]) / 1e6,
              "dur_ms": max(0.0, e["duration_ps"] / 1e9),
              "args": e["args"],
          })
  evs.sort(key=lambda x: x["abs_start_us"])
  return evs


def _extract_tpu0_idle_stats(xs: dict[str, Any]) -> dict[str, Any]:
  """Computes inter-module idle gap (ms) and idle fraction (%) on /device:TPU:0."""
  tpu0_mods = []
  for p in xs["planes"]:
    if p["name"] == "/device:TPU:0":
      for l in p["lines"]:
        if l["name"] == "XLA Modules":
          lts_ps = l["timestamp_ns"] * 1000
          for e in l["events"]:
            if e["name"].startswith("jit_") or "(" in e["name"]:
              s_ms = (lts_ps + e["offset_ps"]) / 1e9
              d_ms = max(0.0, e["duration_ps"] / 1e9)
              if d_ms > 0:
                tpu0_mods.append((s_ms, s_ms + d_ms, d_ms))
  tpu0_mods.sort()
  gaps = []
  idle_pcts = []
  for i in range(len(tpu0_mods) - 1):
    gap = tpu0_mods[i + 1][0] - tpu0_mods[i][1]
    if gap >= 0:
      gaps.append(gap)
      step_total = gap + tpu0_mods[i][2]
      if step_total > 0:
        idle_pcts.append(min(100.0, (gap / step_total) * 100.0))
  return {
      "idle_gap_ms": pct_stats(gaps),
      "idle_pct": pct_stats(idle_pcts),
      "module_dur_ms": pct_stats([m[2] for m in tpu0_mods]),
  }


def parse_pathways_profile_dir(profile_dir: str) -> dict[str, Any]:
  """Parses the 4 Pathways .xplane.pb files in a single plugins/profile/<ts>/ directory."""
  pb_files = glob.glob(os.path.join(profile_dir, "*.xplane.pb"))
  head_pb = proxy_pb = rm_pb = None
  worker_pbs = []
  for p in pb_files:
    b = os.path.basename(p)
    if b.startswith("client."):
      proxy_pb = p
    elif b.startswith("server-") and any(
        b.endswith(f".{i}.{j}.xplane.pb") for i in range(8) for j in range(8)
    ):
      worker_pbs.append(p)
    elif b.startswith("server-"):
      rm_pb = p
    else:
      head_pb = p

  out: dict[str, Any] = {"worker_xplane_count": len(worker_pbs)}
  head_evs = proxy_evs = rm_evs = worker_evs = []

  if head_pb:
    with open(head_pb, "rb") as f:
      head_xs = parse_xspace(f.read())
    head_evs = extract_cpu_events_aligned(head_xs)
    for p in head_xs["planes"]:
      if p["name"] == "/host:metadata":
        out["pv01_xprof_tpu_version"] = p["stats"].get("tpu_version")

    # Pair DoRpc send/recv flows by (rpc_name, int(flow) >> 2)
    flows: dict[tuple[str, int], dict[str, dict[str, Any]]] = (
        collections.defaultdict(dict)
    )
    for e in head_evs:
      args = e["args"]
      if "flow" in args and "dir" in args:
        try:
          fid = int(args["flow"]) >> 2
          flows[(e["name"], fid)][str(args["dir"])] = e
        except (ValueError, TypeError):
          pass

    rpc_rtts: dict[str, list[float]] = collections.defaultdict(list)
    for (rpc_name, _), d in flows.items():
      if "send" in d and "recv" in d:
        rtt_ms = (d["recv"]["abs_start_us"] - d["send"]["abs_start_us"]) / 1000.0
        if 0.0 <= rtt_ms < 500.0:
          rpc_rtts[rpc_name].append(rtt_ms)

    out["cp02_execute_rpc_rtt_ms"] = pct_stats(
        rpc_rtts.get("loaded_executable_execute", [])
    )
    out["cp07_check_value_ready_rtt_ms"] = pct_stats(
        rpc_rtts.get("check_value_ready", [])
    )
    out["cp02_batch_destruct_rtt_ms"] = pct_stats(
        rpc_rtts.get("batch_destruct", [])
    )
    exec_cnt = max(1, len(rpc_rtts.get("loaded_executable_execute", [])))
    out["cp07_checks_per_execute"] = (
        len(rpc_rtts.get("check_value_ready", [])) / exec_cnt
    )

    exec_entry_durs = [
        e["dur_ms"]
        for e in head_evs
        if e["name"] == "IfrtProxyEntrypointLoadedExecutableExecute"
    ][:5]
    out["cp02_client_entrypoint_execute_ms"] = pct_stats(exec_entry_durs)

    stream_writes = [
        e["dur_ms"] for e in head_evs if e["name"] == "grpc_stream_write"
    ]
    out["cp05_grpc_stream_write_ms"] = pct_stats(stream_writes)
    out["cp06_proxy_periodic_flush_count"] = sum(
        1 for e in head_evs if e["name"] == "proxy_periodic_flush"
    )
    out["cp06_batch_destruct_count"] = sum(
        1 for e in head_evs if e["name"] in ("batch_destruct", "batch_delete")
    )

    h2d_spans = [
        e["dur_ms"]
        for e in head_evs
        if e["name"]
        in (
            "IfrtProxyEntrypointMakeArrayFromHostBuffer",
            "GrpcClientHostBufferStore::StoreSync",
            "GrpcClientHostBufferStore::StoreAsync_Send",
        )
    ]
    out["cp04_client_h2d_store_ms"] = pct_stats(h2d_spans)

  if proxy_pb:
    with open(proxy_pb, "rb") as f:
      proxy_xs = parse_xspace(f.read())
    proxy_evs = extract_cpu_events_aligned(proxy_xs)
    out["cp02_proxy_loaded_executable_execute_ms"] = pct_stats([
        e["dur_ms"] for e in proxy_evs if e["name"] == "LoadedExecutable::Execute"
    ])
    out["cp02_proxy_op_process_ms"] = pct_stats([
        e["dur_ms"]
        for e in proxy_evs
        if e["name"] == "ExecuteLoadedExecutableOp::Process"
    ][:5])
    out["dp04_proxy_transfer_ms"] = pct_stats([
        e["dur_ms"]
        for e in proxy_evs
        if e["name"]
        in (
            "Client::HostBufferBroadcastTransfer",
            "Client::HostBufferPartitionedTransfer",
            "Client::HostBufferOneToOneTransfer",
            "MakeShardedBufferFromShardedHostBufferOp::Process",
            "Client::MakeArrayFromHostBuffer",
        )
    ])

  if rm_pb:
    with open(rm_pb, "rb") as f:
      rm_xs = parse_xspace(f.read())
    rm_evs = extract_cpu_events_aligned(rm_xs)
    out["rm03_scheduler_handle_rpc_ms"] = pct_stats([
        e["dur_ms"]
        for e in rm_evs
        if e["name"] == "Scheduler::HandleRpcRequests"
    ][:5])
    out["rm03_scheduler_enqueue_rpcs_ms"] = pct_stats([
        e["dur_ms"] for e in rm_evs if e["name"] == "Scheduler::EnqueueRpcs"
    ][:5])

  if head_evs and proxy_evs:
    c_exec = [
        e
        for e in head_evs
        if e["name"] == "IfrtProxyEntrypointLoadedExecutableExecute"
    ][:5]
    p_exec = [
        e for e in proxy_evs if e["name"] == "LoadedExecutable::Execute"
    ][:5]
    ingress = [
        max(0.0, (p["abs_start_us"] - c["abs_end_us"]) / 1000.0)
        for c, p in zip(c_exec, p_exec)
        if p["abs_start_us"] >= c["abs_end_us"]
    ]
    out["cp02_client_to_proxy_ingress_ms"] = pct_stats(ingress)

  if worker_pbs:
    with open(sorted(worker_pbs)[0], "rb") as f:
      worker_xs = parse_xspace(f.read())
    worker_evs = extract_cpu_events_aligned(worker_xs)
    pe_d0 = [
        e["dur_ms"]
        for e in worker_evs
        if e["name"] == "ProcessExamples"
        and e["args"].get("device_ordinal") == 0
        and "xprof_component" not in str(e["args"].get("component", ""))
    ]
    out["wk12_worker_process_examples_ms"] = pct_stats(pe_d0)
    enq_exec = [
        e["dur_ms"]
        for e in worker_evs
        if e["name"] == "WorkerOpDispatcher::EnqueueWork"
        and e["args"].get("rpc_type") == "execute"
    ][:20]
    out["rm03_worker_enqueue_work_ms"] = pct_stats(enq_exec)

    mem_allocs = [e for e in worker_evs if e["name"] == "MemoryAllocation"]
    if mem_allocs:
      out["wk10_worker_peak_bytes_in_use"] = max(
          int(e["args"].get("peak_bytes_in_use", 0)) for e in mem_allocs
      )
      out["wk10_worker_bytes_allocated"] = max(
          int(e["args"].get("bytes_allocated", 0)) for e in mem_allocs
      )
      out["wk10_worker_bytes_available"] = min(
          int(e["args"].get("bytes_available", 0)) for e in mem_allocs
      )

    idle_info = _extract_tpu0_idle_stats(worker_xs)
    out["cp12_tpu0_idle_gap_ms"] = idle_info["idle_gap_ms"]
    out["cp12_tpu0_idle_pct"] = idle_info["idle_pct"]

  return out


def parse_mcjax_profile_dir(profile_dir: str) -> dict[str, Any]:
  """Parses McJAX .xplane.pb files in a plugins/profile/<ts>/ directory."""
  pb_files = sorted(glob.glob(os.path.join(profile_dir, "*.xplane.pb")))
  out: dict[str, Any] = {"host_xplane_count": len(pb_files)}
  if not pb_files:
    return out
  host_mod_p50s: list[float] = []
  for idx, pb in enumerate(pb_files):
    try:
      with open(pb, "rb") as f:
        xs = parse_xspace(f.read())
      if idx == 0:
        for p in xs["planes"]:
          if p["name"] == "/host:metadata":
            out["pv01_xprof_tpu_version"] = p["stats"].get("tpu_version")
        idle_info = _extract_tpu0_idle_stats(xs)
        out["cp12_tpu0_idle_gap_ms"] = idle_info["idle_gap_ms"]
        out["cp12_tpu0_idle_pct"] = idle_info["idle_pct"]
        if idle_info["module_dur_ms"]["p50"] is not None:
          host_mod_p50s.append(float(idle_info["module_dur_ms"]["p50"]))
      else:
        idle_info = _extract_tpu0_idle_stats(xs)
        if idle_info["module_dur_ms"]["p50"] is not None:
          host_mod_p50s.append(float(idle_info["module_dur_ms"]["p50"]))
    except Exception:  # pylint: disable=broad-exception-caught
      pass
  if host_mod_p50s:
    out["wk14_slowest_host_module_ms"] = pct_stats(host_mod_p50s)
    out["wk14_max_host_p50_ms"] = round(max(host_mod_p50s), 4)
  return out
