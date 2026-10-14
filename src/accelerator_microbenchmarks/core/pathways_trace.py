"""Reads Pathways on Cloud XProf traces (XSpace protos on GCS) as trace events.

On Pathways on Cloud, `jax.profiler.trace(<gs:// dir>)` is patched by
pathwaysutils: the Pathways resource manager and TPU workers record the trace
and each writes one `*.xplane.pb` file to `<dir>/plugins/profile/<session>/`.
The TPU device timelines are in the worker files
(`server-<jobset>-pathways-worker-<job>-<pod>...xplane.pb`); the TPUMS client
only records its own host activity.

TPUMS' step-duration logic (`core/profiler.py`) consumes Chrome trace events
(the `*.trace.json.gz` that McJAX runs produce locally). This module builds the
equivalent events from the TPU device planes of the worker XSpaces, mirroring
XProf's trace-viewer conversion:

* One `process_name` metadata event per TPU device plane, named
  `/device:TPU:<global core id>` using the plane's `core_details` stat. Worker
  planes are named by host-local ordinal (`/device:TPU:0..3` on every host), so
  renaming them makes planes from different hosts distinct and lets callers
  look them up by JAX device id, which on Pathways is the global core id.
* One complete event per XEvent with `ts`/`dur` in microseconds, the event's
  display name as `name` (full name in `args.long_name`), and the event-metadata
  and event stats (e.g. `tf_op`, `hlo_category`, `device_duration_ps`) in
  `args`.
"""

from __future__ import annotations

import functools
import os
import time
from typing import Any, Iterable

TPU_DEVICE_PLANE_PREFIX = "/device:TPU:"
_CORE_DETAILS_STAT = "core_details"
# `CoreDetails.global_core_id` in XProf's op_stats.proto ("unique within mesh").
_GLOBAL_CORE_ID_FIELD = 6
_PS_PER_NS = 1000
_PS_PER_US = 1e6
_POLL_INTERVAL_S = 2.0


def _read_varint(buf: bytes, pos: int) -> tuple[int, int]:
  """Reads a base-128 varint from `buf` at `pos`; returns (value, new_pos)."""
  value, shift = 0, 0
  while True:
    byte = buf[pos]
    pos += 1
    value |= (byte & 0x7F) << shift
    shift += 7
    if not byte & 0x80:
      return value, pos


def decode_varint_fields(buf: bytes) -> dict[int, int]:
  """Returns the top-level varint fields of a serialized protobuf message.

  Used for `core_details`, a serialized XProf `CoreDetails` proto stored as a
  bytes stat, without depending on XProf's proto definitions. Length-delimited
  fields are skipped; parsing stops at the first unsupported wire type.

  Args:
    buf: The serialized message.

  Returns:
    Mapping of field number to value for every varint field. Proto3 omits
    zero-valued fields, so absent fields mean 0.
  """
  fields: dict[int, int] = {}
  pos = 0
  try:
    while pos < len(buf):
      key, pos = _read_varint(buf, pos)
      field_number, wire_type = key >> 3, key & 0x7
      if wire_type == 0:  # varint
        fields[field_number], pos = _read_varint(buf, pos)
      elif wire_type == 2:  # length-delimited
        length, pos = _read_varint(buf, pos)
        pos += length
      else:
        break
  except IndexError:
    pass  # Truncated message: keep what was decoded.
  return fields


def _stat_value(stat: Any, plane: Any) -> Any:
  """Returns a JSON-friendly value of an XStat, or None for bytes values."""
  kind = stat.WhichOneof("value")
  if kind == "ref_value":
    metadata = plane.stat_metadata.get(stat.ref_value)
    return metadata.name if metadata is not None else None
  if kind in ("str_value", "int64_value", "uint64_value", "double_value"):
    return getattr(stat, kind)
  return None


def device_id_of_plane(plane: Any) -> int:
  """Returns the global core id of a `/device:TPU:<ordinal>` plane.

  Falls back to the host-local ordinal in the plane name when the plane has no
  `core_details` stat (e.g. traces from older runtimes).

  Args:
    plane: An `XPlane` whose name starts with `/device:TPU:`.
  """
  for stat in plane.stats:
    metadata = plane.stat_metadata.get(stat.metadata_id)
    if (
        metadata is not None
        and metadata.name == _CORE_DETAILS_STAT
        and stat.WhichOneof("value") == "bytes_value"
    ):
      return decode_varint_fields(stat.bytes_value).get(
          _GLOBAL_CORE_ID_FIELD, 0
      )
  return int(plane.name[len(TPU_DEVICE_PLANE_PREFIX) :])


def xspace_to_trace_events(
    xspaces: Iterable[Any], source_names: Iterable[str] | None = None
) -> list[dict[str, Any]]:
  """Converts the TPU device planes of XSpaces into Chrome trace events.

  Args:
    xspaces: Parsed `XSpace` protos, e.g. one per Pathways worker host.
    source_names: Optional label per XSpace (e.g. file name), stored in the
      `process_name` event args for debugging.

  Returns:
    The trace events. Empty if no XSpace has a TPU device plane.
  """
  xspaces = list(xspaces)
  names = list(source_names) if source_names is not None else []
  device_planes = []
  for index, xspace in enumerate(xspaces):
    source = names[index] if index < len(names) else str(index)
    for plane in xspace.planes:
      if plane.name.startswith(TPU_DEVICE_PLANE_PREFIX):
        device_planes.append((source, plane))

  # Express timestamps relative to the earliest line so that microsecond
  # floats keep sub-nanosecond precision even for epoch-based line timestamps.
  line_timestamps = [
      line.timestamp_ns
      for _, plane in device_planes
      for line in plane.lines
      if line.events
  ]
  base_ns = min(line_timestamps) if line_timestamps else 0

  events: list[dict[str, Any]] = []
  for pid, (source, plane) in enumerate(device_planes, start=1):
    events.append({
        "ph": "M",
        "name": "process_name",
        "pid": pid,
        "args": {
            "name": f"{TPU_DEVICE_PLANE_PREFIX}{device_id_of_plane(plane)}",
            "host_plane": plane.name,
            "source": source,
        },
    })
    stat_names = {
        metadata_id: metadata.name
        for metadata_id, metadata in plane.stat_metadata.items()
    }
    for line in plane.lines:
      line_start_ps = (line.timestamp_ns - base_ns) * _PS_PER_NS
      for event in line.events:
        metadata = plane.event_metadata.get(event.metadata_id)
        if metadata is None:
          continue
        args: dict[str, Any] = {}
        # Metadata stats first, then per-occurrence stats (as XProf does).
        for stat in list(metadata.stats) + list(event.stats):
          value = _stat_value(stat, plane)
          if value is not None:
            args[stat_names.get(stat.metadata_id, str(stat.metadata_id))] = (
                value
            )
        if metadata.display_name:
          name = metadata.display_name
          args["long_name"] = metadata.name
        else:
          name = metadata.name
        events.append({
            "ph": "X",
            "pid": pid,
            "tid": line.id,
            "name": name,
            "ts": (line_start_ps + event.offset_ps) / _PS_PER_US,
            "dur": event.duration_ps / _PS_PER_US,
            "args": args,
        })
  return events


def _list_xplane_files(xprof_dir: str) -> list[str]:
  """Lists the `*.xplane.pb` files of all profile sessions under xprof_dir."""
  from tensorflow.io import gfile  # pylint: disable=g-import-not-at-top

  pattern = os.path.join(xprof_dir, "plugins", "profile", "*", "*.xplane.pb")
  return sorted(gfile.glob(pattern))


def _read_xspace(path: str) -> Any:
  """Reads and parses an XSpace proto from a local or gs:// path."""
  from tensorflow.io import gfile  # pylint: disable=g-import-not-at-top
  from tensorflow.tsl.profiler.protobuf import xplane_pb2  # pylint: disable=g-direct-tensorflow-import,g-import-not-at-top

  xspace = xplane_pb2.XSpace()
  with gfile.GFile(path, "rb") as f:
    xspace.ParseFromString(f.read())
  return xspace


@functools.lru_cache(maxsize=4)
def load_trace(xprof_dir: str, wait_s: float = 30.0) -> dict[str, Any] | None:
  """Loads the TPU device timelines that Pathways wrote under `xprof_dir`.

  Results are cached per directory because device-selection modes may parse
  the same trace once per device.

  Args:
    xprof_dir: The directory passed to `jax.profiler.trace`, usually gs://.
    wait_s: How long to keep polling for worker traces, in case they are still
      being uploaded when this is called.

  Returns:
    A Chrome-trace-like dict (`{"traceEvents": [...]}`) or None if no TPU
    device plane was found.
  """
  deadline = time.monotonic() + wait_s
  while True:
    paths = _list_xplane_files(xprof_dir)
    xspaces, sources = [], []
    for path in paths:
      try:
        xspace = _read_xspace(path)
      except Exception as e:  # pylint: disable=broad-exception-caught
        print(f"Warning: failed to read XProf trace {path}: {e}")
        continue
      if any(p.name.startswith(TPU_DEVICE_PLANE_PREFIX) for p in xspace.planes):
        xspaces.append(xspace)
        sources.append(os.path.basename(path))
    if xspaces or time.monotonic() >= deadline:
      break
    time.sleep(_POLL_INTERVAL_S)

  if not xspaces:
    print(
        f"No Pathways TPU device traces found under {xprof_dir} (looked at"
        f" {len(paths)} *.xplane.pb file(s))."
    )
    return None
  events = xspace_to_trace_events(xspaces, sources)
  print(
      f"Loaded Pathways device traces from {', '.join(sources)}"
      f" ({sum(1 for e in events if e.get('ph') == 'M')} TPU device planes)."
  )
  return {"traceEvents": events}
