"""Unit tests for core/pathways_trace.py (Pathways XSpace trace parsing)."""

import os
import shutil
import tempfile
from unittest import mock

from absl.testing import absltest
from accelerator_microbenchmarks.core import constants
from accelerator_microbenchmarks.core import executor
from accelerator_microbenchmarks.core import pathways
from accelerator_microbenchmarks.core import pathways_trace
from accelerator_microbenchmarks.core import profiler
from tensorflow.tsl.profiler.protobuf import xplane_pb2  # pylint: disable=g-direct-tensorflow-import

_PS_PER_MS = 1_000_000_000

# Stat metadata ids used by the synthetic planes.
_CORE_DETAILS, _TF_OP, _HLO_CATEGORY, _DEVICE_DURATION_PS = 1, 2, 3, 4
# Event metadata ids.
_MODULE, _ALL_GATHER, _COPY = 1, 2, 3


def _core_details(local_ordinal: int, global_core_id: int) -> bytes:
  """Serializes a CoreDetails proto (proto3 omits zero-valued fields)."""
  out = b""
  for field, value in ((2, local_ordinal), (5, global_core_id),
                       (6, global_core_id)):
    if value:
      out += bytes([field << 3, value])
  return out


def _add_device_plane(
    xspace, local_ordinal: int, global_core_id: int | None, op_ms: float
):
  """Adds a TPU plane with 3 module runs, each one MARKER op + one copy."""
  plane = xspace.planes.add()
  plane.id = local_ordinal + 2
  plane.name = f"/device:TPU:{local_ordinal}"
  for stat_id, name in (
      (_CORE_DETAILS, "core_details"),
      (_TF_OP, "tf_op"),
      (_HLO_CATEGORY, "hlo_category"),
      (_DEVICE_DURATION_PS, "device_duration_ps"),
  ):
    plane.stat_metadata[stat_id].id = stat_id
    plane.stat_metadata[stat_id].name = name
  if global_core_id is not None:
    stat = plane.stats.add()
    stat.metadata_id = _CORE_DETAILS
    stat.bytes_value = _core_details(local_ordinal, global_core_id)

  module = plane.event_metadata[_MODULE]
  module.id, module.name = _MODULE, "jit_all_gather_sharded(123)"
  all_gather = plane.event_metadata[_ALL_GATHER]
  all_gather.id = _ALL_GATHER
  all_gather.name = "%all-gather.4 = bf16[32,8,128] all-gather(...)"
  all_gather.display_name = "all-gather.4"
  stat = all_gather.stats.add()
  stat.metadata_id, stat.str_value = _TF_OP, (
      f"jit(all_gather_sharded)/shard_map/{constants.MARKER}/all_gather:"
  )
  stat = all_gather.stats.add()
  stat.metadata_id, stat.str_value = _HLO_CATEGORY, "all-gather"
  copy = plane.event_metadata[_COPY]
  copy.id, copy.name, copy.display_name = _COPY, "%copy.3 = copy(...)", "copy.3"

  modules = plane.lines.add()
  modules.id, modules.name = 2, "XLA Modules"
  ops = plane.lines.add()
  ops.id, ops.name = 3, "XLA Ops"
  for step in range(3):
    start_ps = step * 20 * _PS_PER_MS
    event = modules.events.add()
    event.metadata_id = _MODULE
    event.offset_ps = start_ps
    event.duration_ps = int((op_ms + 3) * _PS_PER_MS)
    event = ops.events.add()
    event.metadata_id = _ALL_GATHER
    event.offset_ps = start_ps + _PS_PER_MS
    event.duration_ps = int(op_ms * _PS_PER_MS)
    stat = event.stats.add()
    stat.metadata_id, stat.int64_value = _DEVICE_DURATION_PS, event.duration_ps
    event = ops.events.add()
    event.metadata_id = _COPY
    event.offset_ps = start_ps + int((op_ms + 1) * _PS_PER_MS)
    event.duration_ps = 2 * _PS_PER_MS
  return plane


def _worker_xspace() -> xplane_pb2.XSpace:
  """A worker trace whose host holds global devices 0 and 5."""
  xspace = xplane_pb2.XSpace()
  xspace.planes.add().name = "/host:CPU"
  _add_device_plane(xspace, local_ordinal=0, global_core_id=0, op_ms=5.0)
  _add_device_plane(xspace, local_ordinal=1, global_core_id=5, op_ms=6.0)
  return xspace


class PathwaysTraceTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    pathways_trace.load_trace.cache_clear()
    self.addCleanup(pathways_trace.load_trace.cache_clear)

  def _write_trace_dir(self, xspaces: dict[str, xplane_pb2.XSpace]) -> str:
    # tempfile rather than create_tempdir(): works under pytest without absl
    # flag parsing.
    root = tempfile.mkdtemp()
    self.addCleanup(shutil.rmtree, root, ignore_errors=True)
    session = os.path.join(root, "plugins", "profile", "2026_10_14_23_16_04")
    os.makedirs(session)
    for name, xspace in xspaces.items():
      with open(os.path.join(session, name), "wb") as f:
        f.write(xspace.SerializeToString())
    return root

  def test_decode_varint_fields(self):
    """Decodes a real core_details value from a v5e Pathways worker trace."""
    self.assertEqual(
        pathways_trace.decode_varint_fields(b"\x10\x03 \x03(\x050\x05"),
        {2: 3, 4: 3, 5: 5, 6: 5},
    )
    self.assertEqual(pathways_trace.decode_varint_fields(b""), {})
    # Truncated input keeps the fields decoded so far.
    self.assertEqual(pathways_trace.decode_varint_fields(b"\x10\x03\x28"), {2: 3})

  def test_device_id_of_plane(self):
    xspace = xplane_pb2.XSpace()
    with_details = _add_device_plane(xspace, 3, 7, op_ms=1.0)
    zero_details = _add_device_plane(xspace, 2, 0, op_ms=1.0)
    no_details = _add_device_plane(xspace, 1, None, op_ms=1.0)
    self.assertEqual(pathways_trace.device_id_of_plane(with_details), 7)
    # An empty CoreDetails means global_core_id == 0 (proto3 default).
    self.assertEqual(pathways_trace.device_id_of_plane(zero_details), 0)
    # Without core_details, fall back to the host-local ordinal.
    self.assertEqual(pathways_trace.device_id_of_plane(no_details), 1)

  def test_xspace_to_trace_events(self):
    events = pathways_trace.xspace_to_trace_events(
        [_worker_xspace()], ["worker.xplane.pb"]
    )
    processes = {
        e["pid"]: e["args"] for e in events if e["name"] == "process_name"
    }
    self.assertEqual(
        {args["name"] for args in processes.values()},
        {"/device:TPU:0", "/device:TPU:5"},
    )
    self.assertEqual(processes[2]["host_plane"], "/device:TPU:1")
    self.assertEqual(processes[2]["source"], "worker.xplane.pb")

    ops = [e for e in events if e.get("name") == "all-gather.4"]
    self.assertLen(ops, 6)
    first = min(ops, key=lambda e: (e["pid"], e["ts"]))
    self.assertEqual(first["ts"], 1000.0)  # 1 ms in microseconds.
    self.assertEqual(first["dur"], 5000.0)
    self.assertIn(constants.MARKER, first["args"]["tf_op"])
    self.assertEqual(first["args"]["hlo_category"], "all-gather")
    self.assertEqual(first["args"]["device_duration_ps"], 5 * _PS_PER_MS)
    self.assertTrue(first["args"]["long_name"].startswith("%all-gather.4"))

  def test_load_trace_and_parse_durations_by_global_device_id(self):
    client = xplane_pb2.XSpace()
    client.planes.add().name = "/host:CPU"
    trace_dir = self._write_trace_dir({
        "client.head.xplane.pb": client,
        "server-jobset-pathways-worker-0-4.0.0.xplane.pb": _worker_xspace(),
    })
    with mock.patch.object(pathways, "_initialized", True):
      dev0 = profiler.parse_xprof_durations(trace_dir, local_device_id=0)
      dev5 = profiler.parse_xprof_durations(trace_dir, local_device_id=5)
      # A device on a host that was not profiled falls back to the first
      # device plane with MARKER events.
      dev9 = profiler.parse_xprof_durations(trace_dir, local_device_id=9)
    for durations in (dev0, dev9):
      self.assertSequenceAlmostEqual(durations, [5.0, 5.0, 5.0])
    self.assertSequenceAlmostEqual(dev5, [6.0, 6.0, 6.0])

  def test_load_trace_without_device_planes(self):
    client = xplane_pb2.XSpace()
    client.planes.add().name = "/host:CPU"
    trace_dir = self._write_trace_dir({"client.head.xplane.pb": client})
    self.assertIsNone(pathways_trace.load_trace(trace_dir, wait_s=0))
    empty_dir = tempfile.mkdtemp()
    self.addCleanup(shutil.rmtree, empty_dir, ignore_errors=True)
    self.assertIsNone(pathways_trace.load_trace(empty_dir, wait_s=0))


class TraceDeviceIdTest(absltest.TestCase):

  def test_trace_device_id(self):
    device = mock.MagicMock(id=12, local_hardware_id=3)
    with mock.patch.object(pathways, "_initialized", False):
      self.assertEqual(executor._trace_device_id(device), 3)
    with mock.patch.object(pathways, "_initialized", True):
      self.assertEqual(executor._trace_device_id(device), 12)


if __name__ == "__main__":
  absltest.main()
