"""Unit tests for pathways.py."""

import json
import os
import sys
import types
from unittest import mock

from absl.testing import absltest
from accelerator_microbenchmarks.core import pathways
from accelerator_microbenchmarks.core import profiler

_WORKER_TRACE = "gs://b/x/plugins/profile/s/server-j-pathways-worker-0-2.xplane.pb"


def _fake_tensorflow(xplane_paths, trace):
  """Returns sys.modules entries faking what `load_trace` uses of TensorFlow."""
  plugin = mock.Mock()
  plugin.xspace_to_tools_data.return_value = (json.dumps(trace).encode(), True)
  return {
      "tensorflow": types.ModuleType("tensorflow"),
      "tensorflow.io": types.SimpleNamespace(
          gfile=mock.Mock(**{"glob.return_value": xplane_paths})
      ),
      "tensorflow.python.profiler.internal": types.SimpleNamespace(
          _pywrap_profiler_plugin=plugin
      ),
  }


class PathwaysTest(absltest.TestCase):

  def test_is_active_follows_jax_platforms(self):
    with mock.patch.dict(os.environ, {"JAX_PLATFORMS": "proxy"}):
      self.assertTrue(pathways.is_active())
    with mock.patch.dict(os.environ, {"JAX_PLATFORMS": "tpu,cpu"}):
      self.assertFalse(pathways.is_active())

  def test_initialize_only_registers_proxy_backend_when_requested(self):
    fake_pathwaysutils = types.SimpleNamespace(initialize=mock.Mock())
    with mock.patch.dict(sys.modules, {"pathwaysutils": fake_pathwaysutils}):
      with mock.patch.dict(os.environ, {"JAX_PLATFORMS": "tpu"}):
        pathways.initialize()
      fake_pathwaysutils.initialize.assert_not_called()
      with mock.patch.dict(os.environ, {"JAX_PLATFORMS": "proxy"}):
        pathways.initialize()
      fake_pathwaysutils.initialize.assert_called_once()

  def test_load_trace_names_processes_like_a_local_trace(self):
    process = {"ph": "M", "pid": 1, "name": "process_name"}
    trace = {
        "traceEvents": [
            {**process, "args": {"name": "worker-0-2 /device:TPU:0"}},
            {"ph": "X", "pid": 1, "name": "jit_fn", "ts": 1.0, "dur": 2.0},
        ]
    }
    with mock.patch.dict(sys.modules, _fake_tensorflow([_WORKER_TRACE], trace)):
      loaded = pathways.load_trace("gs://b/x")
    self.assertEqual(
        loaded["traceEvents"],
        [{**process, "args": {"name": "/device:TPU:0"}}] + trace["traceEvents"][1:],
    )
    self.assertEqual(profiler._get_target_pid(loaded, local_device_id=0), 1)  # pylint: disable=protected-access

  def test_load_trace_without_worker_trace(self):
    with mock.patch.dict(sys.modules, _fake_tensorflow([], {})):
      self.assertIsNone(pathways.load_trace("gs://b/x", wait_s=0))

  def test_profiler_reads_gcs_traces_through_pathways(self):
    with mock.patch.object(pathways, "load_trace") as load_trace:
      trace = profiler._load_xprof_trace("gs://b/x")  # pylint: disable=protected-access
    load_trace.assert_called_once_with("gs://b/x")
    self.assertIs(trace, load_trace.return_value)


if __name__ == "__main__":
  absltest.main()
