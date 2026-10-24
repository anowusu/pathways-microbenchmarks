"""Pathways on Cloud support.

With `JAX_PLATFORMS=proxy` a single TPUMS process (the Pathways client) drives
every TPU in the slice through the Pathways proxy. `pathwaysutils` registers
that backend and redirects `jax.profiler.trace(<gs:// dir>)` to the Pathways
servers, which write XSpace protos (`*.xplane.pb`) to GCS instead of the local
`.trace.json.gz` that `profiler.py` parses; `load_trace` bridges that gap.
"""

import json
import os
import time
from typing import Any

# A worker's trace is `server-<host>.<n>.<m>.xplane.pb`; the head's own server
# trace, `server-<host>.xplane.pb`, has no numeric suffix. The host is the pod
# name with pathwaysutils JobSets and the node name with gcluster's.
_WORKER_TRACE_GLOB = "plugins/profile/*/server-*.[0-9]*.[0-9]*.xplane.pb"


def is_active() -> bool:
  """Whether JAX runs on the Pathways proxy backend."""
  return "proxy" in os.environ.get("JAX_PLATFORMS", "")


def initialize() -> None:
  """Registers the Pathways proxy backend; a no-op for McJAX runs."""
  if is_active():
    import pathwaysutils  # pylint: disable=g-import-not-at-top

    pathwaysutils.initialize()


def load_trace(xprof_dir: str, wait_s: float = 60.0) -> dict[str, Any] | None:
  """Returns the TPU device trace Pathways wrote under `xprof_dir` (gs://).

  Pathways profiles the worker that hosts JAX device 0, so its planes
  `/device:TPU:<n>` are that host's chips and device 0 is `/device:TPU:0`.
  The XSpace is converted with XProf's trace-viewer converter, i.e. into the
  same events a McJAX `.trace.json.gz` holds. Waits up to `wait_s` for the
  worker to finish uploading the trace.
  """
  from tensorflow.io import gfile  # pylint: disable=g-import-not-at-top
  from tensorflow.python.profiler.internal import _pywrap_profiler_plugin  # pylint: disable=g-import-not-at-top

  pattern = os.path.join(xprof_dir, _WORKER_TRACE_GLOB)
  deadline = time.monotonic() + wait_s
  while not (paths := gfile.glob(pattern)) and time.monotonic() < deadline:
    time.sleep(2)
  if not paths:
    print(f"No Pathways worker trace found under {xprof_dir}")
    return None
  data, _ = _pywrap_profiler_plugin.xspace_to_tools_data(
      paths, "trace_viewer@", {}
  )
  trace = json.loads(data)
  for event in trace["traceEvents"]:
    if event.get("name") == "process_name":
      # "<host> /device:TPU:0" -> "/device:TPU:0", as in a local trace.
      event["args"]["name"] = event["args"]["name"].split(" ")[-1]
  return trace
