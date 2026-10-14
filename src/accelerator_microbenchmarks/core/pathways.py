"""Pathways on Cloud support for TPUMS.

TPUMS normally runs as multi-controller JAX ("McJAX"): every TPU VM runs its
own copy of the benchmark, and `jax.distributed.initialize()` stitches the
copies together. On Pathways on Cloud there is exactly one Python process (the
"client"). It runs on a CPU node and drives every TPU device in the slice(s)
through the IFRT proxy. JAX selects this mode when `JAX_PLATFORMS` contains
`proxy`, after `pathwaysutils.initialize()` has registered the proxy backend.

This module keeps the Pathways-specific behavior in one place:

* `maybe_initialize()` registers the proxy backend through pathwaysutils
  before JAX creates any backend. It does nothing for McJAX runs.
* `is_active()` lets the rest of TPUMS branch between single-controller
  (Pathways) and multi-controller (McJAX) behavior.
* `wait_for_devices()` fails fast instead of hanging when the Pathways slice
  never becomes ready.
* `runtime_name()` / `runtime_version()` describe the runtime that actually
  ran the TPU programs, so reports can tell the two apart.
* `resolve_xprof_dir()` adapts XProf trace collection. pathwaysutils routes
  `jax.profiler` traces to the Pathways resource manager and workers, which
  can only write to GCS. `trace_start_delay_s()` gives their device tracers
  time to start. The device traces are read back by `core/pathways_trace.py`.
"""

from __future__ import annotations

import os
import sys
import threading
from typing import Any

import jax

PROXY_PLATFORM = "proxy"
RUNTIME_MCJAX = "mcjax"
RUNTIME_PATHWAYS = "pathways"
# Pathways server/proxy image tag, set by cloud_launch/pathways.
PATHWAYS_VERSION_ENV = "TPUMS_PATHWAYS_VERSION"
TRACE_START_DELAY_ENV = "TPUMS_PATHWAYS_TRACE_START_DELAY_S"
_DEFAULT_TRACE_START_DELAY_S = 1.0
STARTUP_TIMEOUT_ENV = "TPUMS_PATHWAYS_STARTUP_TIMEOUT_S"
_DEFAULT_STARTUP_TIMEOUT_S = 900.0
# Exit code used when the Pathways backend never reports devices.
STARTUP_TIMEOUT_EXIT_CODE = 75

_initialized = False


def _requested_platforms() -> list[str]:
  """Returns the lower-cased entries of `JAX_PLATFORMS` (env or jax.config)."""
  raw = os.environ.get("JAX_PLATFORMS") or jax.config.jax_platforms or ""
  return [p.strip().lower() for p in str(raw).split(",") if p.strip()]


def is_requested() -> bool:
  """Whether this process is configured to use the Pathways proxy backend."""
  return PROXY_PLATFORM in _requested_platforms()


def maybe_initialize() -> bool:
  """Initializes Pathways on Cloud support if `JAX_PLATFORMS` requests it.

  Must run before anything calls `jax.devices()` or otherwise creates a JAX
  backend, because the proxy backend factory has to be registered first.
  Calling it more than once is safe.

  Returns:
    True if this process is a Pathways (single-controller) client.

  Raises:
    ImportError: If `JAX_PLATFORMS` requests `proxy` but pathwaysutils is not
      installed.
  """
  global _initialized
  if _initialized:
    return True
  if not is_requested():
    return False
  try:
    import pathwaysutils  # pylint: disable=g-import-not-at-top
  except ImportError as e:
    raise ImportError(
        "JAX_PLATFORMS requests the Pathways 'proxy' backend, but"
        " pathwaysutils is not installed. Install TPUMS with the Pathways"
        " extra: pip install 'accelerator_microbenchmarks[pathways]'"
    ) from e
  pathwaysutils.initialize()
  _initialized = True
  print(
      "Pathways on Cloud: initialized pathwaysutils (single-controller client,"
      f" JAX_BACKEND_TARGET={os.environ.get('JAX_BACKEND_TARGET', '<unset>')})."
  )
  return True


def is_active() -> bool:
  """Whether TPUMS is running as a Pathways (single-controller) client."""
  return _initialized


def runtime_name() -> str:
  """Returns "pathways" for Pathways clients and "mcjax" otherwise."""
  return RUNTIME_PATHWAYS if is_active() else RUNTIME_MCJAX


def runtime_version() -> str:
  """Returns the version reported by the runtime that runs the TPU programs.

  For Pathways this is the Pathways server build (for example
  "Pathways cl/123456"), forwarded to the client by the IFRT proxy, followed by
  the Pathways image tag when the launcher provides it in
  `TPUMS_PATHWAYS_VERSION` (Cloud builds may report a placeholder build
  number). For McJAX it is the PJRT/libtpu platform version. Returns "unknown"
  if the backend cannot be queried.
  """
  try:
    from jax.extend import backend as jex_backend  # pylint: disable=g-import-not-at-top

    version = jex_backend.get_backend().platform_version
  except Exception:  # pylint: disable=broad-exception-caught
    return "unknown"
  # Pathways reports a multi-line string ("Pathways\ncl/123"); flatten it so it
  # stays on one line in CSV/JSON reports.
  version = " ".join(str(version).split())
  image_tag = os.environ.get(PATHWAYS_VERSION_ENV, "").strip()
  if is_active() and image_tag:
    version = f"{version}; images {image_tag}"
  return version


def _env_seconds(name: str, default: float) -> float:
  """Reads a non-negative number of seconds from env var `name`."""
  raw = os.environ.get(name, "")
  try:
    return max(0.0, float(raw)) if raw else default
  except ValueError:
    print(f"Warning: ignoring invalid {name}={raw!r}; using {default}s.")
    return default


def trace_start_delay_s() -> float:
  """Seconds to wait after starting a Pathways trace, before timed steps.

  Pathways enables the TPU device tracers asynchronously after
  `jax.profiler.start_trace` returns; without a pause the first measured step
  is missing from (or clipped in) the device trace. Configurable through
  `TPUMS_PATHWAYS_TRACE_START_DELAY_S` (default 1 second, 0 disables).
  """
  return _env_seconds(TRACE_START_DELAY_ENV, _DEFAULT_TRACE_START_DELAY_S)


def wait_for_devices(timeout_s: float | None = None) -> list[jax.Device]:
  """Returns `jax.devices()`, exiting the process if Pathways never answers.

  A Pathways client blocks in `jax.devices()` until the resource manager
  reports the TPU slice(s) as ready. If that never happens (for example a
  resource manager that restarted after its JobSet was preempted and
  re-admitted), the client would hang until the Job's deadline while holding
  the slice. This probes the backend from a daemon thread and, after
  `timeout_s` (default `TPUMS_PATHWAYS_STARTUP_TIMEOUT_S`, 900 s; 0 waits
  forever), exits with `STARTUP_TIMEOUT_EXIT_CODE` so that the JobSet fails
  fast and releases the TPUs.

  Args:
    timeout_s: Seconds to wait for the devices; None reads the environment.

  Returns:
    The JAX devices.
  """
  if timeout_s is None:
    timeout_s = _env_seconds(STARTUP_TIMEOUT_ENV, _DEFAULT_STARTUP_TIMEOUT_S)
  if timeout_s <= 0:
    return jax.devices()

  done = threading.Event()
  result: dict[str, Any] = {}

  def _probe() -> None:
    try:
      result["devices"] = jax.devices()
    except BaseException as e:  # pylint: disable=broad-exception-caught
      result["error"] = e
    finally:
      done.set()

  threading.Thread(
      target=_probe, name="pathways-device-probe", daemon=True
  ).start()
  if not done.wait(timeout_s):
    print(
        "ERROR: Pathways on Cloud did not report any TPU devices within"
        f" {timeout_s:g}s (JAX_BACKEND_TARGET="
        f"{os.environ.get('JAX_BACKEND_TARGET', '<unset>')}). The Pathways"
        " resource manager or workers are probably not ready; check the"
        " pathways-rm, pathways-proxy and pathways-worker logs. Exiting so"
        " that the TPU slice is released.",
        flush=True,
    )
    sys.stderr.flush()
    # A clean exit could block on the backend thread, so exit immediately.
    os._exit(STARTUP_TIMEOUT_EXIT_CODE)  # pylint: disable=protected-access
  if "error" in result:
    raise result["error"]
  return result["devices"]


def is_gcs_path(path: str | None) -> bool:
  """Whether `path` is a Google Cloud Storage URI."""
  return bool(path) and str(path).startswith("gs://")


def resolve_xprof_dir(trace_dir: str) -> str | None:
  """Returns the directory to collect an XProf trace into, or None to skip it.

  On McJAX the directory is returned unchanged. On Pathways, pathwaysutils
  patches `jax.profiler.start_trace` so that the resource manager and TPU
  workers record the trace themselves, and they can only write to GCS: a
  non-GCS directory makes `start_trace` raise `ValueError`. In that case XProf
  collection is skipped with a warning and only wall-clock metrics are
  reported.

  Args:
    trace_dir: The per-benchmark trace directory TPUMS would normally use.

  Returns:
    `trace_dir`, or None if the trace cannot be collected on this runtime.
  """
  if not is_active() or is_gcs_path(trace_dir):
    return trace_dir
  print(
      "Warning: XProf trace collection on Pathways requires a gs:// trace"
      f" directory (--xprof_dir), got '{trace_dir}'. Skipping XProf for this"
      " benchmark; only wall-clock metrics will be reported."
  )
  return None
