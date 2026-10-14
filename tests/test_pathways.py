"""Unit tests for core/pathways.py (Pathways on Cloud support)."""

import os
import sys
import threading
import types
from unittest import mock

from absl.testing import absltest
from accelerator_microbenchmarks.core import pathways


class PathwaysTest(absltest.TestCase):
  """Unit tests for pathways.py."""

  def setUp(self):
    super().setUp()
    # Every test starts as a fresh, uninitialized McJAX process.
    self.enter_context(mock.patch.object(pathways, "_initialized", False))

  def _set_platforms(self, value: str):
    self.enter_context(
        mock.patch.dict(os.environ, {"JAX_PLATFORMS": value}, clear=False)
    )

  def _fake_pathwaysutils(self) -> types.ModuleType:
    module = types.ModuleType("pathwaysutils")
    module.initialize = mock.MagicMock()
    self.enter_context(mock.patch.dict(sys.modules, {"pathwaysutils": module}))
    return module

  def test_is_requested(self):
    """Verify that only JAX_PLATFORMS values containing 'proxy' request Pathways."""
    for value in ("proxy", "PROXY", "proxy,cpu", "cpu, proxy"):
      with mock.patch.dict(os.environ, {"JAX_PLATFORMS": value}):
        self.assertTrue(pathways.is_requested(), value)
    for value in ("tpu", "cpu", "tpu,cpu", "proxyx"):
      with mock.patch.dict(os.environ, {"JAX_PLATFORMS": value}):
        self.assertFalse(pathways.is_requested(), value)

  def test_maybe_initialize_noop_for_mcjax(self):
    """Verify McJAX runs neither import nor initialize pathwaysutils."""
    self._set_platforms("tpu")
    fake = self._fake_pathwaysutils()

    self.assertFalse(pathways.maybe_initialize())

    fake.initialize.assert_not_called()
    self.assertFalse(pathways.is_active())
    self.assertEqual(pathways.runtime_name(), pathways.RUNTIME_MCJAX)

  def test_maybe_initialize_pathways(self):
    """Verify Pathways runs initialize pathwaysutils exactly once."""
    self._set_platforms("proxy")
    fake = self._fake_pathwaysutils()

    self.assertTrue(pathways.maybe_initialize())
    self.assertTrue(pathways.maybe_initialize())

    fake.initialize.assert_called_once()
    self.assertTrue(pathways.is_active())
    self.assertEqual(pathways.runtime_name(), pathways.RUNTIME_PATHWAYS)

  def test_maybe_initialize_without_pathwaysutils_raises(self):
    """Verify a missing pathwaysutils yields an actionable ImportError."""
    self._set_platforms("proxy")
    # A None entry in sys.modules makes `import pathwaysutils` raise.
    self.enter_context(mock.patch.dict(sys.modules, {"pathwaysutils": None}))

    with self.assertRaisesRegex(
        ImportError, r"accelerator_microbenchmarks\[pathways\]"
    ):
      pathways.maybe_initialize()
    self.assertFalse(pathways.is_active())

  def test_runtime_version_flattens_multiline_versions(self):
    """Verify multi-line platform versions are reported on a single line."""
    fake_backend = mock.MagicMock()
    fake_backend.platform_version = "Pathways\ncl/123456 "
    with mock.patch(
        "jax.extend.backend.get_backend", return_value=fake_backend
    ):
      self.assertEqual(pathways.runtime_version(), "Pathways cl/123456")

  def test_runtime_version_unknown_on_error(self):
    """Verify backend query failures are reported as 'unknown'."""
    with mock.patch(
        "jax.extend.backend.get_backend", side_effect=RuntimeError("no backend")
    ):
      self.assertEqual(pathways.runtime_version(), "unknown")

  def test_runtime_version_appends_pathways_image_tag(self):
    """Verify the launcher-provided image tag is recorded on Pathways only."""
    fake_backend = mock.MagicMock()
    fake_backend.platform_version = "Pathways\ncl/42"
    self.enter_context(
        mock.patch.dict(
            os.environ, {pathways.PATHWAYS_VERSION_ENV: "20261013-jax_0.11.1"}
        )
    )
    with mock.patch(
        "jax.extend.backend.get_backend", return_value=fake_backend
    ):
      self.assertEqual(pathways.runtime_version(), "Pathways cl/42")
      with mock.patch.object(pathways, "_initialized", True):
        self.assertEqual(
            pathways.runtime_version(),
            "Pathways cl/42; images 20261013-jax_0.11.1",
        )

  def test_trace_start_delay_s(self):
    """Verify the trace start delay default, override and validation."""
    env = pathways.TRACE_START_DELAY_ENV
    with mock.patch.dict(os.environ, {env: ""}):
      self.assertEqual(pathways.trace_start_delay_s(), 1.0)
    with mock.patch.dict(os.environ, {env: "0.25"}):
      self.assertEqual(pathways.trace_start_delay_s(), 0.25)
    with mock.patch.dict(os.environ, {env: "-3"}):
      self.assertEqual(pathways.trace_start_delay_s(), 0.0)
    with mock.patch.dict(os.environ, {env: "soon"}):
      self.assertEqual(pathways.trace_start_delay_s(), 1.0)

  def test_wait_for_devices_returns_devices(self):
    """Verify devices are returned when the backend answers in time."""
    with mock.patch("jax.devices", return_value=["d0", "d1"]):
      self.assertEqual(pathways.wait_for_devices(timeout_s=5), ["d0", "d1"])
      # 0 disables the watchdog.
      self.assertEqual(pathways.wait_for_devices(timeout_s=0), ["d0", "d1"])

  def test_wait_for_devices_reraises_backend_errors(self):
    """Verify backend errors surface in the caller's thread."""
    with mock.patch("jax.devices", side_effect=RuntimeError("proxy down")):
      with self.assertRaisesRegex(RuntimeError, "proxy down"):
        pathways.wait_for_devices(timeout_s=5)

  def test_wait_for_devices_exits_on_timeout(self):
    """Verify a backend that never answers makes the process exit."""
    release = threading.Event()
    self.addCleanup(release.set)

    def _hang():
      release.wait(10)
      return []

    with mock.patch("jax.devices", side_effect=_hang), mock.patch.object(
        pathways.os, "_exit", side_effect=SystemExit
    ) as mock_exit:
      with self.assertRaises(SystemExit):
        pathways.wait_for_devices(timeout_s=0.05)
    mock_exit.assert_called_once_with(pathways.STARTUP_TIMEOUT_EXIT_CODE)

  def test_wait_for_devices_timeout_from_env(self):
    """Verify the timeout is read from TPUMS_PATHWAYS_STARTUP_TIMEOUT_S."""
    with mock.patch.dict(
        os.environ, {pathways.STARTUP_TIMEOUT_ENV: "0"}
    ), mock.patch("jax.devices", return_value=["d0"]) as mock_devices:
      self.assertEqual(pathways.wait_for_devices(), ["d0"])
    mock_devices.assert_called_once()

  def test_is_gcs_path(self):
    """Verify GCS URI detection."""
    self.assertTrue(pathways.is_gcs_path("gs://bucket/dir"))
    for path in ("/tmp/tensorboard", "/cns/xx/home", "", None, "gs:/x"):
      self.assertFalse(pathways.is_gcs_path(path), path)

  def test_resolve_xprof_dir_mcjax_keeps_any_dir(self):
    """Verify McJAX trace directories are returned unchanged."""
    self.assertEqual(
        pathways.resolve_xprof_dir("/tmp/tensorboard/run"),
        "/tmp/tensorboard/run",
    )
    self.assertEqual(
        pathways.resolve_xprof_dir("gs://bucket/run"), "gs://bucket/run"
    )

  def test_resolve_xprof_dir_pathways(self):
    """Verify Pathways keeps gs:// directories and skips local ones."""
    with mock.patch.object(pathways, "_initialized", True):
      self.assertEqual(
          pathways.resolve_xprof_dir("gs://bucket/run"), "gs://bucket/run"
      )
      self.assertIsNone(pathways.resolve_xprof_dir("/tmp/tensorboard/run"))


if __name__ == "__main__":
  absltest.main()
