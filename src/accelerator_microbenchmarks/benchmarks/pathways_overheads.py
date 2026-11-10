"""Pathways vs. McJAX overheads, compile latency, dispatch pipelining, transfer, memory, and parity benchmarks."""

import dataclasses
import hashlib
import os
import time
from typing import Any, Callable, Optional, Sequence

from accelerator_microbenchmarks.core import base
from accelerator_microbenchmarks.core import checksum
from accelerator_microbenchmarks.core import constants
from accelerator_microbenchmarks.core import registry
from accelerator_microbenchmarks.core import report
from accelerator_microbenchmarks.core import runner
from accelerator_microbenchmarks.core import utils
import jax
import jax.numpy as jnp
import numpy as np

_DEVICE_AXIS = "device"


class OverheadsMode(constants.ParamEnum):
  """Sub-benchmark mode executed by `PathwaysOverheadsBenchmark`."""

  CLIENT_INIT = "client_init"
  COMPILE_COLD = "compile_cold"
  COMPILE_WARM = "compile_warm"
  DISPATCH_PIPELINING = "dispatch_pipelining"
  H2D_SHARDED = "h2d_sharded"
  H2D_REPLICATED = "h2d_replicated"
  D2H_SHARDED = "d2h_sharded"
  MEMORY_AND_PARITY = "memory_and_parity"
  ORBAX_CKPT_PROBE = "orbax_ckpt_probe"


@dataclasses.dataclass
class PathwaysOverheadsParams(base.SingleDtypeBenchmarkParams):
  """Parameters for `pathways_overheads` benchmark cases."""

  mode: OverheadsMode = dataclasses.field(
      default=OverheadsMode.DISPATCH_PIPELINING,
      metadata={"help": "Overheads sub-benchmark mode."},
  )
  pipeline_depth: int = dataclasses.field(
      default=1,
      metadata={"min": 1, "help": "Number of back-to-back async dispatches per step."},
  )
  data_size_mib: int = dataclasses.field(
      default=16,
      metadata={"min": 1, "help": "Payload size per device in MiB for transfers/ckpt."},
  )
  matrix_dim: int = dataclasses.field(
      default=1024,
      metadata={"min": 64, "help": "Square matrix dimension for compile/dispatch/parity ops."},
  )
  seed: int = dataclasses.field(
      default=42,
      metadata={"min": 0, "help": "PRNG seed for deterministic parity verification."},
  )


@registry.benchmark_registry.register("pathways_overheads")
class PathwaysOverheadsBenchmark(base.BaseBenchmark[PathwaysOverheadsParams]):
  """Measures client/proxy overheads, compile time, dispatch throughput, transfers, memory, and parity."""

  Config = PathwaysOverheadsParams  # pylint: disable=invalid-name
  derive_chip_bandwidth: bool = False
  REPORT_SCHEMA: Sequence[tuple[str, Callable[[Any], str]]] = (
      ("mode", report.format_str),
      ("dtype", report.format_str),
      ("pipeline_depth", report.format_str),
      ("data_size_mib", report.format_str),
      ("matrix_dim", report.format_str),
      ("wall_clock_p50_ms", report.format_4f),
      ("wall_clock_p90_ms", report.format_4f),
      ("wall_clock_calls_per_s", report.format_2f),
      ("wall_clock_bandwidth_per_device_gb_s", report.format_2f),
      ("wall_clock_bandwidth_per_host_gb_s", report.format_2f),
      ("compiled_temp_bytes", report.format_str),
      ("compiled_arg_bytes", report.format_str),
      ("runtime_peak_hbm_bytes", report.format_str),
      ("parity_gemm_sha256", report.format_str),
      ("parity_allreduce_sha256", report.format_str),
      ("parity_checksum_hex", report.format_str),
      ("backend_platform_version", report.format_str),
  )

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self._compile_events_s: list[float] = []
    self._extra_metrics: dict[str, Any] = {}
    self._transfer_devices: list[jax.Device] = []
    self._sharding_dev0: Optional[jax.sharding.SingleDeviceSharding] = None
    self._sharding_sharded: Optional[jax.sharding.NamedSharding] = None
    self._sharding_replicated: Optional[jax.sharding.NamedSharding] = None
    self._ar_fn: Optional[Callable[[Any], Any]] = None
    self._unique_compile_counter: int = 0
    self._run_checksums: list[str] = []
    self._host_shard_cache: Optional[np.ndarray] = None

  def _on_compile_duration(self, event: str, duration_s: float, **kwargs) -> None:
    del kwargs
    if event == "/jax/core/compile/backend_compile_duration":
      self._compile_events_s.append(float(duration_s))

  def get_run_identifier(self) -> str:
    m = self.config.mode.value
    if self.config.mode == OverheadsMode.DISPATCH_PIPELINING:
      return f"{m}_depth_{self.config.pipeline_depth}_dim_{self.config.matrix_dim}"
    if self.config.mode in (
        OverheadsMode.H2D_SHARDED,
        OverheadsMode.H2D_REPLICATED,
        OverheadsMode.D2H_SHARDED,
        OverheadsMode.ORBAX_CKPT_PROBE,
    ):
      return f"{m}_size_{self.config.data_size_mib}mib"
    return f"{m}_dim_{self.config.matrix_dim}"

  def setup(self) -> None:
    local_devs = list(jax.local_devices())
    # Use the 4 chips of host 0 (all local devices on McJAX, first 4 on Pathways)
    # so every device in self.mesh is 100% addressable by the current process
    # and the per-host shard geometry is identical across McJAX and Pathways.
    self._transfer_devices = local_devs[: min(4, len(local_devs))]
    self.mesh = jax.sharding.Mesh(
        np.array(self._transfer_devices), axis_names=(_DEVICE_AXIS,)
    )
    self._sharding_dev0 = jax.sharding.SingleDeviceSharding(self._transfer_devices[0])
    self._sharding_sharded = jax.sharding.NamedSharding(
        self.mesh, jax.sharding.PartitionSpec(_DEVICE_AXIS, None)
    )
    self._sharding_replicated = jax.sharding.NamedSharding(
        self.mesh, jax.sharding.PartitionSpec()
    )
    try:
      jax.monitoring.register_event_duration_secs_listener(self._on_compile_duration)
    except Exception:  # pylint: disable=broad-exception-caught
      pass

    dtype = utils.parse_dtype(self.config.dtype)
    dim = self.config.matrix_dim

    if self.config.mode in (
        OverheadsMode.DISPATCH_PIPELINING,
        OverheadsMode.COMPILE_WARM,
        OverheadsMode.MEMORY_AND_PARITY,
    ):
      @jax.jit
      def _small_kernel(x, y):
        with jax.named_scope(constants.MARKER):
          return jnp.dot(x, y, preferred_element_type=dtype)

      self._jit_fn = _small_kernel

      if self.config.mode == OverheadsMode.MEMORY_AND_PARITY:
        self._ar_fn = jax.jit(
            jax.shard_map(
                lambda z: jax.lax.psum(z, axis_name=_DEVICE_AXIS),
                mesh=self.mesh,
                in_specs=jax.sharding.PartitionSpec(_DEVICE_AXIS, None),
                out_specs=jax.sharding.PartitionSpec(None, None),
            )
        )
        dummy = jax.device_put(
            np.ones((dim, dim), dtype=np.float32).astype(dtype),
            self._sharding_dev0,
        )
        compiled = self._jit_fn.lower(dummy, dummy).compile()
        mem_stat = compiled.memory_analysis()
        if mem_stat is not None:
          self._extra_metrics["compiled_temp_bytes"] = int(
              getattr(mem_stat, "temp_size_in_bytes", 0)
          )
          self._extra_metrics["compiled_arg_bytes"] = int(
              getattr(mem_stat, "argument_size_in_bytes", 0)
          )
          self._extra_metrics["compiled_output_bytes"] = int(
              getattr(mem_stat, "output_size_in_bytes", 0)
          )
          self._extra_metrics["compiled_code_bytes"] = int(
              getattr(mem_stat, "generated_code_size_in_bytes", 0)
          )
        try:
          dev_mem = self._transfer_devices[0].memory_stats()
          if dev_mem:
            self._extra_metrics["runtime_peak_hbm_bytes"] = int(
                dev_mem.get("peak_bytes_in_use", 0)
            )
            self._extra_metrics["runtime_bytes_in_use"] = int(
                dev_mem.get("bytes_in_use", 0)
            )
        except Exception as e:  # pylint: disable=broad-exception-caught
          self._extra_metrics["runtime_peak_hbm_bytes"] = None
          self._extra_metrics["memory_stats_error"] = str(e)[:160]

        try:
          backend = jax.extend.backend.get_backend()
          self._extra_metrics["backend_platform_version"] = str(
              getattr(backend, "platform_version", "")
          ).replace("\n", "\\n")
        except Exception:  # pylint: disable=broad-exception-caught
          pass

  def teardown(self) -> None:
    try:
      jax.monitoring.unregister_event_duration_secs_listener(self._on_compile_duration)
    except Exception:  # pylint: disable=broad-exception-caught
      pass
    self._jit_fn = None
    self._ar_fn = None
    self._host_shard_cache = None

  def generate_inputs(self) -> tuple[Any, ...]:
    dtype = utils.parse_dtype(self.config.dtype)
    dim = self.config.matrix_dim
    mode = self.config.mode

    if mode in (
        OverheadsMode.CLIENT_INIT,
        OverheadsMode.COMPILE_COLD,
        OverheadsMode.COMPILE_WARM,
    ):
      x = jax.device_put(
          np.ones((dim, dim), dtype=np.float32).astype(dtype),
          self._sharding_dev0,
      )
      jax.block_until_ready(x)
      return (x,)

    if mode == OverheadsMode.DISPATCH_PIPELINING:
      x = jax.device_put(
          np.ones((dim, dim), dtype=np.float32).astype(dtype),
          self._sharding_dev0,
      )
      y = jax.device_put(
          np.ones((dim, dim), dtype=np.float32).astype(dtype),
          self._sharding_dev0,
      )
      jax.block_until_ready((x, y))
      return (x, y)

    if mode in (
        OverheadsMode.H2D_SHARDED,
        OverheadsMode.H2D_REPLICATED,
        OverheadsMode.D2H_SHARDED,
        OverheadsMode.ORBAX_CKPT_PROBE,
    ):
      itemsize = jnp.dtype(dtype).itemsize
      num_devs = (
          len(self._transfer_devices)
          if mode != OverheadsMode.H2D_REPLICATED
          else 1
      )
      rows = (self.config.data_size_mib * 1024 * 1024 // itemsize // 128) * num_devs
      host_arr = utils.random_bits_array((rows, 128), dtype)
      self._host_shard_cache = host_arr
      if mode in (OverheadsMode.D2H_SHARDED, OverheadsMode.ORBAX_CKPT_PROBE):
        dev_arr = jax.device_put(host_arr, self._sharding_sharded)
        jax.block_until_ready(dev_arr)
        return (dev_arr,)
      return (host_arr,)

    if mode == OverheadsMode.MEMORY_AND_PARITY:
      rng = np.random.RandomState(self.config.seed)
      a_np = rng.standard_normal((dim, dim)).astype(np.float32).astype(dtype)
      b_np = rng.standard_normal((dim, dim)).astype(np.float32).astype(dtype)
      a = jax.device_put(a_np, self._sharding_dev0)
      b = jax.device_put(b_np, self._sharding_dev0)
      num_devs = len(self._transfer_devices)
      s_np = np.arange(num_devs * dim, dtype=np.float32).reshape(num_devs, dim)
      s = jax.device_put(s_np, self._sharding_sharded)
      jax.block_until_ready((a, b, s))
      return (a, b, s)

    return ()

  def reset_data(self, *inputs, **kwargs) -> tuple[Any, ...]:
    if self.config.mode == OverheadsMode.D2H_SHARDED and inputs:
      # Recreate device array so jax.device_get does not return a cached host copy.
      host_copy = (
          self._host_shard_cache
          if self._host_shard_cache is not None
          else np.asarray(jax.device_get(inputs[0]))
      )
      try:
        inputs[0].delete()
      except Exception:  # pylint: disable=broad-exception-caught
        pass
      dev_arr = jax.device_put(host_copy, self._sharding_sharded)
      jax.block_until_ready(dev_arr)
      return (dev_arr,)
    return inputs

  def run_op(self, *args, **kwargs) -> Any:
    mode = self.config.mode
    dtype = utils.parse_dtype(self.config.dtype)

    if mode == OverheadsMode.CLIENT_INIT:
      return jax.device_get(args[0][:1, :1])

    if mode == OverheadsMode.COMPILE_COLD:
      self._unique_compile_counter += 1
      c = self._unique_compile_counter

      @jax.jit
      def _fresh_fn(x):
        v = x * jnp.array(1.0 + 1e-4 * float(c), dtype=dtype)
        for _ in range(4):
          v = jnp.dot(v, v, preferred_element_type=dtype) + v
        return v

      t_c0 = time.perf_counter()
      compiled = _fresh_fn.lower(args[0]).compile()
      t_c1 = time.perf_counter()
      self._extra_metrics["compile_only_ms"] = (t_c1 - t_c0) * 1000.0
      return compiled(args[0])

    if mode == OverheadsMode.COMPILE_WARM:
      t_c0 = time.perf_counter()
      compiled = self._jit_fn.lower(args[0], args[0]).compile()
      t_c1 = time.perf_counter()
      self._extra_metrics["compile_only_ms"] = (t_c1 - t_c0) * 1000.0
      return compiled(args[0], args[0])

    if mode == OverheadsMode.DISPATCH_PIPELINING:
      x, y = args[0], args[1]
      t_issue0 = time.perf_counter()
      outs = [self._jit_fn(x, y) for _ in range(self.config.pipeline_depth)]
      t_issue1 = time.perf_counter()
      self._extra_metrics["dispatch_issue_only_ms"] = (t_issue1 - t_issue0) * 1000.0
      return outs[-1]

    if mode == OverheadsMode.H2D_SHARDED:
      return jax.device_put(args[0], self._sharding_sharded)

    if mode == OverheadsMode.H2D_REPLICATED:
      return jax.device_put(args[0], self._sharding_replicated)

    if mode == OverheadsMode.D2H_SHARDED:
      return np.asarray(jax.device_get(args[0]))

    if mode == OverheadsMode.MEMORY_AND_PARITY:
      a, b, s = args[0], args[1], args[2]
      gemm_out = self._jit_fn(a, b)
      ar_out = self._ar_fn(s)
      return (gemm_out, ar_out)

    if mode == OverheadsMode.ORBAX_CKPT_PROBE:
      ckpt_file = f"/tmp/tpums_ckpt_probe_p{jax.process_index()}.npy"
      t_s0 = time.perf_counter()
      host_payload = np.asarray(jax.device_get(args[0]))
      t_s1 = time.perf_counter()
      with open(ckpt_file, "wb") as f:
        np.save(f, host_payload, allow_pickle=False)
        f.flush()
      t_s2 = time.perf_counter()
      with open(ckpt_file, "rb") as f:
        loaded_np = np.load(f, allow_pickle=False)
      restored = jax.device_put(loaded_np, self._sharding_sharded)
      jax.block_until_ready(restored)
      t_s3 = time.perf_counter()
      self._extra_metrics["orbax_save_blocking_ms"] = (t_s1 - t_s0) * 1000.0
      self._extra_metrics["orbax_save_total_ms"] = (t_s2 - t_s0) * 1000.0
      self._extra_metrics["orbax_restore_ms"] = (t_s3 - t_s2) * 1000.0
      total_gb = self.get_total_bytes() / 1e9
      self._extra_metrics["orbax_save_gb_s"] = total_gb / max(1e-6, t_s2 - t_s0)
      self._extra_metrics["orbax_restore_gb_s"] = total_gb / max(1e-6, t_s3 - t_s2)
      return restored

    return None

  def post_run_processing(self, outputs: Any, run_index: int) -> None:
    if self.config.mode == OverheadsMode.MEMORY_AND_PARITY:
      gemm_out, ar_out = outputs
      words = np.asarray(jax.device_get(checksum.checksum_words(gemm_out)))
      combined = checksum.combine_checksum_words(int(words[0]), int(words[1]))
      hex_str = checksum.format_checksum(combined) or ""
      self._run_checksums.append(hex_str)
      self._extra_metrics["parity_checksum_hex"] = hex_str
      self._extra_metrics["run_to_run_deterministic"] = int(
          len(set(self._run_checksums)) == 1
      )
      if run_index == 0:
        gemm_np = np.asarray(jax.device_get(gemm_out))
        ar_np = np.asarray(ar_out.addressable_data(0))
        self._extra_metrics["parity_gemm_sha256"] = hashlib.sha256(
            gemm_np.tobytes()
        ).hexdigest()[:16]
        self._extra_metrics["parity_allreduce_sha256"] = hashlib.sha256(
            ar_np.tobytes()
        ).hexdigest()[:16]

  def get_total_bytes(self) -> float:
    num_devs = max(1, len(self._transfer_devices))
    if self.config.mode in (
        OverheadsMode.H2D_SHARDED,
        OverheadsMode.D2H_SHARDED,
        OverheadsMode.ORBAX_CKPT_PROBE,
    ):
      return float(self.config.data_size_mib * 1024 * 1024 * num_devs)
    if self.config.mode == OverheadsMode.H2D_REPLICATED:
      return float(self.config.data_size_mib * 1024 * 1024)
    return 0.0

  def get_workload_metadata(self) -> dict[str, Any]:
    meta: dict[str, Any] = {
        "num_devices": len(jax.devices()),
        **self._extra_metrics,
    }
    if self.config.mode == OverheadsMode.CLIENT_INIT and runner.INIT_TIMINGS_MS:
      meta.update(runner.INIT_TIMINGS_MS)
    if self._compile_events_s:
      meta["jax_backend_compile_last_ms"] = self._compile_events_s[-1] * 1000.0
    return meta

  def calculate_metrics(self, times_ms: list[float]) -> dict[str, Any]:
    if self.config.mode == OverheadsMode.CLIENT_INIT and runner.INIT_TIMINGS_MS:
      first_dev_ms = runner.INIT_TIMINGS_MS.get("first_jax_devices_ms")
      if first_dev_ms:
        times_ms = [first_dev_ms]
    metrics = super().calculate_metrics(times_ms)
    metrics.update(self._extra_metrics)
    if self._compile_events_s:
      metrics["jax_backend_compile_last_ms"] = self._compile_events_s[-1] * 1000.0
    return metrics

  def calculate_throughput_metrics(
      self, latency_ms: float, prefix: constants.TimingDomain
  ) -> dict[str, Any]:
    out: dict[str, Any] = {}
    latency_s = (
        latency_ms / 1000.0 if latency_ms and latency_ms > 0 else float("inf")
    )
    out[f"{prefix}_calls_per_s"] = self.config.pipeline_depth / latency_s
    total_bytes = self.get_total_bytes()
    if total_bytes > 0:
      num_devs = (
          max(1, len(self._transfer_devices))
          if self.config.mode != OverheadsMode.H2D_REPLICATED
          else 1
      )
      out[f"{prefix}_bandwidth_per_device_gb_s"] = (
          total_bytes / num_devs
      ) / (latency_s * 1e9)
      out[f"{prefix}_bandwidth_per_host_gb_s"] = total_bytes / (
          latency_s * 1e9
      )
    return out
