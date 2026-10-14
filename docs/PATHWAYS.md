# TPUMS on Pathways on Cloud

TPUMS was written for **multi-controller JAX (McJAX)**: one Python process per
TPU VM, each driving its local chips, coordinated through
`jax.distributed.initialize()`. This branch adds support for
**[Pathways on Cloud](https://cloud.google.com/ai-hypercomputer/docs/workloads/pathways-on-cloud/pathways-intro)**,
where a *single* Python client drives every TPU in one or more slices through
the Pathways IFRT proxy. The same configs, benchmarks and reports work in both
modes; Pathways mode is selected purely by the environment
(`JAX_PLATFORMS=proxy`), so McJAX behavior is unchanged.

- To launch benchmarks on a Pathways GKE cluster, see
  [`cloud_launch/pathways/README.md`](../cloud_launch/pathways/README.md).
- This page explains what changed in TPUMS, how to read Pathways results, and
  what was validated.

## How a Pathways run differs

| Aspect | McJAX | Pathways on Cloud |
| :--- | :--- | :--- |
| Processes | One TPUMS process per TPU VM (`process_count` = #hosts) | One TPUMS client on a CPU node (`process_count` = 1, every device is addressable) |
| JAX backend | `tpu` (libtpu in the TPUMS process) | `proxy` → Pathways resource manager + workers; the TPU runtime lives in the Pathways images |
| Bootstrap | `jax.distributed.initialize()` | `pathwaysutils.initialize()` (registers the proxy backend and patches `jax.profiler`) |
| Compilation | In each TPUMS process (libtpu) | In the `pathways-proxy` container |
| `op_flags.yaml` flags | `LIBTPU_INIT_ARGS` in the TPUMS process | Arguments of the `pathways-proxy` container (done by the launcher) |
| XProf traces | `jax.profiler` writes a local `*.trace.json.gz` that TPUMS parses | Pathways components write `*.xplane.pb` to a **gs://** directory; TPUMS parses the worker's TPU planes |
| Wall-clock timing | Host dispatch + device time | Host dispatch + **client→proxy→worker round trip** + device time |

## Code changes

All Pathways behavior is gated on `JAX_PLATFORMS` containing `proxy`; without
it every code path below behaves exactly as before.

### New: [`core/pathways.py`](../src/accelerator_microbenchmarks/core/pathways.py)

The single place that knows about the Pathways runtime:

| Function | What it does |
| :--- | :--- |
| `is_requested()` | `True` if `JAX_PLATFORMS` (env or `jax.config`) contains `proxy`. |
| `maybe_initialize()` | Calls `pathwaysutils.initialize()` once, before any JAX backend is created. Raises an actionable `ImportError` if pathwaysutils is missing. No-op for McJAX. |
| `is_active()` | Whether this process is a Pathways client (used to branch elsewhere). |
| `runtime_name()` / `runtime_version()` | `"pathways"`/`"mcjax"` and the backend `platform_version`, plus the Pathways image tag from `TPUMS_PATHWAYS_VERSION` (set by the launcher; Cloud builds report a placeholder build number). Recorded in every result. |
| `resolve_xprof_dir()` | Returns the trace directory, or `None` (skip XProf with a warning) on Pathways when it is not a `gs://` path, because pathwaysutils' `start_trace` only accepts GCS. |
| `trace_start_delay_s()` | Pause after starting a trace before the timed loop (`TPUMS_PATHWAYS_TRACE_START_DELAY_S`, default 1 s). Pathways enables the device tracers asynchronously; without it the first measured step was missing from the device trace. |
| `wait_for_devices()` | `jax.devices()` with a watchdog (`TPUMS_PATHWAYS_STARTUP_TIMEOUT_S`, default 900 s). If the Pathways slice never becomes ready, the client exits with code 75 so that the JobSet fails and releases the TPUs instead of hanging until its deadline. |

### New: [`core/pathways_trace.py`](../src/accelerator_microbenchmarks/core/pathways_trace.py)

On Pathways, `jax.profiler.trace(gs://dir)` makes the client, the resource
manager and (by default) **one** worker host write
`<dir>/plugins/profile/<session>/*.xplane.pb`. The TPU device timelines are
only in the worker file (`server-<jobset>-pathways-worker-<job>-<pod>...`).
`load_trace()` reads those XSpace protos (with TensorFlow's `xplane_pb2` and
`tf.io.gfile`, both already TPUMS dependencies) and converts their
`/device:TPU:N` planes into the same Chrome trace events that McJAX's
`*.trace.json.gz` contains, so the existing step-duration logic in
`core/profiler.py` (MARKER ops grouped by `jit_*` module executions) is reused
unchanged:

- Worker planes are named by host-local ordinal (`/device:TPU:0..3` on every
  host). They are renamed to `/device:TPU:<global core id>` using the plane's
  `core_details` stat (`CoreDetails.global_core_id`), which equals the JAX
  device id on Pathways, so a benchmark's measured device is found by
  `device.id`. If the measured device is on a host that was not profiled, the
  parser falls back to the first device plane with MARKER ops.
- Event names, `tf_op`, `hlo_category`, `device_duration_ps` and the other
  stats are carried over exactly like XProf's trace-viewer conversion.

### Modified files

| File | Change | Why |
| :--- | :--- | :--- |
| [`cli.py`](../src/accelerator_microbenchmarks/cli.py) | Calls `pathways.maybe_initialize()` before parsing/dispatching commands. | The proxy backend must be registered before `jax.devices()` runs (e.g. in `tpums platform describe`). |
| [`core/runner.py`](../src/accelerator_microbenchmarks/core/runner.py) | `run_benchmarks()` calls `maybe_initialize()` (for library users); `init_jax_distributed()` skips `jax.distributed.initialize()` on Pathways and gets the devices through `wait_for_devices()`; `set_xla_flags()` prints that `LIBTPU_INIT_ARGS` is ignored by a Pathways client. | Pathways is single-controller: there is no coordinator, and op flags must be given to the proxy. |
| [`core/platform.py`](../src/accelerator_microbenchmarks/core/platform.py) | Skips `jax.distributed.initialize()` on Pathways, accepts a `proxy` backend name when Pathways is active, and adds `runtime` / `runtime_version` to `PlatformInfo` (defaults keep the old constructor working). | Results must say which runtime produced them. The two fields show up as new columns in `summary.csv` and in `detailed.json`. |
| [`core/base.py`](../src/accelerator_microbenchmarks/core/base.py) | The per-benchmark trace directory goes through `pathways.resolve_xprof_dir()`; when it is `None` the run skips the profiler and reports XProf metrics as `None` instead of crashing. | With the released JAX, `jax.profiler.trace()` on a Pathways client goes through pathwaysutils, which raises `ValueError` for non-GCS paths — every config with `xprof_timing: true` would have failed. |
| [`core/executor.py`](../src/accelerator_microbenchmarks/core/executor.py) | Looks up trace planes by JAX device id on Pathways (`_trace_device_id`), and sleeps `trace_start_delay_s()` after starting a Pathways trace. | See `pathways_trace.py` and `trace_start_delay_s()` above. |
| [`core/profiler.py`](../src/accelerator_microbenchmarks/core/profiler.py) | `_load_xprof_trace()` reads Pathways traces through `pathways_trace.load_trace()`. | The local `*.json.gz` lookup cannot see GCS and the client's own trace has no TPU planes. |
| [`specs/v5e.py`](../src/accelerator_microbenchmarks/specs/v5e.py), [`specs/schema.py`](../src/accelerator_microbenchmarks/specs/schema.py), [`core/system.py`](../src/accelerator_microbenchmarks/core/system.py) | New `TpuVersion.V5E` (`device_kind` "TPU v5 lite" / "TPU v5e") with peak specs: 197 bf16 TFLOPS, 394 int8 TOPS, float32 = 2-pass bf16, no fp8/int4/fp4; 400 GB/s bidirectional ICI per chip; 819.2 GB/s HBM. | Needed to run on v5e Pathways clusters (the validation cluster is v5e 4x8). Same conventions as the v6e spec. |
| [`configs/v5e/4x8/`](../configs/v5e/4x8/) | `all_gather`, `all_reduce`, `all_to_all` (same sweep as v6e 4x8, sized for 16 GB HBM), `gemm`, `hbm`. | Ready-to-run configs for a v5e 4x8 slice. |
| [`pyproject.toml`](../pyproject.toml) | New optional extra: `pip install '.[pathways]'` → `pathwaysutils>=0.1.12`. | Keeps pathwaysutils out of McJAX installs. |
| Tests | New [`tests/test_pathways.py`](../tests/test_pathways.py) and [`tests/test_pathways_trace.py`](../tests/test_pathways_trace.py) (synthetic worker XSpace, plus a real `core_details` value); Pathways cases in `test_platform.py` and `test_base.py`; v5e cases in `test_system.py`; `test_cli.py` expects the two new `PlatformInfo` keys; [`cloud_launch/pathways/generate_workload_test.py`](../cloud_launch/pathways/generate_workload_test.py) for the launcher. | The rest of the suite has the same results as `main`. |

## Reading Pathways results

- **XProf metrics** (`xprof_*`) are the device-side times of the MARKER ops,
  exactly as on McJAX, and are what should be compared across runtimes. They
  require a `gs://` `--xprof_dir` (`ENABLE_XPROF=true` in the launcher, the
  default).
- **Wall-clock metrics** (`wall_clock_*`) additionally include the client →
  proxy → worker round trip of every step. On the validation runs this added
  3–8 ms per step; it dominates short ops and is a direct measure of Pathways
  dispatch overhead.
- `runtime=pathways` and `runtime_version` (e.g.
  `Pathways cl/42; images 20261013-jax_0.11.1`) identify Pathways rows in
  `summary.csv`. `libtpu_version` describes the client image only.
  `libtpu_init_args` lists the op flags that were applied in the proxy.
- The full traces stay in `<xprof_dir>/<benchmark>_<case>_<ts>/` and can be
  opened in XProf/TensorBoard.

## Validation

All five `configs/v5e/4x8/` configs were run with
`cloud_launch/pathways/pathways_launch.sh` on a Pathways GKE cluster with v5e
4x8 node pools (1 slice, 32 chips, 8 TPU VMs), using:

- TPUMS image built from commit `a178eb2` (jax/jaxlib 0.11.1,
  pathwaysutils 0.1.12 in the client),
- Pathways server/proxy `20261013-jax_0.11.1`,
- the default `OP_FLAGS=auto`: every `op_flags.yaml` entry used by these
  configs was accepted by the proxy.

Every case captured all 5 measured steps in the device trace and found the
measured device (device 0) on the profiled worker. p50 values per step:

| Benchmark | Case | Wall-clock (ms) | XProf device (ms) | XProf metric |
| :--- | :--- | ---: | ---: | :--- |
| `all_gather` | 16 MiB shard | 18.38 | 10.17 | 51.1 GB/s per chip |
| `all_gather` | 32 MiB shard | 30.55 | 20.33 | 51.2 GB/s per chip |
| `all_gather` | 64 MiB shard | 54.02 | 40.65 | 51.2 GB/s per chip |
| `all_reduce` (sum) | 256 MiB shard | 40.36 | 32.44 | 16.0 GB/s per chip |
| `all_reduce` (sum) | 512 MiB shard | 72.59 | 64.86 | 16.0 GB/s per chip |
| `all_reduce` (sum) | 1024 MiB shard | 137.26 | 129.86 | 16.0 GB/s per chip |
| `all_to_all` | 16 MiB shard | 7.93 | 0.67 | 24.2 GB/s per chip |
| `all_to_all` | 32 MiB shard | 8.94 | 1.34 | 24.3 GB/s per chip |
| `all_to_all` | 64 MiB shard | 10.29 | 2.67 | 24.3 GB/s per chip |
| `gemm` bf16, k=n=4096 | m=8192 | 4.19 | 1.41 | 195.0 TFLOPS per chip (99.0% of peak) |
| `gemm` bf16, k=n=4096 | m=16384 | 5.52 | 2.82 | 195.0 TFLOPS per chip (99.0% of peak) |
| `gemm` bf16, k=n=4096 | m=32768 | 9.53 | 5.64 | 194.9 TFLOPS per chip (99.0% of peak) |
| `hbm` copy | 64 Mi elements | 6.06 | 0.41 | 80.3% of HBM peak |
| `hbm` copy | 128 Mi elements | 6.62 | 0.82 | 80.3% of HBM peak |
| `hbm` copy | 256 Mi elements | 7.24 | 1.63 | 80.3% of HBM peak |
| `hbm` copy | 512 Mi elements | 9.35 | 3.26 | 80.3% of HBM peak |

The device-side numbers are what the hardware should deliver (GEMM at 99% of
the 197 TFLOPS bf16 peak, collectives scaling linearly with size), while the
wall-clock column shows the Pathways dispatch overhead on top. A second set of
runs with an earlier build of the same code reproduced the collective numbers
within 0.1%.

## Limitations and follow-ups

- **One profiled host.** `jax.profiler.trace` on Pathways records one worker
  host (`pathwaysutils.profiling.start_trace(max_num_hosts=1)`); in the
  validation runs it was always the host with device 0, which is what
  `FIRST_DEVICE` mode measures. `MAX_DEVICE` mode therefore only considers the
  devices of that host.
- **Host↔device benchmarks** (`host_to_device`, `device_to_host`) time
  `TraceAnnotation` ranges on the host CPU. On Pathways the client is on a
  different machine and those ranges are not in the worker trace, so only
  wall-clock metrics (which include the proxy and network) are reported; they
  are not comparable with McJAX.
- **`gemm_throttling` / soaking executor** needs local thermal telemetry
  parsing; not supported on Pathways.
- **Op flags** are passed to the proxy binary. A flag that the proxy build does
  not define makes the proxy exit at startup; rerun with `OP_FLAGS=none` (or a
  trimmed op flags file) in that case.
- **Multi-slice:** the launcher supports `NUM_SLICES>1`, but the provided
  configs build single-slice meshes, and device planes are matched by
  per-slice core id.
- **Preemption:** with Kueue, higher-priority workloads can preempt a run.
  When the JobSet is re-admitted, the Pathways resource manager may restart and
  the slice never becomes ready; `wait_for_devices()` then fails the run after
  `TPUMS_PATHWAYS_STARTUP_TIMEOUT_S` so you can relaunch it.
