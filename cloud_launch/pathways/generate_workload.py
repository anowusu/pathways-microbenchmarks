#!/usr/bin/env python3
# Copyright 2026 The TPUMS Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Generates the Pathways JobSet that runs one TPUMS config on Pathways on Cloud.

The JobSet is built with `pathwaysutils.experimental.gke.jobset.PathwaysJobSet`,
the JobSet generator that ships with pathways-utils, instead of a hand-written
YAML template. It produces the standard Pathways layout:

* `pathways-head` (1 pod on a CPU node pool): the Pathways resource manager
  (`pathways-rm`) and IFRT proxy (`pathways-proxy`) as native-sidecar init
  containers, plus the TPUMS client in the `user-workload` container.
* `pathways-worker` (one Job per slice, one pod per TPU VM): the Pathways
  worker server that owns the TPU chips.

On top of the generator defaults this script:

* Runs `cloud_launch/pathways/run_benchmark.sh` from the TPUMS image as the
  user workload, which runs `tpums benchmark run-config <config>` and uploads
  the results to GCS.
* Moves the benchmark's `op_flags.yaml` entries to where both the Rapid
  release (`cloud-tpu-v2-images/pathways`, statically linked `libtpu`) and the
  Louhi OSS release (`cloud-tpu-multipod-dev/pathways-prod`, dynamically loaded
  `/lib/libtpu.so`) need them:
  - `flags` are TPU compilation flags. On Rapid images they are parsed from
    `pathways-proxy` CLI `args`; on Louhi OSS images `/opt/pathways/proxy_server`
    loads `/lib/libtpu.so` via `dlopen`, so `--xla_tpu_*` flags live in
    `libtpu.so` (which reads `LIBTPU_INIT_ARGS`) rather than in the proxy
    binary's own `absl` flag table. The generator therefore sets
    `LIBTPU_INIT_ARGS` on the Pathways containers AND passes the flags with a
    matching `--undefok=...` on `pathways-proxy` CLI `args`, so the same
    JobSet works with both Rapid and Louhi OSS images.
  - `env` entries configure the TPU runtime, so they are set on the
    `pathways-worker` containers. (On McJAX both go into the benchmark process
    through `LIBTPU_INIT_ARGS` / `env`, which a Pathways client ignores.)
* Adds the Kueue queue label, a pod priority class, a head node pool, and an
  activeDeadlineSeconds on the head Job so that a hung run cannot hold the
  TPU slice forever.

Example:
  python cloud_launch/pathways/generate_workload.py \\
    --name=tpums-pw-ag --config=configs/v5e/4x8/all_gather.yaml \\
    --image=us-docker.pkg.dev/my-project/my-repo/tpums-pathways:abc123 \\
    --tpu_type=v5e --topology=4x8 \\
    --pathways_version=20261013-jax_0.11.1 \\
    --pathways_dir=gs://my-bucket/tpums-pathways/tpums-pw-ag/scratch \\
    --gcs_output_dir=gs://my-bucket/tpums-pathways/tpums-pw-ag \\
    --output=/tmp/tpums-pw-ag.yaml
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import os
import pathlib
import re
import shlex
import sys
from typing import Any, Sequence

from kubernetes import client
from pathwaysutils.experimental.gke import jobset as pw_jobset
import yaml

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
DEFAULT_OP_FLAGS_PATH = (
    REPO_ROOT / "src" / "accelerator_microbenchmarks" / "op_flags.yaml"
)
# Where cloud_launch/pathways/Dockerfile copies the TPUMS repository.
IMAGE_REPO_ROOT = "/opt/tpums"
RUN_SCRIPT = f"{IMAGE_REPO_ROOT}/cloud_launch/pathways/run_benchmark.sh"

USER_WORKLOAD_CONTAINER = "user-workload"
RM_CONTAINER = "pathways-rm"
PROXY_CONTAINER = "pathways-proxy"
WORKER_CONTAINER = "pathways-worker"
KUEUE_QUEUE_LABEL = "kueue.x-k8s.io/queue-name"
LIBTPU_INIT_ARGS_ENV = "LIBTPU_INIT_ARGS"
# In the Louhi OSS build (`TSL_IS_IN_OSS == 1`), `GrpcServerCredentials()` and
# `GrpcClientCredentials()` delegate to `xla::ifrt::proxy::Get*Credentials()`,
# which defaults to ALTS (`AltsCredentials`) unless `TEST_UNDECLARED_OUTPUTS_DIR`
# is set in the environment. On standard GKE pod networks (without
# `hostNetwork: true`, which `PathwaysJobSet` does not set), the GCE ALTS
# handshaker is unreachable and ALTS handshakes between RM, proxy, and workers
# fail. Setting `TEST_UNDECLARED_OUTPUTS_DIR=/tmp` selects
# `InsecureChannelCredentials()` / `InsecureServerCredentials()` in the Louhi
# OSS build (matching the Rapid build's
# `PATHWAYS_UNSAFE_UNSAFE_OVERRIDE_GRPC_CREDENTIALS=grpc_insecure_override`)
# and is a no-op in the Rapid build.
OSS_INSECURE_GRPC_ENV = ("TEST_UNDECLARED_OUTPUTS_DIR", "/tmp")

# JobSet sets the coordinator label to "<name>-pathways-head-0-0.<name>", which
# must fit in a 63-byte Kubernetes label value (2 * len(name) + 19 <= 63).
MAX_JOBSET_NAME_LENGTH = 22

OP_FLAGS_AUTO = "auto"
OP_FLAGS_NONE = "none"


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
  """Parses command-line flags."""
  parser = argparse.ArgumentParser(
      description=__doc__.split("\n\n", 1)[0],
      formatter_class=argparse.RawDescriptionHelpFormatter,
  )
  workload = parser.add_argument_group("Workload")
  workload.add_argument(
      "--name",
      required=True,
      help=(
          "JobSet name (lowercase letters, digits and '-', at most"
          f" {MAX_JOBSET_NAME_LENGTH} characters)."
      ),
  )
  workload.add_argument("--namespace", default="default")
  workload.add_argument(
      "--config",
      required=True,
      help=(
          "TPUMS config to run, relative to the repository root (e.g."
          " configs/v5e/4x8/all_gather.yaml). It is read locally to resolve"
          " op flags and run from the same path inside the image, so the image"
          " must be built from this checkout."
      ),
  )
  workload.add_argument(
      "--image",
      required=True,
      help="TPUMS Pathways client image (see build_image.sh).",
  )
  workload.add_argument(
      "--gcs_output_dir",
      default="",
      help="gs:// directory to upload summary.csv, detailed.json and logs to.",
  )
  workload.add_argument(
      "--xprof_dir",
      default="",
      help=(
          "gs:// directory for XProf traces. Pathways workers write the device"
          " traces themselves and only support GCS. Empty: no XProf traces,"
          " only wall-clock metrics."
      ),
  )
  # The TPUMS client only dispatches work (Pathways compiles in the proxy), so it
  # needs far less than the 24 CPU / 100G the Pathways examples give the user
  # workload. Smaller head pods leave CPU node-pool quota for other jobs.
  workload.add_argument(
      "--user_cpu",
      default="8",
      help="CPU limit of the user-workload container.",
  )
  workload.add_argument(
      "--user_memory",
      default="32G",
      help="Memory limit of the user-workload container.",
  )

  pathways = parser.add_argument_group("Pathways")
  pathways.add_argument(
      "--tpu_type", required=True, help="TPU type, e.g. v5e, v6e, tpu7x."
  )
  pathways.add_argument(
      "--topology", required=True, help="Slice topology, e.g. 4x8."
  )
  pathways.add_argument("--num_slices", type=int, default=1)
  pathways.add_argument(
      "--pathways_version",
      required=True,
      help=(
          "Tag of the Pathways server and proxy images, e.g."
          " 20261013-jax_0.11.1. Use the release built for the JAX version in"
          " the TPUMS image."
      ),
  )
  pathways.add_argument(
      "--pathways_server_image",
      default=pw_jobset.DEFAULT_PATHWAYS_RM_AND_WORKER_IMAGE,
      help="Resource manager / worker image (without tag).",
  )
  pathways.add_argument(
      "--pathways_proxy_image",
      default=pw_jobset.DEFAULT_PATHWAYS_PROXY_IMAGE,
      help="IFRT proxy image (without tag).",
  )
  pathways.add_argument(
      "--pathways_dir",
      required=True,
      help="gs:// scratch directory for the Pathways servers.",
  )
  pathways.add_argument(
      "--op_flags",
      default=OP_FLAGS_AUTO,
      help=(
          f"'{OP_FLAGS_AUTO}' (default): apply the benchmark's op_flags.yaml"
          " entry (flags -> pathways-proxy args, env -> pathways-worker env)."
          f" '{OP_FLAGS_NONE}': apply nothing. Any other value: path to a"
          " custom op_flags YAML file."
      ),
  )
  pathways.add_argument(
      "--extra_proxy_args",
      default="",
      help=(
          "Additional pathways-proxy flags as one shell-quoted string, e.g."
          " --extra_proxy_args='--xla_tpu_foo=true --xla_tpu_bar=1'."
      ),
  )

  scheduling = parser.add_argument_group("Scheduling")
  scheduling.add_argument(
      "--queue",
      default="multislice-queue",
      help="Kueue LocalQueue name. Empty: do not use Kueue.",
  )
  scheduling.add_argument(
      "--priority_class",
      default="medium",
      help="Pod priorityClassName for head and worker pods. Empty: unset.",
  )
  scheduling.add_argument(
      "--head_node_pool",
      default="cpu-np",
      help="GKE node pool for the pathways-head pod.",
  )
  scheduling.add_argument(
      "--active_deadline_seconds",
      type=int,
      default=1800,
      help=(
          "activeDeadlineSeconds of the pathways-head Job (0 to disable). Time"
          " spent queued in Kueue does not count."
      ),
  )

  output = parser.add_argument_group("Output")
  output.add_argument(
      "--output",
      default="",
      help="Where to write the JobSet YAML. Default: /tmp/<name>.yaml.",
  )
  output.add_argument(
      "--apply",
      action="store_true",
      help=(
          "Also create the JobSet in the current kubectl context with"
          " PathwaysJobSet.apply(recreate=True)."
      ),
  )
  args = parser.parse_args(argv)

  if not re.fullmatch(r"[a-z]([-a-z0-9]*[a-z0-9])?", args.name):
    parser.error(f"--name '{args.name}' is not a valid DNS-1035 label.")
  if len(args.name) > MAX_JOBSET_NAME_LENGTH:
    parser.error(
        f"--name '{args.name}' is longer than {MAX_JOBSET_NAME_LENGTH}"
        " characters."
    )
  for flag in ("pathways_dir", "gcs_output_dir", "xprof_dir"):
    value = getattr(args, flag)
    if value and not value.startswith("gs://"):
      parser.error(f"--{flag} must be a gs:// path, got '{value}'.")
  return args


def _benchmark_name(config_path: pathlib.Path) -> str:
  """Returns `benchmark.name` from a TPUMS config file."""
  with open(config_path, "r") as f:
    data = yaml.safe_load(f)
  try:
    return str(data["benchmark"]["name"])
  except (KeyError, TypeError) as e:
    raise ValueError(
        f"{config_path} is not a TPUMS config (missing benchmark.name)."
    ) from e


def resolve_op_flags(
    benchmark_name: str, op_flags_path: pathlib.Path
) -> tuple[list[str], dict[str, str]]:
  """Returns the (flags, env) op_flags.yaml entry of `benchmark_name`.

  Mirrors `accelerator_microbenchmarks.core.runner.get_op_flags_and_env`
  without importing TPUMS (and therefore JAX) into the launcher.

  Args:
    benchmark_name: The `benchmark.name` of the config, e.g. "all_gather".
    op_flags_path: Path to an op_flags YAML file.

  Returns:
    The flag list and env mapping, both empty if the benchmark has no entry.
  """
  with open(op_flags_path, "r") as f:
    op_flags = yaml.safe_load(f) or {}
  if not isinstance(op_flags, dict):
    raise ValueError(f"{op_flags_path}: expected a mapping at the top level.")
  entry = op_flags.get(benchmark_name) or {}
  if not isinstance(entry, dict):
    raise ValueError(
        f"{op_flags_path}: entry '{benchmark_name}' must be a mapping."
    )
  flags = [str(flag) for flag in entry.get("flags") or []]
  env = {str(k): str(v) for k, v in (entry.get("env") or {}).items()}
  return flags, env


def _find_container(
    pod_spec: client.V1PodSpec, name: str
) -> client.V1Container:
  """Returns the (init) container called `name` in `pod_spec`."""
  for container in (pod_spec.init_containers or []) + (
      pod_spec.containers or []
  ):
    if container.name == name:
      return container
  raise ValueError(f"Container '{name}' not found in pod spec.")


def _undefok_arg(flags: Sequence[str]) -> str | None:
  """Returns an `--undefok=...` flag covering `flags`, or None if empty.

  In the Louhi OSS `proxy_server` image, `libtpu.so` is loaded via `dlopen` and
  reads `LIBTPU_INIT_ARGS`, so `--xla_tpu_*` / `--xla_jf_*` / `--xla_sc_*`
  flags are not registered in the `proxy_server` binary's own `absl` flag table
  when `absl::ParseCommandLine` runs. Passing `--undefok` allows `proxy.args`
  to carry those flags for Rapid (statically-linked) images without crashing
  the Louhi OSS `proxy_server` on startup.
  """
  names: list[str] = []
  for flag in flags:
    name = flag.lstrip("-").split("=", 1)[0]
    if name and name not in names:
      names.append(name)
  return f"--undefok={','.join(names)}" if names else None


def _upsert_env(
    container: client.V1Container, env_vars: dict[str, str]
) -> None:
  """Sets or updates `env_vars` on `container` in order."""
  existing = list(container.env or [])
  index_by_name = {e.name: i for i, e in enumerate(existing)}
  for name, value in env_vars.items():
    env_var = client.V1EnvVar(name=name, value=value)
    if name in index_by_name:
      existing[index_by_name[name]] = env_var
    else:
      index_by_name[name] = len(existing)
      existing.append(env_var)
  container.env = existing


@dataclasses.dataclass(frozen=True)
class Workload:
  """A generated JobSet plus the op flags that were moved into it."""

  jobset: pw_jobset.PathwaysJobSet
  benchmark_name: str
  proxy_flags: list[str]
  worker_env: dict[str, str]


def build_workload(args: argparse.Namespace) -> Workload:
  """Builds the PathwaysJobSet for one TPUMS config."""
  config_path = REPO_ROOT / args.config
  if not config_path.is_file():
    raise FileNotFoundError(f"TPUMS config not found: {config_path}")
  benchmark_name = _benchmark_name(config_path)

  if args.op_flags == OP_FLAGS_NONE:
    op_flags, op_env = [], {}
  else:
    op_flags_path = (
        DEFAULT_OP_FLAGS_PATH
        if args.op_flags == OP_FLAGS_AUTO
        else pathlib.Path(args.op_flags)
    )
    op_flags, op_env = resolve_op_flags(benchmark_name, op_flags_path)

  labels = {"app.kubernetes.io/name": "tpums-pathways"}
  if args.queue:
    labels[KUEUE_QUEUE_LABEL] = args.queue

  head_pod_spec_kwargs: dict[str, Any] = {
      "node_selector": {pw_jobset.GKE_NODEPOOL_KEY: args.head_node_pool},
  }
  worker_pod_spec_kwargs: dict[str, Any] = {}
  if args.priority_class:
    head_pod_spec_kwargs["priority_class_name"] = args.priority_class
    worker_pod_spec_kwargs["priority_class_name"] = args.priority_class

  js = pw_jobset.PathwaysJobSet(
      name=args.name,
      namespace=args.namespace,
      pathways_dir=args.pathways_dir,
      tpu_type=args.tpu_type,
      topology=args.topology,
      num_slices=args.num_slices,
      pathways_version=args.pathways_version,
      pathways_rm_and_worker_image=args.pathways_server_image,
      pathways_proxy_image=args.pathways_proxy_image,
      labels=labels,
      head_pod_spec_kwargs=head_pod_spec_kwargs,
      worker_pod_spec_kwargs=worker_pod_spec_kwargs,
  )

  # The TPUMS client. add_user_workload() also turns pathways-rm and
  # pathways-proxy into native-sidecar init containers, points JAX at the proxy
  # (JAX_PLATFORMS=proxy, JAX_BACKEND_TARGET=grpc://localhost:29000) and makes
  # the JobSet succeed when this container exits 0.
  js.add_user_workload(
      image=args.image,
      command=["bash", RUN_SCRIPT],
      name=USER_WORKLOAD_CONTAINER,
      env={
          "TPUMS_CONFIG": f"{IMAGE_REPO_ROOT}/{args.config}",
          "TPUMS_GCS_OUTPUT_DIR": args.gcs_output_dir,
          "TPUMS_XPROF_DIR": args.xprof_dir,
          "TPUMS_RUN_NAME": args.name,
          # Recorded in results as part of `runtime_version`.
          "TPUMS_PATHWAYS_VERSION": args.pathways_version,
          "PYTHONUNBUFFERED": "1",
          "TF_CPP_MIN_LOG_LEVEL": "1",
      },
      resources=client.V1ResourceRequirements(
          limits={"cpu": args.user_cpu, "memory": args.user_memory}
      ),
  )

  head_pod_spec = js.head_job_template.spec.template.spec
  worker_pod_spec = js.worker_job_template.spec.template.spec

  # op_flags.yaml `flags` are TPU compilation flags:
  # * Rapid release images link `libtpu` statically into `proxy_server` and
  #   parse compilation flags from `pathways-proxy` CLI `args` (ignoring
  #   `LIBTPU_INIT_ARGS`).
  # * Louhi OSS release images load `/lib/libtpu.so` dynamically via `dlopen`
  #   and parse compilation flags from `LIBTPU_INIT_ARGS` inside `libtpu.so`
  #   during `PJRT_Plugin_Initialize`; their `proxy_server` binary rejects
  #   `--xla_tpu_*` CLI flags unless listed in `--undefok`.
  # Setting both `proxy.args` (with `--undefok`) and `LIBTPU_INIT_ARGS` makes
  # the generated JobSet work on both Rapid and Louhi OSS images.
  rm = _find_container(head_pod_spec, RM_CONTAINER)
  proxy = _find_container(head_pod_spec, PROXY_CONTAINER)
  worker = _find_container(worker_pod_spec, WORKER_CONTAINER)
  extra_proxy_args = shlex.split(args.extra_proxy_args)
  proxy_flags = op_flags + extra_proxy_args
  undefok = _undefok_arg(proxy_flags)
  proxy.args = (
      list(proxy.args or []) + ([undefok] if undefok else []) + proxy_flags
  )

  worker_env = dict(op_env)
  if proxy_flags:
    libtpu_init_args = " ".join(proxy_flags)
    if worker_env.get(LIBTPU_INIT_ARGS_ENV):
      worker_env[LIBTPU_INIT_ARGS_ENV] = (
          f"{worker_env[LIBTPU_INIT_ARGS_ENV]} {libtpu_init_args}"
      )
    else:
      worker_env[LIBTPU_INIT_ARGS_ENV] = libtpu_init_args
    _upsert_env(proxy, {LIBTPU_INIT_ARGS_ENV: libtpu_init_args})
    _upsert_env(rm, {LIBTPU_INIT_ARGS_ENV: libtpu_init_args})

  oss_grpc_env = {OSS_INSECURE_GRPC_ENV[0]: OSS_INSECURE_GRPC_ENV[1]}
  _upsert_env(rm, oss_grpc_env)
  _upsert_env(proxy, oss_grpc_env)
  worker_env.setdefault(OSS_INSECURE_GRPC_ENV[0], OSS_INSECURE_GRPC_ENV[1])

  # op_flags.yaml `env` entries (plus `LIBTPU_INIT_ARGS` and
  # `TEST_UNDECLARED_OUTPUTS_DIR` for Louhi OSS) configure the TPU runtime in
  # the Pathways workers.
  if worker_env:
    _upsert_env(worker, worker_env)

  if args.active_deadline_seconds > 0:
    js.head_job_template.spec.active_deadline_seconds = (
        args.active_deadline_seconds
    )
  return Workload(
      jobset=js,
      benchmark_name=benchmark_name,
      proxy_flags=proxy_flags,
      worker_env=worker_env,
  )


def _summarize(args: argparse.Namespace, workload: Workload) -> None:
  """Prints what the generated JobSet will run."""
  js = workload.jobset
  head_pod_spec = js.head_job_template.spec.template.spec
  worker_pod_spec = js.worker_job_template.spec.template.spec
  rm = _find_container(head_pod_spec, "pathways-rm")
  proxy = _find_container(head_pod_spec, PROXY_CONTAINER)
  worker = _find_container(worker_pod_spec, WORKER_CONTAINER)
  print(f"JobSet:            {args.namespace}/{args.name}")
  print(
      f"TPUMS config:      {args.config} (benchmark"
      f" '{workload.benchmark_name}')"
  )
  print(f"TPUMS image:       {args.image}")
  print(f"Pathways RM:       {rm.image}")
  print(f"Pathways proxy:    {proxy.image}")
  print(f"Pathways worker:   {worker.image}")
  print(
      f"Slices:            {args.num_slices} x {args.tpu_type} {args.topology}"
  )
  print(f"Proxy flags added: {workload.proxy_flags or 'none'}")
  print(f"Worker env added:  {workload.worker_env or 'none'}")
  print(f"Results:           {args.gcs_output_dir or '(not uploaded)'}")
  print(f"XProf traces:      {args.xprof_dir or '(disabled)'}")


def main(argv: Sequence[str]) -> int:
  logging.basicConfig(level=logging.INFO)
  args = _parse_args(argv)
  workload = build_workload(args)
  _summarize(args, workload)
  js = workload.jobset

  output = args.output or os.path.join("/tmp", f"{args.name}.yaml")
  js.export_yaml(output)
  print(f"Wrote JobSet YAML: {output}")

  if args.apply:
    js.apply(recreate=True)
    print(f"Applied JobSet {args.namespace}/{args.name}.")
  return 0


if __name__ == "__main__":
  sys.exit(main(sys.argv[1:]))
