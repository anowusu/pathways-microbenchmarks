"""Runs one TPUMS config on Pathways on Cloud (GKE) as a Pathways JobSet.

The Pathways counterpart of gke_job_template.yaml: a single TPUMS client in the
`pathways-head` pod drives every TPU of the slice through the Pathways proxy
(JAX_PLATFORMS=proxy), and the Pathways workers write the XProf traces to GCS.
Results land in <gcs_dir>/results and traces in <gcs_dir>/xprof.

Requires kubectl credentials for the cluster and
`pip install "kubernetes>=28" "pathwaysutils>=0.1.13"` (for `PathwaysJobSet`).

Example (TPU v5e 4x8):
  python templates/pathways_jobset.py --name tpums-v5e-gemm \\
      --config configs/v5e/4x8/gemm.yaml --queue <kueue-local-queue> \\
      --gcs_dir gs://<bucket>/tpums-v5e-gemm --apply

The defaults are the Pathways release images; the tag has to match the JAX
version pinned in pyproject.toml. The open-source Pathways images (built for
JAX 0.12.0 at the time of writing) run the same way, with
`--pathways_version latest` and
  --pathways_server_image us-docker.pkg.dev/cloud-tpu-v2-images-dev/oss-pathways/server
  --pathways_proxy_image us-docker.pkg.dev/cloud-tpu-v2-images-dev/oss-pathways/proxy_server
"""

import argparse
import pathlib
from typing import Any, Optional

from pathwaysutils.experimental.gke import jobset
import yaml

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
_OP_FLAGS = _REPO_ROOT / "src" / "accelerator_microbenchmarks" / "op_flags.yaml"
_REPO_URL = "https://github.com/AI-Hypercomputer/accelerator-microbenchmarks.git"
_SERVER_IMAGE = jobset.DEFAULT_PATHWAYS_RM_AND_WORKER_IMAGE
_PROXY_IMAGE = jobset.DEFAULT_PATHWAYS_PROXY_IMAGE
# Same client image as gke_job_template.yaml. `pip install` pins the JAX version
# of pyproject.toml, which the Pathways image tag has to match.
_IMAGE = "us-docker.pkg.dev/cloud-tpu-images/jax-ai-image/tpu:latest"
# Results (summary.csv, detailed.json, git_commit.txt, execution.log) go to
# $GCS_RESULTS_DIR, the XProf traces the Pathways workers write to
# $GCS_XPROF_DIR; both default to <gcs_dir>/results and <gcs_dir>/xprof.
_RUN = """set -o pipefail
git clone --depth 1 --branch "$TPUMS_REF" "$TPUMS_REPO" /tmp/tpums
cd /tmp/tpums && pip install -q -e '.[pathways]' && mkdir -p /tmp/results
git rev-parse HEAD > /tmp/results/git_commit.txt
tpums benchmark run-config "$TPUMS_CONFIG" --output_dir=/tmp/results \\
    --xprof_dir="$GCS_XPROF_DIR" 2>&1 | tee /tmp/results/execution.log
rc=$?
gcloud storage cp /tmp/results/* "$GCS_RESULTS_DIR/"
exit $rc
"""


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
      description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
  )
  add = parser.add_argument
  add("--name", required=True, help="JobSet name (at most 22 characters).")
  add("--config", required=True, help="TPUMS config, relative to the repo.")
  add("--gcs_dir", required=True, help="gs:// directory of this run.")
  add("--tpu_type", default="v5e")
  add("--topology", default="4x8")
  add("--pathways_version", default="20261028-jax_0.12.0", help="Image tag.")
  add("--pathways_server_image", default=_SERVER_IMAGE, help="RM/worker image.")
  add("--pathways_proxy_image", default=_PROXY_IMAGE, help="Proxy image.")
  add("--queue", default="", help="Kueue LocalQueue name (default: none).")
  add(
      "--pod_priority_class",
      default="",
      help="Pod PriorityClass of the head and worker pods (default: none).",
  )
  add("--repo", default=_REPO_URL, help="TPUMS repository to run.")
  add("--ref", default="main", help="Branch or tag of --repo to run.")
  add("--apply", action="store_true", help="Create the JobSet, not just YAML.")
  return parser.parse_args()


def benchmark_op_flags(config_path: str) -> tuple[list[str], dict[str, str]]:
  """Returns the op_flags.yaml flags and env of the benchmark in `config_path`.

  Args:
    config_path: TPUMS config, relative to the repository root.
  """
  config = yaml.safe_load((_REPO_ROOT / config_path).read_text())
  op_flags = yaml.safe_load(_OP_FLAGS.read_text())
  op_flags = op_flags.get(config["benchmark"]["name"]) or {}
  flags = [str(flag) for flag in op_flags.get("flags") or []]
  env = {key: str(value) for key, value in (op_flags.get("env") or {}).items()}
  return flags, env


def build_pathways_jobset(
    name: str,
    config: str,
    gcs_dir: str,
    *,
    tpu_type: str = "v5e",
    topology: str = "4x8",
    pathways_version: str = "latest",
    pathways_server_image: str = _SERVER_IMAGE,
    pathways_proxy_image: str = _PROXY_IMAGE,
    queue: str = "",
    repo: str = _REPO_URL,
    ref: str = "main",
    image: str = _IMAGE,
    namespace: str = "default",
    results_dir: Optional[str] = None,
    xprof_dir: Optional[str] = None,
    extra_labels: Optional[dict[str, str]] = None,
    extra_env: Optional[dict[str, str]] = None,
    pod_priority_class: str = "",
) -> jobset.PathwaysJobSet:
  """Builds the Pathways JobSet that runs one TPUMS config.

  Args:
    name: JobSet name (at most 22 characters).
    config: TPUMS config, relative to the repository root.
    gcs_dir: `gs://` directory of the run; Pathways scratch goes to
      `<gcs_dir>/pathways`.
    tpu_type: Pathways TPU type, e.g. `v5e`.
    topology: Slice topology, e.g. `4x8`.
    pathways_version: Tag of both Pathways images.
    pathways_server_image: Resource manager / worker image, without tag.
    pathways_proxy_image: Proxy image, without tag.
    queue: Kueue LocalQueue label, or "" for none.
    repo: Repository the head container clones.
    ref: Branch or tag of `repo` to run.
    image: Client image; `pip install` upgrades its JAX to pyproject's pin.
    namespace: Kubernetes namespace.
    results_dir: Where summary.csv, detailed.json, git_commit.txt and
      execution.log go (default `<gcs_dir>/results`).
    xprof_dir: Where the workers write XProf traces (default
      `<gcs_dir>/xprof`); must be a `gs://` path the workers can reach.
    extra_labels: Additional JobSet labels, e.g. a Kueue priority class.
    extra_env: Additional environment variables of the client container.
    pod_priority_class: Pod PriorityClass of the head and worker pods, or ""
      for none. On shared Kueue clusters without WorkloadPriorityClasses this
      is what sets the workload priority (and so who preempts whom).

  Returns:
    The JobSet, ready for `export_yaml()` or `apply()`.
  """
  flags, env = benchmark_op_flags(config)
  # McJAX applies op_flags.yaml to the benchmark process through
  # LIBTPU_INIT_ARGS. On Pathways the proxy compiles and the workers execute,
  # so both take the flags on their command line (the open-source binaries
  # forward --xla_* and --megascale_* flags to libtpu).
  worker_kwargs: dict[str, Any] = {}
  if flags:
    worker_kwargs["args"] = flags
  if env:
    worker_kwargs["env"] = env
  labels = dict(extra_labels or {})
  if queue:
    labels["kueue.x-k8s.io/queue-name"] = queue
  pod_spec_kwargs = (
      {"priority_class_name": pod_priority_class} if pod_priority_class else None
  )
  pathways_jobset = jobset.PathwaysJobSet(
      name=name,
      namespace=namespace,
      pathways_dir=f"{gcs_dir}/pathways",
      tpu_type=tpu_type,
      topology=topology,
      num_slices=1,
      pathways_version=pathways_version,
      pathways_rm_and_worker_image=pathways_server_image,
      pathways_proxy_image=pathways_proxy_image,
      labels=labels or None,
      head_pod_spec_kwargs=pod_spec_kwargs,
      worker_pod_spec_kwargs=pod_spec_kwargs,
      proxy_container_kwargs={"args": flags} if flags else None,
      worker_container_kwargs=worker_kwargs or None,
  )
  return pathways_jobset.add_user_workload(
      image=image,
      command=["bash", "-c", _RUN],
      env={
          "TPUMS_REPO": repo,
          "TPUMS_REF": ref,
          "TPUMS_CONFIG": config,
          "GCS_DIR": gcs_dir,
          "GCS_RESULTS_DIR": results_dir or f"{gcs_dir}/results",
          "GCS_XPROF_DIR": xprof_dir or f"{gcs_dir}/xprof",
          "PYTHONUNBUFFERED": "1",
          **(extra_env or {}),
      },
  )


def build_jobset(args: argparse.Namespace) -> jobset.PathwaysJobSet:
  """Builds the Pathways JobSet that runs `args.config`."""
  return build_pathways_jobset(
      name=args.name,
      config=args.config,
      gcs_dir=args.gcs_dir,
      tpu_type=args.tpu_type,
      topology=args.topology,
      pathways_version=args.pathways_version,
      pathways_server_image=args.pathways_server_image,
      pathways_proxy_image=args.pathways_proxy_image,
      queue=args.queue,
      repo=args.repo,
      ref=args.ref,
      pod_priority_class=args.pod_priority_class,
  )


def main() -> None:
  args = _parse_args()
  pathways_jobset = build_jobset(args)
  pathways_jobset.export_yaml(f"/tmp/{args.name}.yaml")
  print(f"Wrote /tmp/{args.name}.yaml")
  if args.apply:
    pathways_jobset.apply(recreate=True)
    print(
        f"Created JobSet {args.name}. Follow it with: kubectl logs -f"
        f" -l jobset.sigs.k8s.io/jobset-name={args.name},jobset.sigs.k8s.io/"
        "replicatedjob-name=pathways-head -c user-workload"
    )


if __name__ == "__main__":
  main()
