"""Runs one TPUMS config on Pathways on Cloud (GKE) as a Pathways JobSet.

The Pathways counterpart of gke_job_template.yaml: a single TPUMS client in the
`pathways-head` pod drives every TPU of the slice through the Pathways proxy
(JAX_PLATFORMS=proxy), and the Pathways workers write the XProf traces to GCS.
Results land in <gcs_dir>/results and traces in <gcs_dir>/xprof.

Requires kubectl credentials for the cluster and
`pip install kubernetes "pathwaysutils>=0.1.13"` (for `PathwaysJobSet`).

Example (TPU v5e 4x8):
  python templates/pathways_jobset.py --name tpums-v5e-gemm \\
      --config configs/v5e/4x8/gemm.yaml --queue <kueue-local-queue> \\
      --gcs_dir gs://<bucket>/tpums-v5e-gemm --apply
"""

import argparse
import pathlib

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
_RUN = """set -o pipefail
git clone --depth 1 --branch "$TPUMS_REF" "$TPUMS_REPO" /tmp/tpums
cd /tmp/tpums && pip install -q -e '.[pathways]' && mkdir -p /tmp/results
tpums benchmark run-config "$TPUMS_CONFIG" --output_dir=/tmp/results \\
    --xprof_dir="$GCS_DIR/xprof" 2>&1 | tee /tmp/results/execution.log
rc=$?
gcloud storage cp /tmp/results/* "$GCS_DIR/results/"
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
  add("--pathways_version", default="20261018-jax_0.11.1", help="Image tag.")
  add("--pathways_server_image", default=_SERVER_IMAGE, help="RM/worker image.")
  add("--pathways_proxy_image", default=_PROXY_IMAGE, help="Proxy image.")
  add("--queue", default="", help="Kueue LocalQueue name (default: none).")
  add("--repo", default=_REPO_URL, help="TPUMS repository to run.")
  add("--ref", default="main", help="Branch or tag of --repo to run.")
  add("--apply", action="store_true", help="Create the JobSet, not just YAML.")
  return parser.parse_args()


def build_jobset(args: argparse.Namespace) -> jobset.PathwaysJobSet:
  """Builds the Pathways JobSet that runs `args.config`."""
  config = yaml.safe_load((_REPO_ROOT / args.config).read_text())
  op_flags = yaml.safe_load(_OP_FLAGS.read_text())
  op_flags = op_flags.get(config["benchmark"]["name"]) or {}
  flags = [str(flag) for flag in op_flags.get("flags") or []]
  env = {key: str(value) for key, value in (op_flags.get("env") or {}).items()}
  # McJAX applies op_flags.yaml to the benchmark process through
  # LIBTPU_INIT_ARGS. On Pathways the proxy compiles and the workers execute,
  # so both get the flags: Rapid images read them from the proxy command line,
  # Louhi OSS images from LIBTPU_INIT_ARGS (--undefok covers the difference).
  tpu_env = {"LIBTPU_INIT_ARGS": " ".join(flags)} if flags else {}
  undefok = "--undefok=" + ",".join(f.lstrip("-").split("=")[0] for f in flags)
  worker_env = {**env, **tpu_env}
  pathways_jobset = jobset.PathwaysJobSet(
      name=args.name,
      namespace="default",
      pathways_dir=f"{args.gcs_dir}/pathways",
      tpu_type=args.tpu_type,
      topology=args.topology,
      num_slices=1,
      pathways_version=args.pathways_version,
      pathways_rm_and_worker_image=args.pathways_server_image,
      pathways_proxy_image=args.pathways_proxy_image,
      labels={"kueue.x-k8s.io/queue-name": args.queue} if args.queue else None,
      proxy_container_kwargs=(
          {"args": [undefok, *flags], "env": tpu_env} if flags else None
      ),
      worker_container_kwargs={"env": worker_env} if worker_env else None,
  )
  return pathways_jobset.add_user_workload(
      image=_IMAGE,
      command=["bash", "-c", _RUN],
      env={
          "TPUMS_REPO": args.repo,
          "TPUMS_REF": args.ref,
          "TPUMS_CONFIG": args.config,
          "GCS_DIR": args.gcs_dir,
          "PYTHONUNBUFFERED": "1",
      },
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
