#!/bin/bash
# Settings shared by build_image.sh and pathways_launch.sh (the Pathways
# counterpart of ../common.sh). Every value can be overridden by exporting the
# variable before running the scripts. Values shown in comments are the ones
# used to validate this launcher on a v5e 4x8 Pathways cluster.

# --- GCP / GKE -----------------------------------------------------------------
export PROJECT_ID="${PROJECT_ID:-<ADD_YOUR_PROJECT_ID>}"        # cloud-tpu-multipod-dev
export CLUSTER_NAME="${CLUSTER_NAME:-<ADD_YOUR_CLUSTER_NAME>}"  # pw-scale-test-v5e-32
export CLUSTER_LOCATION="${CLUSTER_LOCATION:-<ADD_YOUR_REGION_OR_ZONE>}"  # us-south1
export NAMESPACE="${NAMESPACE:-default}"

# --- TPU slice -----------------------------------------------------------------
export TPU_TYPE="${TPU_TYPE:-v5e}"      # PathwaysJobSet tpu_type: v5e, v6e, tpu7x, ...
export TOPOLOGY="${TOPOLOGY:-4x8}"      # Must match a TPU node pool's topology.
export NUM_SLICES="${NUM_SLICES:-1}"

# --- Pathways ------------------------------------------------------------------
# Pathways server/proxy release. Pick the tag built for the JAX version pinned in
# pyproject.toml (jax 0.11.1), e.g. 20261013-jax_0.11.1.
export PATHWAYS_VERSION="${PATHWAYS_VERSION:-20261013-jax_0.11.1}"
export PATHWAYS_SERVER_IMAGE="${PATHWAYS_SERVER_IMAGE:-us-docker.pkg.dev/cloud-tpu-v2-images/pathways/server}"
export PATHWAYS_PROXY_IMAGE="${PATHWAYS_PROXY_IMAGE:-us-docker.pkg.dev/cloud-tpu-v2-images/pathways/proxy_server}"

# --- Scheduling (Kueue / priorities) -------------------------------------------
export KUEUE_QUEUE="${KUEUE_QUEUE:-multislice-queue}"   # Empty: no Kueue label.
export PRIORITY_CLASS="${PRIORITY_CLASS:-medium}"       # Empty: no priority class.
export HEAD_NODE_POOL="${HEAD_NODE_POOL:-cpu-np}"       # CPU pool for pathways-head.
# Hard limit on a run (pathways-head Job activeDeadlineSeconds), as in the
# Pathways CI workloads. Time spent queued in Kueue does not count.
export ACTIVE_DEADLINE_SECONDS="${ACTIVE_DEADLINE_SECONDS:-1800}"

# --- Images and storage --------------------------------------------------------
# Artifact Registry repository that build_image.sh pushes the TPUMS client image
# to, e.g. us-docker.pkg.dev/cloud-tpu-multipod-dev/pathways-test.
export IMAGE_REPOSITORY="${IMAGE_REPOSITORY:-us-docker.pkg.dev/${PROJECT_ID}/<ADD_YOUR_REPOSITORY>}"
export IMAGE_NAME="${IMAGE_NAME:-tpums-pathways}"
# Full image reference to run. build_image.sh prints the value to export.
export TPUMS_IMAGE="${TPUMS_IMAGE:-${IMAGE_REPOSITORY}/${IMAGE_NAME}:latest}"

# Results land in gs://${GCS_BUCKET}/${GS_PREFIX}/<run name>/ and Pathways
# scratch space in .../<run name>/pathways_scratch.
export GCS_BUCKET="${GCS_BUCKET:-<ADD_YOUR_BUCKET_NAME>}"       # cloud-pathways-staging
export GS_PREFIX="${GS_PREFIX:-tpums-pathways}"

# Collect XProf traces (written to GCS by the Pathways workers).
export ENABLE_XPROF="${ENABLE_XPROF:-true}"

# Python virtualenv for the launcher (see requirements.txt).
export VENV_DIR="${VENV_DIR:-$HOME/tpums_pathways_venv}"
