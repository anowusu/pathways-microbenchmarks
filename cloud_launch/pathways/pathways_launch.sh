#!/bin/bash
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
#
# Runs one TPUMS config on Pathways on Cloud: the Pathways counterpart of
# ../xpk_launch.sh. Where xpk_launch.sh calls `xpk workload create` to start one
# McJAX process per TPU VM, this script generates a Pathways JobSet with
# pathwaysutils' PathwaysJobSet (generate_workload.py), applies it with kubectl,
# waits for it to finish and prints the benchmark log.
#
# Usage:
#   cloud_launch/pathways/pathways_launch.sh <CONFIG> [RUN_NAME]
#
#   CONFIG    TPUMS config relative to the repository root, e.g.
#             configs/v5e/4x8/all_gather.yaml.
#   RUN_NAME  JobSet name. Default: tpums-pw-<config name>.
#
# Settings come from common.sh (override them with environment variables).
# Extra environment knobs:
#   OP_FLAGS=auto|none|<file>  op_flags.yaml handling (see generate_workload.py).
#   EXTRA_PROXY_ARGS="..."     Extra pathways-proxy flags.
#   WAIT=true|false            Wait for the JobSet to finish (default true).
#   CLEANUP=true|false         Delete the JobSet after it finishes (default
#                              false; like xpk_launch.sh, the next launch with
#                              the same RUN_NAME deletes it).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

if [ -z "${1:-}" ]; then
  sed -n '/^# Usage:/,/^$/p' "$0"
  exit 1
fi

CONFIG="$1"
CONFIG_NAME="$(basename "${CONFIG}" .yaml | tr '_.' '--' | tr '[:upper:]' '[:lower:]')"
RUN_NAME="${2:-tpums-pw-${CONFIG_NAME}}"
RUN_NAME="$(echo "${RUN_NAME:0:40}" | sed 's/-*$//')"
OP_FLAGS="${OP_FLAGS:-auto}"
WAIT="${WAIT:-true}"
CLEANUP="${CLEANUP:-false}"

BASE_OUTPUT_DIR="gs://${GCS_BUCKET}/${GS_PREFIX}/${RUN_NAME}"
PATHWAYS_DIR="${BASE_OUTPUT_DIR}/pathways_scratch"
XPROF_DIR=""
if [ "${ENABLE_XPROF}" = "true" ]; then
  XPROF_DIR="${BASE_OUTPUT_DIR}/xprof"
fi
YAML_PATH="/tmp/${RUN_NAME}.yaml"
SELECTOR="jobset.sigs.k8s.io/jobset-name=${RUN_NAME}"

# 1. Launcher virtualenv with pathwaysutils[gke] (PathwaysJobSet).
if [ ! -x "${VENV_DIR}/bin/python" ]; then
  echo "Creating launcher virtualenv ${VENV_DIR} ..."
  python3 -m venv "${VENV_DIR}"
  "${VENV_DIR}/bin/pip" install -q -r "${SCRIPT_DIR}/requirements.txt"
fi

# 2. Use a private kubeconfig for the target cluster, so the launcher neither
#    depends on nor changes your current kubectl context (and is not affected
#    by other tools rewriting ~/.kube/config while it runs).
export KUBECONFIG="${TPUMS_KUBECONFIG:-${HOME}/.kube/tpums-pathways-${CLUSTER_NAME}.yaml}"
if ! kubectl config current-context > /dev/null 2>&1; then
  gcloud container clusters get-credentials "${CLUSTER_NAME}" \
    --location="${CLUSTER_LOCATION}" --project="${PROJECT_ID}"
fi

# Runs kubectl, retrying transient failures (e.g. auth plugin hiccups).
kubectl_retry() {
  local attempt
  for attempt in 1 2 3; do
    if kubectl "$@"; then
      return 0
    fi
    if [ "${attempt}" -lt 3 ]; then
      echo "kubectl $1 failed (attempt ${attempt}); retrying in 10s..."
      sleep 10
    fi
  done
  return 1
}

# 3. Clean up the previous run with the same name (results and JobSet).
echo "Deleting ${BASE_OUTPUT_DIR} if it exists..."
gcloud storage rm -r "${BASE_OUTPUT_DIR}" > /dev/null 2>&1 || true

echo "Deleting JobSet ${RUN_NAME} if it exists..."
kubectl_retry delete jobset "${RUN_NAME}" -n "${NAMESPACE}" --ignore-not-found
WAIT_TIME=0
while true; do
  NUM_PODS=$(kubectl get pods -n "${NAMESPACE}" -l "${SELECTOR}" --no-headers 2>/dev/null | wc -l)
  if [ "${NUM_PODS}" -eq 0 ]; then
    break
  fi
  echo "JobSet ${RUN_NAME} still terminating... ${NUM_PODS} pods left. Waited ${WAIT_TIME}s"
  sleep 10
  WAIT_TIME=$((WAIT_TIME + 10))
done

# 4. Generate the JobSet with PathwaysJobSet and create it.
"${VENV_DIR}/bin/python" "${SCRIPT_DIR}/generate_workload.py" \
  --name="${RUN_NAME}" \
  --namespace="${NAMESPACE}" \
  --config="${CONFIG}" \
  --image="${TPUMS_IMAGE}" \
  --tpu_type="${TPU_TYPE}" \
  --topology="${TOPOLOGY}" \
  --num_slices="${NUM_SLICES}" \
  --pathways_version="${PATHWAYS_VERSION}" \
  --pathways_server_image="${PATHWAYS_SERVER_IMAGE}" \
  --pathways_proxy_image="${PATHWAYS_PROXY_IMAGE}" \
  --pathways_dir="${PATHWAYS_DIR}" \
  --gcs_output_dir="${BASE_OUTPUT_DIR}" \
  --xprof_dir="${XPROF_DIR}" \
  --op_flags="${OP_FLAGS}" \
  --queue="${KUEUE_QUEUE}" \
  --priority_class="${PRIORITY_CLASS}" \
  --head_node_pool="${HEAD_NODE_POOL}" \
  --active_deadline_seconds="${ACTIVE_DEADLINE_SECONDS}" \
  --output="${YAML_PATH}" \
  --extra_proxy_args="${EXTRA_PROXY_ARGS:-}"

kubectl_retry apply -f "${YAML_PATH}"
echo "Created JobSet ${NAMESPACE}/${RUN_NAME} (manifest: ${YAML_PATH})."
echo "Follow the benchmark with:"
echo "  KUBECONFIG=${KUBECONFIG} kubectl logs -n ${NAMESPACE} -f -l ${SELECTOR},jobset.sigs.k8s.io/replicatedjob-name=pathways-head -c user-workload"
LOGGING_QUERY="resource.type=\"k8s_container\"
resource.labels.namespace_name=\"${NAMESPACE}\"
resource.labels.project_id=\"${PROJECT_ID}\"
resource.labels.cluster_name=\"${CLUSTER_NAME}\"
resource.labels.location=\"${CLUSTER_LOCATION}\"
labels.\"k8s-pod/jobset_sigs_k8s_io/jobset-name\"=\"${RUN_NAME}\""
echo "Cloud Logging (all Pathways containers of this run):"
echo "  https://console.cloud.google.com/logs/query;query=$(python3 -c 'import sys, urllib.parse; print(urllib.parse.quote(sys.argv[1], safe=""))' "${LOGGING_QUERY}");project=${PROJECT_ID}"

if [ "${WAIT}" != "true" ]; then
  echo "Results will be uploaded to ${BASE_OUTPUT_DIR}"
  exit 0
fi

# 5. Wait for the JobSet to finish, then show the benchmark log.
STATE=""
WAIT_TIME=0
while [ -z "${STATE}" ]; do
  sleep 30
  WAIT_TIME=$((WAIT_TIME + 30))
  if ! JOBSET_STATUS=$(kubectl get jobset "${RUN_NAME}" -n "${NAMESPACE}" \
      -o jsonpath='{.status.terminalState}' 2>&1); then
    if echo "${JOBSET_STATUS}" | grep -q NotFound; then
      echo "JobSet ${RUN_NAME} was deleted before it finished."
      exit 1
    fi
    JOBSET_STATUS=""  # Transient API error; keep waiting.
  fi
  STATE="${JOBSET_STATUS}"
  PODS=$(kubectl get pods -n "${NAMESPACE}" -l "${SELECTOR}" --no-headers 2>/dev/null | awk '{print $3}' | sort | uniq -c | xargs echo)
  echo "[${WAIT_TIME}s] JobSet ${RUN_NAME}: ${STATE:-running}; pods: ${PODS:-none yet}"
done

HEAD_POD=$(kubectl get pods -n "${NAMESPACE}" \
  -l "${SELECTOR},jobset.sigs.k8s.io/replicatedjob-name=pathways-head" \
  -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || true)
if [ -n "${HEAD_POD}" ]; then
  echo "===== user-workload log (${HEAD_POD}) ====="
  kubectl logs -n "${NAMESPACE}" "${HEAD_POD}" -c user-workload --tail=400 || true
  echo "===== end of user-workload log ====="
fi

echo "JobSet ${RUN_NAME} finished: ${STATE}"
echo "Results: ${BASE_OUTPUT_DIR}"
gcloud storage ls "${BASE_OUTPUT_DIR}" 2>/dev/null || true

if [ "${CLEANUP}" = "true" ]; then
  kubectl delete jobset "${RUN_NAME}" -n "${NAMESPACE}" --ignore-not-found
fi

[ "${STATE}" = "Completed" ]
