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
# Entrypoint of the `user-workload` container in the Pathways head pod (see
# generate_workload.py). It is the Pathways counterpart of the inline command
# in ../xpk_launch.sh: run one TPUMS config, then copy the results to GCS.
#
# Differences from the McJAX flow:
#   * It runs once, in the pathways-head pod, instead of once per TPU VM; JAX
#     reaches every TPU in the slice(s) through the pathways-proxy sidecar
#     (JAX_PLATFORMS=proxy, set by PathwaysJobSet.add_user_workload).
#   * No libtpu override and no LIBTPU_INIT_ARGS: the TPU runtime and compiler
#     live in the Pathways server/proxy images, and op flags are passed to the
#     proxy by generate_workload.py.
#   * XProf traces are written straight to GCS by the Pathways workers, so they
#     are not copied here.
#
# Environment:
#   TPUMS_CONFIG          TPUMS config file inside the image (required).
#   TPUMS_GCS_OUTPUT_DIR  gs:// directory for the results (optional).
#   TPUMS_XPROF_DIR       gs:// directory for XProf traces (optional; empty
#                         disables XProf and reports wall-clock metrics only).
#   TPUMS_EXTRA_ARGS      Extra arguments for `tpums benchmark run-config`.

set -uo pipefail

CONFIG="${TPUMS_CONFIG:?TPUMS_CONFIG must be set}"
GCS_OUTPUT_DIR="${TPUMS_GCS_OUTPUT_DIR:-}"
XPROF_DIR="${TPUMS_XPROF_DIR:-}"
RESULTS_DIR=/tmp/results

mkdir -p "${RESULTS_DIR}"
cd "$(dirname "$0")/../.."  # Repository root inside the image (/opt/tpums).

echo "=== [TPUMS on Pathways] $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
echo "Config:          ${CONFIG}"
echo "Git commit:      ${TPUMS_GIT_COMMIT:-unknown}"
echo "JAX_PLATFORMS:   ${JAX_PLATFORMS:-}"
echo "Backend target:  ${JAX_BACKEND_TARGET:-}"
echo "Results:         ${GCS_OUTPUT_DIR:-(not uploaded)}"
echo "XProf traces:    ${XPROF_DIR:-(disabled)}"

xprof_args=()
if [ -n "${XPROF_DIR}" ]; then
  xprof_args=(--xprof_dir="${XPROF_DIR}")
fi

# shellcheck disable=SC2086  # TPUMS_EXTRA_ARGS is intentionally word-split.
tpums benchmark run-config "${CONFIG}" \
  --output_dir="${RESULTS_DIR}" \
  "${xprof_args[@]}" ${TPUMS_EXTRA_ARGS:-} 2>&1 | tee "${RESULTS_DIR}/execution.log"
rc=${PIPESTATUS[0]}

# `tpums` exits 0 even when every benchmark case failed, so treat a missing
# summary.csv as a failure.
if [ "${rc}" -eq 0 ] && [ ! -s "${RESULTS_DIR}/summary.csv" ]; then
  echo "ERROR: ${RESULTS_DIR}/summary.csv was not written; see the log above."
  rc=1
fi
echo "${TPUMS_GIT_COMMIT:-unknown}" > "${RESULTS_DIR}/git_commit.txt"
cp "${CONFIG}" "${RESULTS_DIR}/config.yaml" 2>/dev/null || true

if [ -n "${GCS_OUTPUT_DIR}" ]; then
  if ! python3 cloud_launch/pathways/upload_results.py "${RESULTS_DIR}" "${GCS_OUTPUT_DIR}"; then
    echo "ERROR: failed to upload results to ${GCS_OUTPUT_DIR}."
    rc=1
  fi
fi

echo "=== [TPUMS on Pathways] finished with exit code ${rc} ==="
exit "${rc}"
