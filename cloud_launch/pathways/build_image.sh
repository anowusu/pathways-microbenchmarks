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
# Builds the TPUMS Pathways client image from this checkout and pushes it to
# ${IMAGE_REPOSITORY}. The tag is <git sha>[-dirty]-<date> unless IMAGE_TAG is
# set. Prints the TPUMS_IMAGE value to export for pathways_launch.sh.
#
# Usage: cloud_launch/pathways/build_image.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/common.sh"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

GIT_COMMIT="$(git -C "${REPO_ROOT}" rev-parse --short=12 HEAD)"
if [ -n "$(git -C "${REPO_ROOT}" status --porcelain)" ]; then
  echo "Warning: the checkout has uncommitted changes; tagging the image -dirty."
  GIT_COMMIT="${GIT_COMMIT}-dirty"
fi
IMAGE_TAG="${IMAGE_TAG:-${GIT_COMMIT}-$(date -u +%Y%m%d)}"
IMAGE="${IMAGE_REPOSITORY}/${IMAGE_NAME}:${IMAGE_TAG}"

echo "Building ${IMAGE} from ${REPO_ROOT} ..."
DOCKER_BUILDKIT=1 docker build \
  -f "${SCRIPT_DIR}/Dockerfile" \
  --build-arg TPUMS_GIT_COMMIT="${GIT_COMMIT}" \
  -t "${IMAGE}" \
  "${REPO_ROOT}"

echo "Pushing ${IMAGE} ..."
docker push "${IMAGE}"

echo
echo "Built and pushed the TPUMS Pathways image. Use it with:"
echo "  export TPUMS_IMAGE=${IMAGE}"
