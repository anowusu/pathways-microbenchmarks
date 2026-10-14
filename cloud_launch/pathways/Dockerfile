# TPUMS client image for Pathways on Cloud.
#
# This image only runs the JAX *client* (the `tpums` CLI) in the
# `user-workload` container of the pathways-head pod. The TPU runtime and the
# XLA TPU compiler come from the Pathways server/proxy images, so the client
# does not need a matching libtpu (the libtpu wheel pulled in by jax[tpu] is
# unused when JAX_PLATFORMS=proxy).
#
# Build from the repository root (see build_image.sh):
#   docker build -f cloud_launch/pathways/Dockerfile -t <image> .
FROM python:3.12-slim

ARG TPUMS_GIT_COMMIT=unknown
ENV PYTHONUNBUFFERED=1 \
    TPUMS_GIT_COMMIT=${TPUMS_GIT_COMMIT}
LABEL org.opencontainers.image.source="https://github.com/AI-Hypercomputer/accelerator-microbenchmarks" \
      org.opencontainers.image.revision="${TPUMS_GIT_COMMIT}" \
      org.opencontainers.image.description="TPUMS client for Pathways on Cloud"

RUN pip install --no-cache-dir uv

# Cache warm-up only: pre-install the heavy dependencies in their own layer so
# that code changes do not re-download them. The install below resolves the
# real requirements from pyproject.toml.
RUN uv pip install --system --no-cache \
      "jax[tpu]==0.11.1" "jaxlib==0.11.1" tensorflow pandas numpy pyyaml \
      simple_parsing absl-py "pathwaysutils>=0.1.12"

WORKDIR /opt/tpums
COPY . /opt/tpums
RUN uv pip install --system --no-cache "/opt/tpums[pathways]" \
    && tpums benchmark list > /dev/null

CMD ["bash", "/opt/tpums/cloud_launch/pathways/run_benchmark.sh"]
