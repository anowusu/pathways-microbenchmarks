# Driver image of the TPUMS nightly runtime comparison (automation/nightly_runner.py).
# It submits the JobSets, polls them, aggregates summary.csv files, publishes
# the GCS report and writes BigQuery rows; the benchmarks themselves run in the
# TPU workload image, not here.
#
# Build and push (from the repository root):
#   gcloud builds submit --project cloud-tpu-multipod-dev \
#     --tag us-docker.pkg.dev/cloud-tpu-multipod-dev/pathways-dev/anowusu/tpums-nightly:latest \
#     --gcs-source-staging-dir gs://cloud-pathways-staging/tpums-nightly/build .
FROM python:3.12-slim

RUN apt-get update \
  && apt-get install -y --no-install-recommends git curl ca-certificates gnupg gcc libc6-dev \
  && curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg \
     | gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg \
  && echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" \
     > /etc/apt/sources.list.d/google-cloud-sdk.list \
  && apt-get update \
  && apt-get install -y --no-install-recommends google-cloud-cli kubectl google-cloud-cli-gke-gcloud-auth-plugin \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY automation/ /app/automation/
COPY templates/ /app/templates/
COPY configs/ /app/configs/
COPY src/accelerator_microbenchmarks/op_flags.yaml /app/src/accelerator_microbenchmarks/op_flags.yaml
COPY automation/k8s/requirements.txt /app/requirements.txt

RUN mkdir -p /opt/tpums \
  && gcc -shared -fPIC -O2 -Wall -Wextra /app/automation/oss_patch/pjrt_profiler_patch.c \
     -ldl -lpthread -o /opt/tpums/libpjrt_profiler_patch.so

# pathwaysutils (PathwaysJobSet) pulls a CPU jax; the TPU wheels are not needed
# to render manifests.
RUN pip install --no-cache-dir -r /app/requirements.txt

ENV PYTHONPATH=/app PYTHONUNBUFFERED=1 USE_GKE_GCLOUD_AUTH_PLUGIN=True
ENTRYPOINT ["python", "-m", "automation.nightly_runner"]
