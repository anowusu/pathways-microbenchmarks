# Nightly McJAX vs Pathways (Louhi, Rapid) Comparison on TPU v5e

A temporary reporting pipeline (branch `nightly` of
`anowusu/pathways-microbenchmarks`) that runs the TPUMS microbenchmarks every
night on three runtimes and reports them side by side, the way uBench does
(p50 / p90 / avg / std, bandwidth, TFLOP/s), until the Pathways runtimes are
onboarded onto uBench itself. It is the upstream `automation/` pipeline
([AUTOMATION.md](AUTOMATION.md)) plus a runtime matrix; it needs no change
outside this repository.

| Runtime | How a config runs | Images |
|---|---|---|
| `mcjax` | One JAX process per TPU host (`automation/jobset_template.yaml`) | client: `us-docker.pkg.dev/cloud-tpu-images/jax-ai-image/tpu:latest` (repo pins JAX 0.12.0) |
| `louhi` | One JAX client in a Pathways head, `PathwaysJobSet` (`templates/pathways_jobset.py`) | `us-docker.pkg.dev/cloud-tpu-v2-images/cloud-pathways/{server,proxy_server}:latest` (public Louhi build) |
| `rapid` | Same as `louhi` | `us-docker.pkg.dev/cloud-tpu-v2-images/pathways/{server,proxy_server}:latest` (internal Rapid build) |

Matrix: `automation/canonical_suite_v5e.yaml` — v5e `4x8` (8 hosts, 32
chips) × `all_gather`, `all_reduce`, `all_to_all`, `gemm`, `hbm` × 3 runtimes
= 15 JobSets, at most 3 in flight, ~15 minutes wall clock in total.

## Where things land

```
gs://cloud-pathways-staging/tpums-nightly/
├── <YYYY-MM-DD>/
│   ├── <runtime>/<config>/            summary.csv, detailed.json, execution*.log,
│   │                                  git_commit.txt, provenance.json, xprof/ (McJAX only),
│   │                                  pathways/ (Pathways scratch: compilation cache)
│   ├── aggregated_results/<category>.csv   one CSV per benchmark class, all runtimes
│   ├── consolidated_report.{md,html}
│   └── dashboard.html                 interactive charts (self-contained, see below)
├── reports/<YYYY-MM-DD>.{md,html}     copies of the consolidated report
├── reports/<YYYY-MM-DD>_dashboard.html
├── reports/latest.{md,html}           always the most recent run
└── reports/latest_dashboard.html
```

`dashboard.html` (`automation/dashboard.py`) is a self-contained page — the
charts are drawn by a little inline JavaScript, no external assets — with the
run's JobSets and outcomes, grouped bars of the p50 step time per case and
runtime (solid: XProf device time, hollow: wall clock), a Pathways / McJAX
ratio heat map, and achieved throughput per case. Open it from the Cloud
Console object page (Authenticated URL) or download it; hovering gives p90,
std, throughput and the slice.

`provenance.json` records the git commit of the benchmark code, the node
pools the pods ran on (`node_pools`; GKE names TPU slice nodes
`gke-tpu-<hash>-<id>`, so the prefix identifies the slice — microbenchmark
numbers can differ between slices, and the comparison table shows it as
`slice`) and, for the Pathways runtimes, the image digests: resolved from
Artifact Registry at submission when the caller may (with every tag alias of
that digest, e.g. `20261029_1817-jax_0.12.0`, `sha-…`), and always from the
digests the kubelet reports for the pods (`client_image_digest` for the JAX
client image too), which is what actually ran. The aggregator stamps these
onto every row as the `runtime`, `pathways_server_digest`,
`client_image_digest`, `node_pools`, … columns.

BigQuery: `cloud-tpu-multipod-dev.tpums_nightly.run_summary_microbenchmarks`
(created on first write) uses uBench's `MicrobenchmarkSchema` column names, so
rows can be copied into uBench's tables later. `run_name` / `run_group` are
`tpums-nightly-<date>`, `run_id` is
`<run_name>_<runtime>_<benchmark>_<hash of the case parameters>` and stable
across re-runs (a re-run of a day, or of one config, replaces its own rows
via a load job instead of duplicating them), the runtime is
`params_other.runtime`, `metrics_other` carries the full `summary.csv` metric
set, and Pathways rows put the server / proxy images and digests in
`configs_container_image` and `params_other`.

```sql
SELECT JSON_VALUE(params_other, '$.runtime') AS runtime, benchmark_name,
       JSON_VALUE(params_other, '$.matrix_dim') AS matrix_dim,
       metrics_time_per_iter_ms_p50, metrics_achieved_bandwidth_gbps_p50,
       metrics_tflops_achieved_p50
FROM `cloud-tpu-multipod-dev.tpums_nightly.run_summary_microbenchmarks`
WHERE run_group = 'tpums-nightly-2026-10-29'
ORDER BY benchmark_name, matrix_dim, runtime;
```

Report sections: the upstream per-category tables plus **Runtimes** (images,
digests, commit) and **Runtime Comparison** (per test case one row per
runtime, Pathways / McJAX ratio on `xprof_p50_ms` when both sides have XProf,
else on `wall_clock_p50_ms`, and an XProf-availability marker). The public
Louhi images do not expose device traces, so Louhi rows are wall-clock only;
the Rapid images do, so Rapid rows carry both (device time matches McJAX,
wall clock adds the proxy round trip).

## Scheduling (GKE CronJob)

`automation/k8s/nightly_cronjob.yaml` runs the driver image inside the
cluster `pw-scale-test-v5e-32` (project `cloud-tpu-multipod-dev`, region
`us-south1`) at 09:00 UTC daily.

* KSA `default/tpums-nightly` → GSA
  `tpums-nightly@cloud-tpu-multipod-dev.iam.gserviceaccount.com` (Workload
  Identity): `storage.objectAdmin` on the bucket, `bigquery.dataEditor` +
  `bigquery.jobUser`, `container.developer`.
* Role `tpums-nightly`: create / watch / delete JobSets, read Jobs, Pods,
  logs and events in `default`.
* Driver image
  `us-docker.pkg.dev/cloud-tpu-multipod-dev/pathways-dev/anowusu/tpums-nightly:{latest,<short-sha>}`
  built with `automation/k8s/cloudbuild.yaml`; it only renders manifests,
  polls and aggregates — the benchmarks run in the TPU workload images.

```bash
# Rebuild the driver after changing automation/ or templates/:
gcloud builds submit --project cloud-tpu-multipod-dev \
  --config automation/k8s/cloudbuild.yaml \
  --substitutions=SHORT_SHA=$(git rev-parse --short HEAD) \
  --gcs-source-staging-dir gs://cloud-pathways-staging/tpums-nightly/build .

# (Re)install the CronJob, run it now, follow it:
kubectl apply -f automation/k8s/nightly_cronjob.yaml
kubectl create job --from=cronjob/tpums-nightly tpums-nightly-manual
kubectl logs -f job/tpums-nightly-manual
```

The workloads clone branch `nightly` of the fork at run time, so changes to
benchmarks, configs or templates take effect the next night without a
rebuild; only driver-side changes (`automation/*.py`) need a new image.

## Running it by hand

```bash
source ~/tpums_launcher_venv/bin/activate      # kubernetes, pathwaysutils, google-cloud-bigquery
export KUBECONFIG=~/.kube/tpums-pathways-pw-scale-test-v5e-32.yaml
PYTHONPATH=.:src python -m automation.nightly_runner \
  --cluster pw-scale-test-v5e-32 --project cloud-tpu-multipod-dev \
  --region us-south1 --zone us-south1-a --namespace default \
  --queue multislice-queue --priority "" --pod-priority-class medium \
  --suite-file automation/canonical_suite_v5e.yaml --runtimes mcjax,louhi,rapid \
  --max-parallel 3 --timeout-minutes 25 \
  --git-repo https://github.com/anowusu/pathways-microbenchmarks.git --git-branch nightly \
  --docker-image us-docker.pkg.dev/cloud-tpu-images/jax-ai-image/tpu:latest \
  --gcs-bucket gs://cloud-pathways-staging/tpums-nightly \
  --bq-table cloud-tpu-multipod-dev.tpums_nightly.run_summary_microbenchmarks
```

Useful variations: `--runtimes rapid --configs configs/v5e/4x8/gemm.yaml` for
one cell of the matrix, `--dry-run` to render the manifests only,
`--no-cleanup` to keep the JobSets, and `python -m automation.metrics_aggregator`
/ `python -m automation.report_generator` to re-aggregate a day without
re-running anything (see [AUTOMATION.md](AUTOMATION.md)).

The day is the unit of reporting: every run overwrites the `summary.csv` of
the configs it ran and then aggregates **everything under the day's prefix**,
so a partial or repeated run merges into the day's report and BigQuery rows
(same `run_id` → replaced) instead of replacing the whole day. The report
header's "JobSets: x/y" still describes the run that produced it.

### Cluster specifics worth knowing

* **Priorities.** The cluster has pod `PriorityClass`es
  (`very-low` … `very-high`) but no Kueue `WorkloadPriorityClass`, so
  `--priority ""` drops the `kueue.x-k8s.io/priority-class` label (Kueue
  rejects unknown names) and `--pod-priority-class medium` sets
  `priorityClassName` on every benchmark pod. Kueue takes the workload
  priority from it; at priority 0 the benchmarks were preempted by every other
  job on this shared cluster.
* **Kueue.** Six LocalQueues exist, so the queue is given explicitly
  (`multislice-queue`). The Pathways head needs 24 CPUs of the `cpu-np`
  flavor quota; when the ClusterQueue is full the JobSet waits (the driver
  gives each JobSet `--timeout-minutes`).
* **Names.** Pathways JobSets are named `tpums-<runtime>-<config>` (≤ 22
  characters) and recreated if one exists; McJAX JobSets use `generateName`.
* **Preemption and retries.** Higher-priority jobs can still preempt a
  `medium` benchmark. When Kueue re-admits a preempted Pathways JobSet, the
  new resource manager registers with the old, still-terminating head (same
  DNS name), its watchdog kills it, and the JAX client hangs on the dead
  backend. The driver fails such a JobSet as soon as a head container has
  restarted (instead of waiting for `--timeout-minutes`) and, with
  `--retries 1`, deletes it and submits the config again under a new name
  (`tpums-rapid-all-gat-r2`), so the retry never shares a DNS name with the
  previous attempt. Only the last attempt is reported; the earlier error is
  kept in the result's message.

## Migration path to uBench / Helm

* `templates/pathways_jobset.py` is the same `PathwaysJobSet` rendering the
  uBench Pathways recipe uses; uBench can call it (or render the same manifest
  with the Helm chart) and keep `op_flags.yaml` as the single source of XLA
  flags.
* `automation/bq_writer.py` already emits uBench's column names; switching to
  uBench's own writer is a matter of pointing it at the same rows.
* `automation/runtimes.py` is the only place that knows the Louhi / Rapid
  image paths; adding a runtime (or pinning a tag) is a one-line change.
