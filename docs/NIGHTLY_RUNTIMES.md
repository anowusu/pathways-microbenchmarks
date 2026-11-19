# Hourly McJAX and Pathways (Louhi, Rapid) comparison on TPU v5e

A temporary reporting pipeline (branch `nightly` of
`anowusu/pathways-microbenchmarks`) that runs the TPUMS microbenchmarks every
hour on three runtimes and reports them side by side, the way uBench does
(p50 / p90 / avg / std, bandwidth, TFLOP/s), until the Pathways runtimes are
onboarded onto uBench itself. It is the upstream `automation/` pipeline
([AUTOMATION.md](AUTOMATION.md)) plus a runtime matrix; it needs no change
outside this repository.

| Runtime | How a config runs | Images |
|---|---|---|
| `mcjax` | One JAX process per TPU host (`automation/jobset_template.yaml`) | client: `us-docker.pkg.dev/cloud-tpu-images/jax-ai-image/tpu:latest` (repo pins JAX 0.12.0) |
| `louhi` | One JAX client in a Pathways head, `PathwaysJobSet` (`templates/pathways_jobset.py`) | `us-docker.pkg.dev/cloud-tpu-v2-images/cloud-pathways/{server,proxy_server}:latest` (public Louhi build) |
| `rapid` | Same as `louhi` | `us-docker.pkg.dev/cloud-tpu-v2-images/pathways/{server,proxy_server}:latest` (internal Rapid build) |

The matrix is `automation/canonical_suite_v5e.yaml`: seven benchmarks on
three runtimes, which makes 21 JobSets with at most 3 in flight. A run takes
about 20 minutes.

| Benchmarks | Where the benchmarks run |
|---|---|
| `all_gather`, `all_reduce`, `all_to_all`, `gemm`, `hbm` | One v5e `4x8` slice (8 hosts, 32 chips) |
| `dcn_all_reduce`, `dcn_all_to_all` | Two v5e `4x8` slices. The collective crosses the data center network (DCN). |

## Where things land

```
gs://cloud-pathways-staging/tpums-nightly/
├── <YYYY-MM-DD>/
│   ├── <runtime>/<config>/            summary.csv, detailed.json, execution*.log,
│   │                                  git_commit.txt, provenance.json, jobset.yaml, XProf traces
│   │                                  (McJAX: tensorboard/, Pathways: xprof/),
│   │                                  pathways/ (Pathways scratch: compilation cache)
│   ├── aggregated_results/<category>.csv   one CSV per benchmark class, all runtimes
│   ├── consolidated_report.{md,html}
│   └── dashboard.html                 interactive charts (self-contained, see below)
├── reports/<YYYY-MM-DD>.{md,html}     copies of the consolidated report
├── reports/<YYYY-MM-DD>_dashboard.html
├── reports/latest.{md,html}           always the most recent run
└── reports/latest_dashboard.html
```

`dashboard.html` (`automation/dashboard.py`) is a self-contained page: inline
JavaScript draws the charts, and the page loads no external assets. The page
shows the following:

*   The JobSets of the run and their outcomes.
*   Grouped bars per case and runtime, in two sections: the benchmarks within
    one slice, and the collectives between slices over the DCN.
*   Three metrics: achieved throughput, XProf device time, and wall-clock
    time. The wall-clock view stacks the two times: the solid bottom of a bar
    is the device time, and the striped top is the time spent outside the TPU.
    To open the page on a metric, add `#thr`, `#dev`, or `#wall` to the URL.
*   The trend of every case over the last runs.
*   A note that names every workload that ran with a debugging override, and
    a dagger (†) on each of its bars. See
    [Debugging override for `dcn_all_reduce`](#debugging-override-for-dcn_all_reduce).

To view the page, open it from the object page in the Google Cloud console
(**Authenticated URL**) or download it. To see the p90, the mean, the
throughput, and the slice of a bar, hold the pointer over the bar.

`jobset.yaml` is the JobSet manifest as the driver submitted it, so that a
run can be read back and repeated. A retry replaces the file with the
manifest of the new attempt.

`provenance.json` records the git commit of the benchmark code, the node
pools the pods ran on (`node_pools`; GKE names TPU slice nodes
`gke-tpu-<hash>-<id>`, so the prefix identifies the slice — microbenchmark
numbers can differ between slices, and the comparison table shows it as
`slice`) and, for the Pathways runtimes, the image digests: resolved from
Artifact Registry at submission when the caller may (with every tag alias of
that digest, e.g. `20261029_1817-jax_0.12.0`, `sha-…`), and always from the
digests the kubelet reports for the pods (`client_image_digest` for the JAX
client image too), which is what actually ran. For a workload that ran with
a debugging override, `provenance.json` also records
`pathways_debug_overrides`: the container and the flag, read from the manifest
that the driver submitted. The aggregator stamps these onto every row as the
`runtime`, `pathways_server_digest`, `pathways_debug_overrides`,
`client_image_digest`, `node_pools`, … columns.

For a Pathways workload, `provenance.json` also records the node of the head
pod (`pathways_head_node`). The client, the proxy, and the resource manager
run there, on a CPU node that other pods share, so the node is part of what
a Pathways wall-clock time measures.

BigQuery: `cloud-tpu-multipod-dev.tpums_nightly.run_summary_microbenchmarks`
(created on first write) uses uBench's `MicrobenchmarkSchema` column names, so
rows can be copied into uBench's tables later. `run_name` / `run_group` are
`tpums-nightly-<date>`, `run_id` is
`<run_name>_<runtime>_<benchmark>_<hash of the case parameters>` and stable
across re-runs (a re-run of a day, or of one config, replaces its own rows
via a load job instead of duplicating them), the runtime is
`params_other.runtime`, `metrics_other` carries the full `summary.csv` metric
set, and Pathways rows put the server / proxy images and digests in
`configs_container_image` and `params_other`. A debugging override is in
`params_other.pathways.debug_overrides`. The override isn't part of the case,
so the `run_id` stays the same with and without the override.

```sql
SELECT JSON_VALUE(params_other, '$.runtime') AS runtime, benchmark_name,
       JSON_VALUE(params_other, '$.matrix_dim') AS matrix_dim,
       metrics_time_per_iter_ms_p50, metrics_achieved_bandwidth_gbps_p50,
       metrics_tflops_achieved_p50
FROM `cloud-tpu-multipod-dev.tpums_nightly.run_summary_microbenchmarks`
WHERE run_group = 'tpums-nightly-2026-10-29'
ORDER BY benchmark_name, matrix_dim, runtime;
```

Report sections: the upstream per-category tables plus the following:

*   **Runtimes**: the images, the digests, and the commit of each runtime.
*   **Debugging overrides**: present only when a workload ran with a debugging
    override. The section lists the runtime, the benchmark, the container, the
    flag, and what the flag does.
*   **Runtime Comparison**: one row per runtime for every case, with the ratio
    of the step time on that runtime to the step time on McJAX. The ratio uses
    `xprof_p50_ms` when both runtimes have XProf device time, and
    `wall_clock_p50_ms` otherwise. The `xprof` column says how many runs the
    device trace covers. The `notes` column marks the rows that ran with a
    debugging override.

All three runtimes report device time. The stock Louhi and Rapid images both
write a device trace, so the pipeline runs the images unmodified.

### Which device the device time comes from

The device time of a collective depends on where the device sits in the mesh,
so every runtime times the same device: JAX device 0.

*   On Pathways, the client times JAX device 0.
*   On McJAX, every host times its own first device, and the host of JAX
    process 0 publishes the results. The pod index and the JAX process index
    differ, so results published from pod 0 come from an arbitrary device.

In the trace of a DCN benchmark, no operation carries the marker that the
other benchmarks use. The device time of a DCN benchmark is the duration of
the XLA module on device 0, which includes the transfer between slices.

## DCN benchmarks

`dcn_all_reduce` and `dcn_all_to_all` run a collective between two slices. The
automation gives every `dcn_*` benchmark two slices of the topology of its
config. To change the number of slices, set `TPUMS_DCN_NUM_SLICES`.

*   On McJAX, the JobSet has one `slice-job` replica per slice, and GKE sets
    the `MEGASCALE_*` variables.
*   On Pathways, `PathwaysJobSet` gets `num_slices`, and the resource manager
    expects that many instances.

Transfers over the network vary more from run to run than work within a
slice. In the v5e validation runs, the device time of one McJAX case differed
by 10 to 12 percent from one run to the next. Compare runtimes on the trend,
not on one run.

### Op flags that differ on an accelerator

`op_flags.yaml` sizes the megascale gRPC premap of `dcn_all_reduce` for tpu7x
and v6e hosts: 128 GiB. A v5e host has 188 GiB of memory, and the kubelet
evicts a pod that holds 128 GiB when the node runs low on memory.

`automation/op_flags/v5e.yaml` holds the entries of `op_flags.yaml` that
differ on v5e: `dcn_all_reduce` premaps 64 GiB. A benchmark that the file
lists runs with the file on every runtime:

*   McJAX and the Pathways client get the file through
    `--xla_flags_file_path`.
*   The Pathways workers and the Pathways proxy get the flags of the file on
    their command lines.

`tests/test_op_flag_overrides.py` keeps each entry identical to
`op_flags.yaml` except for the documented flags.

### Debugging override for `dcn_all_reduce`

`op_flags.yaml` turns the enhanced launch barrier off for `dcn_all_reduce`
(`--xla_tpu_use_enhanced_launch_barrier=false`), and McJAX runs the benchmark
that way. On Pathways, that setting needs a debugging override:

*   Whether the barrier is on is part of the slice topology that a worker
    reports. The resource manager of the Pathways on Cloud images always
    expects the barrier on, and exits when the workers report otherwise ("The
    newly added slice topology is not expected by the resource manager").
*   No flag changes what the resource manager expects. The libtpu flag on the
    command line of the resource manager has no effect, and neither does the
    debugging override of `--pathways_tpu_use_enhanced_launch_barrier`.
*   The resource manager checks worker topologies only while late binding is
    on. The Pathways on Cloud binaries refuse non-default `--pathways_*`
    flags, so the pipeline turns late binding off through the debugging flag
    of the resource manager:

    ```text
    --temporary_flags_for_debugging=temporary_flag_for_debugging_enable_late_binding=false
    ```

With late binding off, the client gets its devices only after the workers
have joined. The proxy and the workers run with the same flags as a McJAX
run. On v5e, `all_gather` times the same with and without the override.

The override is a debugging configuration, not a supported one. The pipeline
applies the override only to a benchmark whose op flags turn the barrier off,
and every output names the workloads that ran with the override:

| Output | Where the override appears |
|---|---|
| `provenance.json` | `pathways_debug_overrides` |
| Aggregated CSV | `pathways_debug_overrides` column |
| Markdown and HTML report | **Debugging overrides** section, and the `notes` column of the comparison tables |
| Dashboard | Note above the charts, and a dagger (†) on each bar |
| BigQuery | `params_other.pathways.debug_overrides` |
| Driver log | Warning when the driver prepares the workload |

## Scheduling (GKE CronJob)

`automation/k8s/nightly_cronjob.yaml` runs the driver image inside the
cluster `pw-scale-test-v5e-32` (project `cloud-tpu-multipod-dev`, region
`us-south1`) every hour, at minute 0 (UTC).

* KSA `default/tpums-nightly` → the project's Compute Engine default service
  account (Workload Identity), which writes to the bucket and to the BigQuery
  dataset and reads the Pathways images.
* The driver pod runs with `priorityClassName: medium`, like the benchmark
  pods. Kueue manages the driver Job too. At priority 0, Kueue preempts the
  driver for any workload that needs `cpu-np` quota, which ends the whole run.
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
the benchmark code take effect in the next run without a rebuild. The driver
renders the manifests from its own copy of `automation/`, `templates/`,
`configs/`, and `op_flags.yaml`, so changes to those files need a new driver
image. Rebuild the driver whenever `op_flags.yaml` or `automation/op_flags/`
changes. Otherwise, the Pathways containers and the McJAX processes run with
different flags.

## Running it by hand

```bash
source ~/tpums_launcher_venv/bin/activate      # kubernetes, pathwaysutils, google-cloud-bigquery
export KUBECONFIG=~/.kube/tpums-pathways-pw-scale-test-v5e-32.yaml
PYTHONPATH=.:src python -m automation.suite_runner \
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
  gives each JobSet `--timeout-minutes`). A DCN JobSet needs two free slices.
* **Names.** Pathways JobSets are named `tpums-<runtime>-<config>` (≤ 22
  characters) and recreated if one exists; McJAX JobSets use `generateName`.
  To keep a manual run apart from the hourly run, set `TPUMS_WORKLOAD_PREFIX`
  to another prefix than `tpums`. For a cluster that needs shorter names, set
  `TPUMS_PATHWAYS_NAME_MAX_LEN`.
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
