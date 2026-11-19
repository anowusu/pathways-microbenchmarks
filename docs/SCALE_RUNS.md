# Run the microbenchmarks on a dynamic slicing cluster

To compare McJAX and Pathways on slices that are larger than one node pool,
run the runtime matrix on a cluster with
[GKE dynamic slicing](https://cloud.google.com/kubernetes-engine/docs/how-to/use-gke-dynamic-slicing).
On such a cluster, the node pools are 4x4x4 cubes of 16 hosts, and Kueue
composes the slice that a workload requests out of whole cubes. The TPU7x
shared-capacity clusters work that way.

The pipeline is the one that [NIGHTLY_RUNTIMES.md](NIGHTLY_RUNTIMES.md)
describes: the same driver, the same benchmarks, and the same reports. This
page covers what differs.

## Limitations

*   A slice topology needs three dimensions that are multiples of 4, smallest
    first, such as `4x4x8` or `8x8x16`.
*   Kueue picks the cubes of a slice for every workload. Two JobSets don't run
    on the same slice unless Kueue happens to pick the same cubes, so a
    comparison of two runtimes also compares two sets of cubes. The node pools
    that a workload ran on are in the `node_pools` field of `provenance.json`.
*   The shared-capacity clusters limit JobSet names to 20 characters.
*   `dcn_all_reduce` runs on Pathways with a debugging override. See
    [Debugging override for `dcn_all_reduce`](NIGHTLY_RUNTIMES.md#debugging-override-for-dcn_all_reduce).

## How the JobSets differ

To shape the JobSets for dynamic slicing, set
`TPUMS_CLUSTER_PROFILE=dynamic-slicing`. The automation then changes the
JobSets of both runtimes (`automation/dynamic_slicing.py`):

| Part of the JobSet | Standard cluster | Dynamic slicing cluster |
|---|---|---|
| Slice selection | `cloud.google.com/gke-tpu-topology` node selector | `cloud.google.com/gke-tpu-slice-topology` annotation on the JobSet, the TPU job, and the pods, plus the Kueue pod-set annotations |
| Cubes | Not applicable | Only cubes whose state is `HEALTHY` |
| Network | McJAX pods use the host network | Every pod uses the host network, the Pathways head included |
| McJAX JobSet name | Generated: `tpums-<topology>-mcjax-<config>-<suffix>` | Fixed: `<prefix>-mcjax-<config>` |
| Slices of a McJAX DCN benchmark | GKE sets the `MEGASCALE_*` variables | The JobSet names its coordinator, and each pod reads its slice from the JobSet labels |
| Pathways workers | Standard `PathwaysJobSet` | The workers skip the TPU webhook and set `MEGASCALE_PORT` |
| Pathways head | Any node of the CPU pool | A node that runs no other Pathways head, because a head binds fixed ports on the host network |

A pod on the host network has the identity of its node. On the
shared-capacity clusters, that identity can write the results bucket. A pod on
the pod network runs as the workload identity pool of the cluster, which can't.

## Build the workload image

The JobSet templates clone the repository and install the package when a pod
starts. Hundreds of hosts can't all do that at the same moment, so a large run
uses a workload image that has the checkout and its dependencies installed.
When the templates find `TPUMS_HOME` in the image, they skip both steps.

To build the image, run the following command from the repository root:

```bash
gcloud builds submit --project PROJECT_ID \
  --config automation/k8s/cloudbuild_workload.yaml \
  --substitutions=_TAG=$(git rev-parse --short HEAD),_COMMIT=$(git rev-parse HEAD),_BASE_IMAGE=BASE_IMAGE \
  --gcs-source-staging-dir gs://STAGING_BUCKET/build .
```

Replace the following:

*   `PROJECT_ID`: the project that runs the build.
*   `BASE_IMAGE`: the client image, by digest, so that the image can be
    rebuilt.
*   `STAGING_BUCKET`: a bucket for the build source.

The image records the commit in `/opt/tpums/GIT_COMMIT` and the installed
packages in `/opt/tpums/PIP_FREEZE.txt`.

## Run a suite

1.  Point `KUBECONFIG` at a kubeconfig file whose current context is the
    cluster. The driver uses the current context.

1.  Run the driver:

    ```bash
    export TPUMS_CLUSTER_PROFILE=dynamic-slicing
    export TPUMS_WORKLOAD_PREFIX=mb
    export TPUMS_PATHWAYS_NAME_MAX_LEN=20
    PYTHONPATH=.:src python -m automation.suite_runner \
      --cluster CLUSTER --project PROJECT_ID --region REGION \
      --namespace NAMESPACE --queue multislice-queue \
      --priority medium --pod-priority-class medium \
      --suite-file automation/scale_suite_tpu7x_8x8x16.yaml \
      --runtimes mcjax,louhi,rapid \
      --max-parallel MAX_PARALLEL --timeout-minutes 30 --retries 1 \
      --docker-image WORKLOAD_IMAGE \
      --gcs-bucket gs://RESULTS_BUCKET/PREFIX \
      --skip-cluster-credentials --cleanup
    ```

    Replace the following:

    *   `CLUSTER`, `PROJECT_ID`, and `REGION`: the cluster, for the report
        header.
    *   `NAMESPACE`: the namespace of your capacity booking, or the namespace
        for ad hoc runs.
    *   `MAX_PARALLEL`: the number of JobSets that fit your capacity at the
        same time. A DCN JobSet needs two slices.
    *   `WORKLOAD_IMAGE`: the workload image, by digest.
    *   `RESULTS_BUCKET` and `PREFIX`: where the results go. The node service
        account of the cluster needs write access.

The suites for TPU7x are the following:

| Suite | Slices |
|---|---|
| `automation/scale_suite_tpu7x_smoke.yaml` | One `4x4x8` slice (128 chips), and two `4x4x4` slices for the DCN benchmarks |
| `automation/scale_suite_tpu7x_8x8x16.yaml` | One `8x8x16` slice (1,024 chips), and two `8x8x16` slices for the DCN benchmarks |

## Device time at scale

The configs of both suites time JAX device 0, like the v5e configs. The
smaller upstream TPU7x configs aggregate the slowest device over all hosts
(`global_slowest_per_step`). A Pathways trace covers the devices of one host,
so device 0 is the device that every runtime can time.
