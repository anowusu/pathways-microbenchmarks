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
"""JobSets for clusters with GKE dynamic slicing.

On a standard GKE TPU cluster a node pool is one slice of a fixed topology,
and a JobSet selects it with `cloud.google.com/gke-tpu-topology`. With dynamic
slicing (the TPU7x shared-capacity clusters) the node pools are 4x4x4 cubes
of 16 hosts, and Kueue's topology-aware scheduling composes the slice a
workload asks for out of whole cubes:

* the JobSet, the TPU job and its pods carry the slice topology in the
  `cloud.google.com/gke-tpu-slice-topology` annotation;
* the pods carry Kueue's pod-set annotations (one topology block, slices of
  one cube: 16 pods);
* every pod uses the host network, like the validated JobSets of these
  clusters. A pod on the host network has the identity of its node, which can
  write the results bucket; a pod on the pod network runs as the cluster's
  workload identity pool, which cannot.

`apply()` reshapes a JobSet manifest of either runtime that way. See
https://cloud.google.com/kubernetes-engine/docs/how-to/use-gke-dynamic-slicing.
"""

from collections.abc import Collection
from typing import Any

SLICE_TOPOLOGY_ANNOTATION = "cloud.google.com/gke-tpu-slice-topology"
# Pathways workers set up the slice themselves (the resource manager tells
# them the topology), so they opt out of the environment the GKE TPU webhook
# injects for JAX processes. McJAX pods need that environment.
_SKIP_TPU_WEBHOOK_ANNOTATION = "cloud.google.com/skip-tpu-webhook-check"
_EXCLUSIVE_TOPOLOGY_ANNOTATION = "alpha.jobset.sigs.k8s.io/exclusive-topology"
# A slice is made of whole 4x4x4 cubes: 64 chips on 16 hosts each.
_CUBE_DIM = 4
_PODSET_ANNOTATIONS = {
    "kueue.x-k8s.io/podset-required-topology": "cloud.google.com/gce-topology-block",
    "kueue.x-k8s.io/podset-slice-required-topology": (
        "cloud.google.com/gke-tpu-partition-4x4x4-id"
    ),
    "kueue.x-k8s.io/podset-slice-size": "16",
}
# Only cubes whose links are all healthy: a degraded cube still forms a slice,
# but it can be slower, which a benchmark must not measure by accident.
_CUBE_STATE_LABEL = "cloud.google.com/gke-tpu-partition-4x4x4-state"
_CUBE_STATE = "HEALTHY"
# Node selector keys of a standard cluster that no node has here.
_STANDARD_NODE_SELECTORS = (
    "cloud.google.com/gke-tpu-topology",
    "cloud.google.com/placement-policy-name",
    "cloud.google.com/reservation-name",
)
_TPU_TOLERATION = {"key": "google.com/tpu", "operator": "Exists"}
# Port of the megascale (DCN) transport of Pathways workers on the host
# network, as the validated TPU7x Pathways JobSets set it.
_PATHWAYS_MEGASCALE_PORT = "60081"
# A Pathways head on the host network binds fixed ports (proxy, resource
# manager) on its node, so two heads cannot share a node. JobSet's exclusive
# topology only keeps the pods of one JobSet apart; this keeps a head off any
# node that already runs a `pathways-head` pod, of any JobSet in any namespace.
_HEAD_JOB_LABEL = "jobset.sigs.k8s.io/replicatedjob-name"
_HEAD_JOB_NAMES = ("pathways-head",)


def validate_topology(topology: str) -> None:
  """Raises ValueError unless Kueue can compose `topology` out of cubes.

  Args:
    topology: Slice topology, e.g. `8x8x16`.
  """
  try:
    dims = [int(d) for d in topology.lower().split("x")]
  except ValueError:
    dims = []
  if (
      len(dims) != 3
      or any(d <= 0 or d % _CUBE_DIM for d in dims)
      or dims != sorted(dims)
  ):
    raise ValueError(
        f"Dynamic slicing composes slices out of {_CUBE_DIM}x{_CUBE_DIM}x"
        f"{_CUBE_DIM} cubes: the topology needs three dimensions that are"
        f" multiples of {_CUBE_DIM}, smallest first (e.g. 4x4x8, 8x8x16);"
        f" got {topology!r}."
    )


def _pod_template(replicated_job: dict[str, Any]) -> dict[str, Any]:
  return replicated_job["template"]["spec"]["template"]


def _set_env(container: dict[str, Any], name: str, entry: dict[str, Any]) -> None:
  """Sets one environment variable of a container, replacing an earlier one."""
  env = [e for e in container.get("env") or [] if e.get("name") != name]
  env.append({"name": name, **entry})
  container["env"] = env


def _label_ref(label: str) -> dict[str, Any]:
  return {"valueFrom": {"fieldRef": {"fieldPath": f"metadata.labels['{label}']"}}}


def _keep_heads_apart(spec: dict[str, Any]) -> None:
  """Adds the anti-affinity that keeps a Pathways head off other heads' nodes."""
  terms = (
      spec.setdefault("affinity", {})
      .setdefault("podAntiAffinity", {})
      .setdefault("requiredDuringSchedulingIgnoredDuringExecution", [])
  )
  terms.append({
      "topologyKey": "kubernetes.io/hostname",
      "namespaceSelector": {},
      "labelSelector": {
          "matchExpressions": [{
              "key": _HEAD_JOB_LABEL,
              "operator": "In",
              "values": list(_HEAD_JOB_NAMES),
          }]
      },
  })


def apply(
    jobset: dict[str, Any],
    *,
    topology: str,
    tpu_jobs: Collection[str],
    pathways: bool,
) -> dict[str, Any]:
  """Reshapes a JobSet manifest for a cluster with dynamic slicing.

  Args:
    jobset: The JobSet manifest of a standard cluster; modified in place.
    topology: Topology of each slice, e.g. `8x8x16`.
    tpu_jobs: Names of the replicated jobs whose pods run on TPU hosts (one
      replica per slice).
    pathways: Whether the TPU pods are Pathways workers (as opposed to JAX
      processes).

  Returns:
    `jobset`.

  Raises:
    ValueError: If Kueue cannot compose `topology` out of cubes, or if the
      manifest has none of `tpu_jobs`.
  """
  validate_topology(topology)
  metadata = jobset["metadata"]
  annotations = metadata.setdefault("annotations", {})
  annotations.pop(_EXCLUSIVE_TOPOLOGY_ANNOTATION, None)
  annotations[SLICE_TOPOLOGY_ANNOTATION] = topology

  found = []
  for job in jobset["spec"]["replicatedJobs"]:
    pod = _pod_template(job)
    spec = pod["spec"]
    spec["hostNetwork"] = True
    spec["dnsPolicy"] = "ClusterFirstWithHostNet"
    if job["name"] not in tpu_jobs:
      if pathways and job["name"] in _HEAD_JOB_NAMES:
        _keep_heads_apart(spec)
      continue
    found.append(job["name"])
    job["template"].setdefault("metadata", {}).setdefault("annotations", {})[
        SLICE_TOPOLOGY_ANNOTATION
    ] = topology
    pod_annotations = pod.setdefault("metadata", {}).setdefault("annotations", {})
    pod_annotations.pop(_EXCLUSIVE_TOPOLOGY_ANNOTATION, None)
    pod_annotations[SLICE_TOPOLOGY_ANNOTATION] = topology
    pod_annotations.update(_PODSET_ANNOTATIONS)
    if pathways:
      pod_annotations[_SKIP_TPU_WEBHOOK_ANNOTATION] = "true"

    node_selector = dict(spec.get("nodeSelector") or {})
    for key in _STANDARD_NODE_SELECTORS:
      node_selector.pop(key, None)
    node_selector[_CUBE_STATE_LABEL] = _CUBE_STATE
    spec["nodeSelector"] = node_selector
    tolerations = list(spec.get("tolerations") or [])
    if not any(
        t.get("key") == _TPU_TOLERATION["key"] and t.get("operator") == "Exists"
        for t in tolerations
    ):
      tolerations.append(dict(_TPU_TOLERATION))
    spec["tolerations"] = tolerations

    num_slices = int(job.get("replicas") or 1)
    for container in spec["containers"]:
      if pathways:
        _set_env(container, "MEGASCALE_PORT", {"value": _PATHWAYS_MEGASCALE_PORT})
      elif num_slices > 1:
        # A standard cluster's TPU webhook connects the slices of a JAX job;
        # here the job names its own coordinator (the first host of the
        # first slice) and each pod reads its slice from the JobSet labels.
        _set_env(container, "MEGASCALE_GRPC_ENABLE_XOR_TRACER", {"value": "false"})
        _set_env(
            container,
            "MEGASCALE_COORDINATOR_ADDRESS",
            _label_ref("jobset.sigs.k8s.io/coordinator"),
        )
        _set_env(
            container,
            "MEGASCALE_NUM_SLICES",
            _label_ref("jobset.sigs.k8s.io/replicatedjob-replicas"),
        )
        _set_env(
            container, "MEGASCALE_SLICE_ID", _label_ref("jobset.sigs.k8s.io/job-index")
        )
    if not pathways and num_slices > 1:
      jobset["spec"]["coordinator"] = {
          "replicatedJob": job["name"],
          "jobIndex": 0,
          "podIndex": 0,
      }

  if not found:
    raise ValueError(
        f"The JobSet has none of the TPU jobs {sorted(tpu_jobs)}: its jobs are"
        f" {[job['name'] for job in jobset['spec']['replicatedJobs']]}."
    )
  return jobset
