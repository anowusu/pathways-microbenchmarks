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
"""Tests for JobSets on dynamic slicing clusters and the tpu7x scale configs."""

import json
import math
import pathlib
import subprocess
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from automation import dynamic_slicing
from automation import runtimes
from automation import utils
from automation import workload_submitter
import yaml

try:
  from pathwaysutils.experimental.gke import jobset as _jobset  # pylint: disable=g-import-not-at-top,unused-import

  _HAS_PATHWAYSUTILS = True
except ImportError:  # pragma: no cover - depends on the test environment.
  _HAS_PATHWAYSUTILS = False

_DYNAMIC = utils.CLUSTER_PROFILE_DYNAMIC_SLICING
_SLICE = dynamic_slicing.SLICE_TOPOLOGY_ANNOTATION
_PODSET = {
    "kueue.x-k8s.io/podset-required-topology": "cloud.google.com/gce-topology-block",
    "kueue.x-k8s.io/podset-slice-required-topology": (
        "cloud.google.com/gke-tpu-partition-4x4x4-id"
    ),
    "kueue.x-k8s.io/podset-slice-size": "16",
}
_CONFIGS = pathlib.Path(utils.PACKAGE_ROOT) / "configs" / "tpu7x"
_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]


def _mcjax(config, topology, num_slices=1, cluster_profile=_DYNAMIC, name="mb-mcjax-x"):
  return yaml.safe_load(
      workload_submitter.generate_jobset_manifest(
          workload_name=name,
          namespace="priority-dev",
          queue_name="multislice-queue",
          priority_class="medium",
          topology=topology,
          docker_image="img:tag",
          config_rel_path=config,
          gcs_output_dir="gs://b/2026-11-19/mcjax/x",
          pod_priority_class="medium",
          num_slices=num_slices,
          cluster_profile=cluster_profile,
      )
  )


def _pathways(config, topology, num_slices=1, cluster_profile=_DYNAMIC):
  return yaml.safe_load(
      workload_submitter.generate_pathways_manifest(
          workload_name="mb-louhi-x",
          namespace="priority-dev",
          queue_name="multislice-queue",
          priority_class="medium",
          topology=topology,
          docker_image="img:tag",
          config_rel_path=config,
          gcs_output_dir="gs://b/2026-11-19/louhi/x",
          runtime=runtimes.LOUHI,
          pod_priority_class="medium",
          num_slices=num_slices,
          cluster_profile=cluster_profile,
      )
  )


def _job(manifest, name):
  return next(j for j in manifest["spec"]["replicatedJobs"] if j["name"] == name)


def _pod(manifest, name):
  return _job(manifest, name)["template"]["spec"]["template"]


def _env(container):
  return {e["name"]: e for e in container.get("env", [])}


class ValidateTopologyTest(parameterized.TestCase):

  @parameterized.parameters("4x4x4", "4x4x8", "4x8x8", "8x8x16")
  def test_accepts_slices_made_of_cubes(self, topology):
    dynamic_slicing.validate_topology(topology)

  @parameterized.parameters("2x2x1", "2x4x4", "4x8x4", "4x4", "8x8x15", "abc", "")
  def test_rejects_everything_else(self, topology):
    with self.assertRaisesRegex(ValueError, "cubes"):
      dynamic_slicing.validate_topology(topology)

  def test_apply_needs_a_tpu_job(self):
    manifest = _mcjax("configs/tpu7x/4x4x8/gemm.yaml", "4x4x8", cluster_profile=None)
    with self.assertRaisesRegex(ValueError, "none of the TPU jobs"):
      dynamic_slicing.apply(
          manifest, topology="4x4x8", tpu_jobs=("worker",), pathways=False
      )


class McjaxManifestTest(absltest.TestCase):

  def test_standard_profile_is_unchanged(self):
    manifest = _mcjax(
        "configs/tpu7x/4x4x4/all_gather.yaml", "4x4x4", cluster_profile=utils.CLUSTER_PROFILE_STANDARD
    )
    self.assertEqual(manifest["metadata"]["generateName"], "mb-mcjax-x-")
    self.assertNotIn(_SLICE, manifest["metadata"]["annotations"])
    selector = _pod(manifest, "slice-job")["spec"]["nodeSelector"]
    self.assertEqual(selector["cloud.google.com/gke-tpu-topology"], "4x4x4")

  def test_single_slice(self):
    manifest = _mcjax("configs/tpu7x/8x8x16/all_gather.yaml", "8x8x16")
    # A fixed name (the cluster limits names to 20 characters).
    self.assertEqual(manifest["metadata"]["name"], "mb-mcjax-x")
    self.assertNotIn("generateName", manifest["metadata"])
    # The slice topology on the JobSet, the job and the pods.
    self.assertEqual(manifest["metadata"]["annotations"], {_SLICE: "8x8x16"})
    job = _job(manifest, "slice-job")
    self.assertEqual(job["template"]["metadata"]["annotations"], {_SLICE: "8x8x16"})
    self.assertEqual(job["replicas"], 1)
    self.assertEqual(job["template"]["spec"]["parallelism"], 256)
    pod = _pod(manifest, "slice-job")
    self.assertEqual(pod["metadata"]["annotations"], {_SLICE: "8x8x16", **_PODSET})
    spec = pod["spec"]
    self.assertTrue(spec["hostNetwork"])
    self.assertEqual(spec["dnsPolicy"], "ClusterFirstWithHostNet")
    self.assertEqual(
        spec["nodeSelector"],
        {
            "cloud.google.com/gke-tpu-accelerator": "tpu7x",
            # The cluster rejects TPU pods without the topology selector.
            "cloud.google.com/gke-tpu-topology": "8x8x16",
            "cloud.google.com/gke-tpu-partition-4x4x4-state": "HEALTHY",
        },
    )
    self.assertIn(
        {"key": "google.com/tpu", "operator": "Exists", "effect": "NoSchedule"},
        spec["tolerations"],
    )
    # One slice: nothing connects slices.
    self.assertNotIn("coordinator", manifest["spec"])
    env = _env(spec["containers"][0])
    self.assertFalse([name for name in env if name.startswith("MEGASCALE_")])
    # The container script survives the round trip.
    script = spec["containers"][0]["args"][0]
    self.assertIn('tpums benchmark run-config "configs/tpu7x/8x8x16/all_gather.yaml"', script)
    self.assertIn('if [ "${PUBLISH:-false}" = "true" ]; then', script)

  def test_two_slices_name_their_coordinator(self):
    manifest = _mcjax("configs/tpu7x/8x8x16/dcn_all_reduce.yaml", "8x8x16", num_slices=2)
    self.assertEqual(_job(manifest, "slice-job")["replicas"], 2)
    self.assertEqual(
        manifest["spec"]["coordinator"],
        {"replicatedJob": "slice-job", "jobIndex": 0, "podIndex": 0},
    )
    env = _env(_pod(manifest, "slice-job")["spec"]["containers"][0])
    for name, label in (
        ("MEGASCALE_COORDINATOR_ADDRESS", "jobset.sigs.k8s.io/coordinator"),
        ("MEGASCALE_NUM_SLICES", "jobset.sigs.k8s.io/replicatedjob-replicas"),
        ("MEGASCALE_SLICE_ID", "jobset.sigs.k8s.io/job-index"),
    ):
      self.assertEqual(
          env[name]["valueFrom"]["fieldRef"]["fieldPath"], f"metadata.labels['{label}']"
      )
    self.assertEqual(env["GLIBC_TUNABLES"]["value"], "glibc.pthread.rseq=0")

  def test_script_uses_the_checkout_of_a_workload_image(self):
    for profile in (utils.CLUSTER_PROFILE_STANDARD, _DYNAMIC):
      manifest = _mcjax("configs/tpu7x/4x4x8/gemm.yaml", "4x4x8", cluster_profile=profile)
      script = _pod(manifest, "slice-job")["spec"]["containers"][0]["args"][0]
      self.assertIn(
          'if [ -n "${TPUMS_HOME:-}" ] && [ -f "${TPUMS_HOME:-}/GIT_COMMIT" ]; then', script
      )
      # The image keeps the McJAX packages in a venv of their own.
      self.assertIn('source "${TPUMS_VENV:-/opt/tpums-venv}/bin/activate"', script)
      # Without one, the pod still clones and installs.
      self.assertIn("git clone --branch", script)
      self.assertLess(script.index("TPUMS_HOME"), script.index("git clone --branch"))


@absltest.skipUnless(_HAS_PATHWAYSUTILS, "pathwaysutils not installed")
class PathwaysManifestTest(absltest.TestCase):

  def test_standard_profile_is_unchanged(self):
    manifest = _pathways(
        "configs/tpu7x/4x4x4/all_gather.yaml", "4x4x4", cluster_profile=utils.CLUSTER_PROFILE_STANDARD
    )
    self.assertNotIn(_SLICE, manifest["metadata"].get("annotations") or {})
    worker = _pod(manifest, "pathways-worker")["spec"]
    self.assertEqual(worker["nodeSelector"]["cloud.google.com/gke-tpu-topology"], "4x4x4")
    self.assertNotIn("hostNetwork", worker)

  def test_two_slices(self):
    manifest = _pathways("configs/tpu7x/8x8x16/dcn_all_to_all.yaml", "8x8x16", num_slices=2)
    self.assertEqual(manifest["metadata"]["name"], "mb-louhi-x")
    self.assertEqual(manifest["metadata"]["annotations"][_SLICE], "8x8x16")
    job = _job(manifest, "pathways-worker")
    self.assertEqual(job["replicas"], 2)
    self.assertEqual(job["template"]["spec"]["parallelism"], 256)
    self.assertEqual(job["template"]["metadata"]["annotations"][_SLICE], "8x8x16")
    pod = _pod(manifest, "pathways-worker")
    self.assertEqual(
        pod["metadata"]["annotations"],
        {
            _SLICE: "8x8x16",
            **_PODSET,
            # Pathways workers do not take the JAX environment of the webhook.
            "cloud.google.com/skip-tpu-webhook-check": "true",
        },
    )
    spec = pod["spec"]
    self.assertTrue(spec["hostNetwork"])
    self.assertEqual(
        spec["nodeSelector"],
        {
            "cloud.google.com/gke-tpu-accelerator": "tpu7x",
            # The cluster rejects TPU pods without the topology selector.
            "cloud.google.com/gke-tpu-topology": "8x8x16",
            "cloud.google.com/gke-tpu-partition-4x4x4-state": "HEALTHY",
        },
    )
    self.assertEqual(spec["tolerations"], [{"key": "google.com/tpu", "operator": "Exists"}])
    env = _env(spec["containers"][0])
    self.assertEqual(env["MEGASCALE_PORT"]["value"], "60081")
    self.assertIn("MEGASCALE_COORDINATOR_ADDRESS", env)

    head = _pod(manifest, "pathways-head")["spec"]
    self.assertTrue(head["hostNetwork"])
    self.assertEqual(head["dnsPolicy"], "ClusterFirstWithHostNet")
    self.assertEqual(head["nodeSelector"], {"cloud.google.com/gke-nodepool": "cpu-np"})
    # A head stays off nodes that run another head, of any namespace.
    (term,) = head["affinity"]["podAntiAffinity"][
        "requiredDuringSchedulingIgnoredDuringExecution"
    ]
    self.assertEqual(term["topologyKey"], "kubernetes.io/hostname")
    self.assertEqual(term["namespaceSelector"], {})
    self.assertEqual(
        term["labelSelector"]["matchExpressions"],
        [{
            "key": "jobset.sigs.k8s.io/replicatedjob-name",
            "operator": "In",
            "values": ["pathways-head"],
        }],
    )
    rm = next(c for c in head["initContainers"] if c["name"] == "pathways-rm")
    self.assertIn("--instance_type=tpu7x:8x8x16", rm["args"])
    self.assertIn("--instance_count=2", rm["args"])

  def test_client_uses_the_checkout_of_a_workload_image(self):
    manifest = _pathways("configs/tpu7x/4x4x8/gemm.yaml", "4x4x8")
    script = _pod(manifest, "pathways-head")["spec"]["containers"][0]["command"][-1]
    self.assertIn('if [ -n "${TPUMS_HOME:-}" ] && [ -f "${TPUMS_HOME:-}/GIT_COMMIT" ]; then', script)
    self.assertIn("git clone --depth 1", script)

  def test_workload_image_installs_the_client_the_way_the_template_does(self):
    # The Pathways client reads worker traces with the TensorFlow of the client
    # image. A workload image that ran the client from the McJAX venv had no
    # device times, so the image has to install the client with the template's
    # own command, and keep the venv off PATH.
    script = _pod(_pathways("configs/tpu7x/4x4x8/gemm.yaml", "4x4x8"), "pathways-head")["spec"][
        "containers"
    ][0]["command"][-1]
    install = "pip install -q -e '.[pathways]'"
    self.assertIn(install, script)
    dockerfile = (_REPO_ROOT / "automation" / "k8s" / "Dockerfile.workload").read_text()
    self.assertIn(f"RUN {install}", dockerfile)
    self.assertIn("_pywrap_profiler_plugin", dockerfile)
    env = dockerfile[dockerfile.rindex("\nENV ") :]
    self.assertIn("TPUMS_VENV=/opt/tpums-venv", env)
    self.assertNotIn("PATH=", env)


class ImagePinsTest(absltest.TestCase):

  _SERVER = "us-docker.pkg.dev/cloud-tpu-v2-images/cloud-pathways/server:latest"
  _PROXY = "us-docker.pkg.dev/cloud-tpu-v2-images/cloud-pathways/proxy_server:latest"
  _A = "sha256:" + "a" * 64
  _B = "sha256:" + "b" * 64

  def test_parse(self):
    self.assertEqual(workload_submitter.parse_image_pins(None), {})
    self.assertEqual(
        workload_submitter.parse_image_pins(f"{self._SERVER}={self._A}, {self._PROXY}={self._B}"),
        {self._SERVER: self._A, self._PROXY: self._B},
    )
    for bad in (self._SERVER, f"{self._SERVER}=latest", f"={self._A}", f"{self._SERVER}=sha256:abc"):
      with self.assertRaisesRegex(ValueError, "Image pin"):
        workload_submitter.parse_image_pins(bad)

  @absltest.skipUnless(_HAS_PATHWAYSUTILS, "pathwaysutils not installed")
  def test_pathways_images_are_pinned_everywhere_they_run(self):
    manifest = yaml.safe_load(
        workload_submitter.generate_pathways_manifest(
            workload_name="mb-louhi-x",
            namespace="priority-dev",
            queue_name="multislice-queue",
            priority_class="medium",
            topology="4x4x8",
            docker_image="img:tag",
            config_rel_path="configs/tpu7x/4x4x8/gemm.yaml",
            gcs_output_dir="gs://b/x",
            runtime=runtimes.LOUHI,
            cluster_profile=_DYNAMIC,
            image_pins={self._SERVER: self._A, self._PROXY: self._B},
        )
    )
    head = _pod(manifest, "pathways-head")["spec"]
    images = {c["name"]: c["image"] for c in head["initContainers"] + head["containers"]}
    worker = _pod(manifest, "pathways-worker")["spec"]["containers"][0]["image"]
    server = f"{self._SERVER.rsplit(':', 1)[0]}@{self._A}"
    self.assertEqual(images["pathways-rm"], server)
    self.assertEqual(worker, server)
    self.assertEqual(
        images["pathways-proxy"], f"{self._PROXY.rsplit(':', 1)[0]}@{self._B}"
    )
    # Images without a pin keep their reference.
    self.assertEqual(images["user-workload"], "img:tag")

  def test_mcjax_template_stays_text_without_a_matching_pin(self):
    kwargs = dict(
        workload_name="tpums-test",
        namespace="default",
        queue_name="multislice-queue",
        priority_class="medium",
        topology="4x4x4",
        docker_image="img:tag",
        config_rel_path="configs/tpu7x/4x4x4/all_gather.yaml",
        gcs_output_dir="gs://b/x",
        cluster_profile=utils.CLUSTER_PROFILE_STANDARD,
    )
    plain = workload_submitter.generate_jobset_manifest(**kwargs, image_pins={})
    other = workload_submitter.generate_jobset_manifest(
        **kwargs, image_pins={self._SERVER: self._A}
    )
    self.assertEqual(plain, other)
    pinned = yaml.safe_load(
        workload_submitter.generate_jobset_manifest(**kwargs, image_pins={"img:tag": self._A})
    )
    self.assertEqual(
        _pod(pinned, "slice-job")["spec"]["containers"][0]["image"], f"img@{self._A}"
    )
    self.assertEqual(pinned["metadata"]["generateName"], "tpums-test-")


class PodQueryTest(absltest.TestCase):

  def _run(self, pods):
    def run(cmd, **kwargs):
      del kwargs  # Unused.
      self.commands.append(cmd)
      return subprocess.CompletedProcess(
          args=cmd, returncode=0, stdout=json.dumps({"items": pods}), stderr=""
      )

    self.commands = []
    return mock.patch("subprocess.run", side_effect=run)

  def test_only_the_pods_of_the_named_jobsets_are_listed(self):
    pod = {"metadata": {"name": "p", "labels": {"jobset.sigs.k8s.io/jobset-name": "mb-a"}}}
    with self._run([pod]):
      grouped = workload_submitter.fetch_namespace_pods("priority-dev", ["mb-b", "mb-a"])
    self.assertEqual(grouped, {"mb-a": [pod]})
    (cmd,) = self.commands
    self.assertEqual(
        cmd[cmd.index("-l") + 1], "jobset.sigs.k8s.io/jobset-name in (mb-a,mb-b)"
    )

  def test_no_jobsets_no_query(self):
    with self._run([]):
      self.assertEqual(workload_submitter.fetch_namespace_pods("priority-dev", []), {})
    self.assertEmpty(self.commands)

  def test_whole_namespace_without_names_or_with_too_many(self):
    with self._run([]):
      workload_submitter.fetch_namespace_pods("default")
      workload_submitter.fetch_namespace_pods("default", [f"js-{i}" for i in range(41)])
    for cmd in self.commands:
      self.assertNotIn("-l", cmd)


class NamesAndRetriesTest(absltest.TestCase):

  def test_short_names_fit_twenty_characters(self):
    names = set()
    for runtime in (runtimes.MCJAX, runtimes.LOUHI, runtimes.RAPID):
      for config in (
          "all_gather", "all_reduce", "all_to_all", "gemm", "hbm",
          "dcn_all_reduce", "dcn_all_to_all",
      ):
        name = workload_submitter.runtime_workload_name(
            runtime, config, "8x8x16", prefix="mb", short=True
        )
        self.assertLessEqual(len(name), 20, name)
        names.add(name)
        retry = workload_submitter.retry_workload_name(name, 2)
        names.add(retry)
    self.assertLen(names, 2 * 3 * 7)  # all distinct, retries included
    self.assertEqual(
        workload_submitter.runtime_workload_name(
            runtimes.MCJAX, "all_gather", "8x8x16", prefix="mb", short=True
        ),
        "mb-mcjax-all-gather",
    )

  @mock.patch.object(utils, "DEFAULT_CLUSTER_PROFILE", _DYNAMIC)
  @mock.patch.object(
      runtimes, "runtime_provenance", return_value={"runtime": "mcjax", "git_commit": "c"}
  )
  def test_mcjax_retry_gets_a_new_fixed_name(self, _):
    manager = workload_submitter.WorkloadManager(runtimes=[runtimes.MCJAX])
    manager.prepare_workloads([_CONFIGS / "8x8x16" / "all_gather.yaml"])
    (workload,) = manager.pending_workloads.values()
    self.assertTrue(workload.fixed_name)
    name = yaml.safe_load(workload.manifest_yaml)["metadata"]["name"]
    self.assertEqual(name, workload.workload_name)
    # The short form, under the default prefix here.
    self.assertEqual(name, "tpums-mcjax-all-gather")

    workload.status, workload.error_message = "FAILED", "boom"
    retry = workload.retry_copy()
    self.assertEqual(retry.workload_name, "tpums-mcjax-all-gat-r2")
    self.assertEqual(
        yaml.safe_load(retry.manifest_yaml)["metadata"]["name"], retry.workload_name
    )

  def test_profile_helper(self):
    self.assertTrue(utils.uses_dynamic_slicing(_DYNAMIC))
    self.assertFalse(utils.uses_dynamic_slicing(utils.CLUSTER_PROFILE_STANDARD))
    self.assertFalse(utils.uses_dynamic_slicing())


class ScaleConfigsTest(parameterized.TestCase):

  @parameterized.parameters(("4x4x8", 128), ("8x8x16", 1024))
  def test_collective_meshes_cover_every_device(self, topology, chips):
    devices = 2 * chips  # tpu7x: two JAX devices per chip
    for name in ("all_gather", "all_reduce", "all_to_all"):
      config = yaml.safe_load((_CONFIGS / topology / f"{name}.yaml").read_text())["benchmark"]
      mesh = [int(d) for d in config["params"]["mesh_shape"].split("x")]
      self.assertEqual(math.prod(mesh), devices, name)
      for strategy in config["sweep"]["sharding_strategy"]:
        dims = [int(d) for d in strategy.split("x")]
        self.assertLen(dims, len(mesh))
        # An axis takes part entirely or not at all.
        self.assertTrue(all(d in (1, m) for d, m in zip(dims, mesh)), strategy)
      # Device 0 on every runtime, not a cross-host aggregate.
      self.assertNotIn("xprof_device_timing_aggregation", config)
      self.assertTrue(config["xprof_timing"])

  @parameterized.parameters(("4x4x8", 128), ("8x8x16", 1024))
  def test_all_gather_result_fits_a_device(self, topology, chips):
    config = yaml.safe_load((_CONFIGS / topology / "all_gather.yaml").read_text())["benchmark"]
    shard_bytes = max(config["sweep"]["matrix_dim"]) * 1024 * 2  # bfloat16
    per_device_gib = shard_bytes * 2 * chips / 2**30
    self.assertLessEqual(per_device_gib, 16)

  @parameterized.parameters(
      "scale_suite_tpu7x_smoke.yaml", "scale_suite_tpu7x_8x8x16.yaml"
  )
  def test_suites_list_existing_tpu7x_configs(self, suite):
    path = utils.AUTOMATION_DIR / suite
    configs = workload_submitter.load_canonical_configs(str(path))
    self.assertLen(configs, 7)
    for config in configs:
      self.assertTrue((pathlib.Path(utils.PACKAGE_ROOT) / config).is_file(), config)
      self.assertEqual(utils.infer_accelerator(None, config_path=config), "tpu7x")
      dynamic_slicing.validate_topology(workload_submitter.infer_topology_from_config(config))
    self.assertEqual(
        workload_submitter.load_suite_runtimes(str(path)), ["mcjax", "louhi", "rapid"]
    )

  def test_dcn_configs_run_on_two_slices(self):
    for topology in ("4x4x4", "8x8x16"):
      for name in ("dcn_all_reduce", "dcn_all_to_all"):
        config = f"configs/tpu7x/{topology}/{name}.yaml"
        self.assertEqual(utils.config_num_slices(config), 2)
        body = yaml.safe_load((pathlib.Path(utils.PACKAGE_ROOT) / config).read_text())["benchmark"]
        self.assertTrue(body["xprof_timing"])
        self.assertLessEqual(body["params"]["num_runs"], 20)


if __name__ == "__main__":
  absltest.main()
