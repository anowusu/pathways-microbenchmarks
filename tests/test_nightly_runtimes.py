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
"""Tests for the runtime matrix: runtimes, naming, comparison and BigQuery rows."""

import dataclasses
import datetime
import json
import os
import subprocess
import tempfile
import time
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from automation import bq_writer
from automation import dashboard
from automation import metrics_aggregator
from automation import runtime_comparison
from automation import runtimes
from automation import utils
from automation import workload_submitter
import yaml

try:
  from pathwaysutils.experimental.gke import jobset as _jobset  # pylint: disable=g-import-not-at-top,unused-import

  _HAS_PATHWAYSUTILS = True
except ImportError:  # pragma: no cover - depends on the test environment.
  _HAS_PATHWAYSUTILS = False


def _row(runtime, xprof_p50, wall_p50, **extra):
  row = {
      "runtime": runtime,
      "benchmark": "AllGatherBenchmark",
      "topology": "4x8",
      "mesh_shape": "4x8",
      "sharding_strategy": "4x8",
      "dtype": "bfloat16",
      "matrix_dim": "8192",
      "xprof_p50_ms": xprof_p50,
      "xprof_p90_ms": xprof_p50,
      "wall_clock_p50_ms": wall_p50,
      "wall_clock_p90_ms": wall_p50,
      "wall_clock_avg_ms": wall_p50,
      "wall_clock_std_ms": "0.1",
      "xprof_captured_runs": "5" if xprof_p50 else "0",
      "xprof_expected_runs": "5",
      "wall_clock_bandwidth_per_chip_gb_s": "27.1",
      "xprof_bandwidth_per_chip_gb_s": "51.1" if xprof_p50 else "",
  }
  row.update(extra)
  return row


class RuntimesTest(parameterized.TestCase):

  def test_parse_runtimes_defaults_to_mcjax(self):
    self.assertEqual(runtimes.parse_runtimes(None), [runtimes.MCJAX])
    self.assertEqual(runtimes.parse_runtimes([]), [runtimes.MCJAX])

  def test_parse_runtimes_accepts_csv_and_dedupes(self):
    parsed = runtimes.parse_runtimes(["rapid,louhi", "RAPID"])
    self.assertEqual([r.name for r in parsed], ["rapid", "louhi"])

  def test_parse_runtimes_rejects_unknown(self):
    with self.assertRaisesRegex(ValueError, "Unknown runtime"):
      runtimes.parse_runtimes(["xpk"])

  def test_pathways_runtimes_have_both_images(self):
    for runtime in (runtimes.LOUHI, runtimes.RAPID):
      self.assertTrue(runtime.is_pathways)
      refs = runtime.image_refs()
      self.assertEqual(set(refs), {"server", "proxy"})
      for ref in refs.values():
        self.assertTrue(ref.endswith(":latest"), ref)
    self.assertFalse(runtimes.MCJAX.is_pathways)
    self.assertEqual(runtimes.MCJAX.image_refs(), {})

  def test_runtime_sort_key_puts_mcjax_first(self):
    names = sorted(["rapid", "custom", "mcjax", "louhi"], key=runtimes.runtime_sort_key)
    self.assertEqual(names, ["mcjax", "louhi", "rapid", "custom"])

  def test_runtime_provenance_resolves_images_and_commit(self):
    def fake_run(cmd, **unused_kwargs):
      if cmd[0] == "git":
        return subprocess.CompletedProcess(cmd, 0, stdout="abc123\trefs/heads/nightly\n", stderr="")
      if "describe" in cmd:
        return subprocess.CompletedProcess(
            cmd, 0, stdout=json.dumps({"image_summary": {"digest": "sha256:feed"}}), stderr=""
        )
      return subprocess.CompletedProcess(
          cmd, 0, stdout=json.dumps([{"tags": ["latest", "20261029_1012"]}]), stderr=""
      )

    with mock.patch.object(subprocess, "run", side_effect=fake_run):
      prov = runtimes.runtime_provenance(runtimes.LOUHI, "https://x/y.git", "nightly")
    self.assertEqual(prov["git_commit"], "abc123")
    self.assertEqual(prov["pathways_server_digest"], "sha256:feed")
    self.assertEqual(prov["pathways_proxy_tags"], ["20261029_1012", "latest"])
    self.assertEqual(prov["runtime"], "louhi")

  def test_runtime_provenance_survives_registry_errors(self):
    failed = subprocess.CompletedProcess(["gcloud"], 1, stdout="", stderr="denied")
    with mock.patch.object(subprocess, "run", return_value=failed):
      prov = runtimes.runtime_provenance(runtimes.RAPID, "https://x/y.git", "main")
    self.assertIsNone(prov["pathways_server_digest"])
    self.assertIsNone(prov["git_commit"])


class WorkloadNamingTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ("rapid_all_gather", runtimes.RAPID, "all_gather", "tpums-rapid-all-gather"),
      ("louhi_all_to_all", runtimes.LOUHI, "all_to_all", "tpums-louhi-all-to-all"),
      ("rapid_hbm", runtimes.RAPID, "hbm", "tpums-rapid-hbm"),
  )
  def test_pathways_names_fit_22_characters(self, runtime, config_name, expected):
    name = workload_submitter.runtime_workload_name(runtime, config_name, "4x8")
    self.assertEqual(name, expected)
    self.assertLessEqual(len(name), workload_submitter.PATHWAYS_WORKLOAD_NAME_MAX_LEN)

  def test_mcjax_names_keep_topology_and_runtime(self):
    self.assertEqual(
        workload_submitter.runtime_workload_name(runtimes.MCJAX, "all_gather", "4x8"),
        "tpums-4x8-mcjax-all-gather",
    )

  def test_load_suite_runtimes(self):
    with tempfile.TemporaryDirectory() as tmp:
      suite = os.path.join(tmp, "suite.yaml")
      with open(suite, "w", encoding="utf-8") as f:
        f.write("configs: [a.yaml]\nruntimes: [mcjax, rapid]\n")
      self.assertEqual(workload_submitter.load_suite_runtimes(suite), ["mcjax", "rapid"])
      plain = os.path.join(tmp, "plain.yaml")
      with open(plain, "w", encoding="utf-8") as f:
        f.write("configs: [a.yaml]\n")
      self.assertEqual(workload_submitter.load_suite_runtimes(plain), [])

  @absltest.skipUnless(_HAS_PATHWAYSUTILS, "pathwaysutils not installed")
  def test_pathways_manifest_routes_results_and_flags(self):
    manifest = workload_submitter.generate_pathways_manifest(
        workload_name="tpums-rapid-all-gather",
        namespace="default",
        queue_name="multislice-queue",
        priority_class="medium",
        topology="4x8",
        docker_image="img:latest",
        config_rel_path="configs/v5e/4x8/all_gather.yaml",
        gcs_output_dir="gs://b/2026-10-29/rapid/all_gather",
        runtime=runtimes.RAPID,
        git_repo="https://x/y.git",
        git_branch="nightly",
    )
    doc = yaml.safe_load(manifest)
    self.assertEqual(doc["metadata"]["name"], "tpums-rapid-all-gather")
    self.assertEqual(
        doc["metadata"]["labels"],
        {"kueue.x-k8s.io/queue-name": "multislice-queue", "kueue.x-k8s.io/priority-class": "medium"},
    )
    head, worker = doc["spec"]["replicatedJobs"]
    head_spec = head["template"]["spec"]["template"]["spec"]
    client = next(c for c in head_spec["containers"] if c["name"] == "user-workload")
    env = {e["name"]: e.get("value") for e in client["env"]}
    self.assertEqual(env["GCS_RESULTS_DIR"], "gs://b/2026-10-29/rapid/all_gather")
    self.assertEqual(env["GCS_XPROF_DIR"], "gs://b/2026-10-29/rapid/all_gather/xprof")
    self.assertEqual(env["TPUMS_REPO"], "https://x/y.git")
    self.assertEqual(env["TPUMS_REF"], "nightly")
    proxy = next(c for c in head_spec["initContainers"] if c["name"] == "pathways-proxy")
    worker_container = worker["template"]["spec"]["template"]["spec"]["containers"][0]
    proxy_flags = [a for a in proxy["args"] if a.startswith("--xla_")]
    worker_flags = [a for a in worker_container["args"] if a.startswith("--xla_")]
    self.assertNotEmpty(proxy_flags)
    self.assertEqual(proxy_flags, worker_flags)
    self.assertTrue(proxy["image"].startswith(runtimes.RAPID.pathways_proxy_image))
    # No pod priority class requested: neither pod template sets one.
    self.assertNotIn("priorityClassName", head_spec)
    self.assertNotIn("priorityClassName", worker["template"]["spec"]["template"]["spec"])

  @absltest.skipUnless(_HAS_PATHWAYSUTILS, "pathwaysutils not installed")
  def test_pathways_manifest_sets_pod_priority_class(self):
    manifest = workload_submitter.generate_pathways_manifest(
        workload_name="tpums-louhi-gemm",
        namespace="default",
        queue_name="multislice-queue",
        priority_class="",
        topology="4x8",
        docker_image="img:latest",
        config_rel_path="configs/v5e/4x8/gemm.yaml",
        gcs_output_dir="gs://b/2026-10-29/louhi/gemm",
        runtime=runtimes.LOUHI,
        pod_priority_class="medium",
    )
    doc = yaml.safe_load(manifest)
    # Empty Kueue priority: only the queue label remains.
    self.assertEqual(
        doc["metadata"]["labels"], {"kueue.x-k8s.io/queue-name": "multislice-queue"}
    )
    for replicated_job in doc["spec"]["replicatedJobs"]:
      pod_spec = replicated_job["template"]["spec"]["template"]["spec"]
      self.assertEqual(pod_spec["priorityClassName"], "medium", replicated_job["name"])

  def test_mcjax_manifest_pod_priority_class_is_optional(self):
    common = dict(
        workload_name="tpums-4x8-mcjax-gemm",
        namespace="default",
        queue_name="multislice-queue",
        priority_class="",
        topology="4x8",
        docker_image="img:latest",
        config_rel_path="configs/v5e/4x8/gemm.yaml",
        gcs_output_dir="gs://b/2026-10-29/mcjax/gemm",
        accelerator="v5e",
    )
    without = yaml.safe_load(workload_submitter.generate_jobset_manifest(**common))
    with_prio = yaml.safe_load(
        workload_submitter.generate_jobset_manifest(**common, pod_priority_class="medium")
    )
    pod_spec = lambda doc: doc["spec"]["replicatedJobs"][0]["template"]["spec"]["template"]["spec"]
    self.assertNotIn("priorityClassName", pod_spec(without))
    self.assertNotIn("kueue.x-k8s.io/priority-class", without["metadata"]["labels"])
    self.assertEqual(pod_spec(with_prio)["priorityClassName"], "medium")
    self.assertEqual(pod_spec(with_prio)["restartPolicy"], "Never")
    self.assertTrue(pod_spec(with_prio)["hostNetwork"])


class RetryTest(absltest.TestCase):
  """A failed or timed-out workload is deleted and resubmitted `retries` times."""

  _FAILED = {"tpums-rapid-all-gather": {"status": {"conditions": [
      {"type": "Failed", "status": "True", "reason": "Preempted", "message": "rm crashed"}]}}}
  _COMPLETED = {"tpums-rapid-all-gather": {"status": {"conditions": [
      {"type": "Completed", "status": "True"}]}}}

  def _manager(self, retries):
    wl = workload_submitter.ActiveWorkload(
        workload_name="tpums-rapid-all-gather",
        config_name="all_gather",
        config_rel_path="configs/v5e/4x8/all_gather.yaml",
        topology="4x8",
        manifest_yaml="kind: JobSet\nname: tpums-rapid-all-gather",
        gcs_output_dir="gs://b/2026-10-29/rapid/all_gather",
        start_time=0.0,
        total_pods=9,
        runtime="rapid",
        fixed_name=True,
        base_name="tpums-rapid-all-gather",
        render_manifest=lambda name: f"kind: JobSet\nname: {name}",
    )
    return workload_submitter.WorkloadManager(
        timeout_minutes=1,
        poll_interval_seconds=1,
        tick_interval_seconds=0,
        cleanup=True,
        retries=retries,
        pending_workloads={wl.workload_name: wl},
    )

  def test_retry_workload_name_fits_and_differs(self):
    self.assertEqual(
        workload_submitter.retry_workload_name("tpums-rapid-all-gather", 2),
        "tpums-rapid-all-gat-r2",
    )
    self.assertEqual(workload_submitter.retry_workload_name("tpums-louhi-hbm", 3), "tpums-louhi-hbm-r3")
    self.assertLessEqual(
        len(workload_submitter.retry_workload_name("tpums-rapid-all-gather", 10)),
        workload_submitter.PATHWAYS_WORKLOAD_NAME_MAX_LEN,
    )

  @mock.patch.object(workload_submitter, "upload_provenance")
  @mock.patch.object(workload_submitter, "_delete_existing_jobset")
  @mock.patch.object(workload_submitter, "delete_jobset", return_value=True)
  @mock.patch.object(workload_submitter, "apply_jobset")
  @mock.patch.object(workload_submitter, "fetch_namespace_pods", return_value={})
  @mock.patch.object(workload_submitter, "fetch_namespace_jobsets")
  def test_failed_workload_is_resubmitted_once(
      self, mock_jobsets, mock_pods, mock_apply, mock_delete, mock_delete_existing, mock_upload
  ):
    del mock_pods, mock_upload  # Unused.
    mock_apply.side_effect = lambda manifest: manifest.split("name: ")[1]
    retry_name = "tpums-rapid-all-gat-r2"
    completed_retry = {retry_name: {"status": {"conditions": [
        {"type": "Completed", "status": "True"}]}}}
    mock_jobsets.side_effect = [self._FAILED, completed_retry, completed_retry]
    manager = self._manager(retries=1)
    manager.submit_all()
    results = manager.poll()

    self.assertLen(results, 1)
    self.assertEqual(results[0].status, "SUCCESS")
    self.assertIsNone(results[0].error_message)
    # The retry runs under a new JobSet name with a re-rendered manifest, so
    # its head does not share a DNS name with the terminating old head.
    self.assertEqual(results[0].workload_name, retry_name)
    self.assertEqual(
        [c.args[0] for c in mock_apply.call_args_list],
        ["kind: JobSet\nname: tpums-rapid-all-gather", f"kind: JobSet\nname: {retry_name}"],
    )
    # The failed JobSet is deleted before the retry is created.
    mock_delete.assert_any_call("tpums-rapid-all-gather", utils.DEFAULT_NAMESPACE)
    self.assertEqual(
        [c.args[0] for c in mock_delete_existing.call_args_list],
        ["tpums-rapid-all-gather", retry_name],
    )

  @mock.patch.object(workload_submitter, "upload_provenance")
  @mock.patch.object(workload_submitter, "_delete_existing_jobset")
  @mock.patch.object(workload_submitter, "delete_jobset", return_value=True)
  @mock.patch.object(workload_submitter, "apply_jobset")
  @mock.patch.object(workload_submitter, "fetch_namespace_pods", return_value={})
  @mock.patch.object(workload_submitter, "fetch_namespace_jobsets")
  def test_errors_of_every_attempt_are_reported(
      self, mock_jobsets, mock_pods, mock_apply, mock_delete, mock_delete_existing, mock_upload
  ):
    del mock_pods, mock_delete, mock_delete_existing, mock_upload  # Unused.
    mock_apply.side_effect = lambda manifest: manifest.split("name: ")[1]
    failed_any = {"status": {"conditions": [
        {"type": "Failed", "status": "True", "reason": "Preempted", "message": "rm crashed"}]}}
    mock_jobsets.return_value = {
        "tpums-rapid-all-gather": failed_any, "tpums-rapid-all-gat-r2": failed_any,
    }
    manager = self._manager(retries=1)
    manager.submit_all()
    results = manager.poll()

    self.assertLen(results, 1)
    self.assertEqual(results[0].status, "FAILED")
    self.assertEqual(mock_apply.call_count, 2)
    self.assertRegex(
        results[0].error_message,
        r"attempt 1: FAILED: .*rm crashed; attempt 2: .*rm crashed",
    )

  def test_no_retry_by_default(self):
    manager = self._manager(retries=0)
    wl = manager.pending_workloads.pop("tpums-rapid-all-gather")
    wl.status, wl.terminal, wl.error_message = "TIMEOUT", True, "slow"
    manager.active_workloads = {wl.workload_name: wl}
    self.assertEqual(manager._requeue_failed(), 0)
    self.assertIn(wl.workload_name, manager.active_workloads)


class PodDigestTest(absltest.TestCase):
  """Image digests come from the pods' imageID, no registry permission needed."""

  _SERVER = "sha256:" + "a" * 64
  _PROXY = "sha256:" + "b" * 64
  _CLIENT = "sha256:" + "c" * 64

  def _pods(self):
    head = {"status": {
        "initContainerStatuses": [
            {"name": "pathways-rm", "imageID": f"us-docker.pkg.dev/p/pathways/server@{self._SERVER}"},
            {"name": "pathways-proxy", "imageID": f"us-docker.pkg.dev/p/pathways/proxy_server@{self._PROXY}"},
        ],
        "containerStatuses": [
            {"name": "user-workload", "imageID": f"docker-pullable://us-docker.pkg.dev/c/tpu@{self._CLIENT}"},
        ],
    }}
    worker = {"status": {"containerStatuses": [
        {"name": "pathways-worker", "imageID": f"us-docker.pkg.dev/p/pathways/server@{self._SERVER}"},
        {"name": "sidecar", "imageID": "other@sha256:" + "d" * 64},
    ]}}
    return [head, worker]

  def test_pathways_head_restart_fails_fast(self):
    head_label = {"jobset.sigs.k8s.io/replicatedjob-name": "pathways-head"}
    worker_label = {"jobset.sigs.k8s.io/replicatedjob-name": "pathways-worker"}
    healthy = [
        {"metadata": {"labels": head_label}, "status": {"phase": "Running",
            "initContainerStatuses": [{"name": "pathways-rm", "restartCount": 0}],
            "containerStatuses": [{"name": "user-workload", "restartCount": 0}]}},
        {"metadata": {"labels": worker_label}, "status": {"phase": "Running",
            "containerStatuses": [{"name": "pathways-worker", "restartCount": 2}]}},
    ]
    self.assertEqual(workload_submitter.pathways_head_restarts(healthy), [])
    crashed = [
        {"metadata": {"labels": head_label}, "status": {"phase": "Running",
            "initContainerStatuses": [{"name": "pathways-rm", "restartCount": 1,
                "lastState": {"terminated": {"exitCode": 139, "reason": "Error"}}}],
            "containerStatuses": [{"name": "user-workload", "restartCount": 0}]}},
    ]
    self.assertEqual(
        workload_submitter.pathways_head_restarts(crashed), ["pathways-rm (exit 139)"]
    )
    name = "tpums-rapid-all-gather"
    wl = workload_submitter.ActiveWorkload(
        workload_name=name, config_name="all_gather", config_rel_path="c",
        topology="4x8", manifest_yaml="", gcs_output_dir="gs://b", start_time=time.time(),
        total_pods=9, runtime="rapid",
    )
    # Not terminal from the JobSet's conditions: the restart fails it.
    done = workload_submitter.update_workload_states(
        {name: wl}, {name: {"status": {"conditions": []}}}, {name: crashed}, timeout_minutes=25
    )
    self.assertTrue(done)
    self.assertEqual(wl.status, "FAILED")
    self.assertIn("pathways-rm (exit 139)", wl.error_message)
    # A JobSet that completed anyway keeps its success.
    wl2 = dataclasses.replace(wl, status="SUBMITTED", terminal=False, error_message=None)
    workload_submitter.update_workload_states(
        {name: wl2},
        {name: {"status": {"conditions": [{"type": "Completed", "status": "True"}]}}},
        {name: crashed},
        timeout_minutes=25,
    )
    self.assertEqual(wl2.status, "SUCCESS")

  def test_pod_node_pools_and_placement(self):
    pods = [
        {"spec": {"nodeName": "gke-pw-scale-test-v5e-32-default-pool-ddc42050-0kwu"}},
        {"spec": {"nodeName": "gke-tpu-31dbadc5-jt36"}},
        {"spec": {"nodeName": "gke-tpu-31dbadc5-q8xp"}},
        {"spec": {}},  # not scheduled yet
    ]
    self.assertEqual(
        workload_submitter.pod_node_pools(pods),
        ["gke-pw-scale-test-v5e-32-default-pool-ddc42050", "gke-tpu-31dbadc5"],
    )
    wl = workload_submitter.ActiveWorkload(
        workload_name="w", config_name="gemm", config_rel_path="c", topology="4x8",
        manifest_yaml="", gcs_output_dir="gs://b", start_time=0.0, total_pods=9,
    )
    self.assertFalse(workload_submitter.record_pod_placement(wl, [{"spec": {}}]))
    self.assertTrue(workload_submitter.record_pod_placement(wl, pods))
    self.assertEqual(
        wl.provenance["node_pools"],
        "gke-pw-scale-test-v5e-32-default-pool-ddc42050,gke-tpu-31dbadc5",
    )
    self.assertFalse(workload_submitter.record_pod_placement(wl, pods))
    # The comparison shows the TPU slice, not the CPU pool of the head.
    self.assertEqual(
        runtime_comparison._tpu_slice({"node_pools": wl.provenance["node_pools"]}),
        "tpu-31dbadc5",
    )
    self.assertEqual(
        runtime_comparison._tpu_slice(
            {"node_pools": "gke-pw-scale-test-v5e-32-cpu-np-157a9698,gke-tpu-ee5b4152"}
        ),
        "tpu-ee5b4152",
    )
    self.assertEqual(runtime_comparison._tpu_slice({}), "")

  def test_pod_image_digests(self):
    self.assertEqual(
        workload_submitter.pod_image_digests(self._pods()),
        {
            "pathways_server_digest": self._SERVER,
            "pathways_proxy_digest": self._PROXY,
            "client_image_digest": self._CLIENT,
        },
    )
    self.assertEqual(workload_submitter.pod_image_digests([{"status": {}}]), {})

  def test_record_pod_digests_keeps_registry_digest_when_tag_moved(self):
    wl = workload_submitter.ActiveWorkload(
        workload_name="tpums-rapid-gemm", config_name="gemm",
        config_rel_path="configs/v5e/4x8/gemm.yaml", topology="4x8",
        manifest_yaml="", gcs_output_dir="gs://b/d/rapid/gemm", start_time=0.0,
        total_pods=9, runtime="rapid",
        provenance={"runtime": "rapid", "pathways_server_digest": "sha256:old"},
    )
    self.assertTrue(workload_submitter.record_pod_digests(wl, self._pods()))
    self.assertEqual(wl.provenance["pathways_server_digest"], self._SERVER)
    self.assertEqual(wl.provenance["pathways_server_digest_at_submission"], "sha256:old")
    self.assertEqual(wl.provenance["pathways_proxy_digest"], self._PROXY)
    self.assertEqual(wl.provenance["client_image_digest"], self._CLIENT)
    # Idempotent once recorded.
    self.assertFalse(workload_submitter.record_pod_digests(wl, self._pods()))

  @mock.patch.object(workload_submitter, "upload_provenance")
  @mock.patch.object(workload_submitter, "delete_jobset", return_value=True)
  @mock.patch.object(workload_submitter, "fetch_namespace_pods")
  @mock.patch.object(workload_submitter, "fetch_namespace_jobsets")
  def test_poll_reuploads_provenance_with_pod_digests(
      self, mock_jobsets, mock_pods, mock_delete, mock_upload
  ):
    del mock_delete  # Unused.
    name = "tpums-4x8-mcjax-gemm-abcde"
    wl = workload_submitter.ActiveWorkload(
        workload_name=name, config_name="gemm",
        config_rel_path="configs/v5e/4x8/gemm.yaml", topology="4x8",
        manifest_yaml="", gcs_output_dir="gs://b/d/mcjax/gemm",
        start_time=time.time(), total_pods=1, runtime="mcjax",
        provenance={"runtime": "mcjax", "git_commit": "abc"},
        uploaded_provenance={"runtime": "mcjax", "git_commit": "abc"},
    )
    mock_jobsets.return_value = {name: {"status": {"conditions": [
        {"type": "Completed", "status": "True"}]}}}
    mock_pods.return_value = {name: [{"status": {"phase": "Succeeded", "containerStatuses": [
        {"name": "jax-tpu", "imageID": f"us-docker.pkg.dev/c/tpu@{self._CLIENT}"}]}}]}
    manager = workload_submitter.WorkloadManager(
        timeout_minutes=1, poll_interval_seconds=1, tick_interval_seconds=0,
        cleanup=True, active_workloads={name: wl},
    )
    results = manager.poll()
    self.assertEqual(results[0].status, "SUCCESS")
    self.assertEqual(results[0].provenance["client_image_digest"], self._CLIENT)
    mock_upload.assert_called_once_with(
        "gs://b/d/mcjax/gemm",
        {"runtime": "mcjax", "git_commit": "abc", "client_image_digest": self._CLIENT},
    )


class DashboardTest(absltest.TestCase):

  def _results(self):
    ok = workload_submitter.BenchmarkResult(
        workload_name="tpums-rapid-gemm", config_name="gemm", config_rel_path="c",
        topology="4x8", status="SUCCESS", duration_seconds=95.0, gcs_artifact_path="gs://b",
        runtime="rapid", provenance={"node_pools": "gke-cpu-np-1,gke-tpu-abcd1234"},
    )
    bad = workload_submitter.BenchmarkResult(
        workload_name="tpums-louhi-hbm", config_name="hbm", config_rel_path="c",
        topology="4x8", status="TIMEOUT", duration_seconds=1500.0, gcs_artifact_path="gs://b",
        runtime="louhi", error_message="attempt 1: TIMEOUT: Exceeded timeout of 25 minutes",
    )
    return [ok, bad]

  def test_render_dashboard_embeds_rows_and_jobsets(self):
    rows = {"collectives": [_row("mcjax", "1.0", "2.0"), _row("rapid", "1.1", "8.0", node_pools="gke-tpu-abcd1234"),
                            _row("louhi", "", "9.0")], "gemm": [], "hbm": []}
    html = dashboard.render_dashboard(
        rows, self._results(), date_str="2026-10-29", cluster="pw-scale-test-v5e-32",
        report_uri="2026-10-29.html", history_dates=["2026-10-30", "2026-10-29"],
    )
    self.assertIn("TPUMS nightly — 2026-10-29", html)
    self.assertIn("pw-scale-test-v5e-32", html)
    self.assertIn('<a href="2026-10-29.html">2026-10-29.html</a>', html)
    self.assertIn('<a href="2026-10-30_dashboard.html">2026-10-30</a>', html)
    self.assertIn("<b>2026-10-29</b>", html)
    embedded = json.loads(html.split("const ROWS = ")[1].split(";\n")[0])
    self.assertLen(embedded, 3)
    self.assertEqual({r["runtime"] for r in embedded}, {"mcjax", "rapid", "louhi"})
    self.assertNotIn("xprof_p50_ms", embedded[2])  # empty cells are dropped
    jobsets = json.loads(html.split("const JOBSETS = ")[1].split(";\n")[0])
    self.assertEqual([j["status"] for j in jobsets], ["SUCCESS", "TIMEOUT"])
    self.assertEqual(jobsets[0]["slice"], "tpu-abcd1234")
    self.assertIn("Exceeded timeout", jobsets[1]["error"])
    history = json.loads(html.split("const HISTORY = ")[1].split(";\n")[0])
    self.assertLen(history, 1)
    self.assertEqual(history[0]["run_id"], "2026-10-29")
    # Self-contained: no external scripts or styles.
    self.assertNotIn("<script src=", html)
    self.assertNotIn("<link ", html)

  @mock.patch.object(subprocess, "run")
  def test_update_history_appends_and_sorts_snapshots(self, mock_run):
    existing = [{"run_id": "2026-10-29 21:55 UTC", "rows": [_row("mcjax", "1.0", "2.0")]}]
    mock_run.side_effect = [
        subprocess.CompletedProcess(args=[], returncode=0, stdout=json.dumps(existing), stderr=""),
        subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr=""),
    ]
    updated = dashboard.update_history(
        "gs://b/tpums-nightly",
        "2026-10-30 01:00 UTC",
        {"collectives": [_row("rapid", "1.1", "8.0")]},
    )
    self.assertEqual([h["run_id"] for h in updated], ["2026-10-29 21:55 UTC", "2026-10-30 01:00 UTC"])

  @mock.patch.object(subprocess, "run")
  def test_list_report_dates_extracts_sorted_dates(self, mock_run):
    mock_run.return_value = subprocess.CompletedProcess(
        args=[], returncode=0,
        stdout=(
            "gs://b/tpums-nightly/reports/2026-10-29_dashboard.html\n"
            "gs://b/tpums-nightly/reports/2026-10-30_dashboard.html\n"
            "gs://b/tpums-nightly/reports/latest_dashboard.html\n"
        ),
        stderr="",
    )
    self.assertEqual(
        dashboard.list_report_dates("gs://b/tpums-nightly"),
        ["2026-10-30", "2026-10-29"],
    )

  @mock.patch.object(subprocess, "run")
  def test_publish_dashboard_writes_day_and_latest(self, mock_run):
    mock_run.return_value = subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")
    written = dashboard.publish_dashboard("gs://b/tpums-nightly", "2026-10-29", "<html>")
    self.assertEqual(written, [
        "gs://b/tpums-nightly/2026-10-29/dashboard.html",
        "gs://b/tpums-nightly/reports/2026-10-29_dashboard.html",
        "gs://b/tpums-nightly/reports/latest_dashboard.html",
    ])
    for call in mock_run.call_args_list:
      self.assertEqual(
          call.args[0][:5],
          [
              "gcloud",
              "storage",
              "cp",
              "--content-type=text/html",
              "--cache-control=no-cache,max-age=0",
          ],
      )
      self.assertEqual(call.kwargs["input"], "<html>")


class RuntimeComparisonTest(absltest.TestCase):

  def test_build_comparison_ratios_and_xprof_status(self):
    rows = [_row("mcjax", "10.0", "19.0"), _row("louhi", "", "19.9"), _row("rapid", "10.5", "19.2")]
    tables = runtime_comparison.build_comparison({"collectives": rows})
    self.assertLen(tables, 1)
    table = tables[0]
    self.assertNotIn("wall_clock_std_ms", table.headers)
    self.assertIn("xprof_avg_ms", table.headers)
    self.assertEqual(table.runtimes, ["mcjax", "louhi", "rapid"])
    by_runtime = {r["runtime"]: r for r in table.rows}
    self.assertEqual(by_runtime["mcjax"]["vs_mcjax"], "baseline")
    self.assertEqual(by_runtime["rapid"]["vs_mcjax"], "1.0500x (xprof)")
    self.assertEqual(by_runtime["louhi"]["vs_mcjax"], "1.0474x (wall)")
    self.assertEqual(by_runtime["louhi"]["xprof"], "wall-clock only")
    self.assertEqual(by_runtime["rapid"]["xprof"], "device 5/5")
    self.assertEqual(by_runtime["rapid"]["throughput"], "51.100 GB/s/chip")
    self.assertEqual(by_runtime["louhi"]["throughput"], "27.100 GB/s/chip (wall)")
    self.assertStartsWith(by_runtime["mcjax"]["case"], "AllGatherBenchmark topology=4x8")

  def test_rows_without_runtime_count_as_mcjax(self):
    rows = [{k: v for k, v in _row("mcjax", "1.0", "2.0").items() if k != "runtime"}]
    tables = runtime_comparison.build_comparison({"collectives": rows})
    self.assertEqual(tables[0].rows[0]["runtime"], "mcjax")
    self.assertEqual(tables[0].rows[0]["vs_mcjax"], "baseline")

  def test_provenance_summary_shortens_digests(self):
    rows = [
        _row("rapid", "1", "2", git_commit="abcdef0123456789", pathways_server_digest="sha256:0123456789abcdef"),
        _row("mcjax", "1", "2", git_commit="abcdef0123456789"),
    ]
    summary = runtime_comparison.runtime_provenance_summary({"collectives": rows})
    self.assertEqual([r["runtime"] for r in summary], ["mcjax", "rapid"])
    self.assertEqual(summary[1]["pathways_server_digest"], "0123456789ab")
    self.assertEqual(summary[0]["git_commit"], "abcdef012345")


class BqWriterTest(absltest.TestCase):

  def test_row_to_record_maps_ubench_columns(self):
    row = _row(
        "rapid",
        "10.0",
        "19.0",
        num_runs="5",
        total_devices="32",
        jax_version="0.12.0",
        libtpu_version="0.0.50",
        libtpu_init_args="--xla_a=1",
        git_commit="abc",
        pathways_server_image="srv:latest",
        pathways_server_digest="sha256:1",
        pathways_proxy_image="proxy:latest",
        pathways_proxy_digest="sha256:2",
    )
    record = bq_writer.row_to_record(
        row,
        "collectives",
        run_name="tpums-nightly-2026-10-29",
        run_id="r0",
        started_at=datetime.datetime(2026, 10, 29, tzinfo=datetime.timezone.utc),
        accelerator="v5e",
        container_image="img",
        artifacts_dir="gs://b/x",
        region="us-south1",
        cluster="c",
        user="me",
    )
    self.assertEqual(record["benchmark_type"], "collective")
    self.assertEqual(record["benchmark_name"], "AllGatherBenchmark")
    self.assertEqual(record["metrics_time_per_iter_ms_p50"], 10.0)
    self.assertEqual(record["metrics_achieved_bandwidth_gbps_p50"], 51.1)
    self.assertIsNone(record["metrics_tflops_achieved_p50"])
    self.assertEqual(record["hardware_id"], "v5litepod")
    self.assertEqual(record["hardware_num_chips"], 32)
    self.assertEqual(record["hardware_num_nodes"], 8)
    self.assertEqual(record["hardware_topology"], "4x8")
    self.assertEqual(record["params_num_runs"], 5)
    self.assertEqual(record["params_dtype"], "bfloat16")
    self.assertEqual(record["configs_xla_flags"], "--xla_a=1")
    params = json.loads(record["params_other"])
    self.assertEqual(params["runtime"], "rapid")
    self.assertEqual(params["pathways"]["server_image"], "srv:latest")
    self.assertEqual(params["matrix_dim"], "8192")
    self.assertEqual(params["platform"]["jax_version"], "0.12.0")
    metrics = json.loads(record["metrics_other"])
    self.assertEqual(metrics["wall_clock_p50_ms"], "19.0")
    self.assertNotIn("matrix_dim", metrics)
    versions = {v["name"]: v["version"] for v in json.loads(record["configs_software_versions"])}
    self.assertEqual(versions["accelerator_microbenchmarks"], "abc")
    self.assertEqual(set(record), {name for name, _ in bq_writer.SCHEMA})

  def test_build_records_falls_back_to_wall_clock_and_keys_artifacts(self):
    rows = [_row("louhi", "", "19.9", total_devices="32")]
    records = bq_writer.build_records(
        {"collectives": rows},
        {("louhi", "AllGatherBenchmark"): "gs://b/2026-10-29/louhi/all_gather"},
        date_str="2026-10-29",
        started_at=datetime.datetime.now(datetime.timezone.utc),
        accelerator="v5e",
        container_image="img",
        region="r",
        cluster="c",
        user="me",
    )
    self.assertLen(records, 1)
    self.assertEqual(records[0]["metrics_time_per_iter_ms_p50"], 19.9)
    self.assertEqual(records[0]["logs_artifact_directory"], "gs://b/2026-10-29/louhi/all_gather")
    self.assertEqual(records[0]["run_name"], "tpums-nightly-2026-10-29")
    self.assertRegex(
        records[0]["run_id"],
        r"^tpums-nightly-2026-10-29_louhi_AllGatherBenchmark_[0-9a-f]{10}$",
    )

  def test_case_run_id_is_stable_and_ignores_metrics_and_provenance(self):
    base = _row("rapid", "1.0", "2.0", client_image_digest="sha256:aaa", git_commit="c1",
                start="2026-10-29T20:08:59+00:00", node_pools="gke-tpu-a",
                test_name="AllGatherBenchmark_1793306671")
    rerun = _row("rapid", "1.1", "2.3", client_image_digest="sha256:bbb", git_commit="c2",
                 start="2026-10-29T20:24:24+00:00", node_pools="gke-tpu-b",
                 test_name="AllGatherBenchmark_1793307999")
    other_case = _row("rapid", "1.0", "2.0", matrix_dim="16384")
    ids = []
    for row in (base, rerun, other_case):
      record = bq_writer.row_to_record(
          row, "collectives", run_name="n", started_at=datetime.datetime.now(datetime.timezone.utc),
          accelerator="v5e", container_image="img", artifacts_dir="gs://a", region="r",
          cluster="c", user="me",
      )
      ids.append(record["run_id"])
      params = json.loads(record["params_other"])
      self.assertNotIn("git_commit", params)
      self.assertEqual(params["runtime"], "rapid")
    self.assertEqual(ids[0], ids[1])  # same case, different night metrics/build
    self.assertNotEqual(ids[0], ids[2])  # different matrix_dim
    self.assertEqual(
        json.loads(bq_writer.row_to_record(
            base, "collectives", run_name="n", started_at=datetime.datetime.now(datetime.timezone.utc),
            accelerator="v5e", container_image="img", artifacts_dir="gs://a", region="r",
            cluster="c", user="me",
        )["params_other"])["client_image_digest"],
        "sha256:aaa",
    )


class AggregatorRuntimeStampTest(absltest.TestCase):

  def test_normalize_row_stamps_runtime_columns_first(self):
    row = {"benchmark": "GemmBenchmark", "xprof_p50_ms": "1", "_source_path": "gs://x/gemm/summary.csv"}
    norm = metrics_aggregator.normalize_row(
        row, "gemm", runtime="rapid", provenance={"pathways_server_image": "srv:latest", "git_commit": "abc"}
    )
    self.assertEqual(list(norm)[: len(metrics_aggregator.RUNTIME_COLUMNS)], list(metrics_aggregator.RUNTIME_COLUMNS))
    self.assertEqual(norm["runtime"], "rapid")
    self.assertEqual(norm["pathways_server_image"], "srv:latest")
    self.assertEqual(norm["pathways_proxy_digest"], "")
    self.assertNotIn("_source_path", norm)

  def test_normalize_row_without_runtime_is_unchanged(self):
    row = {"benchmark": "GemmBenchmark", "xprof_p50_ms": "1"}
    self.assertEqual(metrics_aggregator.normalize_row(row, "gemm"), row)


if __name__ == "__main__":
  absltest.main()
