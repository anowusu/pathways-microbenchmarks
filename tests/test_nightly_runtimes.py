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

import datetime
import json
import os
import subprocess
import tempfile
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from automation import bq_writer
from automation import metrics_aggregator
from automation import runtime_comparison
from automation import runtimes
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


class RuntimeComparisonTest(absltest.TestCase):

  def test_build_comparison_ratios_and_xprof_status(self):
    rows = [_row("mcjax", "10.0", "19.0"), _row("louhi", "", "19.9"), _row("rapid", "10.5", "19.2")]
    tables = runtime_comparison.build_comparison({"collectives": rows})
    self.assertLen(tables, 1)
    table = tables[0]
    self.assertEqual(table.runtimes, ["mcjax", "louhi", "rapid"])
    by_runtime = {r["runtime"]: r for r in table.rows}
    self.assertEqual(by_runtime["mcjax"]["vs_mcjax"], "baseline")
    self.assertEqual(by_runtime["rapid"]["vs_mcjax"], "1.05x (xprof)")
    self.assertEqual(by_runtime["louhi"]["vs_mcjax"], "1.05x (wall)")
    self.assertEqual(by_runtime["louhi"]["xprof"], "wall-clock only")
    self.assertEqual(by_runtime["rapid"]["xprof"], "device 5/5")
    self.assertEqual(by_runtime["rapid"]["throughput"], "51.10 GB/s/chip")
    self.assertEqual(by_runtime["louhi"]["throughput"], "27.10 GB/s/chip (wall)")
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
    self.assertEqual(records[0]["run_id"], "tpums-nightly-2026-10-29_louhi_collectives_0")


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
