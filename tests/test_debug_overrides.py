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
"""Tests for the debugging override: where it applies, and that reports say so.

dcn_all_reduce turns the enhanced launch barrier off, which the Pathways
resource manager only accepts with late binding off, a debugging override (see
templates/pathways_jobset.py). A run that uses it is not a supported
configuration, so the provenance, both reports, the dashboard and BigQuery
have to name it.
"""

import datetime
import json
import pathlib

from absl.testing import absltest
from automation import bq_writer
from automation import dashboard
from automation import metrics_aggregator
from automation import report_generator
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

_FLAG = "--temporary_flags_for_debugging"
_LATE_BINDING_OFF = "temporary_flag_for_debugging_enable_late_binding=false"
_OVERRIDE = f"{_FLAG}={_LATE_BINDING_OFF}"
_PROVENANCE = f"pathways-rm: {_OVERRIDE}"
_V5E = "configs/v5e/4x8"


def _pathways_manifest(config, runtime=runtimes.LOUHI, topology="4x8"):
  return workload_submitter.generate_pathways_manifest(
      workload_name="tpums-louhi-x",
      namespace="default",
      queue_name="multislice-queue",
      priority_class="",
      topology=topology,
      docker_image="img:latest",
      config_rel_path=config,
      gcs_output_dir="gs://b/2026-11-18/louhi/x",
      runtime=runtime,
      num_slices=utils.config_num_slices(config),
  )


def _containers(manifest):
  """Maps container names of a Pathways manifest to their specs."""
  found = {}
  for job in yaml.safe_load(manifest)["spec"]["replicatedJobs"]:
    spec = job["template"]["spec"]["template"]["spec"]
    for container in spec.get("initContainers", []) + spec["containers"]:
      found[container["name"]] = container
  return found


def _dcn_row(runtime, benchmark, matrix_dim, xprof, wall, override=""):
  return {
      "runtime": runtime,
      "benchmark": benchmark,
      "topology": "4x8",
      "replica_group_rank": "2",
      "dtype": "bfloat16",
      "matrix_dim": matrix_dim,
      "shard_size_mib": "256.0",
      "xprof_p50_ms": xprof,
      "xprof_p90_ms": xprof,
      "wall_clock_p50_ms": wall,
      "wall_clock_p90_ms": wall,
      "wall_clock_avg_ms": wall,
      "xprof_captured_runs": "5",
      "xprof_expected_runs": "5",
      "xprof_bandwidth_per_host_gbps": "58.7",
      "wall_clock_bandwidth_per_host_gbps": "46.0",
      runtimes.DEBUG_OVERRIDES_KEY: override,
  }


def _categorized(with_override=True):
  override = _PROVENANCE if with_override else ""
  return {
      "dcn_collectives": [
          _dcn_row("mcjax", "DcnAllReduceBenchmark", "131072", "135.1", "154.4"),
          _dcn_row("louhi", "DcnAllReduceBenchmark", "131072", "149.8", "175.9", override),
          _dcn_row("rapid", "DcnAllReduceBenchmark", "131072", "134.1", "168.6", override),
          _dcn_row("mcjax", "DcnAllReduceBenchmark", "262144", "289.4", "331.0"),
          _dcn_row("louhi", "DcnAllReduceBenchmark", "262144", "260.4", "335.6", override),
          _dcn_row("mcjax", "DcnAllToAllBenchmark", "131072", "80.5", "84.8"),
          _dcn_row("louhi", "DcnAllToAllBenchmark", "131072", "87.7", "102.4"),
      ],
  }


@absltest.skipUnless(_HAS_PATHWAYSUTILS, "pathwaysutils not installed")
class ManifestTest(absltest.TestCase):

  def test_dcn_all_reduce_keeps_upstream_flags_and_overrides_the_rm(self):
    manifest = _pathways_manifest(f"{_V5E}/dcn_all_reduce.yaml")
    containers = _containers(manifest)
    # Workers and proxy run the op flags as McJAX does: barrier off.
    for name in ("pathways-worker", "pathways-proxy"):
      self.assertIn(
          "--xla_tpu_use_enhanced_launch_barrier=false", containers[name]["args"], name
      )
      self.assertFalse([a for a in containers[name]["args"] if a.startswith(_FLAG)], name)
    # The resource manager gets the override, once, and no libtpu flag.
    rm_args = containers["pathways-rm"]["args"]
    self.assertEqual([a for a in rm_args if a.startswith(_FLAG)], [_OVERRIDE])
    self.assertFalse(
        [a for a in rm_args if a.startswith(("--xla_", "--megascale_", "--pathways_"))]
    )
    self.assertIn("--instance_type=tpuv5e:4x8", rm_args)
    self.assertEqual(workload_submitter.manifest_debug_overrides(manifest), _PROVENANCE)

  def test_rapid_gets_the_same_override(self):
    manifest = _pathways_manifest(f"{_V5E}/dcn_all_reduce.yaml", runtime=runtimes.RAPID)
    self.assertEqual(workload_submitter.manifest_debug_overrides(manifest), _PROVENANCE)

  def test_benchmarks_that_keep_the_barrier_get_no_override(self):
    for config in ("dcn_all_to_all", "all_gather", "all_reduce", "gemm", "hbm"):
      manifest = _pathways_manifest(f"{_V5E}/{config}.yaml")
      self.assertNotIn(_FLAG, manifest, config)
      self.assertEqual(workload_submitter.manifest_debug_overrides(manifest), "", config)

  def test_client_reads_the_op_flags_file_the_workers_run_with(self):
    client = _containers(_pathways_manifest(f"{_V5E}/dcn_all_reduce.yaml"))["user-workload"]
    env = {e["name"]: e.get("value") for e in client["env"]}
    self.assertEqual(env["TPUMS_OP_FLAGS_FILE"], "automation/op_flags/v5e.yaml")
    self.assertIn(
        '${TPUMS_OP_FLAGS_FILE:+--xla_flags_file_path="$TPUMS_OP_FLAGS_FILE"}',
        client["command"][-1],
    )
    # Without a file for the accelerator, the client uses op_flags.yaml too.
    client = _containers(
        _pathways_manifest("configs/tpu7x/2x2x1/dcn_all_reduce.yaml", topology="2x2x1")
    )["user-workload"]
    env = {e["name"]: e.get("value") for e in client["env"]}
    self.assertFalse(env["TPUMS_OP_FLAGS_FILE"])

  def test_provenance_names_the_override_of_that_workload_only(self):
    manager = workload_submitter.WorkloadManager(
        dry_run=True, runtimes=[runtimes.MCJAX, runtimes.LOUHI]
    )
    configs = [
        pathlib.Path(utils.PACKAGE_ROOT) / _V5E / f"{name}.yaml"
        for name in ("all_gather", "dcn_all_reduce")
    ]
    results = manager.run(configs)
    overrides = {
        (r.runtime, r.config_name): r.provenance.get(runtimes.DEBUG_OVERRIDES_KEY)
        for r in results
    }
    self.assertEqual(
        overrides,
        {
            ("mcjax", "all_gather"): None,
            ("mcjax", "dcn_all_reduce"): None,
            ("louhi", "all_gather"): None,
            ("louhi", "dcn_all_reduce"): _PROVENANCE,
        },
    )


class McjaxManifestTest(absltest.TestCase):

  def test_mcjax_manifest_has_no_override(self):
    manifest = workload_submitter.generate_jobset_manifest(
        workload_name="tpums-test",
        namespace="default",
        queue_name="multislice-queue",
        priority_class="medium",
        topology="4x8",
        docker_image="img:latest",
        config_rel_path=f"{_V5E}/dcn_all_reduce.yaml",
        gcs_output_dir="gs://b/x",
        num_slices=2,
    )
    self.assertEqual(workload_submitter.manifest_debug_overrides(manifest), "")


class ReportTest(absltest.TestCase):

  def test_override_is_a_runtime_column(self):
    norm = metrics_aggregator.normalize_row(
        {"benchmark": "DcnAllReduceBenchmark"},
        "dcn_collectives",
        runtime="louhi",
        provenance={runtimes.DEBUG_OVERRIDES_KEY: _PROVENANCE},
    )
    self.assertEqual(norm[runtimes.DEBUG_OVERRIDES_KEY], _PROVENANCE)
    plain = metrics_aggregator.normalize_row(
        {"benchmark": "DcnAllReduceBenchmark"}, "dcn_collectives", runtime="mcjax", provenance={}
    )
    self.assertEqual(plain[runtimes.DEBUG_OVERRIDES_KEY], "")

  def test_summary_lists_each_use_once_and_says_what_it_does(self):
    summary = runtime_comparison.debug_override_summary(_categorized())
    self.assertEqual(
        [(s["runtime"], s["benchmark"], s["container"], s["override"]) for s in summary],
        [
            ("louhi", "DcnAllReduceBenchmark", "pathways-rm", _OVERRIDE),
            ("rapid", "DcnAllReduceBenchmark", "pathways-rm", _OVERRIDE),
        ],
    )
    for entry in summary:
      self.assertIn("Late binding is off", entry["effect"])
      self.assertIn("not a supported one", entry["effect"])
    self.assertEmpty(runtime_comparison.debug_override_summary(_categorized(False)))

  def test_summary_splits_several_overrides_and_flags_unknown_ones(self):
    rows = {
        "dcn_collectives": [
            _dcn_row(
                "louhi", "DcnAllReduceBenchmark", "131072", "1", "2",
                f"pathways-rm: {_FLAG}={_LATE_BINDING_OFF};;;temporary_flag_for_debugging_x=1"
                f", pathways-proxy: {_FLAG}=temporary_flag_for_debugging_y=2",
            )
        ]
    }
    summary = runtime_comparison.debug_override_summary(rows)
    self.assertEqual(
        sorted((s["container"], s["override"]) for s in summary),
        [
            ("pathways-proxy", f"{_FLAG}=temporary_flag_for_debugging_y=2"),
            ("pathways-rm", _OVERRIDE),
            ("pathways-rm", f"{_FLAG}=temporary_flag_for_debugging_x=1"),
        ],
    )
    unknown = [s for s in summary if s["override"].endswith("_x=1")][0]
    self.assertIn("Not described", unknown["effect"])

  def test_dcn_cases_stay_apart_with_per_host_bandwidth_and_notes(self):
    rows = _categorized()["dcn_collectives"]
    (table,) = runtime_comparison.build_comparison({"dcn_collectives": rows})
    self.assertLen(table.rows, len(rows))
    self.assertIn("notes", table.headers)
    by_key = {(r["case"], r["runtime"]): r for r in table.rows}
    case = "DcnAllReduceBenchmark topology=4x8 replica_group_rank=2 dtype=bfloat16 matrix_dim=131072"
    self.assertEqual(by_key[(case, "louhi")]["throughput"], "58.700 Gbps/host")
    self.assertEqual(by_key[(case, "louhi")]["notes"], "debugging override")
    self.assertEqual(by_key[(case, "mcjax")]["notes"], "")
    self.assertEqual(by_key[(case, "rapid")]["vs_mcjax"], "0.99260x (xprof)")
    a2a = [r for r in table.rows if r["case"].startswith("DcnAllToAll")]
    self.assertEqual({r["notes"] for r in a2a}, {""})

  def test_both_reports_name_the_override(self):
    report_md = report_generator.format_consolidated_report(
        [], categorized_metrics=_categorized(), cluster="c", project="p", region="r"
    )
    self.assertIn("## Debugging overrides", report_md)
    self.assertIn("did not run in a supported Pathways configuration", report_md)
    self.assertIn(f"| louhi | DcnAllReduceBenchmark | pathways-rm | {_OVERRIDE} |", report_md)
    self.assertIn(f"| rapid | DcnAllReduceBenchmark | pathways-rm | {_OVERRIDE} |", report_md)
    # The section comes before the numbers it qualifies.
    self.assertLess(
        report_md.index("## Debugging overrides"), report_md.index("## Runtime Comparison")
    )
    report_html = report_generator.generate_html_report(
        [], categorized_metrics=_categorized(), cluster="c", project="p", region="r"
    )
    self.assertIn("Debugging overrides</h3>", report_html)
    self.assertIn(_OVERRIDE, report_html)
    self.assertIn("did not run in a supported Pathways configuration", report_html)

  def test_reports_without_an_override_have_no_section(self):
    report_md = report_generator.format_consolidated_report(
        [], categorized_metrics=_categorized(False), cluster="c", project="p", region="r"
    )
    self.assertNotIn("Debugging overrides", report_md)
    report_html = report_generator.generate_html_report(
        [], categorized_metrics=_categorized(False), cluster="c", project="p", region="r"
    )
    self.assertNotIn("Debugging overrides", report_html)

  def test_dashboard_names_the_override_and_marks_its_rows(self):
    html = dashboard.render_dashboard(_categorized(), [], date_str="2026-11-18")
    overrides = json.loads(html.split("const OVERRIDES = ")[1].split(";\n")[0])
    self.assertEqual(
        [(o["runtime"], o["benchmark"], o["override"]) for o in overrides],
        [
            ("louhi", "DcnAllReduceBenchmark", _OVERRIDE),
            ("rapid", "DcnAllReduceBenchmark", _OVERRIDE),
        ],
    )
    rows = json.loads(html.split("const ROWS = ")[1].split(";\n")[0])
    marked = {
        (r["runtime"], r["benchmark"]) for r in rows if r.get(runtimes.DEBUG_OVERRIDES_KEY)
    }
    self.assertEqual(
        marked, {("louhi", "DcnAllReduceBenchmark"), ("rapid", "DcnAllReduceBenchmark")}
    )
    # The note sits above the charts, and DCN benchmarks have their own section.
    self.assertLess(html.index('id="overrides"'), html.index('id="charts"'))
    self.assertLess(html.index('id="charts"'), html.index('id="dcn-charts"'))
    self.assertIn("Debugging override in effect.", html)
    self.assertNotIn("__OVERRIDES__", html)

  def test_dashboard_without_an_override_embeds_none(self):
    html = dashboard.render_dashboard(_categorized(False), [], date_str="2026-11-18")
    self.assertEqual(json.loads(html.split("const OVERRIDES = ")[1].split(";\n")[0]), [])

  def test_bigquery_keeps_the_override_but_not_in_the_case_id(self):
    records = []
    for override in (_PROVENANCE, ""):
      row = _dcn_row("louhi", "DcnAllReduceBenchmark", "131072", "149.8", "175.9", override)
      records.append(
          bq_writer.row_to_record(
              row, "dcn_collectives", run_name="n",
              started_at=datetime.datetime.now(datetime.timezone.utc),
              accelerator="v5e", container_image="img", artifacts_dir="gs://a",
              region="r", cluster="c", user="me",
          )
      )
    with_override, without = (json.loads(r["params_other"]) for r in records)
    self.assertEqual(with_override["pathways"]["debug_overrides"], _PROVENANCE)
    self.assertNotIn("debug_overrides", without["pathways"])
    self.assertNotIn(runtimes.DEBUG_OVERRIDES_KEY, with_override)
    self.assertEqual(records[0]["run_id"], records[1]["run_id"])


if __name__ == "__main__":
  absltest.main()
