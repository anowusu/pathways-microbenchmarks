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

"""Automated benchmark runner for TPUMS on GKE TPU clusters via kubectl & Kueue.

Orchestrates the benchmark pipeline by coordinating:
  - Workload Submitter: Resolves configs from canonical suite YAMLs or
    configs/<accelerator>/ (tpu7x or v6e), generates parameterized JobSet
    manifests, submits via kubectl, and polls status.
  - Metrics Aggregator: Ingests each workload's GCS summary.csv, normalizes
    schemas, and aggregates category metrics (collectives, hbm, host_device,
    device_to_device, gemm).
  - Report Generator: Generates consolidated Markdown summary tables
    and responsive HTML preview reports.
"""

import argparse
from collections.abc import Sequence
import dataclasses
import datetime
import logging
import os
import subprocess
import sys
from typing import Any, Optional

from automation import bq_writer
from automation import dashboard
from automation import metrics_aggregator
from automation import oss_image_patcher
from automation import report_generator
from automation import runtimes as runtimes_lib
from automation import utils
from automation import workload_submitter

DEFAULT_CLEANUP = True
# Dated and rolling copies of the HTML/Markdown report, under the bucket.
REPORTS_SUBDIR = "reports"


def publish_report_copies(
    gcs_bucket: str, date_str: str, output_report: str
) -> None:
  """Copies the night's reports to `<bucket>/reports/<date>.*` and `latest.*`.

  The per-date `consolidated_report.*` written by the report generator stays
  where it is; these copies give one stable URL for the newest report.
  """
  html_path = os.path.splitext(output_report)[0] + ".html"
  for local_path, ext in ((output_report, "md"), (html_path, "html")):
    if not os.path.exists(local_path):
      continue
    for name in (f"{date_str}.{ext}", f"latest.{ext}"):
      target = utils.format_gcs_uri(gcs_bucket, REPORTS_SUBDIR, name)
      proc = subprocess.run(
          [
              "gcloud",
              "storage",
              "cp",
              "--cache-control=no-cache,max-age=0",
              local_path,
              target,
          ],
          capture_output=True,
          text=True,
          check=False,
          stdin=subprocess.DEVNULL,
          timeout=utils.CLI_TIMEOUT_SECONDS,
      )
      if proc.returncode == 0:
        logging.info("Published %s", target)
      else:
        logging.error(
            "Failed to publish %s (gcloud exit %d): %s",
            target,
            proc.returncode,
            (proc.stderr or "").strip(),
        )


def publish_dashboard(
    gcs_bucket: str,
    date_str: str,
    categorized_metrics: dict[str, list[dict[str, Any]]],
    results: Sequence[workload_submitter.BenchmarkResult],
    *,
    cluster: str = "",
    run_id: str = "",
) -> list[str]:
  """Renders and uploads the interactive dashboard; failures are logged."""
  try:
    history = dashboard.update_history(
        gcs_bucket, run_id or date_str, categorized_metrics
    )
    history_dates = dashboard.list_report_dates(gcs_bucket)
    html = dashboard.render_dashboard(
        categorized_metrics,
        results,
        date_str=date_str,
        cluster=cluster,
        report_uri=f"{date_str}.html",
        history_dates=history_dates,
        history=history,
    )
    return dashboard.publish_dashboard(gcs_bucket, date_str, html)
  except Exception:  # pylint: disable=broad-exception-caught
    # The report and BigQuery rows are the primary outputs.
    logging.exception("Dashboard generation failed.")
    return []


def write_bigquery(
    bq_table: str,
    results: Sequence[workload_submitter.BenchmarkResult],
    categorized_metrics: dict[str, list[dict[str, Any]]],
    *,
    date_str: str,
    started_at: datetime.datetime,
    accelerator: str,
    docker_image: str,
    region: str,
    cluster: str,
) -> int:
  """Writes the night's aggregated rows to BigQuery; failures are logged."""
  artifacts_by_key: dict[tuple[str, str], str] = {}
  for r in results:
    for row in r.metrics:
      artifacts_by_key[(r.runtime, str(row.get("benchmark")))] = (
          r.gcs_artifact_path
      )
  records = bq_writer.build_records(
      categorized_metrics,
      artifacts_by_key,
      date_str=date_str,
      started_at=started_at,
      accelerator=accelerator,
      container_image=docker_image,
      region=region,
      cluster=cluster,
  )
  try:
    return bq_writer.write_records(bq_table, records)
  except Exception:  # pylint: disable=broad-exception-caught
    # The GCS report is the primary output; a BigQuery outage must not fail
    # the night.
    logging.exception("Writing %d row(s) to %s failed.", len(records), bq_table)
    return 0


def _default_output_report(
    now_utc: Optional[datetime.datetime] = None,
) -> str:
  """Derives the dated default Markdown report path from DEFAULT_OUTPUT_REPORT."""
  if now_utc is None:
    now_utc = datetime.datetime.now(datetime.timezone.utc)
  base, ext = report_generator.DEFAULT_OUTPUT_REPORT.rsplit(".", 1)
  return f"{base}_{now_utc.strftime('%Y%m%d')}.{ext}"


def _result_from_gcs_entry(
    entry: dict[str, Any],
) -> workload_submitter.BenchmarkResult:
  """Wraps a `discover_and_fetch_gcs_results` entry as a BenchmarkResult."""
  return workload_submitter.BenchmarkResult(
      workload_name=entry["workload_name"],
      config_name=entry["config_name"],
      config_rel_path="",
      topology=entry["topology"],
      status="SUCCESS",
      duration_seconds=0.0,
      gcs_artifact_path=entry["gcs_artifact_path"],
      metrics=entry["metrics"],
      runtime=entry.get("runtime") or runtimes_lib.MCJAX.name,
      provenance=entry.get("provenance") or {},
  )


def day_results(
    gcs_bucket: str,
    date_str: str,
    run_results: Sequence[workload_submitter.BenchmarkResult],
) -> list[workload_submitter.BenchmarkResult]:
  """Returns every result of the day found on GCS, else this run's results.

  The day's GCS prefix is the unit of reporting: a re-run of the day, or of a
  single config, overwrites that config's `summary.csv` and then aggregates
  the whole day again, so the report and BigQuery always show the day's
  latest measurement per runtime and config rather than only this run's.
  """
  entries = metrics_aggregator.discover_and_fetch_gcs_results(
      gcs_bucket, date_str=date_str
  )
  if not entries:
    return list(run_results)
  by_key = {(r.runtime, r.config_name): r for r in run_results}
  merged = []
  for e in entries:
    r = _result_from_gcs_entry(e)
    live = by_key.get((r.runtime, r.config_name))
    if live is not None:
      r.duration_seconds = live.duration_seconds
      r.status = live.status
      r.error_message = live.error_message
    merged.append(r)
  return merged


def run_pipeline(
    config_dir: Optional[str] = None,
    configs: Optional[list[str]] = None,
    categories: Optional[list[str]] = None,
    topologies: Optional[list[str]] = None,
    suite_file: Optional[str] = None,
    cluster: str = utils.DEFAULT_CLUSTER,
    project: str = utils.DEFAULT_PROJECT,
    region: str = utils.DEFAULT_REGION,
    zone: str = utils.DEFAULT_ZONE,
    namespace: str = utils.DEFAULT_NAMESPACE,
    queue: str = utils.DEFAULT_QUEUE,
    priority: str = utils.DEFAULT_PRIORITY,
    docker_image: str = utils.DEFAULT_DOCKER_IMAGE,
    gcs_bucket: str = utils.DEFAULT_GCS_BUCKET,
    timeout_minutes: Optional[int] = utils.DEFAULT_TIMEOUT_MINUTES,
    declared_duration_minutes: int = utils.DEFAULT_DECLARED_DURATION_MINUTES,
    poll_interval_seconds: int = utils.DEFAULT_POLL_INTERVAL_SECONDS,
    cleanup: bool = DEFAULT_CLEANUP,
    upload_trace: bool = False,
    output_report: Optional[str] = None,
    dry_run: bool = False,
    git_commit: Optional[str] = None,
    git_branch: str = utils.DEFAULT_GIT_BRANCH,
    jax_version: Optional[str] = None,
    libtpu_version: Optional[str] = None,
    reservation_name: Optional[str] = utils.DEFAULT_RESERVATION_NAME,
    accelerator: Optional[str] = None,
    git_repo: str = utils.DEFAULT_GIT_REPO,
    runtimes: Optional[Sequence[runtimes_lib.Runtime]] = None,
    bq_table: Optional[str] = None,
    max_parallel: Optional[int] = None,
    skip_cluster_credentials: bool = False,
    pod_priority_class: str = utils.DEFAULT_POD_PRIORITY_CLASS,
    retries: int = utils.DEFAULT_RETRIES,
    patch_oss_image: bool = False,
    patched_server_image: str = oss_image_patcher.DEFAULT_PATCHED_SERVER_IMAGE,
) -> list[workload_submitter.BenchmarkResult]:
  """Executes the benchmark automation pipeline (Workload Submission, Metrics Aggregation, Report Generation)."""
  # Resolve the selection first so flag misuse fails before any GCS or
  # cluster access. Logs the scanned config directory or suite file.
  config_files = workload_submitter.resolve_config_files(
      config_dir=config_dir,
      configs=configs,
      categories=categories,
      topologies=topologies,
      suite_file=suite_file,
      accelerator=accelerator,
  )
  if not config_files:
    raise ValueError(
        "No YAML configuration files found matching selection criteria."
    )
  acc = workload_submitter.infer_configs_accelerator(config_files, accelerator)
  if runtimes is None:
    runtimes = workload_submitter.resolve_runtimes(None, suite_file)

  if patch_oss_image:
    runtimes = oss_image_patcher.maybe_patch_runtimes_on_the_fly(
        runtimes,
        enabled=True,
        dry_run=dry_run,
        patched_server_image=patched_server_image,
    )

  if not dry_run and gcs_bucket:
    metrics_aggregator.verify_gcs_bucket_accessible(gcs_bucket)

  now_utc = datetime.datetime.now(datetime.timezone.utc)
  date_str = now_utc.strftime("%Y-%m-%d")
  if not output_report:
    output_report = _default_output_report(now_utc)

  logging.info("=" * 80)
  logging.info("TPUMS External GKE Automated Benchmark Pipeline")
  logging.info(
      "Target Cluster   : %s (%s / %s / %s)", cluster, project, region, zone
  )
  logging.info("Accelerator      : %s", acc)
  logging.info(
      "Namespace / Queue: %s / %s (Kueue priority: %r, pod priority: %r)",
      namespace,
      queue,
      priority,
      pod_priority_class,
  )
  if reservation_name:
    logging.info("Reservation Name : %s", reservation_name)
  logging.info("Docker Image     : %s", docker_image)
  logging.info("Runtimes         : %s", ", ".join(r.name for r in runtimes))
  logging.info("Git Repo / Branch: %s @ %s", git_repo, git_branch)
  logging.info("GCS Bucket       : %s", gcs_bucket)
  if bq_table:
    logging.info("BigQuery Table   : %s", bq_table)
  logging.info("Report Output    : %s", output_report)
  logging.info("=" * 80)
  logging.info(
      "Resolved %d benchmark config(s) for execution...", len(config_files)
  )

  # Fetch cluster credentials first, unless kubectl is already configured
  # (e.g. the in-cluster service account of the nightly CronJob).
  if skip_cluster_credentials:
    logging.info("Using the existing kubectl configuration for %s.", cluster)
  else:
    workload_submitter.fetch_cluster_credentials(
        cluster=cluster,
        project=project,
        region=region,
        zone=zone,
        dry_run=dry_run,
    )

  # ----------------------------------------------------------------------------
  # Workload Submitter: Submit workloads, poll status, cleanup
  # ----------------------------------------------------------------------------
  results = workload_submitter.submit_workloads(
      config_files=config_files,
      namespace=namespace,
      queue=queue,
      priority=priority,
      docker_image=docker_image,
      gcs_bucket=gcs_bucket,
      timeout_minutes=timeout_minutes,
      declared_duration_minutes=declared_duration_minutes,
      poll_interval_seconds=poll_interval_seconds,
      cleanup=cleanup,
      upload_trace=upload_trace,
      git_branch=git_branch,
      dry_run=dry_run,
      reservation_name=reservation_name,
      accelerator=acc,
      git_repo=git_repo,
      runtimes=runtimes,
      max_parallel=max_parallel,
      pod_priority_class=pod_priority_class,
      retries=retries,
  )

  # Ingest summary.csv metrics; a JobSet that exited 0 without producing any
  # rows did not benchmark anything, so it is reported as failed.
  if not dry_run:
    for r in results:
      if r.status != "SUCCESS":
        continue
      r.metrics = metrics_aggregator.fetch_gcs_raw_metrics(r.gcs_artifact_path)
      if not r.metrics:
        r.status = "FAILED"
        r.error_message = (
            f"JobSet succeeded but {r.gcs_artifact_path}/summary.csv has no"
            " rows."
        )
        logging.error("%s: %s", r.workload_name, r.error_message)

  total_jobsets = len(results)
  if dry_run:
    logging.info("Dry-run preview completed for %d JobSet(s).", total_jobsets)
  else:
    successful_jobsets, failed_jobsets = report_generator.count_statuses(
        results
    )
    logging.info(
        "JobSet execution finished: %d/%d succeeded (%d failed).",
        successful_jobsets,
        total_jobsets,
        failed_jobsets,
    )

  # ----------------------------------------------------------------------------
  # Metrics Aggregator: Ingest GCS metrics and aggregate by category
  # ----------------------------------------------------------------------------
  # Aggregate the whole day found on GCS (this run's results included), so a
  # partial or repeated run merges into the day's report instead of
  # replacing it.
  aggregated_results = (
      day_results(gcs_bucket, date_str, results)
      if gcs_bucket and not dry_run
      else results
  )
  raw_result_dicts = [dataclasses.asdict(r) for r in aggregated_results]
  categorized_metrics = (
      metrics_aggregator.aggregate_and_upload_category_results(
          raw_results=raw_result_dicts,
          gcs_bucket=gcs_bucket,
          date_str=date_str,
          dry_run=dry_run,
      )
  )

  # ----------------------------------------------------------------------------
  # Report Generator: Publish Markdown and HTML summary reports
  # ----------------------------------------------------------------------------
  report_generator.generate_and_publish_report(
      results=results,
      categorized_metrics=categorized_metrics,
      cluster=cluster,
      project=project,
      region=region,
      namespace=namespace,
      output_file=output_report,
      gcs_bucket=gcs_bucket,
      date_str=date_str,
      dry_run=dry_run,
      git_commit=git_commit,
      jax_version=jax_version,
      libtpu_version=libtpu_version,
  )

  if not dry_run and gcs_bucket:
    publish_report_copies(gcs_bucket, date_str, output_report)
    publish_dashboard(
        gcs_bucket,
        date_str,
        categorized_metrics,
        aggregated_results,
        cluster=cluster,
        run_id=now_utc.strftime("%Y-%m-%d %H:%M UTC"),
    )

  if bq_table and not dry_run:
    write_bigquery(
        bq_table,
        aggregated_results,
        categorized_metrics,
        date_str=date_str,
        started_at=now_utc,
        accelerator=acc or "",
        docker_image=docker_image,
        region=region,
        cluster=cluster,
    )

  logging.info("=" * 80)
  logging.info(
      "Daily Pipeline Execution Finished. Report saved to %s", output_report
  )
  logging.info("=" * 80)

  return results


def main(argv: Sequence[str]) -> None:
  default_report = _default_output_report()

  parser = argparse.ArgumentParser(
      description="TPUMS Automated Benchmark Pipeline Orchestrator."
  )
  for flag, default, help_text in (
      ("--cluster", utils.DEFAULT_CLUSTER, "GKE cluster name"),
      ("--project", utils.DEFAULT_PROJECT, "GCP project ID"),
      ("--region", utils.DEFAULT_REGION, "GCP region"),
      ("--zone", utils.DEFAULT_ZONE, "GCP zone"),
  ):
    parser.add_argument(
        flag, default=default, help=f"{help_text} (default: {default!r})."
    )
  utils.add_workload_args(parser, default_cleanup=DEFAULT_CLEANUP)
  utils.add_selection_args(parser)
  parser.add_argument(
      "--output-report",
      default=default_report,
      help=f"Local report filepath (default: {default_report}).",
  )
  parser.add_argument("--git-commit", default=None, help="GitHub commit hash.")
  parser.add_argument(
      "--jax-version", default=None, help="JAX runtime version."
  )
  parser.add_argument("--libtpu-version", default=None, help="libtpu version.")
  parser.add_argument(
      "--skip-cluster-credentials",
      action="store_true",
      default=bool(os.environ.get("SKIP_CLUSTER_CREDENTIALS")),
      help=(
          "Do not run `gcloud container clusters get-credentials`; use the"
          " kubectl configuration already in place (in-cluster CronJob)."
      ),
  )
  parser.add_argument(
      "--bq-table",
      default=os.environ.get("BQ_TABLE"),
      help=(
          "BigQuery table (project.dataset.table) that receives one row per"
          " benchmark case in uBench's microbenchmark schema (default: none)."
      ),
  )
  parser.add_argument(
      "--patch-oss-image",
      action=argparse.BooleanOptionalAction,
      default=bool(os.environ.get("TPUMS_PATCH_OSS_IMAGE")),
      help=(
          "Dynamically patch the live OSS Pathways image"
          " (us-docker.pkg.dev/cloud-tpu-v2-images/cloud-pathways/server:latest)"
          " on the fly at the start of the run by injecting"
          " libpjrt_profiler_patch.so for PJRT TPU profiler registration."
      ),
  )
  parser.add_argument(
      "--patched-server-image",
      default=os.environ.get(
          "TPUMS_PATCHED_SERVER_IMAGE",
          oss_image_patcher.DEFAULT_PATCHED_SERVER_IMAGE,
      ),
      help=(
          "Destination Artifact Registry repository (without tag) for the"
          " on-the-fly patched OSS Pathways server image."
      ),
  )

  args = parser.parse_args(argv[1:])
  utils.validate_workload_args(args, require_cluster=True)
  logging.basicConfig(
      level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
  )

  logging.info(
      "Starting TPUMS Pipeline Orchestrator (Dry-Run: %s)...", args.dry_run
  )
  run_pipeline(
      config_dir=args.config_dir,
      configs=args.configs,
      categories=args.categories,
      topologies=args.topologies,
      suite_file=args.suite_file,
      cluster=args.cluster,
      project=args.project,
      region=args.region,
      zone=args.zone,
      namespace=args.namespace,
      queue=args.queue,
      priority=args.priority,
      docker_image=args.docker_image,
      gcs_bucket=args.gcs_bucket,
      timeout_minutes=args.timeout_minutes,
      declared_duration_minutes=args.declared_duration_minutes,
      poll_interval_seconds=args.poll_interval,
      cleanup=args.cleanup,
      upload_trace=args.upload_trace,
      output_report=args.output_report,
      dry_run=args.dry_run,
      git_commit=args.git_commit,
      git_branch=args.git_branch,
      jax_version=args.jax_version,
      libtpu_version=args.libtpu_version,
      reservation_name=args.reservation_name,
      accelerator=args.accelerator,
      git_repo=args.git_repo,
      runtimes=workload_submitter.resolve_runtimes(
          args.runtimes, args.suite_file
      ),
      bq_table=args.bq_table,
      max_parallel=args.max_parallel,
      skip_cluster_credentials=args.skip_cluster_credentials,
      pod_priority_class=args.pod_priority_class,
      retries=args.retries,
      patch_oss_image=args.patch_oss_image,
      patched_server_image=args.patched_server_image,
  )


if __name__ == "__main__":
  main(sys.argv)
