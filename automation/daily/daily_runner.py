"""Daily Pathways benchmarking metrics pipeline orchestrator (`automation/daily/daily_runner.py`).

Executes the 5 canonical device-time benchmarks + `pathways_overheads.yaml` on
`mcjax` and `louhi` (never `rapid`), captures pre-cleanup Kubernetes & container
log telemetry, parses `.xplane.pb` traces, appends both case-level (`perf`) and
86-metric (`manifest_metric`) rows to BigQuery `pathways_metrics_daily`, and
publishes the manifest-complete HTML/MD/JSON/CSV report to GCS.
"""

import argparse
from collections.abc import Sequence
import dataclasses
import datetime
import logging
import os
import pathlib
import sys
from typing import Any, Optional

from automation import bq_writer
from automation import metrics_aggregator
from automation import nightly_runner
from automation import report_generator
from automation import runtimes as runtimes_lib
from automation import utils
from automation import workload_submitter
from automation.daily import collectors
from automation.daily import report as daily_report_lib
import yaml

DEFAULT_MANIFEST_FILE = "automation/daily/metrics_manifest.yaml"
DEFAULT_DAILY_SUITE = "automation/daily_suite_v5e.yaml"
DEFAULT_DAILY_PREFIX = "tpmd"
DEFAULT_DAILY_DRIVER_IMAGE = (
    "us-docker.pkg.dev/cloud-tpu-multipod-dev/pathways-dev/anowusu/"
    "tpums-pathways-metrics-daily:latest"
)


def _resolve_manifest_path(manifest_file: str) -> pathlib.Path:
  p = pathlib.Path(manifest_file)
  if p.is_Absolute() if hasattr(p, "is_Absolute") else p.is_absolute():
    if p.exists():
      return p
  for base in (utils.PACKAGE_ROOT, pathlib.Path.cwd()):
    cand = base / p
    if cand.exists():
      return cand
  raise FileNotFoundError(f"Could not find manifest file: {manifest_file}")


def load_manifest(manifest_file: str = DEFAULT_MANIFEST_FILE) -> list[dict[str, Any]]:
  path = _resolve_manifest_path(manifest_file)
  with open(path, "r", encoding="utf-8") as f:
    data = yaml.safe_load(f)
  if not isinstance(data, list) or not data:
    raise ValueError(f"Invalid or empty metrics manifest at {path}")
  return data


def run_daily_pipeline(
    config_dir: Optional[str] = None,
    configs: Optional[list[str]] = None,
    categories: Optional[list[str]] = None,
    topologies: Optional[list[str]] = None,
    suite_file: Optional[str] = DEFAULT_DAILY_SUITE,
    manifest_file: str = DEFAULT_MANIFEST_FILE,
    cluster: str = utils.DEFAULT_CLUSTER,
    project: str = utils.DEFAULT_PROJECT,
    region: str = utils.DEFAULT_REGION,
    zone: str = utils.DEFAULT_ZONE,
    namespace: str = utils.DEFAULT_NAMESPACE,
    queue: str = utils.DEFAULT_QUEUE,
    priority: str = utils.DEFAULT_PRIORITY,
    docker_image: str = utils.DEFAULT_DOCKER_IMAGE,
    gcs_bucket: str = "gs://cloud-pathways-staging/tpums-pathways-metrics-daily",
    timeout_minutes: Optional[int] = 25,
    declared_duration_minutes: int = utils.DEFAULT_DECLARED_DURATION_MINUTES,
    poll_interval_seconds: int = utils.DEFAULT_POLL_INTERVAL_SECONDS,
    cleanup: bool = True,
    upload_trace: bool = True,
    output_report: Optional[str] = None,
    dry_run: bool = False,
    git_commit: Optional[str] = None,
    git_branch: str = "pathways-metrics-daily",
    reservation_name: Optional[str] = utils.DEFAULT_RESERVATION_NAME,
    accelerator: Optional[str] = None,
    git_repo: str = utils.DEFAULT_GIT_REPO,
    runtimes: Optional[Sequence[runtimes_lib.Runtime]] = None,
    bq_table: Optional[str] = "cloud-tpu-multipod-dev.tpums_nightly.pathways_metrics_daily",
    max_parallel: Optional[int] = 1,
    skip_cluster_credentials: bool = False,
    pod_priority_class: str = "medium",
    retries: int = 1,
    workload_prefix: str = DEFAULT_DAILY_PREFIX,
    run_source: str = bq_writer.DAILY_RUN_SOURCE,
    run_type: str = bq_writer.DAILY_RUN_TYPE,
    bq_replace: bool = False,
) -> list[workload_submitter.BenchmarkResult]:
  """Executes the full R5 daily Pathways benchmark metrics pipeline."""
  manifest = load_manifest(manifest_file)
  config_files = workload_submitter.resolve_config_files(
      config_dir=config_dir,
      configs=configs,
      categories=categories,
      topologies=topologies,
      suite_file=suite_file,
      accelerator=accelerator,
  )
  if not config_files:
    raise ValueError("No YAML configuration files found matching selection criteria.")
  acc = workload_submitter.infer_configs_accelerator(config_files, accelerator)
  if runtimes is None:
    runtimes = workload_submitter.resolve_runtimes("mcjax,louhi", suite_file)
  if any(r.name == "rapid" for r in runtimes):
    raise ValueError("R5 daily pipeline forbids the 'rapid' runtime.")

  if not dry_run and gcs_bucket:
    metrics_aggregator.verify_gcs_bucket_accessible(gcs_bucket)

  now_utc = datetime.datetime.now(datetime.timezone.utc)
  date_str = now_utc.strftime("%Y-%m-%d")
  run_stamp = now_utc.strftime("%Y%m%dT%H%M%SZ")
  run_group = f"{run_source}-{date_str}"
  run_name = f"{run_source}-{run_stamp}"

  logging.info("=" * 80)
  logging.info("TPUMS Daily Pathways Benchmark Metrics Pipeline (R5)")
  logging.info("Target Cluster   : %s (%s / %s / %s)", cluster, project, region, zone)
  logging.info("Runtimes         : %s", ", ".join(r.name for r in runtimes))
  logging.info("Manifest Entries : %d", len(manifest))
  logging.info("Configs          : %d", len(config_files))
  logging.info("Git Repo / Branch: %s @ %s", git_repo, git_branch)
  logging.info("GCS Bucket       : %s", gcs_bucket)
  logging.info("BigQuery Table   : %s (replace=%s)", bq_table, bq_replace)
  logging.info("=" * 80)

  if not skip_cluster_credentials:
    workload_submitter.fetch_cluster_credentials(
        cluster=cluster,
        project=project,
        region=region,
        zone=zone,
        dry_run=dry_run,
    )

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
      workload_prefix=workload_prefix,
      on_workload_terminal=collectors.capture_workload_k8s_telemetry,
  )

  if dry_run:
    logging.info("Dry-run completed for %d JobSet(s).", len(results))
    return results

  for r in results:
    if r.status != "SUCCESS":
      continue
    r.metrics = metrics_aggregator.fetch_gcs_raw_metrics(r.gcs_artifact_path)
    if not r.metrics:
      r.status = "FAILED"
      r.error_message = (
          f"JobSet succeeded but {r.gcs_artifact_path}/summary.csv has no rows."
      )
      logging.error("%s: %s", r.workload_name, r.error_message)

  ok_cnt, fail_cnt = report_generator.count_statuses(results)
  logging.info(
      "JobSet execution finished: %d/%d succeeded (%d failed).",
      ok_cnt,
      len(results),
      fail_cnt,
  )

  # Preserve in-memory k8s_telemetry across day_results merge
  telem_by_key = {
      (r.runtime, r.config_name): (r.provenance or {}).get("k8s_telemetry")
      for r in results
      if (r.provenance or {}).get("k8s_telemetry")
  }
  aggregated_results = (
      nightly_runner.day_results(gcs_bucket, date_str, results)
      if gcs_bucket
      else list(results)
  )
  for r in aggregated_results:
    if (r.runtime, r.config_name) in telem_by_key:
      r.provenance.setdefault(
          "k8s_telemetry", telem_by_key[(r.runtime, r.config_name)]
      )

  raw_result_dicts = [dataclasses.asdict(r) for r in aggregated_results]
  categorized_metrics = (
      metrics_aggregator.aggregate_and_upload_category_results(
          raw_results=raw_result_dicts,
          gcs_bucket=gcs_bucket,
          date_str=date_str,
          dry_run=False,
      )
  )

  # Run post-workload collectors & reconcile against manifest
  measurements, runtime_prov_by_rt = collectors.collect_all_manifest_metrics(
      manifest,
      categorized_metrics,
      aggregated_results,
      gcs_bucket=gcs_bucket,
      date_str=date_str,
  )

  # Write both case-level (`perf`) and manifest (`manifest_metric`) rows to BigQuery
  if bq_table:
    artifacts_by_key: dict[tuple[str, str], str] = {}
    for r in aggregated_results:
      for row in r.metrics:
        artifacts_by_key[(r.runtime, str(row.get("benchmark")))] = (
            r.gcs_artifact_path
        )
    case_records = bq_writer.build_records(
        categorized_metrics,
        artifacts_by_key,
        date_str=date_str,
        started_at=now_utc,
        accelerator=acc or "v5e",
        container_image=docker_image,
        region=region,
        cluster=cluster,
        run_source=run_source,
        run_type=run_type,
        run_name=run_name,
        run_group=run_group,
    )
    metric_records = bq_writer.build_metric_records(
        manifest,
        measurements,
        date_str=date_str,
        started_at=now_utc,
        accelerator=acc or "v5e",
        topology="4x8",
        total_devices=32,
        container_image=docker_image,
        default_artifacts_dir=utils.format_gcs_uri(gcs_bucket, date_str),
        region=region,
        cluster=cluster,
        run_source=run_source,
        run_type=run_type,
        run_name=run_name,
        run_group=run_group,
        runtime_provenance_by_rt=runtime_prov_by_rt,
    )
    all_records = case_records + metric_records
    try:
      bq_writer.write_records(bq_table, all_records, replace=bq_replace)
    except Exception:  # pylint: disable=broad-exception-caught
      logging.exception(
          "Failed to write %d record(s) to BigQuery %s.",
          len(all_records),
          bq_table,
      )

  # Build run metadata and publish manifest-complete reports to GCS
  louhi_prov = runtime_prov_by_rt.get("louhi", {})
  mcjax_prov = runtime_prov_by_rt.get("mcjax", {})
  resolved_commit = (
      git_commit
      or louhi_prov.get("git_commit")
      or mcjax_prov.get("git_commit")
      or runtimes_lib.resolve_git_commit(git_repo, git_branch)
      or ""
  )
  run_meta = {
      "run_name": run_name,
      "run_group": run_group,
      "date_str": date_str,
      "collected_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
      "cronjob_name": "tpums-pathways-metrics-daily",
      "cluster": cluster,
      "git_branch": git_branch,
      "git_commit": resolved_commit,
      "driver_image": os.environ.get(
          "TPUMS_DRIVER_IMAGE", DEFAULT_DAILY_DRIVER_IMAGE
      ),
      "louhi_server_image": louhi_prov.get(
          "pathways_server_image",
          os.environ.get("TPUMS_LOUHI_SERVER_IMAGE", ""),
      ),
      "louhi_server_digest": louhi_prov.get("pathways_server_digest", ""),
      "louhi_proxy_image": louhi_prov.get(
          "pathways_proxy_image",
          os.environ.get("TPUMS_LOUHI_PROXY_IMAGE", ""),
      ),
      "louhi_proxy_digest": louhi_prov.get("pathways_proxy_digest", ""),
      "client_image": docker_image,
      "client_image_digest": (
          louhi_prov.get("client_image_digest")
          or mcjax_prov.get("client_image_digest")
          or ""
      ),
      "gcs_prefix": gcs_bucket,
      "bq_table": bq_table or "",
  }
  daily_report_lib.render_and_publish_daily_reports(
      manifest,
      measurements,
      gcs_bucket=gcs_bucket,
      date_str=date_str,
      run_meta=run_meta,
      bq_table=bq_table,
      output_dir=os.path.dirname(output_report) if output_report else "/tmp",
  )

  logging.info("=" * 80)
  logging.info("Daily Pathways Metrics Pipeline Finished Successfully.")
  logging.info("=" * 80)
  return results


def main(argv: Sequence[str]) -> None:
  parser = argparse.ArgumentParser(
      description="TPUMS Daily Pathways Benchmark Metrics Orchestrator (R5)."
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
  utils.add_workload_args(parser, default_cleanup=True)
  utils.add_selection_args(parser)
  parser.add_argument(
      "--manifest-file",
      default=DEFAULT_MANIFEST_FILE,
      help=f"Path to metrics_manifest.yaml (default: {DEFAULT_MANIFEST_FILE}).",
  )
  parser.add_argument(
      "--workload-prefix",
      default=os.environ.get("TPUMS_WORKLOAD_PREFIX", DEFAULT_DAILY_PREFIX),
      help="JobSet name prefix (default: tpmd).",
  )
  parser.add_argument(
      "--run-source",
      default=bq_writer.DAILY_RUN_SOURCE,
      help="BigQuery run_source value.",
  )
  parser.add_argument(
      "--run-type",
      default=bq_writer.DAILY_RUN_TYPE,
      help="BigQuery run_type value.",
  )
  parser.add_argument(
      "--bq-replace",
      action=argparse.BooleanOptionalAction,
      default=False,
      help="Replace same-run_id rows in BigQuery (default: False, append).",
  )
  parser.add_argument(
      "--output-report",
      default=None,
      help="Optional local report filepath.",
  )
  parser.add_argument("--git-commit", default=None, help="GitHub commit hash.")
  parser.add_argument(
      "--skip-cluster-credentials",
      action="store_true",
      default=bool(os.environ.get("SKIP_CLUSTER_CREDENTIALS")),
      help="Use in-cluster kubectl config without calling gcloud get-credentials.",
  )
  parser.add_argument(
      "--bq-table",
      default=os.environ.get(
          "BQ_TABLE",
          "cloud-tpu-multipod-dev.tpums_nightly.pathways_metrics_daily",
      ),
      help="Target BigQuery table (project.dataset.table).",
  )

  args = parser.parse_args(argv[1:])
  utils.validate_workload_args(args, require_cluster=True)
  logging.basicConfig(
      level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
  )

  suite_file = args.suite_file or (
      DEFAULT_DAILY_SUITE
      if not (args.config_dir or args.configs or args.categories)
      else None
  )
  run_daily_pipeline(
      config_dir=args.config_dir,
      configs=args.configs,
      categories=args.categories,
      topologies=args.topologies,
      suite_file=suite_file,
      manifest_file=args.manifest_file,
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
      reservation_name=args.reservation_name,
      accelerator=args.accelerator,
      git_repo=args.git_repo,
      runtimes=workload_submitter.resolve_runtimes(args.runtimes, suite_file),
      bq_table=args.bq_table,
      max_parallel=args.max_parallel,
      skip_cluster_credentials=args.skip_cluster_credentials,
      pod_priority_class=args.pod_priority_class,
      retries=args.retries,
      workload_prefix=args.workload_prefix,
      run_source=args.run_source,
      run_type=args.run_type,
      bq_replace=args.bq_replace,
  )


if __name__ == "__main__":
  main(sys.argv)
