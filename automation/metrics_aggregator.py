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

"""Ingest, Normalize, and Aggregate TPUMS Benchmark Metrics Across Categories.

Categorizes results strictly via the directory and file paths in the GCS bucket,
standardized across benchmark categories.
"""

import argparse
from collections.abc import Sequence
import csv
import datetime
import io
import itertools
import json
import logging
import subprocess
import sys
from typing import Any, Optional

from automation import runtimes as runtimes_lib
from automation import utils


def verify_gcs_bucket_accessible(gcs_bucket: str) -> None:
  """Fails fast if the GCS bucket is missing or not readable by this identity.

  Args:
    gcs_bucket: Bucket name or gs:// prefix to probe.

  Raises:
    RuntimeError: If the bucket cannot be listed.
  """
  bucket_uri = utils.format_gcs_uri(gcs_bucket)
  # Probe the bucket itself: a prefix that has no objects yet (first night
  # under a new prefix) makes `ls` fail with "matched no objects".
  bucket_only = "gs://" + bucket_uri.removeprefix("gs://").split("/", 1)[0]
  proc = subprocess.run(
      ["gcloud", "storage", "ls", bucket_only],
      capture_output=True,
      text=True,
      check=False,
      stdin=subprocess.DEVNULL,
      timeout=utils.CLI_TIMEOUT_SECONDS,
  )
  if proc.returncode != 0:
    raise RuntimeError(
        f"GCS bucket {bucket_uri} is not accessible (exit"
        f" {proc.returncode}): {proc.stderr.strip()}"
    )


# Runtime discriminator columns added to every row (see runtimes.py); the
# Pathways provenance ones are only filled for Pathways runtimes, the client
# image digest and node pools whenever the pods reported them.
RUNTIME_COLUMNS: tuple[str, ...] = (
    "runtime",
    "pathways_server_image",
    "pathways_proxy_image",
    "pathways_server_digest",
    "pathways_proxy_digest",
    "client_image_digest",
    "node_pools",
    "git_commit",
)
# Cross-workload discriminator and metric columns placed before the remaining
# raw summary.csv columns.
CATEGORY_LEADING_COLUMNS: dict[str, tuple[str, ...]] = {
    "collectives": (
        "runtime",
        "topology",
        "mesh_shape",
        "sharding_strategy",
        "benchmark",
        "dtype",
        "matrix_dim",
        "replica_group_rank",
        "shard_size_mib",
        "xprof_p50_ms",
        "xprof_bandwidth_per_chip_gb_s",
    ),
    "hbm": (
        "runtime",
        "benchmark",
        "dtype",
        "num_elements",
        "data_size_mib",
        "xprof_p50_ms",
        "xprof_bandwidth_per_device_gb_s",
    ),
    "host_device": (
        "runtime",
        "benchmark",
        "dtype",
        "data_size_mib",
        "transfer_backend",
        "xprof_p50_ms",
        "xprof_bandwidth_per_device_gb_s",
    ),
    "device_to_device": (
        "runtime",
        "benchmark",
        "dtype",
        "direction",
        "src_device_index",
        "dst_device_index",
        "data_size_mib",
        "xprof_p50_ms",
        "xprof_bandwidth_per_device_gb_s",
    ),
    "gemm": (
        "runtime",
        "benchmark",
        "in_dtype",
        "out_dtype",
        "m",
        "k",
        "n",
        "xprof_p50_ms",
        "xprof_tflops_per_device",
    ),
}


# Maps a report category to the path fragments that identify it. Fragments use
# underscore separators only; `infer_category_from_path` normalizes hyphens
# before matching.
_CATEGORY_PATTERNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "collectives",
        (
            "collectives",
            "all_gather",
            "all_reduce",
            "all_to_all",
            "reduce_scatter",
        ),
    ),
    ("hbm", ("hbm",)),
    (
        "host_device",
        (
            "host_to_device",
            "device_to_host",
            "host_device",
        ),
    ),
    ("device_to_device", ("device_to_device",)),
    ("gemm", ("gemm",)),
)

SUPPORTED_CATEGORIES: tuple[str, ...] = tuple(
    cat for cat, _ in _CATEGORY_PATTERNS
)


def ordered_columns(category: str, rows: Sequence[dict[str, Any]]) -> list[str]:
  """Returns the union of row keys: category leading columns first, then the rest.

  Args:
    category: Report category key in `CATEGORY_LEADING_COLUMNS`.
    rows: Metric rows; trailing columns keep their first-seen order.

  Returns:
    Column names.
  """
  discovered = list(dict.fromkeys(itertools.chain.from_iterable(rows)))
  # .get(): report categories come from GCS CSV names and may be unknown.
  leading = [
      c for c in CATEGORY_LEADING_COLUMNS.get(category, ()) if c in discovered
  ]
  return leading + [c for c in discovered if c not in leading]


def infer_category_from_path(gcs_or_file_path: str) -> Optional[str]:
  """Categorizes results purely through directory and file path in GCS."""
  path_lower = (
      str(gcs_or_file_path).lower().replace("\\", "/").replace("-", "_")
  )
  for cat, patterns in _CATEGORY_PATTERNS:
    if any(p in path_lower for p in patterns):
      return cat
  return None


def normalize_row(
    row: dict[str, Any],
    category: str,
    topology: Optional[str] = None,
    runtime: Optional[str] = None,
    provenance: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
  """Returns a copy of `row` ready for aggregation.

  Drops the internal `_source_path` key, fills in a missing collectives
  `topology` from `topology`, casts D2D device indices to int, and stamps the
  runtime discriminator and provenance columns (`RUNTIME_COLUMNS`).

  Args:
    row: Raw summary.csv row.
    category: Report category of the row.
    topology: Result-level topology, if known.
    runtime: Runtime the row was measured on (default McJAX).
    provenance: The workload's provenance (images, digests, commit).

  Returns:
    The normalized row.
  """
  norm = dict(row)
  norm.pop("_source_path", None)
  # Single-runtime pipelines (no runtime or provenance known) keep the plain
  # summary.csv columns; runtime matrices stamp every row so the rows of one
  # config on different runtimes stay distinguishable.
  if runtime or provenance:
    provenance = provenance or {}
    stamped = {
        "runtime": (
            runtime or provenance.get("runtime") or runtimes_lib.MCJAX.name
        )
    }
    for col in RUNTIME_COLUMNS[1:]:
      value = provenance.get(col)
      stamped[col] = "" if value is None else str(value)
    # Runtime columns lead; summary.csv columns keep their order after them.
    norm = {**stamped, **{k: v for k, v in norm.items() if k not in stamped}}

  if category == "collectives":
    if topology:
      norm.setdefault("topology", topology)

  elif category == "device_to_device":
    for dev_col in ("src_device_index", "dst_device_index"):
      if norm.get(dev_col) not in (None, ""):
        try:
          norm[dev_col] = int(float(norm[dev_col]))
        except (ValueError, TypeError):
          pass

  return norm


# The only per-workload metrics file written by `tpums benchmark run-config`.
_SUMMARY_CSV = "summary.csv"


def fetch_gcs_raw_metrics(gcs_output_dir: str) -> list[dict[str, Any]]:
  """Downloads and parses the summary.csv of a workload output directory."""
  m_file = utils.format_gcs_uri(gcs_output_dir, _SUMMARY_CSV)
  proc = subprocess.run(
      ["gcloud", "storage", "cat", m_file],
      capture_output=True,
      text=True,
      check=False,
      stdin=subprocess.DEVNULL,
      timeout=utils.CLI_TIMEOUT_SECONDS,
  )
  if proc.returncode != 0:
    logging.warning(
        "Could not read %s (gcloud exit %d): %s",
        m_file,
        proc.returncode,
        (proc.stderr or "").strip() or "<no stderr>",
    )
    return []
  if not proc.stdout.strip():
    return []
  reader = csv.DictReader(io.StringIO(proc.stdout))
  return [dict(r, _source_path=m_file) for r in reader]


_PROVENANCE_JSON = "provenance.json"


def fetch_gcs_provenance(gcs_output_dir: str) -> dict[str, Any]:
  """Reads the provenance.json the submitter wrote next to a workload's results."""
  target = utils.format_gcs_uri(gcs_output_dir, _PROVENANCE_JSON)
  proc = subprocess.run(
      ["gcloud", "storage", "cat", target],
      capture_output=True,
      text=True,
      check=False,
      stdin=subprocess.DEVNULL,
      timeout=utils.CLI_TIMEOUT_SECONDS,
  )
  if proc.returncode != 0 or not proc.stdout.strip():
    return {}
  try:
    data = json.loads(proc.stdout)
  except json.JSONDecodeError:
    logging.warning("Ignoring malformed %s", target)
    return {}
  return data if isinstance(data, dict) else {}


def _list_summary_files(target_prefix: str) -> list[str]:
  """Lists summary.csv files one level (`<workload>/`) and two levels
  (`<runtime>/<config>/`) below `target_prefix`."""
  found: list[str] = []
  failures: list[str] = []
  for pattern in (f"*/{_SUMMARY_CSV}", f"*/*/{_SUMMARY_CSV}"):
    res = subprocess.run(
        ["gcloud", "storage", "ls", f"{target_prefix}/{pattern}"],
        capture_output=True,
        text=True,
        check=False,
        stdin=subprocess.DEVNULL,
        timeout=utils.CLI_TIMEOUT_SECONDS,
    )
    if res.returncode != 0:
      # Also covers "matched no objects"; stderr tells it apart from auth
      # errors.
      failures.append(
          f"Could not list {target_prefix}/{pattern} (gcloud exit"
          f" {res.returncode}): {(res.stderr or '').strip() or '<no stderr>'}"
      )
      continue
    found.extend(res.stdout.split())
  # One layout matching is normal; neither matching deserves the details.
  for failure in failures:
    if found:
      logging.info("%s", failure)
    else:
      logging.warning("%s", failure)
  return list(dict.fromkeys(found))


def discover_and_fetch_gcs_results(
    gcs_base: str, date_str: str
) -> list[dict[str, Any]]:
  """Discovers all workload summary.csv files on GCS.

  Supports both layouts: `<date>/<workload>/summary.csv` (one runtime) and
  `<date>/<runtime>/<config>/summary.csv` (runtime matrix). The topology comes
  from the path when it carries one, else from the rows' `topology` column.
  """
  target_prefix = utils.format_gcs_uri(gcs_base, date_str)
  metric_files = _list_summary_files(target_prefix)

  if not metric_files:
    logging.warning("No metric files found on GCS under %s", target_prefix)
    return []

  discovered = []
  for mf in metric_files:
    parts = mf.split("/")
    wl_name = parts[-2]
    runtime = parts[-3] if parts[-3] in runtimes_lib.RUNTIMES else None
    gcs_artifact_path = mf.rsplit("/", 1)[0]
    metrics = fetch_gcs_raw_metrics(gcs_artifact_path)
    if not metrics:
      continue
    try:
      topo = utils.extract_topology(wl_name, mf)
    except ValueError:
      topo = str(metrics[0].get("topology") or "")
      if not topo:
        logging.warning("Skipping metric file %s: no topology found.", mf)
        continue
    provenance = fetch_gcs_provenance(gcs_artifact_path)
    entry: dict[str, Any] = {
        "workload_name": wl_name,
        "config_name": wl_name,
        "gcs_artifact_path": gcs_artifact_path,
        "topology": topo,
        "status": "SUCCESS",
        "metrics": metrics,
    }
    runtime = runtime or provenance.get("runtime")
    if runtime or provenance:
      entry["runtime"] = runtime or runtimes_lib.MCJAX.name
      entry["provenance"] = provenance
    discovered.append(entry)

  logging.info(
      "Discovered %d workload metric file(s) on GCS containing %d total"
      " row(s).",
      len(discovered),
      sum(len(r["metrics"]) for r in discovered),
  )
  return discovered


def aggregate_and_upload_category_results(
    raw_results: list[dict[str, Any]],
    gcs_bucket: str,
    date_str: Optional[str] = None,
    dry_run: bool = False,
) -> dict[str, list[dict[str, Any]]]:
  """Categorizes results strictly by GCS/file path and uploads aggregated CSVs to GCS."""
  if not gcs_bucket and not dry_run:
    raise ValueError(
        "gcs_bucket is required when dry_run is False; callers must resolve it"
        " from --gcs-bucket or the GCS_BUCKET environment variable."
    )
  if not date_str:
    date_str = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d")

  if not raw_results and not dry_run:
    raw_results = discover_and_fetch_gcs_results(gcs_bucket, date_str=date_str)

  categorized: dict[str, list[dict[str, Any]]] = {
      cat: [] for cat in SUPPORTED_CATEGORIES
  }

  for r in raw_results:
    metrics = r.get("metrics", [])
    topology = r.get("topology")
    runtime = r.get("runtime")
    provenance = r.get("provenance") or {}

    for row in metrics:
      # Set on every row by fetch_gcs_raw_metrics().
      source_path = row["_source_path"]
      cat = infer_category_from_path(source_path)
      if cat is None:
        logging.warning(
            "Unknown or unmapped category for path '%s'; skipping row.",
            source_path,
        )
        continue
      norm_row = normalize_row(
          row,
          category=cat,
          topology=topology,
          runtime=runtime,
          provenance=provenance,
      )
      categorized[cat].append(
          {k: norm_row[k] for k in ordered_columns(cat, [norm_row])}
      )

  if dry_run:
    print(
        "\n[Dry-Run: Metrics Aggregator - Path-Based Aggregation Output Plan"
        " (CSV-Derived Schema)]"
    )
    for cat in SUPPORTED_CATEGORIES:
      sample_gcs = utils.format_gcs_uri(
          gcs_bucket,
          date_str,
          "aggregated_results",
          f"{cat}.csv",
      )
      rows = categorized[cat]
      cols = (
          list(rows[0].keys()) if rows else list(CATEGORY_LEADING_COLUMNS[cat])
      )
      cols_preview = f" (Columns: {', '.join(cols[:6])}...)" if cols else ""
      print(f"* Category '{cat:<15}' -> {sample_gcs}{cols_preview}")
    print()
    return categorized

  upload_failures: list[str] = []
  for cat, rows in categorized.items():
    if not rows:
      continue
    csv_buffer = io.StringIO()
    writer = csv.DictWriter(
        csv_buffer,
        fieldnames=ordered_columns(cat, rows),
        extrasaction="ignore",
    )
    writer.writeheader()
    writer.writerows(rows)

    gcs_uri = utils.format_gcs_uri(
        gcs_bucket, date_str, "aggregated_results", f"{cat}.csv"
    )
    # Stream from stdin: a fixed local temp file would race when the tpu7x
    # and v6e nightly pipelines run concurrently on the same host.
    try:
      proc = subprocess.run(
          ["gcloud", "storage", "cp", "-", gcs_uri],
          input=csv_buffer.getvalue(),
          capture_output=True,
          text=True,
          check=False,
          timeout=utils.CLI_TIMEOUT_SECONDS,
      )
    except subprocess.TimeoutExpired as e:
      # Keep uploading the remaining categories; raise once at the end.
      upload_failures.append(cat)
      logging.error(
          "Timed out uploading aggregated %s CSV to %s after %ss.",
          cat,
          gcs_uri,
          e.timeout,
      )
      continue
    if proc.returncode == 0:
      logging.info(
          "Saved aggregated %s CSV (%d rows) to %s", cat, len(rows), gcs_uri
      )
    else:
      upload_failures.append(cat)
      logging.error(
          "Failed to upload aggregated %s CSV (%d rows) to %s (gcloud exit"
          " %d): %s",
          cat,
          len(rows),
          gcs_uri,
          proc.returncode,
          proc.stderr.strip() or "<no stderr>",
      )

  if upload_failures:
    raise RuntimeError(
        "Failed to upload aggregated CSVs for categories: "
        + ", ".join(sorted(upload_failures))
    )

  return categorized


def main(argv: Sequence[str]) -> None:
  parser = argparse.ArgumentParser(
      description="Aggregate TPUMS benchmark metrics by GCS path."
  )
  parser.add_argument(
      "--gcs-bucket",
      default=utils.DEFAULT_GCS_BUCKET,
      help="GCS bucket/prefix for metrics.",
  )
  parser.add_argument(
      "--date",
      default=None,
      help="Target date YYYY-MM-DD (default: current UTC date).",
  )
  parser.add_argument(
      "--dry-run",
      action="store_true",
      help="Print aggregation target schemas without network writes.",
  )

  args = parser.parse_args(argv[1:])
  if not args.gcs_bucket and not args.dry_run:
    raise ValueError(
        "Must specify --gcs-bucket or set GCS_BUCKET environment variable."
    )
  logging.basicConfig(
      level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
  )
  if not args.dry_run:
    verify_gcs_bucket_accessible(args.gcs_bucket)
  aggregate_and_upload_category_results(
      raw_results=[],
      gcs_bucket=args.gcs_bucket,
      date_str=args.date,
      dry_run=args.dry_run,
  )


if __name__ == "__main__":
  main(sys.argv)
