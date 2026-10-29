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
"""Writes aggregated nightly rows to BigQuery in uBench's microbenchmark schema.

The table mirrors the column names of uBench's
`run_summary_microbenchmarks` (`MicrobenchmarkSchema`), so a month of nightly
history can later move into uBench's table with one `INSERT ... SELECT`.
Free-form data lives in the `*_other` JSON string columns, as in uBench:
`params_other` holds the benchmark parameters, platform versions, flags and
the runtime provenance (`runtime`, `pathways` images/digests); `metrics_other`
holds every timing and throughput column of summary.csv.
"""

from collections.abc import Mapping, Sequence
import datetime
import getpass
import json
import logging
import math
from typing import Any, Optional

from automation import runtimes as runtimes_lib

RUN_SOURCE = "tpums-nightly"
RUN_TYPE = "nightly"
WORKLOAD_TYPE = "microbenchmarks"

# BigQuery schema: (name, type). Names follow uBench's MicrobenchmarkSchema.
SCHEMA: tuple[tuple[str, str], ...] = (
    ("run_id", "STRING"),
    ("run_name", "STRING"),
    ("benchmark_type", "STRING"),
    ("benchmark_name", "STRING"),
    ("start_time", "TIMESTAMP"),
    ("end_time", "TIMESTAMP"),
    ("result_success", "BOOL"),
    ("result_error", "STRING"),
    ("metrics_other", "STRING"),
    ("params_other", "STRING"),
    ("run_type", "STRING"),
    ("run_source", "STRING"),
    ("run_group", "STRING"),
    ("run_mode", "STRING"),
    ("recipe_type", "STRING"),
    ("is_run_externally_visible", "BOOL"),
    ("params_num_runs", "INTEGER"),
    ("params_dtype", "STRING"),
    ("metrics_time_per_iter_ms_p50", "FLOAT"),
    ("metrics_tflops_achieved_p50", "FLOAT"),
    ("metrics_achieved_bandwidth_gbps_p50", "FLOAT"),
    ("update_person_ldap", "STRING"),
    ("update_timestamp", "TIMESTAMP"),
    ("hardware_id", "STRING"),
    ("hardware_num_chips", "INTEGER"),
    ("hardware_num_nodes", "INTEGER"),
    ("hardware_topology", "STRING"),
    ("configs_software_versions", "STRING"),
    ("configs_container_image", "STRING"),
    ("configs_xla_flags", "STRING"),
    ("logs_artifact_directory", "STRING"),
    ("workload_type", "STRING"),
    ("cloud_region", "STRING"),
    ("cluster_name", "STRING"),
)

# uBench's precedence: XProf device time, then wall clock.
_STEP_TIME_KEYS = ("xprof_p50_ms", "xprof_avg_ms", "wall_clock_p50_ms")
_TFLOPS_KEYS = (
    "xprof_tflops_per_chip",
    "xprof_tflops_per_device",
    "wall_clock_tflops_per_chip",
    "wall_clock_tflops_per_device",
)
_BANDWIDTH_KEYS = (
    "xprof_bandwidth_per_chip_gb_s",
    "xprof_bandwidth_per_device_gb_s",
    "wall_clock_bandwidth_per_chip_gb_s",
    "wall_clock_bandwidth_per_device_gb_s",
)
_METRIC_PREFIXES = ("xprof_", "wall_clock_")
_METRIC_KEYS = frozenset({
    "actual_runs",
    "total_duration_s",
    "data_transferred_bytes",
    "shard_size_mib",
    "data_size_mib",
    "total_flops",
})
# summary.csv columns that describe the platform rather than the benchmark.
_PLATFORM_KEYS = (
    "tpu_type",
    "topology",
    "total_devices",
    "local_devices",
    "process_count",
    "process_index",
    "python_version",
    "jax_version",
    "jaxlib_version",
    "libtpu_version",
    "xla_flags",
    "libtpu_init_args",
)
# uBench hardware ids per TPU family (hardware.get_hardware_id_for_db).
_HARDWARE_IDS = {"v5e": "v5litepod", "v6e": "v6e", "tpu7x": "tpu7x"}


def _number(value: Any) -> Optional[float]:
  if value in (None, ""):
    return None
  try:
    number = float(value)
  except (TypeError, ValueError):
    return None
  return number if math.isfinite(number) else None


def _first_number(row: Mapping[str, Any], keys: Sequence[str]) -> Optional[float]:
  for key in keys:
    number = _number(row.get(key))
    if number is not None:
      return number
  return None


def _int(value: Any) -> Optional[int]:
  number = _number(value)
  return int(number) if number is not None else None


def _is_metric(key: str) -> bool:
  return key.startswith(_METRIC_PREFIXES) or key in _METRIC_KEYS


def benchmark_type_of(category: str) -> str:
  """Maps an aggregator category to uBench's `collective` / `compute`."""
  return "collective" if category == "collectives" else "compute"


def row_to_record(
    row: Mapping[str, Any],
    category: str,
    *,
    run_name: str,
    run_id: str,
    started_at: datetime.datetime,
    accelerator: str,
    container_image: str,
    artifacts_dir: str,
    region: str,
    cluster: str,
    user: str,
) -> dict[str, Any]:
  """Converts one aggregated summary.csv row into a BigQuery record.

  Args:
    row: Aggregated row (with the runtime columns stamped).
    category: Aggregator category of the row.
    run_name: The night's run name, shared by all its rows.
    run_id: Unique id of this row.
    started_at: When the night started (summary.csv has no timestamps).
    accelerator: TPU family, e.g. `v5e`.
    container_image: Client image.
    artifacts_dir: `gs://` directory of the workload's results.
    region: GCP region of the cluster.
    cluster: GKE cluster name.
    user: Reporter identity.
  """
  runtime = str(row.get("runtime") or runtimes_lib.MCJAX.name)
  metrics = {k: v for k, v in row.items() if _is_metric(k)}
  params = {
      k: v
      for k, v in row.items()
      if not _is_metric(k)
      and k not in _PLATFORM_KEYS
      and k not in ("runtime", "benchmark", "git_commit")
      and not k.startswith("pathways_")
  }
  params["test_name"] = row.get("benchmark", "")
  params["runtime"] = runtime
  params["platform"] = {k: row.get(k, "") for k in _PLATFORM_KEYS}
  if runtime != runtimes_lib.MCJAX.name:
    params["pathways"] = {
        "server_image": row.get("pathways_server_image", ""),
        "server_digest": row.get("pathways_server_digest", ""),
        "proxy_server_image": row.get("pathways_proxy_image", ""),
        "proxy_digest": row.get("pathways_proxy_digest", ""),
    }
  total_devices = _int(row.get("total_devices"))
  chips_per_node = 4 if accelerator in ("v5e", "v6e", "tpu7x") else None
  num_nodes = (
      max(1, total_devices // chips_per_node)
      if total_devices and chips_per_node
      else None
  )
  versions = [
      {"name": "jax", "version": str(row.get("jax_version", ""))},
      {"name": "jaxlib", "version": str(row.get("jaxlib_version", ""))},
      {"name": "libtpu", "version": str(row.get("libtpu_version", ""))},
      {"name": "accelerator_microbenchmarks", "version": str(row.get("git_commit", ""))},
  ]
  now = datetime.datetime.now(datetime.timezone.utc)
  return {
      "run_id": run_id,
      "run_name": run_name,
      "benchmark_type": benchmark_type_of(category),
      "benchmark_name": row.get("benchmark", ""),
      "start_time": started_at.isoformat(),
      "end_time": None,
      "result_success": True,
      "result_error": None,
      "metrics_other": json.dumps(metrics, sort_keys=True),
      "params_other": json.dumps(params, sort_keys=True, default=str),
      "run_type": RUN_TYPE,
      "run_source": RUN_SOURCE,
      "run_group": run_name,
      "run_mode": "cloud",
      "recipe_type": "perf",
      "is_run_externally_visible": False,
      "params_num_runs": _int(row.get("num_runs")),
      "params_dtype": row.get("dtype") or row.get("in_dtype") or None,
      "metrics_time_per_iter_ms_p50": _first_number(row, _STEP_TIME_KEYS),
      "metrics_tflops_achieved_p50": _first_number(row, _TFLOPS_KEYS),
      "metrics_achieved_bandwidth_gbps_p50": _first_number(
          row, _BANDWIDTH_KEYS
      ),
      "update_person_ldap": user,
      "update_timestamp": now.isoformat(),
      "hardware_id": _HARDWARE_IDS.get(accelerator, accelerator),
      "hardware_num_chips": total_devices,
      "hardware_num_nodes": num_nodes,
      "hardware_topology": row.get("topology") or None,
      "configs_software_versions": json.dumps(versions),
      "configs_container_image": container_image,
      "configs_xla_flags": row.get("libtpu_init_args") or None,
      "logs_artifact_directory": artifacts_dir,
      "workload_type": WORKLOAD_TYPE,
      "cloud_region": region,
      "cluster_name": cluster,
  }


def build_records(
    categorized_metrics: Mapping[str, Sequence[Mapping[str, Any]]],
    artifacts_by_key: Mapping[tuple[str, str], str],
    *,
    date_str: str,
    started_at: datetime.datetime,
    accelerator: str,
    container_image: str,
    region: str,
    cluster: str,
    user: Optional[str] = None,
) -> list[dict[str, Any]]:
  """Builds the BigQuery records of one night.

  Args:
    categorized_metrics: Aggregated rows per category.
    artifacts_by_key: `(runtime, benchmark)` -> `gs://` results directory,
      where `benchmark` is the class name reported in summary.csv.
    date_str: The night's date (`YYYY-MM-DD`), part of `run_name`.
    started_at: When the night started.
    accelerator: TPU family.
    container_image: Client image.
    region: GCP region.
    cluster: GKE cluster.
    user: Reporter identity (default: the OS user).
  """
  user = user or getpass.getuser()
  run_name = f"{RUN_SOURCE}-{date_str}"
  records = []
  for category, rows in categorized_metrics.items():
    for index, row in enumerate(rows):
      runtime = str(row.get("runtime") or runtimes_lib.MCJAX.name)
      artifacts = artifacts_by_key.get((runtime, str(row.get("benchmark"))), "")
      if not artifacts:
        # Fall back to any artifact dir of that runtime (same night).
        artifacts = next(
            (v for (rt, _), v in artifacts_by_key.items() if rt == runtime), ""
        )
      records.append(
          row_to_record(
              row,
              category,
              run_name=run_name,
              run_id=f"{run_name}_{runtime}_{category}_{index}",
              started_at=started_at,
              accelerator=accelerator,
              container_image=container_image,
              artifacts_dir=artifacts,
              region=region,
              cluster=cluster,
              user=user,
          )
      )
  return records


def write_records(table: str, records: Sequence[Mapping[str, Any]]) -> int:
  """Streams `records` into `table` (`project.dataset.table`), creating it.

  Args:
    table: Fully-qualified BigQuery table id.
    records: Records from `build_records`.

  Returns:
    The number of rows written.

  Raises:
    RuntimeError: If BigQuery rejects any row.
  """
  if not records:
    logging.warning("No BigQuery records to write to %s.", table)
    return 0
  # Imported lazily: the BigQuery client is only needed with --bq-table.
  from google.cloud import bigquery  # pylint: disable=g-import-not-at-top

  project, dataset_id, table_id = table.split(".")
  client = bigquery.Client(project=project)
  dataset = bigquery.Dataset(f"{project}.{dataset_id}")
  dataset.location = "US"
  client.create_dataset(dataset, exists_ok=True)
  schema = [bigquery.SchemaField(name, kind) for name, kind in SCHEMA]
  bq_table = bigquery.Table(table, schema=schema)
  bq_table.time_partitioning = bigquery.TimePartitioning(field="start_time")
  client.create_table(bq_table, exists_ok=True)
  errors = client.insert_rows_json(table, list(records))
  if errors:
    raise RuntimeError(f"BigQuery rejected {len(errors)} row(s): {errors[:3]}")
  logging.info("Wrote %d row(s) to BigQuery table %s.", len(records), table)
  return len(records)
