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
"""Side-by-side comparison of the runtimes a benchmark case ran on.

Rows of one category are grouped by benchmark case (the category's parameter
columns); within a case, one row per runtime is emitted with its timing,
throughput and XProf availability, plus the ratio of its step time to McJAX's
on the same case. Ratios use XProf device time when both runtimes have it and
wall-clock time otherwise, and say which.
"""

from collections.abc import Mapping, Sequence
import dataclasses
import math
from typing import Any, Optional

from automation import runtimes as runtimes_lib

BASELINE_RUNTIME = runtimes_lib.MCJAX.name

# Parameter columns that identify a benchmark case, per category. Everything
# not listed here (metrics, versions, flags) varies per runtime.
CASE_COLUMNS: dict[str, tuple[str, ...]] = {
    "collectives": (
        "benchmark",
        "topology",
        "mesh_shape",
        "sharding_strategy",
        "dtype",
        "matrix_dim",
    ),
    "hbm": ("benchmark", "dtype", "num_elements", "data_size_mib"),
    "host_device": ("benchmark", "dtype", "data_size_mib", "transfer_backend"),
    "device_to_device": (
        "benchmark",
        "dtype",
        "direction",
        "src_device_index",
        "dst_device_index",
        "data_size_mib",
    ),
    "gemm": ("benchmark", "in_dtype", "out_dtype", "m", "k", "n"),
}

# Throughput columns in order of preference (XProf before wall clock).
_THROUGHPUT_COLUMNS: tuple[tuple[str, str], ...] = (
    ("xprof_bandwidth_per_chip_gb_s", "GB/s/chip"),
    ("xprof_bandwidth_per_device_gb_s", "GB/s/device"),
    ("xprof_tflops_per_device", "TFLOP/s/device"),
    ("xprof_tflops_per_chip", "TFLOP/s/chip"),
    ("wall_clock_bandwidth_per_chip_gb_s", "GB/s/chip (wall)"),
    ("wall_clock_bandwidth_per_device_gb_s", "GB/s/device (wall)"),
    ("wall_clock_tflops_per_device", "TFLOP/s/device (wall)"),
    ("wall_clock_tflops_per_chip", "TFLOP/s/chip (wall)"),
)

COMPARISON_HEADERS: tuple[str, ...] = (
    "case",
    "runtime",
    "xprof_p50_ms",
    "xprof_p90_ms",
    "wall_clock_p50_ms",
    "wall_clock_p90_ms",
    "wall_clock_avg_ms",
    "wall_clock_std_ms",
    "throughput",
    "xprof",
    "vs_mcjax",
)


@dataclasses.dataclass
class ComparisonTable:
  """One category's comparison rows."""

  category: str
  headers: tuple[str, ...]
  rows: list[dict[str, Any]]
  runtimes: list[str]


def _to_float(value: Any) -> Optional[float]:
  """Parses a CSV cell to a finite float, else None."""
  if value is None or value == "":
    return None
  try:
    number = float(value)
  except (TypeError, ValueError):
    return None
  return number if math.isfinite(number) else None


def _fmt(value: Optional[float], digits: int = 3) -> str:
  return "" if value is None else f"{value:.{digits}f}"


def case_key(row: Mapping[str, Any], category: str) -> tuple[str, ...]:
  """Returns the values of the category's case columns present in `row`."""
  return tuple(
      str(row.get(col, "")) for col in CASE_COLUMNS.get(category, ())
  )


def case_label(row: Mapping[str, Any], category: str) -> str:
  """Formats the case columns as `benchmark dtype=... matrix_dim=...`."""
  cols = CASE_COLUMNS.get(category, ())
  parts = []
  for col in cols:
    value = row.get(col, "")
    if value in ("", None):
      continue
    parts.append(str(value) if col == "benchmark" else f"{col}={value}")
  return " ".join(parts) or "(all)"


def _throughput(row: Mapping[str, Any]) -> str:
  for col, unit in _THROUGHPUT_COLUMNS:
    value = _to_float(row.get(col))
    if value is not None:
      return f"{value:.2f} {unit}"
  return ""


def _xprof_status(row: Mapping[str, Any]) -> str:
  captured = _to_float(row.get("xprof_captured_runs"))
  expected = _to_float(row.get("xprof_expected_runs")) or _to_float(
      row.get("actual_runs")
  )
  if _to_float(row.get("xprof_p50_ms")) is not None:
    if captured is not None and expected:
      return f"device {int(captured)}/{int(expected)}"
    return "device"
  return "wall-clock only"


def _ratio(
    row: Mapping[str, Any], baseline: Optional[Mapping[str, Any]]
) -> str:
  """Step-time ratio to the baseline: XProf if both have it, else wall clock."""
  if baseline is None:
    return "n/a"
  if baseline is row:
    return "baseline"
  for col, label in (("xprof_p50_ms", "xprof"), ("wall_clock_p50_ms", "wall")):
    own = _to_float(row.get(col))
    base = _to_float(baseline.get(col))
    if own is not None and base:
      return f"{own / base:.2f}x ({label})"
  return "n/a"


def build_comparison(
    categorized_metrics: Mapping[str, Sequence[Mapping[str, Any]]],
) -> list[ComparisonTable]:
  """Builds the per-category runtime comparison tables.

  Args:
    categorized_metrics: Aggregated rows per category, each row carrying a
      `runtime` column (rows without one count as McJAX).

  Returns:
    One table per category that has rows, in the input order. Categories
    whose rows all share one runtime are included too (their ratio column is
    `baseline`/`n/a`), so the section doubles as a compact per-case summary.
  """
  tables: list[ComparisonTable] = []
  for category, rows in categorized_metrics.items():
    if not rows:
      continue
    by_case: dict[tuple[str, ...], dict[str, Mapping[str, Any]]] = {}
    for row in rows:
      runtime = str(row.get("runtime") or BASELINE_RUNTIME)
      by_case.setdefault(case_key(row, category), {})[runtime] = row
    runtimes_seen = sorted(
        {rt for per_runtime in by_case.values() for rt in per_runtime},
        key=runtimes_lib.runtime_sort_key,
    )
    out_rows: list[dict[str, Any]] = []
    for per_runtime in by_case.values():
      baseline = per_runtime.get(BASELINE_RUNTIME)
      first = next(iter(per_runtime.values()))
      label = case_label(first, category)
      for runtime in sorted(per_runtime, key=runtimes_lib.runtime_sort_key):
        row = per_runtime[runtime]
        out_rows.append({
            "case": label,
            "runtime": runtime,
            "xprof_p50_ms": _fmt(_to_float(row.get("xprof_p50_ms"))),
            "xprof_p90_ms": _fmt(_to_float(row.get("xprof_p90_ms"))),
            "wall_clock_p50_ms": _fmt(_to_float(row.get("wall_clock_p50_ms"))),
            "wall_clock_p90_ms": _fmt(_to_float(row.get("wall_clock_p90_ms"))),
            "wall_clock_avg_ms": _fmt(_to_float(row.get("wall_clock_avg_ms"))),
            "wall_clock_std_ms": _fmt(_to_float(row.get("wall_clock_std_ms"))),
            "throughput": _throughput(row),
            "xprof": _xprof_status(row),
            "vs_mcjax": _ratio(row, baseline),
        })
    tables.append(
        ComparisonTable(
            category=category,
            headers=COMPARISON_HEADERS,
            rows=out_rows,
            runtimes=runtimes_seen,
        )
    )
  return tables


def runtime_provenance_summary(
    categorized_metrics: Mapping[str, Sequence[Mapping[str, Any]]],
) -> list[dict[str, str]]:
  """Returns one row per runtime with its images/digests/commit, from the rows."""
  seen: dict[str, dict[str, str]] = {}
  for rows in categorized_metrics.values():
    for row in rows:
      runtime = str(row.get("runtime") or BASELINE_RUNTIME)
      if runtime in seen:
        continue
      entry = {"runtime": runtime}
      for col in (
          "git_commit",
          "jax_version",
          "libtpu_version",
          "pathways_server_image",
          "pathways_server_digest",
          "pathways_proxy_image",
          "pathways_proxy_digest",
      ):
        value = row.get(col)
        if value not in (None, ""):
          text = str(value)
          if col.endswith("digest") or col == "git_commit":
            text = text.replace("sha256:", "")[:12]
          entry[col] = text
      seen[runtime] = entry
  return [seen[rt] for rt in sorted(seen, key=runtimes_lib.runtime_sort_key)]
