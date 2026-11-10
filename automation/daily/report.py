"""Manifest-complete HTML, Markdown, JSON, and CSV report renderer for R5 (`automation/daily/report.py`).

Enforces the PROJECT.md report row contract:
  id, name, layer, tier, runtime, status, value_p50, value_p90, value_mean,
  unit, blocker, artifact_uri, run_id, collected_at
with `set(report IDs) == set(manifest IDs)` and zero `rapid` rows.
"""

from collections.abc import Mapping, Sequence
import csv
import datetime
import html
import io
import json
import logging
import math
import os
import subprocess
from typing import Any, Optional

from automation import utils

REPORT_ROW_FIELDS: tuple[str, ...] = (
    "id",
    "name",
    "layer",
    "tier",
    "runtime",
    "status",
    "value_p50",
    "value_p90",
    "value_mean",
    "unit",
    "blocker",
    "artifact_uri",
    "run_id",
    "collected_at",
)

LAYER_TITLES: dict[str, str] = {
    "client_proxy": "Layer 1: Client / Proxy (CP)",
    "rm_scheduler": "Layer 2: Resource Manager / Scheduler (RM)",
    "worker_tpu": "Layer 3: Worker / TPU (WK)",
    "datapath_sidecar": "Layer 4: Data Path & Colocated Python Sidecar (DP)",
    "storage_ckpt": "Layer 5: Storage / Checkpointing (ST)",
    "cluster_orch": "Layer 6: Cluster / Orchestration (CO)",
    "provenance": "Layer 7: Image, Flag & Version Provenance (PV)",
    "parity": "Numerical Correctness & Cross-Runtime Parity (PA)",
}

DEFAULT_NIGHTLY_TABLE = "cloud-tpu-multipod-dev.tpums_nightly.run_summary_microbenchmarks"


def _num(val: Any) -> Optional[float]:
  if val in (None, "") or isinstance(val, bool):
    return None
  try:
    f = float(val)
    return f if math.isfinite(f) else None
  except (TypeError, ValueError):
    return None


def build_report_rows(
    manifest: Sequence[Mapping[str, Any]],
    measurements: Mapping[tuple[str, str], Mapping[str, Any]],
    *,
    run_name: str,
    default_artifact_uri: str,
) -> list[dict[str, Any]]:
  """Builds normalized 14-column report rows for every (manifest entry, runtime)."""
  now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
  rows: list[dict[str, Any]] = []

  for entry in manifest:
    mid = str(entry["id"])
    name = str(entry.get("name", ""))
    layer = str(entry.get("layer", "unknown"))
    tier = str(entry.get("tier", "full"))
    unit = str(entry.get("unit", ""))
    runtimes = [r for r in (entry.get("runtimes") or ["louhi"]) if r != "rapid"]
    for rt in runtimes:
      meas = measurements.get((mid, rt)) or {}
      val_p50 = _num(meas.get("value_p50"))
      val_p90 = _num(meas.get("value_p90"))
      val_mean = _num(meas.get("value_mean"))
      val_text = meas.get("value_text")
      is_measured = bool(
          meas.get("status") == "measured"
          and (
              val_p50 is not None
              or val_mean is not None
              or val_text not in (None, "")
          )
      )
      status = "measured" if is_measured else "not_collectable_yet"
      blocker = (
          None
          if is_measured
          else str(
              meas.get("blocker")
              or entry.get("blocker")
              or "requires future instrumentation outside the single-slice daily run"
          )
      )
      row: dict[str, Any] = {
          "id": mid,
          "name": name,
          "layer": layer,
          "tier": tier,
          "runtime": rt,
          "status": status,
          "value_p50": val_p50 if is_measured else None,
          "value_p90": val_p90 if is_measured else None,
          "value_mean": val_mean if is_measured else None,
          "unit": unit,
          "blocker": blocker,
          "artifact_uri": str(meas.get("artifact_uri") or default_artifact_uri),
          "run_id": f"{run_name}_{rt}_{mid}",
          "collected_at": str(meas.get("collected_at") or now_iso),
      }
      if val_text not in (None, ""):
        row["value_text"] = str(val_text)
      rows.append(row)

  manifest_ids = {str(m["id"]) for m in manifest}
  report_ids = {str(r["id"]) for r in rows}
  if manifest_ids != report_ids:
    raise RuntimeError(
        f"Report ID invariant violated: missing={manifest_ids - report_ids}, "
        f"extra={report_ids - manifest_ids}"
    )
  if any(r["runtime"] == "rapid" for r in rows):
    raise RuntimeError("Forbidden runtime='rapid' found in report rows")
  return rows


def query_plausibility_join(
    daily_table: str,
    nightly_table: str = DEFAULT_NIGHTLY_TABLE,
    project: str = "cloud-tpu-multipod-dev",
) -> list[dict[str, Any]]:
  """Runs the 32-case 20% plausibility SQL join between `daily_table` and `nightly_table`."""
  sql = f"""
WITH daily_ranked AS (
  SELECT
    benchmark_name,
    CASE benchmark_name
      WHEN 'GeneralizedGemmBenchmark' THEN 'gemm'
      WHEN 'HBMBandwidthBenchmark'    THEN 'hbm'
      WHEN 'AllGatherBenchmark'       THEN 'all_gather'
      WHEN 'AllReduceBenchmark'       THEN 'all_reduce'
      WHEN 'AllToAllBenchmark'        THEN 'all_to_all'
      ELSE benchmark_name
    END AS bench,
    JSON_VALUE(params_other, '$.runtime') AS runtime,
    COALESCE(
      JSON_VALUE(params_other, '$.matrix_dim'),
      CONCAT('m', JSON_VALUE(params_other, '$.m'), '_n', JSON_VALUE(params_other, '$.n'), '_k', JSON_VALUE(params_other, '$.k')),
      CONCAT('elems', JSON_VALUE(params_other, '$.num_elements'))
    ) AS case_key,
    COALESCE(params_dtype, JSON_VALUE(params_other, '$.dtype'), JSON_VALUE(params_other, '$.in_dtype')) AS dtype,
    COALESCE(JSON_VALUE(params_other, '$.mesh_shape'), '') AS mesh_shape,
    COALESCE(JSON_VALUE(params_other, '$.sharding_strategy'), '') AS sharding_strategy,
    COALESCE(JSON_VALUE(params_other, '$.op_type'), JSON_VALUE(params_other, '$.reduce_op'), '') AS op_variant,
    run_group,
    start_time,
    update_timestamp,
    COALESCE(
      SAFE_CAST(JSON_VALUE(metrics_other, '$.xprof_p50_ms') AS FLOAT64),
      metrics_time_per_iter_ms_p50
    ) AS device_p50_ms,
    SAFE_CAST(JSON_VALUE(metrics_other, '$.wall_clock_p50_ms') AS FLOAT64) AS wall_p50_ms,
    metrics_tflops_achieved_p50 AS tflops_p50,
    metrics_achieved_bandwidth_gbps_p50 AS gbps_p50,
    ROW_NUMBER() OVER (
      PARTITION BY
        benchmark_name,
        JSON_VALUE(params_other, '$.runtime'),
        COALESCE(
          JSON_VALUE(params_other, '$.matrix_dim'),
          CONCAT('m', JSON_VALUE(params_other, '$.m'), '_n', JSON_VALUE(params_other, '$.n'), '_k', JSON_VALUE(params_other, '$.k')),
          CONCAT('elems', JSON_VALUE(params_other, '$.num_elements'))
        ),
        COALESCE(params_dtype, JSON_VALUE(params_other, '$.dtype'), JSON_VALUE(params_other, '$.in_dtype')),
        COALESCE(JSON_VALUE(params_other, '$.mesh_shape'), ''),
        COALESCE(JSON_VALUE(params_other, '$.sharding_strategy'), ''),
        COALESCE(JSON_VALUE(params_other, '$.op_type'), JSON_VALUE(params_other, '$.reduce_op'), '')
      ORDER BY start_time DESC, update_timestamp DESC
    ) AS rn
  FROM `{daily_table}`
  WHERE result_success = TRUE
    AND JSON_VALUE(params_other, '$.runtime') IN ('mcjax', 'louhi')
    AND benchmark_name IN (
      'GeneralizedGemmBenchmark',
      'HBMBandwidthBenchmark',
      'AllGatherBenchmark',
      'AllReduceBenchmark',
      'AllToAllBenchmark'
    )
),
nightly_ranked AS (
  SELECT
    benchmark_name,
    JSON_VALUE(params_other, '$.runtime') AS runtime,
    COALESCE(
      JSON_VALUE(params_other, '$.matrix_dim'),
      CONCAT('m', JSON_VALUE(params_other, '$.m'), '_n', JSON_VALUE(params_other, '$.n'), '_k', JSON_VALUE(params_other, '$.k')),
      CONCAT('elems', JSON_VALUE(params_other, '$.num_elements'))
    ) AS case_key,
    COALESCE(params_dtype, JSON_VALUE(params_other, '$.dtype'), JSON_VALUE(params_other, '$.in_dtype')) AS dtype,
    COALESCE(JSON_VALUE(params_other, '$.mesh_shape'), '') AS mesh_shape,
    COALESCE(JSON_VALUE(params_other, '$.sharding_strategy'), '') AS sharding_strategy,
    COALESCE(JSON_VALUE(params_other, '$.op_type'), JSON_VALUE(params_other, '$.reduce_op'), '') AS op_variant,
    run_group AS nightly_run_group,
    COALESCE(
      SAFE_CAST(JSON_VALUE(metrics_other, '$.xprof_p50_ms') AS FLOAT64),
      metrics_time_per_iter_ms_p50
    ) AS nightly_p50_ms,
    ROW_NUMBER() OVER (
      PARTITION BY
        benchmark_name,
        JSON_VALUE(params_other, '$.runtime'),
        COALESCE(
          JSON_VALUE(params_other, '$.matrix_dim'),
          CONCAT('m', JSON_VALUE(params_other, '$.m'), '_n', JSON_VALUE(params_other, '$.n'), '_k', JSON_VALUE(params_other, '$.k')),
          CONCAT('elems', JSON_VALUE(params_other, '$.num_elements'))
        ),
        COALESCE(params_dtype, JSON_VALUE(params_other, '$.dtype'), JSON_VALUE(params_other, '$.in_dtype')),
        COALESCE(JSON_VALUE(params_other, '$.mesh_shape'), ''),
        COALESCE(JSON_VALUE(params_other, '$.sharding_strategy'), ''),
        COALESCE(JSON_VALUE(params_other, '$.op_type'), JSON_VALUE(params_other, '$.reduce_op'), '')
      ORDER BY start_time DESC, update_timestamp DESC
    ) AS rn
  FROM `{nightly_table}`
  WHERE result_success = TRUE
    AND JSON_VALUE(params_other, '$.runtime') IN ('mcjax', 'louhi')
    AND benchmark_name IN (
      'GeneralizedGemmBenchmark',
      'HBMBandwidthBenchmark',
      'AllGatherBenchmark',
      'AllReduceBenchmark',
      'AllToAllBenchmark'
    )
)
SELECT
  d.bench,
  d.case_key,
  d.runtime,
  d.dtype,
  ROUND(d.device_p50_ms, 4) AS daily_p50_ms,
  ROUND(n.nightly_p50_ms, 4) AS nightly_p50_ms,
  ROUND(d.wall_p50_ms, 4) AS daily_wall_p50_ms,
  ROUND(SAFE_DIVIDE(d.device_p50_ms, n.nightly_p50_ms), 4) AS ratio_daily_vs_nightly,
  ROUND(ABS(SAFE_DIVIDE(d.device_p50_ms - n.nightly_p50_ms, n.nightly_p50_ms)) * 100.0, 2) AS rel_diff_pct,
  (
    d.device_p50_ms > 0
    AND n.nightly_p50_ms > 0
    AND ABS(SAFE_DIVIDE(d.device_p50_ms - n.nightly_p50_ms, n.nightly_p50_ms)) <= 0.20
  ) AS within_20pct,
  (
    d.device_p50_ms > 0
    AND (d.wall_p50_ms IS NULL OR d.wall_p50_ms >= d.device_p50_ms)
    AND (d.tflops_p50 IS NULL OR (d.tflops_p50 > 0 AND d.tflops_p50 <= 197.0))
    AND (d.gbps_p50 IS NULL OR (d.gbps_p50 > 0 AND d.gbps_p50 <= 819.2))
  ) AS hw_bounds_ok
FROM (SELECT * FROM daily_ranked WHERE rn = 1) d
INNER JOIN (SELECT * FROM nightly_ranked WHERE rn = 1) n
  USING (benchmark_name, runtime, case_key, dtype, mesh_shape, sharding_strategy, op_variant)
ORDER BY d.bench, d.case_key, d.runtime
"""
  try:
    from google.cloud import bigquery  # pylint: disable=g-import-not-at-top
    client = bigquery.Client(project=project)
    job = client.query(sql)
    return [dict(row.items()) for row in job.result()]
  except Exception as e:  # pylint: disable=broad-exception-caught
    logging.warning("BigQuery client query failed (%s); falling back to bq CLI.", e)
    try:
      proc = subprocess.run(
          [
              "bq",
              f"--project_id={project}",
              "query",
              "--use_legacy_sql=false",
              "--format=json",
              "--max_rows=500",
              "--quiet",
              sql,
          ],
          capture_output=True,
          text=True,
          check=False,
          timeout=60,
      )
      if proc.returncode == 0 and proc.stdout.strip():
        idx = proc.stdout.find("[")
        if idx >= 0:
          return json.loads(proc.stdout[idx:])
    except Exception as sub_e:  # pylint: disable=broad-exception-caught
      logging.warning("bq CLI query also failed: %s", sub_e)
  return []


def _fmt_cell(row: Optional[Mapping[str, Any]]) -> str:
  if not row:
    return "Pathways-only"
  if row.get("status") != "measured":
    return "not collectable yet"
  v_text = row.get("value_text")
  p50 = _num(row.get("value_p50"))
  p90 = _num(row.get("value_p90"))
  mean = _num(row.get("value_mean"))
  unit = str(row.get("unit") or "")
  if v_text and unit in ("string", "digest", "bool"):
    return str(v_text)
  if p50 is not None:
    base = f"{p50:.4g} {unit}".strip()
    if p90 is not None and mean is not None and (p90 != p50 or mean != p50):
      base += f" (p90={p90:.4g}, mean={mean:.4g})"
    if v_text:
      base += f" [{v_text}]"
    return base
  return str(v_text or "measured")


def render_csv(report_rows: Sequence[Mapping[str, Any]]) -> str:
  buf = io.StringIO()
  writer = csv.DictWriter(
      buf, fieldnames=list(REPORT_ROW_FIELDS), extrasaction="ignore"
  )
  writer.writeheader()
  for r in report_rows:
    writer.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in REPORT_ROW_FIELDS})
  return buf.getvalue()


def render_json(report_rows: Sequence[Mapping[str, Any]]) -> str:
  return json.dumps(list(report_rows), indent=2, sort_keys=True)


def render_markdown(
    manifest: Sequence[Mapping[str, Any]],
    report_rows: Sequence[Mapping[str, Any]],
    plausibility_rows: Sequence[Mapping[str, Any]],
    run_meta: Mapping[str, Any],
) -> str:
  rows_by_id_rt = {(str(r["id"]), str(r["runtime"])): r for r in report_rows}
  measured_ids = {str(r["id"]) for r in report_rows if r.get("status") == "measured"}
  total_ids = len(manifest)
  blocked_ids = total_ids - len(measured_ids)
  plaus_pass = sum(
      1
      for r in plausibility_rows
      if str(r.get("within_20pct")).lower() == "true"
  )

  lines: list[str] = []
  lines.append(
      f"# Pathways Benchmarking Metrics — Daily Report ({run_meta.get('date_str', '')})"
  )
  lines.append("")
  lines.append("## Run & Provenance Record")
  lines.append(f"- **Run Name / Group**: `{run_meta.get('run_name', '')}` / `{run_meta.get('run_group', '')}`")
  lines.append(f"- **CronJob**: `default/{run_meta.get('cronjob_name', 'tpums-pathways-metrics-daily')}` on `{run_meta.get('cluster', 'pw-scale-test-v5e-32')}`")
  lines.append(f"- **Fork Branch & Commit**: `{run_meta.get('git_branch', 'pathways-metrics-daily')}@{str(run_meta.get('git_commit', ''))[:12]}`")
  lines.append(f"- **Driver Image**: `{run_meta.get('driver_image', '')}`")
  lines.append(f"- **Louhi Server Image**: `{run_meta.get('louhi_server_image', '')}` (`{run_meta.get('louhi_server_digest', '')}`)")
  lines.append(f"- **Louhi Proxy Image**: `{run_meta.get('louhi_proxy_image', '')}` (`{run_meta.get('louhi_proxy_digest', '')}`)")
  lines.append(f"- **Client Image**: `{run_meta.get('client_image', '')}` (`{run_meta.get('client_image_digest', '')}`)")
  lines.append(f"- **GCS Prefix**: `{run_meta.get('gcs_prefix', '')}`")
  lines.append(f"- **BigQuery Table**: `{run_meta.get('bq_table', '')}`")
  lines.append(
      f"- **Coverage**: **{len(measured_ids)} / {total_ids}** catalogued metrics measured today; "
      f"**{blocked_ids}** not collectable yet (with concrete blockers); "
      f"**{plaus_pass} / {len(plausibility_rows)}** canonical cases within 20% of nightly."
  )
  lines.append("")

  lines.append("## 1. Today vs. Nightly Plausibility Check (5 Canonical Benchmarks)")
  lines.append("")
  lines.append("| Benchmark | Case Key | Runtime | Daily Device p50 (ms) | Nightly Device p50 (ms) | Ratio | Rel Diff (%) | Within 20%? | HW Bounds OK? |")
  lines.append("|---|---|---|---:|---:|---:|---:|---|---|")
  for r in plausibility_rows:
    w20 = "PASS" if str(r.get("within_20pct")).lower() == "true" else "FAIL"
    hw = "PASS" if str(r.get("hw_bounds_ok")).lower() == "true" else "FAIL"
    lines.append(
        f"| `{r.get('bench')}` | `{r.get('case_key')}` | `{r.get('runtime')}` | "
        f"{r.get('daily_p50_ms')} | {r.get('nightly_p50_ms')} | "
        f"{r.get('ratio_daily_vs_nightly')}x | {r.get('rel_diff_pct')}% | {w20} | {hw} |"
    )
  lines.append("")

  for layer_key, layer_title in LAYER_TITLES.items():
    layer_entries = [e for e in manifest if e.get("layer") == layer_key]
    if not layer_entries:
      continue
    lines.append(f"## {layer_title}")
    lines.append("")
    lines.append("| ID | Metric Name | Tier | Priority / Effort | Unit | Method | McJAX (Baseline) | Louhi (OSS Pathways) | Status / Concrete Blocker |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for e in layer_entries:
      mid = str(e["id"])
      mc_row = rows_by_id_rt.get((mid, "mcjax"))
      lo_row = rows_by_id_rt.get((mid, "louhi"))
      mc_cell = _fmt_cell(mc_row) if "mcjax" in (e.get("runtimes") or []) else "Pathways-only"
      lo_cell = _fmt_cell(lo_row)
      any_measured = (mc_row and mc_row.get("status") == "measured") or (
          lo_row and lo_row.get("status") == "measured"
      )
      if any_measured and (not lo_row or lo_row.get("status") == "measured"):
        status_str = "measured"
      else:
        blk = (lo_row or mc_row or {}).get("blocker") or e.get("blocker") or ""
        status_str = f"not collectable yet: {blk}"
      lines.append(
          f"| `{mid}` | {e.get('name', '')} | `{e.get('tier', '')}` | "
          f"`{e.get('priority', '')}` / `{e.get('effort', '')}` | `{e.get('unit', '')}` | "
          f"`{e.get('method', '')}` | {mc_cell} | {lo_cell} | {status_str} |"
      )
    lines.append("")

  return "\n".join(lines) + "\n"


def render_html(
    manifest: Sequence[Mapping[str, Any]],
    report_rows: Sequence[Mapping[str, Any]],
    plausibility_rows: Sequence[Mapping[str, Any]],
    run_meta: Mapping[str, Any],
) -> str:
  rows_by_id_rt = {(str(r["id"]), str(r["runtime"])): r for r in report_rows}
  measured_ids = {str(r["id"]) for r in report_rows if r.get("status") == "measured"}
  total_ids = len(manifest)
  blocked_ids = total_ids - len(measured_ids)
  plaus_pass = sum(
      1
      for r in plausibility_rows
      if str(r.get("within_20pct")).lower() == "true"
  )
  esc = html.escape

  plaus_trs: list[str] = []
  for r in plausibility_rows:
    w20 = str(r.get("within_20pct")).lower() == "true"
    hw = str(r.get("hw_bounds_ok")).lower() == "true"
    w20_badge = (
        '<span class="badge badge-measured">PASS</span>'
        if w20
        else '<span class="badge badge-blocked">FAIL</span>'
    )
    hw_badge = (
        '<span class="badge badge-measured">PASS</span>'
        if hw
        else '<span class="badge badge-blocked">FAIL</span>'
    )
    plaus_trs.append(
        "<tr>"
        f"<td><code>{esc(str(r.get('bench', '')))}</code></td>"
        f"<td><code>{esc(str(r.get('case_key', '')))}</code></td>"
        f"<td><code>{esc(str(r.get('runtime', '')))}</code></td>"
        f'<td class="num">{esc(str(r.get("daily_p50_ms", "")))}</td>'
        f'<td class="num">{esc(str(r.get("nightly_p50_ms", "")))}</td>'
        f'<td class="num">{esc(str(r.get("ratio_daily_vs_nightly", "")))}×</td>'
        f'<td class="num">{esc(str(r.get("rel_diff_pct", "")))}%</td>'
        f"<td>{w20_badge}</td>"
        f"<td>{hw_badge}</td>"
        "</tr>"
    )

  layer_sections: list[str] = []
  for layer_key, layer_title in LAYER_TITLES.items():
    layer_entries = [e for e in manifest if e.get("layer") == layer_key]
    if not layer_entries:
      continue
    trs: list[str] = []
    for e in layer_entries:
      mid = str(e["id"])
      mc_row = rows_by_id_rt.get((mid, "mcjax"))
      lo_row = rows_by_id_rt.get((mid, "louhi"))
      has_mc = "mcjax" in (e.get("runtimes") or [])
      mc_cell = (
          esc(_fmt_cell(mc_row))
          if has_mc
          else '<span class="badge badge-pwonly">Pathways-only</span>'
      )
      lo_cell = esc(_fmt_cell(lo_row))
      lo_meas = bool(lo_row and lo_row.get("status") == "measured")
      mc_meas = bool(mc_row and mc_row.get("status") == "measured")
      if lo_meas:
        status_html = '<span class="badge badge-measured">measured</span>'
      elif mc_meas and not lo_meas:
        blk = esc(str((lo_row or {}).get("blocker") or e.get("blocker") or ""))
        status_html = (
            '<span class="badge badge-measured">measured (McJAX)</span> '
            f'<span class="badge badge-blocked">Louhi: not collectable yet</span><br><small>{blk}</small>'
        )
      else:
        blk = esc(
            str(
                (lo_row or mc_row or {}).get("blocker")
                or e.get("blocker")
                or ""
            )
        )
        status_html = (
            '<span class="badge badge-blocked">not collectable yet</span>'
            f"<br><small>{blk}</small>"
        )
      art_uri = str((lo_row or mc_row or {}).get("artifact_uri") or "")
      art_short = "/".join(art_uri.rstrip("/").split("/")[-2:]) if art_uri else "-"
      trs.append(
          "<tr>"
          f"<td><code>{esc(mid)}</code></td>"
          f"<td><strong>{esc(str(e.get('name', '')))}</strong></td>"
          f"<td><code>{esc(str(e.get('tier', '')))}</code></td>"
          f"<td><code>{esc(str(e.get('priority', '')))}</code> / <code>{esc(str(e.get('effort', '')))}</code></td>"
          f"<td><code>{esc(str(e.get('unit', '')))}</code></td>"
          f"<td><code>{esc(str(e.get('method', '')))}</code></td>"
          f'<td class="num">{mc_cell}</td>'
          f'<td class="num">{lo_cell}</td>'
          f"<td>{status_html}</td>"
          f"<td><code>{esc(art_short)}</code></td>"
          "</tr>"
      )
    layer_sections.append(
        f'<section id="layer-{esc(layer_key)}">'
        f"<h2>{esc(layer_title)}</h2>"
        "<table><thead><tr>"
        "<th>ID</th><th>Metric Name</th><th>Tier</th><th>Priority / Effort</th>"
        "<th>Unit</th><th>Method</th><th>McJAX (Baseline)</th>"
        "<th>Louhi (OSS Pathways)</th><th>Status / Concrete Blocker</th><th>Artifact</th>"
        "</tr></thead><tbody>"
        + "\n".join(trs)
        + "</tbody></table></section>"
    )

  return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Pathways Daily Benchmark Report — {esc(str(run_meta.get('date_str', '')))}</title>
  <style>
    :root {{
      --bg: #f8fafc; --card: #ffffff; --text: #0f172a; --muted: #475569;
      --border: #e2e8f0; --mcjax: #2563eb; --louhi: #d97706;
      --ok-bg: #dcfce7; --ok-fg: #166534; --warn-bg: #fef3c7; --warn-fg: #92400e;
      --na-bg: #f1f5f9; --na-fg: #475569;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0; padding: 24px; background: var(--bg); color: var(--text);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      font-size: 14px; line-height: 1.5;
    }}
    .container {{ max-width: 1480px; margin: 0 auto; }}
    header, section {{
      background: var(--card); border: 1px solid var(--border);
      border-radius: 8px; padding: 20px; margin-bottom: 20px;
    }}
    h1 {{ margin: 0 0 8px 0; font-size: 22px; }}
    h2 {{ margin: 0 0 12px 0; font-size: 17px; border-bottom: 1px solid var(--border); padding-bottom: 8px; }}
    .meta-grid {{
      display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
      gap: 10px; font-size: 13px; color: var(--muted);
    }}
    .meta-grid code {{ color: var(--text); background: #f1f5f9; padding: 2px 5px; border-radius: 4px; word-break: break-all; }}
    .kpi-strip {{ display: flex; flex-wrap: wrap; gap: 12px; margin-top: 14px; }}
    .kpi-card {{
      flex: 1 1 180px; border: 1px solid var(--border); border-radius: 6px;
      padding: 12px; background: #fcfcfd;
    }}
    .kpi-card .label {{ font-size: 12px; color: var(--muted); }}
    .kpi-card .val {{ font-size: 20px; font-weight: 700; margin-top: 4px; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
    th, td {{
      border: 1px solid var(--border); padding: 8px 10px;
      text-align: left; vertical-align: top;
    }}
    th {{ background: #f1f5f9; font-weight: 600; }}
    .badge {{
      display: inline-block; padding: 2px 8px; border-radius: 999px;
      font-size: 11px; font-weight: 600;
    }}
    .badge-measured {{ background: var(--ok-bg); color: var(--ok-fg); }}
    .badge-blocked {{ background: var(--warn-bg); color: var(--warn-fg); }}
    .badge-pwonly {{ background: var(--na-bg); color: var(--na-fg); }}
    .num {{ font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }}
  </style>
</head>
<body>
<div class="container">
  <header>
    <h1>Pathways Benchmarking Metrics — Daily Report ({esc(str(run_meta.get('date_str', '')))})</h1>
    <div class="meta-grid">
      <div><strong>Run Name / Group:</strong> <code>{esc(str(run_meta.get('run_name', '')))}</code> / <code>{esc(str(run_meta.get('run_group', '')))}</code></div>
      <div><strong>CronJob:</strong> <code>default/{esc(str(run_meta.get('cronjob_name', 'tpums-pathways-metrics-daily')))}</code> on <code>{esc(str(run_meta.get('cluster', 'pw-scale-test-v5e-32')))}</code></div>
      <div><strong>Fork Branch &amp; Commit:</strong> <code>{esc(str(run_meta.get('git_branch', 'pathways-metrics-daily')))}@{esc(str(run_meta.get('git_commit', ''))[:12])}</code></div>
      <div><strong>Driver Image:</strong> <code>{esc(str(run_meta.get('driver_image', '')))}</code></div>
      <div><strong>Louhi Server Image:</strong> <code>{esc(str(run_meta.get('louhi_server_image', '')))}</code> (<code>{esc(str(run_meta.get('louhi_server_digest', ''))[:23])}</code>)</div>
      <div><strong>Louhi Proxy Image:</strong> <code>{esc(str(run_meta.get('louhi_proxy_image', '')))}</code> (<code>{esc(str(run_meta.get('louhi_proxy_digest', ''))[:23])}</code>)</div>
      <div><strong>Client Image:</strong> <code>{esc(str(run_meta.get('client_image', '')))}</code> (<code>{esc(str(run_meta.get('client_image_digest', ''))[:23])}</code>)</div>
      <div><strong>Sinks:</strong> <code>{esc(str(run_meta.get('gcs_prefix', '')))}</code> · <code>{esc(str(run_meta.get('bq_table', '')))}</code></div>
    </div>
    <div class="kpi-strip">
      <div class="kpi-card"><div class="label">Catalogued Metrics (All Tiers)</div><div class="val">{total_ids}</div></div>
      <div class="kpi-card"><div class="label">Measured Today</div><div class="val">{len(measured_ids)}</div></div>
      <div class="kpi-card"><div class="label">Not Collectable Yet (with Blocker)</div><div class="val">{blocked_ids}</div></div>
      <div class="kpi-card"><div class="label">Nightly 20% Plausibility Check</div><div class="val">{plaus_pass}/{len(plausibility_rows)} PASS</div></div>
    </div>
  </header>

  <section id="plausibility">
    <h2>1. Today vs. Nightly Plausibility Check (5 Canonical Benchmarks × {{McJAX, Louhi}})</h2>
    <table>
      <thead>
        <tr>
          <th>Benchmark</th><th>Case Key</th><th>Runtime</th>
          <th>Daily Device p50 (ms)</th><th>Nightly Device p50 (ms)</th>
          <th>Ratio (Daily / Nightly)</th><th>Rel Diff (%)</th>
          <th>Within 20%?</th><th>HW Bounds OK?</th>
        </tr>
      </thead>
      <tbody>
        {"".join(plaus_trs)}
      </tbody>
    </table>
  </section>

  {"".join(layer_sections)}
</div>
</body>
</html>
"""


def _upload_text(content: str, gcs_uri: str) -> None:
  proc = subprocess.run(
      [
          "gcloud",
          "storage",
          "cp",
          "--cache-control=no-cache,max-age=0",
          "-",
          gcs_uri,
      ],
      input=content,
      capture_output=True,
      text=True,
      check=False,
      timeout=utils.CLI_TIMEOUT_SECONDS,
  )
  if proc.returncode == 0:
    logging.info("Published %s", gcs_uri)
  else:
    logging.error(
        "Failed to publish %s (exit %d): %s",
        gcs_uri,
        proc.returncode,
        (proc.stderr or "").strip(),
    )


def render_and_publish_daily_reports(
    manifest: Sequence[Mapping[str, Any]],
    measurements: Mapping[tuple[str, str], Mapping[str, Any]],
    *,
    gcs_bucket: str,
    date_str: str,
    run_meta: Mapping[str, Any],
    bq_table: Optional[str] = None,
    output_dir: str = "/tmp",
) -> dict[str, str]:
  """Renders HTML, Markdown, JSON, and CSV daily reports locally and uploads them to GCS."""
  default_art_uri = utils.format_gcs_uri(gcs_bucket, date_str)
  run_name = str(run_meta.get("run_name") or f"tpums-pathways-metrics-daily-{date_str}")

  report_rows = build_report_rows(
      manifest,
      measurements,
      run_name=run_name,
      default_artifact_uri=default_art_uri,
  )
  plausibility_rows: list[dict[str, Any]] = []
  if bq_table:
    plausibility_rows = query_plausibility_join(bq_table)

  html_str = render_html(manifest, report_rows, plausibility_rows, run_meta)
  md_str = render_markdown(manifest, report_rows, plausibility_rows, run_meta)
  json_str = render_json(report_rows)
  csv_str = render_csv(report_rows)

  os.makedirs(output_dir, exist_ok=True)
  local_paths: dict[str, str] = {}
  for ext, text in (
      ("html", html_str),
      ("md", md_str),
      ("json", json_str),
      ("csv", csv_str),
  ):
    lpath = os.path.join(output_dir, f"daily_report.{ext}")
    with open(lpath, "w", encoding="utf-8") as f:
      f.write(text)
    local_paths[ext] = lpath

  if gcs_bucket:
    # Per-date copies
    _upload_text(
        html_str,
        utils.format_gcs_uri(gcs_bucket, date_str, "consolidated_report.html"),
    )
    _upload_text(
        md_str,
        utils.format_gcs_uri(gcs_bucket, date_str, "consolidated_report.md"),
    )
    for ext, text in (
        ("html", html_str),
        ("md", md_str),
        ("json", json_str),
        ("csv", csv_str),
    ):
      _upload_text(
          text,
          utils.format_gcs_uri(gcs_bucket, date_str, f"daily_report.{ext}"),
      )
      _upload_text(
          text,
          utils.format_gcs_uri(gcs_bucket, "reports", f"{date_str}.{ext}"),
      )
      _upload_text(
          text,
          utils.format_gcs_uri(gcs_bucket, "reports", f"latest.{ext}"),
      )
    # Also publish HTML to *_dashboard.html so serve_dashboard.py `/` rewrite works
    _upload_text(
        html_str,
        utils.format_gcs_uri(
            gcs_bucket, "reports", f"{date_str}_dashboard.html"
        ),
    )
    _upload_text(
        html_str,
        utils.format_gcs_uri(gcs_bucket, "reports", "latest_dashboard.html"),
    )

  return local_paths
