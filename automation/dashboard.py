"""Interactive HTML dashboard of a night's runtime comparison.

A self-contained page (inline SVG drawn by a little JavaScript, no external
assets, so it renders straight from GCS) with:

* the night's JobSets and their outcome, one cell each;
* per benchmark, grouped bars of the p50 step time of every runtime (solid:
  XProf device time, hollow: wall clock as the JAX client sees it);
* a Pathways / McJAX ratio heat map per case (device time and wall clock);
* per benchmark, achieved throughput (bandwidth or TFLOP/s).

It is published next to the Markdown/HTML report as
`<date>/dashboard.html`, `reports/<date>_dashboard.html` and
`reports/latest_dashboard.html`.
"""

from collections.abc import Mapping, Sequence
import json
import logging
import subprocess
from typing import Any

from automation import utils

DASHBOARD_FILE = "dashboard.html"

_TEMPLATE = r"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>__TITLE__</title>
<style>
  :root { color-scheme: light dark; --bg: #f6f8fa; --card: #fff; --fg: #1f2328; --muted: #656d76; --border: #d0d7de; --mutedbg: #eaeef2; }
  @media (prefers-color-scheme: dark) { :root { --bg: #0d1117; --card: #161b22; --fg: #e6edf3; --muted: #8d96a0; --border: #30363d; --mutedbg: #21262d; } }
  body { margin: 0; padding: 20px; background: var(--bg); color: var(--fg); font: 14px/1.45 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; }
  .wrap { max-width: 1200px; margin: 0 auto; }
  .card { background: var(--card); border: 1px solid var(--border); border-radius: 12px; padding: 18px 20px; margin-bottom: 18px; }
  h1 { font-size: 20px; margin: 0 0 6px; } h2 { font-size: 16px; margin: 0 0 4px; }
  .muted { color: var(--muted); font-size: 13px; }
  .grid2 { display: grid; grid-template-columns: repeat(auto-fit, minmax(480px, 1fr)); gap: 14px; margin-top: 12px; }
  .panel { border: 1px solid var(--border); border-radius: 8px; padding: 8px; }
  .legend span { display: inline-flex; align-items: center; margin-right: 16px; } .sw { display: inline-block; width: 12px; height: 12px; border-radius: 3px; margin-right: 6px; }
  .tip { position: fixed; pointer-events: none; z-index: 50; max-width: 380px; font-size: 12px; line-height: 1.35; background: var(--card); color: var(--fg); border: 1px solid var(--border); border-radius: 8px; padding: 8px 10px; box-shadow: 0 4px 14px rgba(0,0,0,.25); display: none; white-space: pre-line; }
  svg text { font-family: inherit; } .axis { stroke: var(--border); } .grid { stroke: var(--border); stroke-opacity: .6; stroke-dasharray: 2 3; }
  .lbl { fill: var(--muted); font-size: 11px; } .lbl2 { fill: var(--fg); font-size: 12px; font-weight: 600; }
  a { color: inherit; }
</style>
</head>
<body><div class="wrap">
  <div class="card">
    <h1>__TITLE__</h1>
    <div class="muted">__SUBTITLE__</div>
    <div class="legend" id="legend" style="margin-top:10px"></div>
    <div class="muted" id="prov" style="margin-top:6px"></div>
  </div>
  <div class="card">
    <h2>JobSets of this run</h2>
    <div class="muted">One cell per JobSet (hover for duration, slice and errors).</div>
    <div id="runs" style="margin-top:10px"></div>
  </div>
  <div class="card">
    <h2>Step time per case (p50, ms)</h2>
    <div class="muted">Solid bar = XProf device time; hollow bar = wall clock seen by the JAX client. Runtimes without device traces show wall clock only. Hover for p90 / std / throughput / slice.</div>
    <div id="bars" class="grid2"></div>
  </div>
  <div class="card">
    <h2>Pathways / McJAX ratio per case</h2>
    <div class="muted">1.00 = same as McJAX. Device-time ratios isolate the TPU work; wall-clock ratios include the proxy round trip.</div>
    <div id="heat" style="margin-top:10px;overflow-x:auto"></div>
  </div>
  <div class="card">
    <h2>Throughput per case</h2>
    <div class="muted">Achieved bandwidth (GB/s per chip or device) or TFLOP/s per device; from device time where available, else wall clock (hatched).</div>
    <div id="thr" class="grid2"></div>
  </div>
</div>
<div class="tip" id="tip"></div>
<script>
const ROWS = __ROWS__;
const JOBSETS = __JOBSETS__;
const COLORS = ["#4285F4", "#F9AB00", "#34A853", "#A142F4", "#EA4335", "#00ACC1"];
const RT = [...new Set(ROWS.map(r => r.runtime || "mcjax"))].sort((a, b) => (a === "mcjax" ? -1 : b === "mcjax" ? 1 : a.localeCompare(b)));
const COLOR = Object.fromEntries(RT.map((rt, i) => [rt, COLORS[i % COLORS.length]]));
const BENCH = [...new Set(ROWS.map(r => r.benchmark))].map(b => [b, b.replace(/Benchmark$/, "")]);
const CASE_KEYS = ["matrix_dim", "num_elements", "data_size_mib", "m", "k", "n", "shard_size_mib", "dtype", "in_dtype", "out_dtype", "src_device_index", "dst_device_index", "mesh_shape", "sharding_strategy"];
const num = v => (v === "" || v == null) ? null : Number(v);
const f = (v, d = 2) => (v == null || Number.isNaN(Number(v))) ? "–" : Number(v).toFixed(d);
const slice = r => (r.node_pools || "").split(",").filter(p => p.startsWith("gke-tpu-")).map(p => p.replace("gke-", "")).join(", ");
const caseLabel = r => { if (r.m && r.k && r.n) return `m=${r.m} k=${r.k} n=${r.n}`; if (r.matrix_dim) return `dim ${r.matrix_dim}`; if (r.data_size_mib) return `${Number(r.data_size_mib)} MiB`; if (r.num_elements) return `${r.num_elements} elems`; return CASE_KEYS.filter(k => r[k]).map(k => `${k}=${r[k]}`).join(" ") || "(all)"; };
const caseKey = r => CASE_KEYS.map(k => r[k] || "").join("|");
const thr = r => { const pairs = [["xprof_tflops_per_device","TFLOP/s/dev",false],["xprof_tflops_per_chip","TFLOP/s/chip",false],["wall_clock_tflops_per_device","TFLOP/s/dev (wall)",true],["wall_clock_tflops_per_chip","TFLOP/s/chip (wall)",true],["xprof_bandwidth_per_chip_gb_s","GB/s/chip",false],["xprof_bandwidth_per_device_gb_s","GB/s/dev",false],["wall_clock_bandwidth_per_chip_gb_s","GB/s/chip (wall)",true],["wall_clock_bandwidth_per_device_gb_s","GB/s/dev (wall)",true]]; for (const [k,u,w] of pairs) if (num(r[k]) != null) return {v: num(r[k]), u, wall: w}; return null; };
const tip = document.getElementById("tip");
const showTip = (e, html) => { tip.innerHTML = html; tip.style.display = "block"; tip.style.left = Math.min(e.clientX + 14, window.innerWidth - 400) + "px"; tip.style.top = (e.clientY + 14) + "px"; };
const hideTip = () => tip.style.display = "none";
const svgEl = (tag, attrs = {}, text) => { const el = document.createElementNS("http://www.w3.org/2000/svg", tag); for (const k in attrs) el.setAttribute(k, attrs[k]); if (text != null) el.textContent = text; return el; };
const hover = (el, text) => { el.addEventListener("mousemove", e => showTip(e, text)); el.addEventListener("mouseleave", hideTip); return el; };

document.getElementById("legend").innerHTML = RT.map(rt => `<span><span class="sw" style="background:${COLOR[rt]}"></span>${rt}</span>`).join("");
(function(){ const by = {}; for (const r of ROWS) if (!by[r.runtime || "mcjax"]) by[r.runtime || "mcjax"] = r;
  const parts = RT.filter(rt => by[rt]).map(rt => { const r = by[rt]; const d = (r.pathways_server_digest || "").replace("sha256:", "").slice(0, 12); const c = (r.client_image_digest || "").replace("sha256:", "").slice(0, 12); return `<b>${rt}</b>: ${[c && "client " + c, d && "server " + d, r.git_commit && "commit " + String(r.git_commit).slice(0, 12)].filter(Boolean).join(", ")}`; });
  const slices = [...new Set(ROWS.map(slice).filter(Boolean))];
  document.getElementById("prov").innerHTML = parts.join(" · ") + (slices.length ? ". Slices: " + slices.join(", ") : ""); })();

// JobSet strip
(function(){ const el = document.getElementById("runs"); if (!JOBSETS.length) { el.innerHTML = '<span class="muted">(no JobSets in this run)</span>'; return; }
  const STATUS = { SUCCESS: "#34A853", FAILED: "#EA4335", TIMEOUT: "#EA4335", DRY_RUN: "#9aa0a6" };
  const cw = 46, H = 70, W = Math.max(JOBSETS.length * cw, 200);
  const svg = svgEl("svg", {width: W, height: H, viewBox: `0 0 ${W} ${H}`});
  JOBSETS.forEach((j, i) => { const x = i * cw; const dur = j.duration_seconds ? `${Math.floor(j.duration_seconds / 60)}m ${Math.round(j.duration_seconds % 60)}s` : "";
    svg.appendChild(hover(svgEl("rect", {x: x + 2, y: 4, width: cw - 6, height: 26, rx: 5, fill: STATUS[j.status] || "#999"}), `<b>${j.runtime}/${j.config}</b> — ${j.status}${dur ? " in " + dur : ""}\n${j.workload}${j.slice ? "\nslice: " + j.slice : ""}${j.error ? "\n" + j.error : ""}`));
    svg.appendChild(svgEl("text", {x: x + cw / 2 - 1, y: 44, class: "lbl", "text-anchor": "middle"}, j.config.length > 7 ? j.config.slice(0, 6) + "…" : j.config));
    svg.appendChild(svgEl("text", {x: x + cw / 2 - 1, y: 58, class: "lbl", "text-anchor": "middle"}, j.runtime)); });
  el.appendChild(svg);
  const ok = JOBSETS.filter(j => j.status === "SUCCESS").length;
  const lg = document.createElement("div"); lg.className = "muted legend"; lg.style.marginTop = "6px";
  lg.innerHTML = `<span><span class="sw" style="background:#34A853"></span>succeeded</span><span><span class="sw" style="background:#EA4335"></span>failed / timed out</span><span>${ok}/${JOBSETS.length} succeeded</span>`; el.appendChild(lg); })();

function panel(bench, short, mode) {
  const rows = ROWS.filter(r => r.benchmark === bench);
  const keys = [...new Set(rows.map(caseKey))];
  const label = {}; rows.forEach(r => label[caseKey(r)] = caseLabel(r));
  keys.sort((a, b) => { const na = Number((label[a].match(/[\d.]+/) || [0])[0]), nb = Number((label[b].match(/[\d.]+/) || [0])[0]); return na - nb || label[a].localeCompare(label[b]); });
  const W = 560, H = 250, left = 54, right = 10, top = 28, bottom = 36;
  const svg = svgEl("svg", {width: "100%", viewBox: `0 0 ${W} ${H}`, style: "max-width:600px"});
  const val = (r, kind) => mode === "time" ? num(kind === "dev" ? r.xprof_p50_ms : r.wall_clock_p50_ms) : (thr(r) ? thr(r).v : null);
  let max = 0; rows.forEach(r => { for (const k of ["dev", "wall"]) { const v = val(r, k); if (v != null && v > max) max = v; } }); if (!max) max = 1;
  const yv = v => top + (H - top - bottom) * (1 - v / (max * 1.08));
  const gw = (W - left - right) / Math.max(keys.length, 1);
  for (let t = 0; t <= 4; t++) { const v = max * 1.08 * t / 4; svg.appendChild(svgEl("line", {x1: left, x2: W - right, y1: yv(v), y2: yv(v), class: "grid"})); svg.appendChild(svgEl("text", {x: left - 6, y: yv(v) + 4, class: "lbl", "text-anchor": "end"}, v >= 100 ? v.toFixed(0) : v >= 10 ? v.toFixed(1) : v.toFixed(2))); }
  svg.appendChild(svgEl("line", {x1: left, x2: W - right, y1: yv(0), y2: yv(0), class: "axis"}));
  const unit = mode === "time" ? "p50 step time (ms)" : ((rows.map(thr).find(Boolean) || {u: ""}).u.replace(" (wall)", ""));
  svg.appendChild(svgEl("text", {x: left, y: 14, class: "lbl2"}, `${short} — ${unit}`));
  if (mode !== "time") { const defs = svgEl("defs"); RT.forEach(rt => { const p = svgEl("pattern", {id: `hatch-${rt}`, width: 6, height: 6, patternUnits: "userSpaceOnUse", patternTransform: "rotate(45)"}); p.appendChild(svgEl("rect", {width: 6, height: 6, fill: COLOR[rt], opacity: .25})); p.appendChild(svgEl("line", {x1: 0, y1: 0, x2: 0, y2: 6, stroke: COLOR[rt], "stroke-width": 2})); defs.appendChild(p); }); svg.appendChild(defs); }
  keys.forEach((key, i) => { const gx = left + i * gw; svg.appendChild(svgEl("text", {x: gx + gw / 2, y: H - 14, class: "lbl", "text-anchor": "middle"}, label[key]));
    const bw = (gw - 16) / Math.max(RT.length, 1);
    RT.forEach((rt, j) => { const r = rows.find(q => (q.runtime || "mcjax") === rt && caseKey(q) === key); if (!r) return; const bx = gx + 8 + j * bw; const dev = val(r, "dev"), wall = val(r, "wall"); const t = thr(r);
      const tt = `<b>${short} · ${rt}</b>\n${label[key]}\nxprof p50/p90: ${f(r.xprof_p50_ms,3)} / ${f(r.xprof_p90_ms,3)} ms\nwall p50/p90/avg±std: ${f(r.wall_clock_p50_ms,3)} / ${f(r.wall_clock_p90_ms,3)} / ${f(r.wall_clock_avg_ms,3)} ± ${f(r.wall_clock_std_ms,3)} ms` + (t ? `\nthroughput: ${f(t.v)} ${t.u}` : "") + (slice(r) ? `\nslice: ${slice(r)}` : "");
      if (mode === "time") { if (wall != null) svg.appendChild(hover(svgEl("rect", {x: bx + 1, y: yv(wall), width: Math.max(bw - 3, 1), height: yv(0) - yv(wall), fill: "none", stroke: COLOR[rt], "stroke-width": 1.5, rx: 2}), tt)); if (dev != null) svg.appendChild(hover(svgEl("rect", {x: bx + 1, y: yv(dev), width: Math.max(bw - 3, 1), height: yv(0) - yv(dev), fill: COLOR[rt], opacity: .9, rx: 2}), tt)); }
      else if (t) svg.appendChild(hover(svgEl("rect", {x: bx + 1, y: yv(t.v), width: Math.max(bw - 3, 1), height: yv(0) - yv(t.v), fill: t.wall ? `url(#hatch-${rt})` : COLOR[rt], stroke: COLOR[rt], "stroke-width": t.wall ? 1 : 0, rx: 2}), tt)); }); });
  const wrap = document.createElement("div"); wrap.className = "panel"; wrap.appendChild(svg); return wrap;
}
for (const [b, s] of BENCH) document.getElementById("bars").appendChild(panel(b, s, "time"));
for (const [b, s] of BENCH) document.getElementById("thr").appendChild(panel(b, s, "thr"));

// Ratio heat map
(function(){ const others = RT.filter(rt => rt !== "mcjax"); const el = document.getElementById("heat");
  if (!others.length || !RT.includes("mcjax")) { el.innerHTML = '<span class="muted">(needs McJAX and at least one other runtime)</span>'; return; }
  const cols = []; for (const rt of others) { cols.push([rt, "xprof_p50_ms", `${rt} / mcjax (device)`]); cols.push([rt, "wall_clock_p50_ms", `${rt} / mcjax (wall)`]); }
  const allCases = []; for (const [b, s] of BENCH) { const rows = ROWS.filter(r => r.benchmark === b); const keys = [...new Set(rows.map(caseKey))];
    for (const key of keys) { const per = {}; rows.filter(r => caseKey(r) === key).forEach(r => per[r.runtime || "mcjax"] = r); const any = rows.find(r => caseKey(r) === key); allCases.push({label: `${s} ${caseLabel(any)}`, per}); } }
  const cases = allCases;
  // Drop columns without a single ratio (e.g. device time of a runtime that has no traces).
  const hasData = ([rt, key]) => cases.some(cs => cs.per[rt] && num(cs.per[rt][key]) != null && cs.per["mcjax"] && num(cs.per["mcjax"][key]));
  const liveCols = cols.filter(hasData); if (!liveCols.length) { el.innerHTML = '<span class="muted">(no comparable cases)</span>'; return; }
  cols.length = 0; cols.push(...liveCols);
  const color = v => { if (v == null) return "var(--mutedbg)"; if (v < 0.97) return "#1e8e3e"; const t = Math.max(0, Math.min(1, Math.log2(v) / 2)); return `rgb(${Math.round(52 + t * 182)},${Math.round(168 - t * 101)},${Math.round(83 - t * 30)})`; };
  const rowH = 24, left = 280, cw = 150, top = 34, W = left + cols.length * cw + 10, H = top + cases.length * rowH + 8;
  const svg = svgEl("svg", {width: W, height: H, viewBox: `0 0 ${W} ${H}`});
  cols.forEach((c, j) => svg.appendChild(svgEl("text", {x: left + j * cw + cw / 2, y: 20, class: "lbl2", "text-anchor": "middle"}, c[2])));
  cases.forEach((cs, i) => { const y = top + i * rowH; svg.appendChild(svgEl("text", {x: left - 8, y: y + 16, class: "lbl", "text-anchor": "end"}, cs.label));
    cols.forEach(([rt, key], j) => { const r = cs.per[rt], base = cs.per["mcjax"]; const a = r ? num(r[key]) : null, b = base ? num(base[key]) : null; const v = (a != null && b) ? a / b : null;
      svg.appendChild(hover(svgEl("rect", {x: left + j * cw + 2, y: y + 2, width: cw - 4, height: rowH - 4, rx: 4, fill: color(v), opacity: v == null ? .5 : .92}), `<b>${cs.label}</b>\n${rt}: ${f(a,3)} ms vs mcjax ${f(b,3)} ms (${key.includes("xprof") ? "device" : "wall clock"})` + (r && slice(r) ? `\n${rt} slice: ${slice(r)}` : "") + (base && slice(base) ? `\nmcjax slice: ${slice(base)}` : "")));
      svg.appendChild(svgEl("text", {x: left + j * cw + cw / 2, y: y + 16, "text-anchor": "middle", style: "fill:#fff;font-size:12px;font-weight:600;pointer-events:none"}, v == null ? "n/a" : v.toFixed(2) + "x")); }); });
  el.appendChild(svg); })();
</script>
</body>
</html>
"""

_ROW_KEYS = (
    "runtime", "benchmark", "matrix_dim", "num_elements", "data_size_mib",
    "m", "k", "n", "in_dtype", "out_dtype", "dtype", "shard_size_mib",
    "mesh_shape", "sharding_strategy", "src_device_index", "dst_device_index",
    "xprof_p50_ms", "xprof_p90_ms", "wall_clock_p50_ms", "wall_clock_p90_ms",
    "wall_clock_avg_ms", "wall_clock_std_ms",
    "xprof_bandwidth_per_chip_gb_s", "wall_clock_bandwidth_per_chip_gb_s",
    "xprof_bandwidth_per_device_gb_s", "wall_clock_bandwidth_per_device_gb_s",
    "xprof_tflops_per_device", "wall_clock_tflops_per_device",
    "xprof_tflops_per_chip", "wall_clock_tflops_per_chip",
    "node_pools", "pathways_server_digest", "client_image_digest", "git_commit",
)


def _tpu_slice(node_pools: Any) -> str:
  pools = [p for p in str(node_pools or "").split(",") if p.startswith("gke-tpu-")]
  return ", ".join(p.replace("gke-", "", 1) for p in pools)


def list_report_dates(gcs_bucket: str) -> list[str]:
  """Lists available `<YYYY-MM-DD>` dashboard dates under `<bucket>/reports/`."""
  pattern = utils.format_gcs_uri(gcs_bucket, "reports", f"*_{DASHBOARD_FILE}")
  try:
    proc = subprocess.run(
        ["gcloud", "storage", "ls", pattern],
        capture_output=True,
        text=True,
        check=False,
        stdin=subprocess.DEVNULL,
        timeout=utils.CLI_TIMEOUT_SECONDS,
    )
  except (OSError, subprocess.TimeoutExpired):
    return []
  if proc.returncode != 0:
    return []
  dates = set()
  suffix = f"_{DASHBOARD_FILE}"
  for line in proc.stdout.splitlines():
    name = line.strip().rsplit("/", 1)[-1]
    if name.endswith(suffix) and name != f"latest_{DASHBOARD_FILE}":
      dates.add(name[: -len(suffix)])
  return sorted(dates, reverse=True)


def render_dashboard(
    categorized_metrics: Mapping[str, Sequence[Mapping[str, Any]]],
    results: Sequence[Any],
    *,
    date_str: str,
    cluster: str = "",
    report_uri: str = "",
    history_dates: Sequence[str] = (),
) -> str:
  """Returns the dashboard HTML for one night.

  Args:
    categorized_metrics: Aggregated rows per category (`runtime` stamped).
    results: `BenchmarkResult`s of the run (status, duration, error) for the
      JobSet strip; anything with the matching attributes works.
    date_str: The night's date, shown in the title.
    cluster: Cluster name, shown in the subtitle.
    report_uri: Where the Markdown/HTML report lives, linked in the subtitle.
    history_dates: Optional list of `<YYYY-MM-DD>` dates to link in the header.
  """
  rows = []
  for category_rows in categorized_metrics.values():
    for row in category_rows:
      rows.append({k: row[k] for k in _ROW_KEYS if row.get(k) not in (None, "")})
  jobsets = []
  for r in results:
    prov = getattr(r, "provenance", None) or {}
    jobsets.append({
        "runtime": getattr(r, "runtime", "mcjax"),
        "config": getattr(r, "config_name", ""),
        "workload": getattr(r, "workload_name", ""),
        "status": getattr(r, "status", ""),
        "duration_seconds": getattr(r, "duration_seconds", 0.0),
        "error": getattr(r, "error_message", None) or "",
        "slice": _tpu_slice(prov.get("node_pools")),
    })
  runtimes = sorted({str(r.get("runtime") or "mcjax") for r in rows})
  all_dates = sorted({date_str, *history_dates}, reverse=True)[:14]
  history_html = ""
  if len(all_dates) > 1:
    links = [
        f'<a href="{d}_{DASHBOARD_FILE}"><b>{d}</b></a>'
        if d == date_str
        else f'<a href="{d}_{DASHBOARD_FILE}">{d}</a>'
        for d in all_dates
    ]
    history_html = "runs: " + " · ".join(links)
  subtitle = " · ".join(
      s for s in (
          cluster and f"cluster {cluster}",
          f"{len(jobsets)} JobSet(s)",
          f"runtimes: {', '.join(runtimes)}" if runtimes else "",
          report_uri and f'report: <a href="{report_uri}">{report_uri}</a>',
          history_html,
      ) if s
  )
  return (
      _TEMPLATE.replace("__TITLE__", f"TPUMS nightly — {date_str}")
      .replace("__SUBTITLE__", subtitle)
      .replace("__ROWS__", json.dumps(rows))
      .replace("__JOBSETS__", json.dumps(jobsets))
  )


def publish_dashboard(gcs_bucket: str, date_str: str, html: str) -> list[str]:
  """Uploads the dashboard to the day's prefix and the reports/ aliases.

  Returns:
    The URIs written; failures are logged and skipped.
  """
  targets = (
      utils.format_gcs_uri(gcs_bucket, date_str, DASHBOARD_FILE),
      utils.format_gcs_uri(gcs_bucket, "reports", f"{date_str}_{DASHBOARD_FILE}"),
      utils.format_gcs_uri(gcs_bucket, "reports", f"latest_{DASHBOARD_FILE}"),
  )
  written = []
  for uri in targets:
    try:
      proc = subprocess.run(
          [
              "gcloud",
              "storage",
              "cp",
              "--content-type=text/html",
              "--cache-control=no-cache,max-age=0",
              "-",
              uri,
          ],
          input=html,
          capture_output=True,
          text=True,
          check=False,
          timeout=utils.CLI_TIMEOUT_SECONDS,
      )
    except subprocess.TimeoutExpired:
      logging.warning("Timed out uploading %s.", uri)
      continue
    if proc.returncode == 0:
      logging.info("Published %s", uri)
      written.append(uri)
    else:
      logging.warning("Could not upload %s: %s", uri, (proc.stderr or "").strip())
  return written
