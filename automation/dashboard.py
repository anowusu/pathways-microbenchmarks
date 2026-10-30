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
  :root { color-scheme: light dark; --bg: #f8fafc; --card: #ffffff; --fg: #0f172a; --muted: #64748b; --border: #e2e8f0; --mutedbg: #f1f5f9; --accent: #2563eb; --good: #16a34a; --goodbg: #dcfce7; }
  @media (prefers-color-scheme: dark) { :root { --bg: #0b0f17; --card: #151c28; --fg: #f1f5f9; --muted: #94a3b8; --border: #263142; --mutedbg: #1e293b; --accent: #3b82f6; --good: #4ade80; --goodbg: #14532d; } }
  * { box-sizing: border-box; }
  body { margin: 0; padding: 24px 16px; background: var(--bg); color: var(--fg); font: 14px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; }
  .wrap { max-width: 1180px; margin: 0 auto; }
  .card { background: var(--card); border: 1px solid var(--border); border-radius: 14px; padding: 20px 22px; margin-bottom: 16px; }
  .top { display: flex; flex-wrap: wrap; justify-content: space-between; align-items: flex-start; gap: 12px; }
  h1 { font-size: 20px; font-weight: 700; margin: 0 0 4px; letter-spacing: -0.01em; }
  h2 { font-size: 15px; font-weight: 600; margin: 0; }
  .muted { color: var(--muted); font-size: 13px; }
  .kpis { display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 12px; margin-top: 16px; }
  .kpi { border: 1px solid var(--border); border-radius: 10px; padding: 12px 14px; background: var(--mutedbg); }
  .kpi-head { display: flex; align-items: center; justify-content: space-between; font-weight: 600; font-size: 13px; }
  .kpi-val { font-size: 20px; font-weight: 700; margin: 4px 0 2px; }
  .badge { display: inline-block; font-size: 11px; font-weight: 600; padding: 2px 8px; border-radius: 999px; background: var(--goodbg); color: var(--good); }
  .bar-ctrl { display: flex; flex-wrap: wrap; align-items: center; justify-content: space-between; gap: 12px; margin-bottom: 14px; }
  .seg { display: inline-flex; background: var(--mutedbg); border: 1px solid var(--border); border-radius: 10px; padding: 3px; gap: 2px; }
  .seg button { border: 0; background: transparent; color: var(--muted); font: 600 12.5px/1 inherit; padding: 8px 12px; border-radius: 7px; cursor: pointer; transition: all .12s; }
  .seg button.active { background: var(--card); color: var(--fg); box-shadow: 0 1px 3px rgba(0,0,0,.12); }
  .legend { display: flex; flex-wrap: wrap; align-items: center; gap: 16px; font-size: 13px; font-weight: 500; }
  .legend span { display: inline-flex; align-items: center; }
  .sw { display: inline-block; width: 12px; height: 12px; border-radius: 3px; margin-right: 6px; }
  .grid2 { display: grid; grid-template-columns: repeat(auto-fit, minmax(510px, 1fr)); gap: 14px; }
  .panel { border: 1px solid var(--border); border-radius: 12px; padding: 14px 16px 10px; background: var(--card); }
  .panel-head { display: flex; align-items: baseline; justify-content: space-between; gap: 8px; margin-bottom: 2px; }
  .panel-title { font-size: 15px; font-weight: 650; }
  .dir { font-size: 11.5px; font-weight: 600; color: var(--good); }
  .panel-sub { font-size: 12.5px; color: var(--muted); margin-bottom: 8px; }
  .tip { position: fixed; pointer-events: none; z-index: 50; max-width: 360px; font-size: 12px; line-height: 1.4; background: var(--card); color: var(--fg); border: 1px solid var(--border); border-radius: 8px; padding: 9px 11px; box-shadow: 0 6px 18px rgba(0,0,0,.22); display: none; white-space: pre-line; }
  svg text { font-family: inherit; }
  .axis { stroke: var(--border); stroke-width: 1; }
  .grid { stroke: var(--border); stroke-opacity: .55; stroke-dasharray: 2 3; }
  .lbl { fill: var(--muted); font-size: 11px; }
  .lbl-x { fill: var(--fg); font-size: 11.5px; font-weight: 600; }
  .lbl-xsub { fill: var(--muted); font-size: 10.5px; }
  .val-lbl { fill: var(--fg); font-size: 11px; font-weight: 600; }
  details summary { cursor: pointer; font-weight: 600; font-size: 13px; color: var(--muted); user-select: none; }
  a { color: var(--accent); text-decoration: none; }
  a:hover { text-decoration: underline; }
</style>
</head>
<body><div class="wrap">
  <div class="card">
    <div class="top">
      <div>
        <h1>__TITLE__</h1>
        <div class="muted">__SUBTITLE__</div>
      </div>
      <div class="legend" id="legend"></div>
    </div>
    <div class="kpis" id="kpis"></div>
  </div>

  <div class="card">
    <div class="bar-ctrl">
      <div style="display:flex;flex-wrap:wrap;gap:10px;align-items:center">
        <div class="seg" role="tablist" aria-label="Metric">
          <button id="btn-thr" class="active" onclick="setMode('thr')">Hardware Speed (GB/s · TFLOP/s)</button>
          <button id="btn-dev" onclick="setMode('dev')">TPU Chip Time (ms)</button>
          <button id="btn-wall" onclick="setMode('wall')">Client Wall Clock = Chip + Host/Proxy (ms)</button>
        </div>
        <div class="seg" role="tablist" aria-label="Statistic">
          <button id="stat-p50" class="active" onclick="setStat('p50')">p50 (Median)</button>
          <button id="stat-p90" onclick="setStat('p90')">p90 (90th %ile)</button>
          <button id="stat-avg" onclick="setStat('avg')">Mean ± Std</button>
        </div>
      </div>
      <div class="muted" id="mode-hint"></div>
    </div>
    <div id="charts" class="grid2"></div>
  </div>

  <div class="card">
    <details id="health-details">
      <summary id="health-summary">Run health &amp; build provenance</summary>
      <div class="muted" style="margin-top:8px">One cell per JobSet (hover for duration, TPU slice, and errors).</div>
      <div id="runs" style="margin-top:8px;overflow-x:auto"></div>
      <div class="muted" id="prov" style="margin-top:8px;font-size:12px"></div>
    </details>
  </div>
</div>
<div class="tip" id="tip"></div>
<script>
const ROWS = __ROWS__;
const JOBSETS = __JOBSETS__;
const COLORS = { mcjax: "#3b82f6", louhi: "#f59e0b", rapid: "#10b981" };
const PALETTE = ["#3b82f6", "#f59e0b", "#10b981", "#8b5cf6", "#ef4444", "#06b6d4"];
const RT_META = {
  mcjax: { name: "McJAX (Baseline)", desc: "Runs directly on TPU hosts" },
  louhi: { name: "Louhi (OSS Pathways)", desc: "Open-source Pathways container" },
  rapid: { name: "Rapid (Internal Pathways)", desc: "Internal release Pathways container" },
};
const BENCH_META = {
  AllGatherBenchmark: {
    title: "All-Gather · Combine data across 32 chips",
    what: "Each chip shares its piece so all 32 chips get the full tensor",
    unit: "Network speed (GB/s per chip)",
    axisTitle: "Data size per chip (bfloat16)",
  },
  AllReduceBenchmark: {
    title: "All-Reduce · Sum data across 32 chips",
    what: "Adds tensors across all 32 chips and returns the sum to every chip",
    unit: "Network speed (GB/s per chip)",
    axisTitle: "Data size per chip (bfloat16)",
  },
  AllToAllBenchmark: {
    title: "All-to-All · Shuffle data across 32 chips",
    what: "Every chip sends a distinct slice to every other chip",
    unit: "Network speed (GB/s per chip)",
    axisTitle: "Data size per chip (bfloat16)",
  },
  GeneralizedGemmBenchmark: {
    title: "Matrix Multiply (GEMM) · Compute speed",
    what: "Multiplies two bfloat16 matrices on each TPU chip's matrix unit (MXU)",
    unit: "Compute speed (TFLOP/s per chip)",
    axisTitle: "Matrix shape (M × K × N)",
  },
  HBMBandwidthBenchmark: {
    title: "HBM Memory Copy · On-chip memory speed",
    what: "Copies a large tensor inside each chip's High-Bandwidth Memory",
    unit: "Memory speed (GB/s per chip)",
    axisTitle: "Buffer size copied per chip",
  },
};
const RT = [...new Set(ROWS.map(r => r.runtime || "mcjax"))].sort((a, b) => (a === "mcjax" ? -1 : b === "mcjax" ? 1 : a.localeCompare(b)));
const COLOR = Object.fromEntries(RT.map((rt, i) => [rt, COLORS[rt] || PALETTE[i % PALETTE.length]]));
const rtName = rt => (RT_META[rt] && RT_META[rt].name) || rt;
const BENCH = [...new Set(ROWS.map(r => r.benchmark))];
const CASE_KEYS = ["matrix_dim", "num_elements", "data_size_mib", "m", "k", "n", "shard_size_mib", "dtype", "in_dtype", "out_dtype", "src_device_index", "dst_device_index", "mesh_shape", "sharding_strategy"];
const num = v => (v === "" || v == null) ? null : Number(v);
const f = (v, d = 1) => (v == null || Number.isNaN(Number(v))) ? "–" : Number(v).toFixed(d);
const fmtVal = v => v == null ? "" : (v >= 100 ? v.toFixed(0) : v >= 10 ? v.toFixed(1) : v.toFixed(2));
const fmtMib = mib => { const n = Number(mib); if (!n) return ""; return n >= 1024 ? `${(n / 1024).toFixed(n % 1024 ? 1 : 0)} GB` : `${n.toFixed(0)} MB`; };
const fmtK = n => { const v = Number(n); return v >= 1024 && v % 1024 === 0 ? `${v / 1024}K` : String(n); };
const slice = r => (r.node_pools || "").split(",").filter(p => p.startsWith("gke-tpu-")).map(p => p.replace("gke-", "")).join(", ");
const caseKey = r => CASE_KEYS.map(k => r[k] || "").join("|");
const caseLabels = r => {
  if (r.m && r.k && r.n) {
    const gflops = (2 * Number(r.m) * Number(r.k) * Number(r.n)) / 1e9;
    const ops = gflops >= 1000 ? `${(gflops / 1000).toFixed(1)} TFLOP` : `${gflops.toFixed(0)} GFLOP`;
    return { main: `${fmtK(r.m)}×${fmtK(r.k)}×${fmtK(r.n)}`, sub: ops };
  }
  if (r.shard_size_mib) {
    const perChip = fmtMib(r.shard_size_mib);
    return { main: `${perChip}/chip`, sub: r.matrix_dim ? `dim ${Number(r.matrix_dim).toLocaleString()}` : "" };
  }
  if (r.data_size_mib) return { main: fmtMib(r.data_size_mib), sub: "per chip" };
  if (r.matrix_dim) return { main: `dim ${Number(r.matrix_dim).toLocaleString()}`, sub: "" };
  if (r.num_elements) return { main: `${Number(r.num_elements).toLocaleString()} elems`, sub: "" };
  return { main: CASE_KEYS.filter(k => r[k]).map(k => `${k}=${r[k]}`).join(" ") || "(all)", sub: "" };
};
const thr = r => {
  const pairs = [
    ["xprof_tflops_per_device", "TFLOP/s", false],
    ["xprof_tflops_per_chip", "TFLOP/s", false],
    ["xprof_bandwidth_per_chip_gb_s", "GB/s", false],
    ["xprof_bandwidth_per_device_gb_s", "GB/s", false],
    ["wall_clock_tflops_per_device", "TFLOP/s (wall)", true],
    ["wall_clock_tflops_per_chip", "TFLOP/s (wall)", true],
    ["wall_clock_bandwidth_per_chip_gb_s", "GB/s (wall)", true],
    ["wall_clock_bandwidth_per_device_gb_s", "GB/s (wall)", true],
  ];
  for (const [k, u, w] of pairs) if (num(r[k]) != null) return { v: num(r[k]), u, wall: w };
  return null;
};
const tip = document.getElementById("tip");
const showTip = (e, html) => { tip.innerHTML = html; tip.style.display = "block"; tip.style.left = Math.min(e.clientX + 14, window.innerWidth - 380) + "px"; tip.style.top = (e.clientY + 14) + "px"; };
const hideTip = () => tip.style.display = "none";
const svgEl = (tag, attrs = {}, text) => { const el = document.createElementNS("http://www.w3.org/2000/svg", tag); for (const k in attrs) el.setAttribute(k, attrs[k]); if (text != null) el.textContent = text; return el; };
const hover = (el, text) => { el.addEventListener("mousemove", e => showTip(e, text)); el.addEventListener("mouseleave", hideTip); return el; };

// Legend
document.getElementById("legend").innerHTML = RT.map(rt => `<span><span class="sw" style="background:${COLOR[rt]}"></span>${rtName(rt)}</span>`).join("");

let currentMode = "thr";
let currentStat = "p50";
const STAT_LABEL = { p50: "p50", p90: "p90", avg: "mean" };
const devTime = (r, st = currentStat) => num(r[`xprof_${st}_ms`]) ?? num(r.xprof_p50_ms);
const wallTime = (r, st = currentStat) => num(r[`wall_clock_${st}_ms`]) ?? num(r.wall_clock_p50_ms);
const thrAt = (r, st = currentStat) => {
  const base = thr(r);
  if (!base || st === "p50") return base;
  // Scale achieved throughput inversely with the chosen step-time statistic (rate ∝ 1 / time).
  const t50 = base.wall ? num(r.wall_clock_p50_ms) : num(r.xprof_p50_ms);
  const tSt = base.wall ? wallTime(r, st) : devTime(r, st);
  if (t50 && tSt && tSt > 0) return { v: base.v * (t50 / tSt), u: base.u, wall: base.wall };
  return base;
};

// Top KPI summary cards
function renderKpis() {
  const st = STAT_LABEL[currentStat];
  const baseByKey = {};
  ROWS.filter(r => (r.runtime || "mcjax") === "mcjax").forEach(r => baseByKey[r.benchmark + "|" + caseKey(r)] = r);
  const kpiEl = document.getElementById("kpis");
  kpiEl.innerHTML = "";
  RT.forEach(rt => {
    const rtRows = ROWS.filter(r => (r.runtime || "mcjax") === rt);
    const meta = RT_META[rt] || { name: rt, desc: "" };
    let headline = `${rtRows.length} cases measured`;
    let badgeText = "Baseline";
    const ratios = [];
    const hostGaps = [];
    const proxyDeltas = [];
    rtRows.forEach(r => {
      const b = baseByKey[r.benchmark + "|" + caseKey(r)];
      const tRt = thrAt(r), tBase = b ? thrAt(b) : null;
      if (tRt && tBase && tBase.v > 0 && !tRt.wall && !tBase.wall) ratios.push(tRt.v / tBase.v);
      const dev = devTime(r), wall = wallTime(r);
      if (dev != null && wall != null && wall >= dev) {
        const gap = wall - dev;
        hostGaps.push(gap);
        if (rt !== "mcjax" && b) {
          const bDev = devTime(b), bWall = wallTime(b);
          if (bDev != null && bWall != null && bWall >= bDev) {
            proxyDeltas.push(gap - (bWall - bDev));
          }
        }
      }
    });
    if (rt !== "mcjax" && ratios.length) {
      const geo = Math.exp(ratios.reduce((s, x) => s + Math.log(x), 0) / ratios.length);
      const pct = Math.round(geo * 100);
      headline = `${pct}% of McJAX chip speed (${st})`;
      badgeText = pct >= 97 ? "Matches McJAX" : `${pct}% vs McJAX`;
    }
    let sub = meta.desc;
    let tipText = `<b>${meta.name}</b>\n${meta.desc}`;
    const medOf = arr => { const s = [...arr].sort((a, b) => a - b); return s[Math.floor(s.length / 2)]; };
    if (rt === "mcjax" && hostGaps.length) {
      const medGap = medOf(hostGaps);
      sub = `${meta.desc} (no proxy) · +${medGap.toFixed(1)} ms local host sync (${st})`;
      tipText += `\n\n<b>Local host sync (no proxy in McJAX):</b>\nhost_sync = McJAX Wall-Clock (${st}) − McJAX Chip XProf (${st})\nMedian across ${hostGaps.length} cases: +${medGap.toFixed(2)} ms (Python block_until_ready + PJRT host callback)`;
    } else if (rt !== "mcjax" && hostGaps.length) {
      const medGap = medOf(hostGaps);
      const medProxy = proxyDeltas.length ? medOf(proxyDeltas) : medGap;
      sub = `${meta.desc} · +${medProxy.toFixed(1)} ms proxy hop over McJAX (+${medGap.toFixed(1)} ms total wall−chip, ${st})`;
      tipText += `\n\n<b>How the Pathways proxy hop is calculated (${st}):</b>\n1. Total host+proxy gap = Wall-Clock (${st}) − Chip XProf (${st}) (median +${medGap.toFixed(2)} ms)\n2. Subtract McJAX's local host sync on the same case:\n   proxy_hop = (Pathways Wall − Chip) − (McJAX Wall − Chip)\n   Median added proxy hop across ${proxyDeltas.length} cases: <b>+${medProxy.toFixed(2)} ms</b>`;
    }
    const div = document.createElement("div");
    div.className = "kpi";
    div.innerHTML = `<div class="kpi-head"><span><span class="sw" style="background:${COLOR[rt]}"></span>${meta.name}</span><span class="badge">${badgeText}</span></div><div class="kpi-val">${headline}</div><div class="muted" style="font-size:12px">${sub}</div>`;
    hover(div, tipText);
    kpiEl.appendChild(div);
  });
}

// Provenance & JobSet strip
(function(){
  const by = {};
  for (const r of ROWS) if (!by[r.runtime || "mcjax"]) by[r.runtime || "mcjax"] = r;
  const parts = RT.filter(rt => by[rt]).map(rt => {
    const r = by[rt];
    const d = (r.pathways_server_digest || "").replace("sha256:", "").slice(0, 12);
    const c = (r.client_image_digest || "").replace("sha256:", "").slice(0, 12);
    return `<b>${rt}</b>: ${[c && "client " + c, d && "server " + d, r.git_commit && "commit " + String(r.git_commit).slice(0, 12)].filter(Boolean).join(", ")}`;
  });
  const slices = [...new Set(ROWS.map(slice).filter(Boolean))];
  document.getElementById("prov").innerHTML = parts.join(" · ") + (slices.length ? ". Slices: " + slices.join(", ") : "");

  const el = document.getElementById("runs");
  const ok = JOBSETS.filter(j => j.status === "SUCCESS").length;
  const sumEl = document.getElementById("health-summary");
  if (JOBSETS.length) {
    sumEl.textContent = `Run health & build provenance (${ok}/${JOBSETS.length} JobSets succeeded)`;
    if (ok < JOBSETS.length) document.getElementById("health-details").open = true;
  }
  if (!JOBSETS.length) { el.innerHTML = '<span class="muted">(no JobSets in this run)</span>'; return; }
  const STATUS = { SUCCESS: "#16a34a", FAILED: "#dc2626", TIMEOUT: "#dc2626", DRY_RUN: "#94a3b8" };
  const cw = 52, H = 68, W = Math.max(JOBSETS.length * cw, 200);
  const svg = svgEl("svg", { width: W, height: H, viewBox: `0 0 ${W} ${H}` });
  JOBSETS.forEach((j, i) => {
    const x = i * cw;
    const dur = j.duration_seconds ? `${Math.floor(j.duration_seconds / 60)}m ${Math.round(j.duration_seconds % 60)}s` : "";
    svg.appendChild(hover(svgEl("rect", { x: x + 2, y: 4, width: cw - 6, height: 24, rx: 5, fill: STATUS[j.status] || "#999" }), `<b>${j.runtime}/${j.config}</b> — ${j.status}${dur ? " in " + dur : ""}\n${j.workload}${j.slice ? "\nslice: " + j.slice : ""}${j.error ? "\n" + j.error : ""}`));
    svg.appendChild(svgEl("text", { x: x + cw / 2 - 1, y: 42, class: "lbl", "text-anchor": "middle" }, j.config.length > 8 ? j.config.slice(0, 7) + "…" : j.config));
    svg.appendChild(svgEl("text", { x: x + cw / 2 - 1, y: 56, class: "lbl", "text-anchor": "middle" }, j.runtime));
  });
  el.appendChild(svg);
})();

const MODE_HINTS = {
  thr: "Showing on-chip hardware speed from XProf traces. Higher bars = faster hardware execution.",
  dev: "Showing pure TPU chip execution time (XProf). Lower bars = faster.",
  wall: "Solid bottom = TPU chip time (XProf); striped top = Wall − Chip (local host sync in McJAX; host sync + gRPC proxy hop in Louhi/Rapid).",
};

function renderPanel(bench, mode) {
  const st = STAT_LABEL[currentStat];
  const rows = ROWS.filter(r => r.benchmark === bench);
  const meta = BENCH_META[bench] || {
    title: bench.replace(/Benchmark$/, ""),
    what: "",
    unit: (rows.map(thr).find(Boolean) || { u: "Speed" }).u,
    axisTitle: "Workload size",
  };
  const keys = [...new Set(rows.map(caseKey))];
  const labels = {};
  const sortNum = {};
  rows.forEach(r => {
    const k = caseKey(r);
    labels[k] = caseLabels(r);
    sortNum[k] = Number(r.shard_size_mib || r.data_size_mib || r.m || r.matrix_dim || r.num_elements || 0);
  });
  keys.sort((a, b) => sortNum[a] - sortNum[b]);

  const W = 520, H = 240, left = 48, right = 12, top = 22, bottom = 52;
  const svg = svgEl("svg", { width: "100%", viewBox: `0 0 ${W} ${H}` });
  const defs = svgEl("defs");
  RT.forEach(rt => {
    const p = svgEl("pattern", { id: `stripe-${bench}-${rt}`, width: 6, height: 6, patternUnits: "userSpaceOnUse", patternTransform: "rotate(45)" });
    p.appendChild(svgEl("rect", { width: 6, height: 6, fill: COLOR[rt], opacity: 0.22 }));
    p.appendChild(svgEl("line", { x1: 0, y1: 0, x2: 0, y2: 6, stroke: COLOR[rt], "stroke-width": 2 }));
    defs.appendChild(p);
  });
  svg.appendChild(defs);

  const valOf = r => {
    if (mode === "thr") { const t = thrAt(r); return t ? t.v : null; }
    if (mode === "dev") return devTime(r) ?? wallTime(r);
    return wallTime(r);
  };
  const stdOf = r => {
    if (currentStat !== "avg" || mode === "thr") return null;
    return mode === "dev" ? num(r.xprof_std_ms) : num(r.wall_clock_std_ms);
  };
  let max = 0;
  rows.forEach(r => {
    const v = valOf(r), sd = stdOf(r) || 0;
    if (v != null && v + sd > max) max = v + sd;
  });
  if (!max) max = 1;
  const yMax = max * 1.18;
  const yv = v => top + (H - top - bottom) * (1 - v / yMax);
  const gw = (W - left - right) / Math.max(keys.length, 1);

  for (let t = 0; t <= 4; t++) {
    const v = (max * t) / 4;
    svg.appendChild(svgEl("line", { x1: left, x2: W - right, y1: yv(v), y2: yv(v), class: "grid" }));
    svg.appendChild(svgEl("text", { x: left - 6, y: yv(v) + 4, class: "lbl", "text-anchor": "end" }, fmtVal(v)));
  }
  svg.appendChild(svgEl("line", { x1: left, x2: W - right, y1: yv(0), y2: yv(0), class: "axis" }));

  keys.forEach((key, i) => {
    const gx = left + i * gw;
    const cx = gx + gw / 2;
    svg.appendChild(svgEl("text", { x: cx, y: H - 30, class: "lbl-x", "text-anchor": "middle" }, labels[key].main));
    if (labels[key].sub) {
      svg.appendChild(svgEl("text", { x: cx, y: H - 17, class: "lbl-xsub", "text-anchor": "middle" }, labels[key].sub));
    }
    const pad = Math.min(22, gw * 0.16);
    const bw = (gw - pad * 2) / Math.max(RT.length, 1);
    RT.forEach((rt, j) => {
      const r = rows.find(q => (q.runtime || "mcjax") === rt && caseKey(q) === key);
      if (!r) return;
      const bx = gx + pad + j * bw;
      const barW = Math.max(bw - 4, 4);
      const v = valOf(r);
      if (v == null) return;
      const dev = devTime(r), wall = wallTime(r), t = thrAt(r), sd = stdOf(r);
      const overhead = (dev != null && wall != null && wall > dev) ? (wall - dev) : null;
      const baseRow = rows.find(q => (q.runtime || "mcjax") === "mcjax" && caseKey(q) === key);
      const baseGap = (baseRow && devTime(baseRow) != null && wallTime(baseRow) != null)
        ? (wallTime(baseRow) - devTime(baseRow)) : null;
      const proxyHop = (rt !== "mcjax" && overhead != null && baseGap != null) ? (overhead - baseGap) : null;
      const overheadLine = overhead == null ? "" : (
        rt === "mcjax"
          ? `\nLocal host sync (${st} wall − chip, no proxy): <b>+${f(overhead, 2)} ms</b> (${f(wall, 2)} − ${f(dev, 2)} ms)`
          : `\nTotal host + proxy gap (${st} wall − chip): <b>+${f(overhead, 2)} ms</b> (${f(wall, 2)} − ${f(dev, 2)} ms)` +
            (proxyHop != null ? `\nAdded Pathways proxy hop vs McJAX (${st}): <b>+${f(proxyHop, 2)} ms</b> (${f(overhead, 2)} − ${f(baseGap, 2)} ms)` : "")
      );
      const devStats = num(r.xprof_p50_ms) != null
        ? `\nTPU chip (p50 / p90 / mean±std): <b>${f(r.xprof_p50_ms, 2)}</b> / <b>${f(r.xprof_p90_ms, 2)}</b> / <b>${f(r.xprof_avg_ms ?? r.xprof_p50_ms, 2)}±${f(r.xprof_std_ms ?? 0, 2)} ms</b>`
        : "\nTPU chip time: not captured";
      const wallStats = num(r.wall_clock_p50_ms) != null
        ? `\nClient wall (p50 / p90 / mean±std): <b>${f(r.wall_clock_p50_ms, 2)}</b> / <b>${f(r.wall_clock_p90_ms, 2)}</b> / <b>${f(r.wall_clock_avg_ms, 2)}±${f(r.wall_clock_std_ms, 2)} ms</b>`
        : "";
      const tt = `<b>${meta.title.split(" · ")[0]} · ${rtName(rt)}</b>\nSize: ${labels[key].main} ${labels[key].sub ? "(" + labels[key].sub + ")" : ""}` +
        (t ? `\nHardware speed (${st}): <b>${f(t.v, 1)} ${t.u}</b>` : "") +
        devStats + wallStats + overheadLine +
        (slice(r) ? `\nTPU slice: ${slice(r)}` : "");

      if (mode === "wall" && dev != null && wall != null && wall > dev) {
        svg.appendChild(hover(svgEl("rect", { x: bx + 2, y: yv(wall), width: barW, height: yv(dev) - yv(wall), fill: `url(#stripe-${bench}-${rt})`, stroke: COLOR[rt], "stroke-width": 1, rx: 3 }), tt));
        svg.appendChild(hover(svgEl("rect", { x: bx + 2, y: yv(dev), width: barW, height: yv(0) - yv(dev), fill: COLOR[rt], rx: 2 }), tt));
      } else {
        const isWallFallback = (mode === "thr" && t && t.wall) || (mode === "dev" && dev == null);
        svg.appendChild(hover(svgEl("rect", { x: bx + 2, y: yv(v), width: barW, height: Math.max(yv(0) - yv(v), 2), fill: isWallFallback ? `url(#stripe-${bench}-${rt})` : COLOR[rt], stroke: COLOR[rt], "stroke-width": isWallFallback ? 1 : 0, rx: 3 }), tt));
      }
      if (sd != null && sd > 0) {
        const cxBar = bx + 2 + barW / 2;
        const yHi = yv(v + sd), yLo = yv(Math.max(0, v - sd));
        svg.appendChild(svgEl("line", { x1: cxBar, x2: cxBar, y1: yHi, y2: yLo, stroke: "var(--fg)", "stroke-width": 1.3 }));
        svg.appendChild(svgEl("line", { x1: cxBar - 3, x2: cxBar + 3, y1: yHi, y2: yHi, stroke: "var(--fg)", "stroke-width": 1.3 }));
        svg.appendChild(svgEl("line", { x1: cxBar - 3, x2: cxBar + 3, y1: yLo, y2: yLo, stroke: "var(--fg)", "stroke-width": 1.3 }));
      }
      const lblY = sd != null && sd > 0 ? Math.max(yv(v + sd) - 4, 11) : Math.max(yv(v) - 5, 11);
      svg.appendChild(svgEl("text", { x: bx + 2 + barW / 2, y: lblY, class: "val-lbl", "text-anchor": "middle" }, fmtVal(v)));
    });
  });
  svg.appendChild(svgEl("text", { x: left + (W - left - right) / 2, y: H - 3, class: "lbl", "text-anchor": "middle" }, meta.axisTitle));

  const unitLabel = mode === "thr" ? `${meta.unit} (${st})` : mode === "dev" ? `TPU chip time (ms, ${st})` : `Client wall-clock (${st}) = solid chip + striped host/proxy gap (ms)`;
  const dirLabel = mode === "thr" ? "↑ Higher is better" : "↓ Lower is better";
  const wrap = document.createElement("div");
  wrap.className = "panel";
  wrap.innerHTML = `<div class="panel-head"><span class="panel-title">${meta.title}</span><span class="dir">${dirLabel}</span></div><div class="panel-sub">${meta.what ? meta.what + " · " : ""}<b>${unitLabel}</b></div>`;
  wrap.appendChild(svg);
  return wrap;
}

function refreshAll() {
  for (const id of ["thr", "dev", "wall"]) {
    document.getElementById("btn-" + id).classList.toggle("active", id === currentMode);
  }
  for (const st of ["p50", "p90", "avg"]) {
    document.getElementById("stat-" + st).classList.toggle("active", st === currentStat);
  }
  document.getElementById("mode-hint").textContent = MODE_HINTS[currentMode];
  renderKpis();
  const container = document.getElementById("charts");
  container.innerHTML = "";
  for (const b of BENCH) container.appendChild(renderPanel(b, currentMode));
}
function setMode(m) { currentMode = m; refreshAll(); }
function setStat(s) { currentStat = s; refreshAll(); }
refreshAll();
</script>
</body>
</html>
"""

_ROW_KEYS = (
    "runtime", "benchmark", "matrix_dim", "num_elements", "data_size_mib",
    "m", "k", "n", "in_dtype", "out_dtype", "dtype", "shard_size_mib",
    "mesh_shape", "sharding_strategy", "src_device_index", "dst_device_index",
    "xprof_p50_ms", "xprof_p90_ms", "xprof_avg_ms", "xprof_std_ms",
    "wall_clock_p50_ms", "wall_clock_p90_ms",
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
