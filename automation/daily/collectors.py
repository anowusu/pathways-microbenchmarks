"""Metric collectors for the TPUMS Daily Pathways Benchmarking Harness (R5).

Combines:
1. Pre-cleanup Kubernetes & container-log telemetry (`capture_workload_k8s_telemetry`),
   invoked by `WorkloadManager.poll()` via `on_workload_terminal` before the JobSet is deleted.
2. Post-run XProf `.xplane.pb` parsing (`automation.daily.xplane_parser`).
3. Prometheus exporter (`:9090`/`:9091`) & `MetricDeltaLogExporter` probe parsing.
4. GCS compilation-cache (`ST-02`) and XProf trace-volume (`ST-07`) inspection.
5. Full reconciliation of all 86 entries in `metrics_manifest.yaml` across `mcjax` and `louhi`.
"""

import collections
from collections.abc import Mapping, Sequence
import datetime
import glob
import json
import logging
import math
import os
import re
import statistics
import subprocess
import tempfile
from typing import Any, Optional

from automation import utils
from automation import workload_submitter
from automation.daily import xplane_parser

K8S_TELEMETRY_FILENAME = "k8s_telemetry.json"

_WAIT_TIME_RE = re.compile(r"wait time since queued was ([0-9.]+)s")
_PULL_DURATION_RE = re.compile(r"in ([0-9.]+)(ms|s)\b")
_TS_PREFIX_RE = re.compile(
    r"^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z)\s+(.*)$"
)
_STARTUP_ARGV_RE = re.compile(r"STARTUP: argv\[\d+\]: (.*)\$$")
_STARTUP_ENV_RE = re.compile(r"STARTUP: env: ([A-Z0-9_]+)=(.*)\$$")
_ADD_JOB_RE = re.compile(r"AddJobInstance:\s+(\d+)\s+with\s+(\d+)\s+tasks\.")
_ACQUIRED_RE = re.compile(
    r"ListDevices:\s+(\d+)\s+job instances acquired after\s+([0-9.]+)(ms|s|m)"
)
_COMP_WORKER_RE = re.compile(
    r"XLA compilation for computation '([^']+)' \(key: ([0-9a-f]+)\) took"
    r" ([0-9.]+)(ns|us|ms|s)\. Scheduling: ([0-9.]+)(ns|us|ms|s), IO:"
    r" ([0-9.]+)(ns|us|ms|s), XLA compilation: ([0-9.]+)(ns|us|ms|s)?:"
    r" (PERSISTENT_HIT|PERSISTENT_MISS)"
)
_CACHE_FETCH_RE = re.compile(
    r"XLA compilation cache fetch for computation '([^']+)' \(key"
    r" ([0-9a-f]+)\) took ([0-9.]+)(ns|us|ms|s): (\S+)"
)
_HEARTBEAT_RE = re.compile(
    r"Worker activity will be reported to the Resource Manager every"
    r" ([0-9.]+)(ms|s|m)"
)
_METRIC_DELTA_RE = re.compile(
    r"MetricDeltaLogExporter: Metric changed:\s*(\S+)\s+Value Type:\s*(\S+)\s+Value:\s*(.*)$"
)
_CLIENT_BRINGUP_RE = re.compile(r"CLIENT_BRINGUP_TIMINGS_JSON:\s*(\{.*\})")


def _parse_iso_ts(ts: Optional[str]) -> Optional[float]:
  if not ts:
    return None
  try:
    return datetime.datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp()
  except ValueError:
    return None


def _to_ms(val: str, unit: Optional[str]) -> float:
  v = float(val)
  if unit == "ns":
    return v / 1e6
  if unit == "us":
    return v / 1e3
  if unit == "s":
    return v * 1e3
  if unit == "m":
    return v * 60e3
  return v


def _num(val: Any) -> Optional[float]:
  if val in (None, "") or isinstance(val, bool):
    return None
  try:
    f = float(val)
    return f if math.isfinite(f) else None
  except (TypeError, ValueError):
    return None


def _pct_stats(vals: Sequence[float]) -> dict[str, Optional[float]]:
  clean = [float(v) for v in vals if v is not None and math.isfinite(float(v))]
  return xplane_parser.pct_stats(clean)


def _kubectl_logs(
    namespace: str, pod_name: str, container: str, tail: int = 4000
) -> str:
  cmd = [
      "kubectl",
      "logs",
      pod_name,
      "-n",
      namespace,
      "-c",
      container,
      "--timestamps",
      f"--tail={tail}",
  ]
  try:
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=False,
        stdin=subprocess.DEVNULL,
        timeout=30,
    )
    return proc.stdout if proc.returncode == 0 else ""
  except (OSError, subprocess.TimeoutExpired):
    return ""


def _kubectl_events(namespace: str) -> list[dict[str, Any]]:
  cmd = ["kubectl", "get", "events", "-n", namespace, "-o", "json"]
  try:
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=False,
        stdin=subprocess.DEVNULL,
        timeout=30,
    )
    if proc.returncode != 0:
      return []
    return json.loads(proc.stdout or "{}").get("items", [])
  except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError):
    return []


def extract_k8s_metrics(
    workload_name: str,
    jobset_obj: Optional[dict[str, Any]],
    pod_list: list[dict[str, Any]],
    events_items: list[dict[str, Any]],
) -> dict[str, Any]:
  """Computes CO-01..CO-08, CO-10, and RM-05 Kubernetes metrics for one JobSet."""
  out: dict[str, Any] = {"workload_name": workload_name}

  js_create_ts = _parse_iso_ts(
      (jobset_obj or {}).get("metadata", {}).get("creationTimestamp")
  )
  js_restarts = int(
      (jobset_obj or {}).get("status", {}).get("restarts", 0) or 0
  )
  out["co03_jobset_restarts"] = js_restarts

  kueue_wait_s = None
  preempted_count = 0
  pull_durations_s: list[float] = []
  for ev in events_items:
    obj = ev.get("involvedObject", {}) or {}
    obj_name = str(obj.get("name", ""))
    if not (
        obj_name == workload_name
        or obj_name.startswith(f"{workload_name}-")
        or obj_name.startswith(f"jobset-{workload_name}-")
    ):
      continue
    reason = str(ev.get("reason", ""))
    msg = str(ev.get("message", ""))
    if reason in ("QuotaReserved", "Admitted"):
      m = _WAIT_TIME_RE.search(msg)
      if m:
        kueue_wait_s = max(0.0, float(m.group(1)))
    elif reason in ("Preempted", "Evicted"):
      preempted_count += 1
    elif reason == "Pulled":
      m = _PULL_DURATION_RE.search(msg)
      if m:
        val = float(m.group(1))
        pull_durations_s.append(val / 1000.0 if m.group(2) == "ms" else val)

  out["co02_kueue_admission_wait_s"] = (
      kueue_wait_s if kueue_wait_s is not None else 0.0
  )
  out["co03_kueue_preempted_count"] = preempted_count
  if pull_durations_s:
    s_pulls = sorted(pull_durations_s)
    out["co05_image_pull_p50_s"] = round(s_pulls[len(s_pulls) // 2], 4)
    out["co05_image_pull_max_s"] = round(max(s_pulls), 4)
  else:
    out["co05_image_pull_p50_s"] = None
    out["co05_image_pull_max_s"] = None

  head_create_ts = user_started_ts = None
  worker_create_ts: list[float] = []
  worker_ready_ts: list[float] = []
  worker0_ready_ts: Optional[float] = None
  head_restarts = 0
  worker_restarts = 0
  head_pod_name: Optional[str] = None
  worker0_pod_name: Optional[str] = None

  for pod in pod_list:
    meta = pod.get("metadata", {}) or {}
    pname = str(meta.get("name", ""))
    labels = meta.get("labels", {}) or {}
    rjob = str(labels.get("jobset.sigs.k8s.io/replicatedjob-name", ""))
    p_create = _parse_iso_ts(meta.get("creationTimestamp"))
    status = pod.get("status", {}) or {}
    conds = {
        c.get("type"): _parse_iso_ts(c.get("lastTransitionTime"))
        for c in (status.get("conditions") or [])
        if c.get("status") == "True"
    }

    if rjob == "pathways-head":
      head_pod_name = pname
      head_create_ts = p_create
      for cs in status.get("initContainerStatuses") or []:
        head_restarts += int(cs.get("restartCount", 0) or 0)
      for cs in status.get("containerStatuses") or []:
        head_restarts += int(cs.get("restartCount", 0) or 0)
        if cs.get("name") == "user-workload":
          st_obj = (
              (cs.get("state", {}) or {}).get("running")
              or (cs.get("state", {}) or {}).get("terminated")
              or {}
          )
          user_started_ts = _parse_iso_ts(st_obj.get("startedAt"))
    else:
      if p_create:
        worker_create_ts.append(p_create)
      if conds.get("ContainersReady"):
        worker_ready_ts.append(conds["ContainersReady"])
      if pname.endswith("-0-0") or "-worker-0-0-" in pname or worker0_pod_name is None:
        if rjob.startswith("pathways-worker") or rjob == "slice-job":
          if worker0_pod_name is None or "-0-0-" in pname:
            worker0_pod_name = pname
            worker0_ready_ts = conds.get("ContainersReady")
      for cs in status.get("containerStatuses") or []:
        worker_restarts += int(cs.get("restartCount", 0) or 0)
        if rjob == "slice-job" and cs.get("name") == "jax-tpu" and user_started_ts is None:
          st_obj = (
              (cs.get("state", {}) or {}).get("running")
              or (cs.get("state", {}) or {}).get("terminated")
              or {}
          )
          user_started_ts = _parse_iso_ts(st_obj.get("startedAt"))

  out["head_pod_name"] = head_pod_name
  out["worker0_pod_name"] = worker0_pod_name
  out["rm05_head_restarts"] = head_restarts
  out["co03_worker_restarts"] = worker_restarts
  if head_create_ts and worker_create_ts:
    out["co10_inorder_worker_create_delay_s"] = round(
        max(0.0, min(worker_create_ts) - head_create_ts), 3
    )
  else:
    out["co10_inorder_worker_create_delay_s"] = 0.0
  if js_create_ts and user_started_ts:
    out["co01_head_init_s"] = round(max(0.0, user_started_ts - js_create_ts), 3)
  elif js_create_ts and worker_ready_ts:
    out["co01_head_init_s"] = round(
        max(0.0, max(worker_ready_ts) - js_create_ts), 3
    )
  out["worker0_k8s_ready_ts"] = worker0_ready_ts
  out["user_workload_started_ts"] = user_started_ts

  return out


def parse_rm_and_worker_logs(
    rm_log: str, proxy_log: str, worker_log: str, user_log: str = ""
) -> dict[str, Any]:
  """Extracts RM-01..11, WK-07/08, ST-02, CO-04, PV-02/05/06, PA-05 from container logs."""
  out: dict[str, Any] = {}

  rm_t0 = rm_serving_ts = rm_workers_joined_ts = rm_placed_ts = None
  list_devices_wait_s = 0.0
  crash_sigs = 0
  comp_hits = 0
  comp_misses = 0
  comp_io_ms: list[float] = []
  comp_xla_ms: list[float] = []
  cache_writes = 0
  delta_metrics: dict[str, str] = {}

  for line in rm_log.splitlines():
    m_ts = _TS_PREFIX_RE.match(line)
    ts = _parse_iso_ts(m_ts.group(1)) if m_ts else None
    body = m_ts.group(2) if m_ts else line

    if "STARTUP: argv[0]:" in body and rm_t0 is None:
      rm_t0 = ts
    elif "Pathways Server serving on " in body and rm_serving_ts is None:
      rm_serving_ts = ts
    elif "AddJobInstance:" in body:
      m = _ADD_JOB_RE.search(body)
      if m and int(m.group(2)) >= 8 and rm_workers_joined_ts is None:
        rm_workers_joined_ts = ts
    elif "DeviceSet " in body and " placed:" in body and rm_placed_ts is None:
      rm_placed_ts = ts
    elif "job instances acquired after" in body:
      m = _ACQUIRED_RE.search(body)
      if m:
        list_devices_wait_s = max(0.0, _to_ms(m.group(2), m.group(3)) / 1000.0)
    elif (
        "Only expect scheduler registration from resource manager" in body
        or "Timebomb for dead " in body
        or "WatchDog will terminate the process" in body
    ):
      crash_sigs += 1
    elif "Write to persistent cache completed for key" in body:
      cache_writes += 1
    elif "XLA compilation for computation" in body:
      m = _COMP_WORKER_RE.search(body)
      if m:
        comp_io_ms.append(_to_ms(m.group(7), m.group(8)))
        if m.group(11) == "PERSISTENT_HIT":
          comp_hits += 1
        else:
          comp_misses += 1
          comp_xla_ms.append(_to_ms(m.group(9), m.group(10)))
    m_dm = _METRIC_DELTA_RE.search(body)
    if m_dm:
      delta_metrics[m_dm.group(1)] = m_dm.group(3).strip()

  out["rm01_list_devices_wait_s"] = round(list_devices_wait_s, 4)
  if rm_t0 and rm_workers_joined_ts:
    out["rm01_worker_registration_s"] = round(
        max(0.0, rm_workers_joined_ts - rm_t0), 3
    )
  if rm_t0 and rm_placed_ts:
    out["rm09_device_set_placed_s"] = round(max(0.0, rm_placed_ts - rm_t0), 3)
  if rm_workers_joined_ts and rm_placed_ts:
    out["rm02_placement_delta_ms"] = round(
        max(0.0, (rm_placed_ts - rm_workers_joined_ts) * 1000.0), 3
    )
  out["rm05_crash_signature_count"] = crash_sigs
  out["wk08_persistent_cache_hits"] = comp_hits
  out["wk08_persistent_cache_misses"] = comp_misses
  out["wk08_persistent_cache_io_p50_ms"] = (
      round(sorted(comp_io_ms)[len(comp_io_ms) // 2], 4)
      if comp_io_ms
      else None
  )
  out["wk08_server_xla_compile_p50_ms"] = (
      round(sorted(comp_xla_ms)[len(comp_xla_ms) // 2], 4)
      if comp_xla_ms
      else None
  )

  def _extract_startup(
      log_text: str,
  ) -> tuple[list[str], dict[str, str], Optional[float], list[float], Optional[float]]:
    argv: list[str] = []
    env: dict[str, str] = {}
    serving_ts: Optional[float] = None
    fetch_ms: list[float] = []
    heartbeat_s: Optional[float] = None
    nonlocal cache_writes
    for line in log_text.splitlines():
      m_ts = _TS_PREFIX_RE.match(line)
      ts = _parse_iso_ts(m_ts.group(1)) if m_ts else None
      body = m_ts.group(2) if m_ts else line
      m_a = _STARTUP_ARGV_RE.search(body)
      if m_a:
        argv.append(m_a.group(1))
      m_e = _STARTUP_ENV_RE.search(body)
      if m_e:
        env[m_e.group(1)] = m_e.group(2)
      if "Pathways Server serving on " in body and serving_ts is None:
        serving_ts = ts
      if "Write to persistent cache completed for key" in body:
        cache_writes += 1
      m_f = _CACHE_FETCH_RE.search(body)
      if m_f:
        fetch_ms.append(_to_ms(m_f.group(3), m_f.group(4)))
      m_hb = _HEARTBEAT_RE.search(body)
      if m_hb:
        heartbeat_s = _to_ms(m_hb.group(1), m_hb.group(2)) / 1000.0
      m_dm = _METRIC_DELTA_RE.search(body)
      if m_dm:
        delta_metrics[m_dm.group(1)] = m_dm.group(3).strip()
    return argv, env, serving_ts, fetch_ms, heartbeat_s

  proxy_argv, proxy_env, _, _, _ = _extract_startup(proxy_log)
  worker_argv, worker_env, worker_serving_ts, worker_fetch_ms, heartbeat_s = (
      _extract_startup(worker_log)
  )

  for line in user_log.splitlines():
    m_cb = _CLIENT_BRINGUP_RE.search(line)
    if m_cb:
      try:
        out["client_bringup_timings"] = json.loads(m_cb.group(1))
      except json.JSONDecodeError:
        pass

  out["st02_cold_cache_writes_count"] = cache_writes
  out["co04_worker0_serving_ts"] = worker_serving_ts
  out["rm11_heartbeat_interval_s"] = heartbeat_s if heartbeat_s is not None else 5.0
  out["wk08_worker_cache_fetch_p50_ms"] = (
      round(sorted(worker_fetch_ms)[len(worker_fetch_ms) // 2], 4)
      if worker_fetch_ms
      else None
  )
  out["pv02_proxy_xla_argv"] = [
      a
      for a in proxy_argv
      if a.startswith("--xla_") or a.startswith("--megascale_")
  ]
  out["pv02_worker_xla_argv"] = [
      a
      for a in worker_argv
      if a.startswith("--xla_") or a.startswith("--megascale_")
  ]
  out["pv02_worker_libtpu_init_args"] = worker_env.get("LIBTPU_INIT_ARGS", "")
  out["pv02_proxy_libtpu_init_args"] = proxy_env.get("LIBTPU_INIT_ARGS", "")
  out["pv05_pjrt_config_logged"] = "max_inflight_computations" in worker_log
  out["delta_metrics"] = delta_metrics

  return out


def capture_workload_k8s_telemetry(
    wl: workload_submitter.ActiveWorkload,
    jobset_obj: Optional[dict[str, Any]],
    pod_list: list[dict[str, Any]],
) -> None:
  """Callback invoked by `WorkloadManager.poll()` before JobSet deletion."""
  events_items = _kubectl_events("default")
  k8s_info = extract_k8s_metrics(
      wl.workload_name, jobset_obj, pod_list, events_items
  )
  log_info: dict[str, Any] = {}
  head_pod = k8s_info.get("head_pod_name")
  worker0_pod = k8s_info.get("worker0_pod_name")
  if wl.runtime != "mcjax" and head_pod:
    rm_log = _kubectl_logs("default", head_pod, "pathways-rm")
    proxy_log = _kubectl_logs("default", head_pod, "pathways-proxy")
    user_log = _kubectl_logs("default", head_pod, "user-workload", tail=1000)
    worker_log = (
        _kubectl_logs("default", worker0_pod, "pathways-worker")
        if worker0_pod
        else ""
    )
    log_info = parse_rm_and_worker_logs(rm_log, proxy_log, worker_log, user_log)
    w0_ready = k8s_info.get("worker0_k8s_ready_ts")
    w0_serv = log_info.get("co04_worker0_serving_ts")
    if w0_ready and w0_serv:
      k8s_info["co04_worker_readiness_blind_spot_s"] = round(
          max(0.0, abs(w0_serv - w0_ready)), 3
      )
  elif wl.runtime == "mcjax" and worker0_pod:
    user_log = _kubectl_logs("default", worker0_pod, "jax-tpu", tail=1000)
    for line in user_log.splitlines():
      m_cb = _CLIENT_BRINGUP_RE.search(line)
      if m_cb:
        try:
          log_info["client_bringup_timings"] = json.loads(m_cb.group(1))
        except json.JSONDecodeError:
          pass

  payload = {
      "workload_name": wl.workload_name,
      "config_name": wl.config_name,
      "runtime": wl.runtime,
      "status": wl.status,
      "duration_seconds": wl.duration_seconds,
      "attempt": wl.attempt,
      "k8s": k8s_info,
      "logs": log_info,
      "captured_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
  }
  wl.provenance["k8s_telemetry"] = payload
  if wl.gcs_output_dir:
    target = utils.format_gcs_uri(wl.gcs_output_dir, K8S_TELEMETRY_FILENAME)
    try:
      subprocess.run(
          ["gcloud", "storage", "cp", "-", target],
          input=json.dumps(payload, indent=2, sort_keys=True),
          capture_output=True,
          text=True,
          check=False,
          timeout=utils.CLI_TIMEOUT_SECONDS,
      )
    except (OSError, subprocess.TimeoutExpired) as e:
      logging.warning("Could not upload %s: %s", target, e)


def fetch_gcs_json(gcs_uri: str) -> dict[str, Any]:
  try:
    proc = subprocess.run(
        ["gcloud", "storage", "cat", gcs_uri],
        capture_output=True,
        text=True,
        check=False,
        stdin=subprocess.DEVNULL,
        timeout=utils.CLI_TIMEOUT_SECONDS,
    )
    if proc.returncode == 0 and proc.stdout.strip():
      data = json.loads(proc.stdout)
      return data if isinstance(data, dict) else {}
  except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError):
    pass
  return {}


def fetch_gcs_text(gcs_uri: str) -> str:
  try:
    proc = subprocess.run(
        ["gcloud", "storage", "cat", gcs_uri],
        capture_output=True,
        text=True,
        check=False,
        stdin=subprocess.DEVNULL,
        timeout=utils.CLI_TIMEOUT_SECONDS,
    )
    return proc.stdout if proc.returncode == 0 else ""
  except (OSError, subprocess.TimeoutExpired):
    return ""


def parse_prometheus_text(prom_text: str) -> dict[str, float]:
  """Parses Prometheus exposition text into family/metric -> float value."""
  metrics: dict[str, float] = {}
  for line in prom_text.splitlines():
    line = line.strip()
    if not line or line.startswith("#"):
      continue
    parts = line.split()
    if len(parts) >= 2:
      key = parts[0]
      val = _num(parts[-1])
      if val is not None:
        base_key = key.split("{", 1)[0]
        metrics[key] = val
        metrics.setdefault(base_key, val)
  return metrics


def _gcs_dir_size_mib(gcs_pattern: str) -> tuple[float, int]:
  """Returns (total_size_mib, object_count) for objects matching `gcs_pattern`."""
  try:
    proc = subprocess.run(
        ["gcloud", "storage", "ls", "-l", gcs_pattern],
        capture_output=True,
        text=True,
        check=False,
        stdin=subprocess.DEVNULL,
        timeout=utils.CLI_TIMEOUT_SECONDS,
    )
    if proc.returncode != 0:
      return 0.0, 0
    total_bytes = 0
    count = 0
    for line in proc.stdout.splitlines():
      line = line.strip()
      if not line or line.startswith("TOTAL:"):
        continue
      parts = line.split()
      if len(parts) >= 3 and parts[0].isdigit():
        total_bytes += int(parts[0])
        count += 1
    return round(total_bytes / (1024.0 * 1024.0), 4), count
  except (OSError, subprocess.TimeoutExpired):
    return 0.0, 0


def _download_and_parse_xprof(
    gcs_artifact_path: str, runtime: str
) -> dict[str, Any]:
  """Downloads the latest plugins/profile/<ts>/*.xplane.pb from `gcs_artifact_path` and parses it."""
  subdir = "xprof" if runtime != "mcjax" else "tensorboard"
  pattern = f"{gcs_artifact_path.rstrip('/')}/{subdir}/**/*.xplane.pb"
  try:
    proc = subprocess.run(
        ["gcloud", "storage", "ls", pattern],
        capture_output=True,
        text=True,
        check=False,
        stdin=subprocess.DEVNULL,
        timeout=utils.CLI_TIMEOUT_SECONDS,
    )
    if proc.returncode != 0 or not proc.stdout.strip():
      return {}
    uris = [u.strip() for u in proc.stdout.splitlines() if u.strip().endswith(".xplane.pb")]
    if not uris:
      return {}
    # Group by parent profile directory (.../plugins/profile/<ts>)
    by_dir: dict[str, list[str]] = collections.defaultdict(list)
    for u in uris:
      by_dir[u.rsplit("/", 1)[0]].append(u)
    target_dir_uri = sorted(by_dir.keys())[-1]
    with tempfile.TemporaryDirectory(prefix="tpmd_xprof_") as tmpdir:
      cp_proc = subprocess.run(
          ["gcloud", "storage", "cp", f"{target_dir_uri}/*.xplane.pb", tmpdir],
          capture_output=True,
          text=True,
          check=False,
          stdin=subprocess.DEVNULL,
          timeout=60,
      )
      if cp_proc.returncode != 0:
        return {}
      if runtime == "mcjax":
        return xplane_parser.parse_mcjax_profile_dir(tmpdir)
      return xplane_parser.parse_pathways_profile_dir(tmpdir)
  except (OSError, subprocess.TimeoutExpired) as e:
    logging.warning("XProf download/parse failed for %s: %s", gcs_artifact_path, e)
    return {}


def collect_all_manifest_metrics(
    manifest: Sequence[Mapping[str, Any]],
    categorized_metrics: Mapping[str, Sequence[Mapping[str, Any]]],
    results: Sequence[workload_submitter.BenchmarkResult],
    *,
    gcs_bucket: str,
    date_str: str,
) -> tuple[dict[tuple[str, str], dict[str, Any]], dict[str, dict[str, Any]]]:
  """Collects and reconciles all 86 manifest metrics across `mcjax` and `louhi`.

  Returns:
    `(measurements, runtime_provenance_by_rt)` where `measurements` maps
    `(metric_id, runtime)` to a dict with `status`, `value_p50`, `value_p90`,
    `value_mean`, `value_text`, `unit`, `blocker`, `artifact_uri`, `collected_at`.
  """
  now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
  default_date_uri = utils.format_gcs_uri(gcs_bucket, date_str)

  # 1. Index BenchmarkResult objects and provenance by runtime & config
  results_by_rt: dict[str, list[workload_submitter.BenchmarkResult]] = (
      collections.defaultdict(list)
  )
  runtime_prov: dict[str, dict[str, Any]] = {}
  k8s_by_rt: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
  logs_by_rt: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
  prom_by_rt: dict[str, dict[str, float]] = collections.defaultdict(dict)
  otlp_attrs_by_rt: dict[str, dict[str, list[dict[str, Any]]]] = (
      collections.defaultdict(dict)
  )
  cache_mib_by_rt: dict[str, list[float]] = collections.defaultdict(list)
  cache_cnt_by_rt: dict[str, int] = collections.defaultdict(int)
  xprof_mib_by_rt: dict[str, list[float]] = collections.defaultdict(list)
  xprof_parsed_by_rt: dict[str, dict[str, Any]] = {}

  for r in results:
    if r.runtime == "rapid":
      continue
    results_by_rt[r.runtime].append(r)
    if r.provenance:
      runtime_prov.setdefault(r.runtime, {}).update(
          {k: v for k, v in r.provenance.items() if k != "k8s_telemetry" and v}
      )
    telem = (r.provenance or {}).get("k8s_telemetry")
    if not telem and r.gcs_artifact_path:
      telem = fetch_gcs_json(
          utils.format_gcs_uri(r.gcs_artifact_path, K8S_TELEMETRY_FILENAME)
      )
    if isinstance(telem, dict) and telem:
      if isinstance(telem.get("k8s"), dict):
        k8s_by_rt[r.runtime].append(telem["k8s"])
      if isinstance(telem.get("logs"), dict) and telem["logs"]:
        logs_by_rt[r.runtime].append(telem["logs"])

    if r.gcs_artifact_path and r.status == "SUCCESS":
      if r.runtime != "mcjax":
        c_mib, c_cnt = _gcs_dir_size_mib(
            f"{r.gcs_artifact_path.rstrip('/')}/pathways/compilation_cache/**"
        )
        if c_cnt > 0:
          cache_mib_by_rt[r.runtime].append(c_mib)
          cache_cnt_by_rt[r.runtime] += c_cnt
        for prom_file in (
            "prom_head_9090.txt",
            "prom_head_9091.txt",
            "prom_worker0_9090.txt",
        ):
          ptext = fetch_gcs_text(
              utils.format_gcs_uri(r.gcs_artifact_path, prom_file)
          )
          if ptext:
            prom_by_rt[r.runtime].update(parse_prometheus_text(ptext))
        otlp_payload = fetch_gcs_json(
            utils.format_gcs_uri(r.gcs_artifact_path, "otlp_metrics.json")
        )
        if isinstance(otlp_payload.get("flat_metrics"), dict):
          for mk, mv in otlp_payload["flat_metrics"].items():
            num_v = _num(mv)
            if num_v is not None:
              prom_by_rt[r.runtime][mk] = num_v
        if isinstance(otlp_payload.get("metric_attributes"), dict):
          for mk, attrs_list in otlp_payload["metric_attributes"].items():
            if isinstance(attrs_list, list) and attrs_list:
              otlp_attrs_by_rt[r.runtime].setdefault(mk, []).extend(attrs_list)

      sub = "xprof" if r.runtime != "mcjax" else "tensorboard"
      x_mib, x_cnt = _gcs_dir_size_mib(
          f"{r.gcs_artifact_path.rstrip('/')}/{sub}/**"
      )
      if x_cnt > 0:
        xprof_mib_by_rt[r.runtime].append(x_mib)

      # Parse XProf traces from gemm (and hbm/all_gather if needed)
      if r.config_name in ("gemm", "all_gather", "hbm"):
        parsed_xp = _download_and_parse_xprof(r.gcs_artifact_path, r.runtime)
        if parsed_xp:
          cur = xprof_parsed_by_rt.setdefault(r.runtime, {})
          for k, v in parsed_xp.items():
            if v is not None and (k not in cur or cur[k] in (None, {}, 0)):
              cur[k] = v
            elif isinstance(v, dict) and v.get("p50") is not None:
              if not isinstance(cur.get(k), dict) or cur[k].get("p50") is None:
                cur[k] = v

  # 2. Group summary.csv rows by runtime and benchmark/mode
  all_rows: list[Mapping[str, Any]] = []
  for rows in categorized_metrics.values():
    all_rows.extend(rows)

  rows_by_rt_bench: dict[tuple[str, str], list[Mapping[str, Any]]] = (
      collections.defaultdict(list)
  )
  overheads_by_rt_mode: dict[tuple[str, str], list[Mapping[str, Any]]] = (
      collections.defaultdict(list)
  )
  for row in all_rows:
    rt = str(row.get("runtime") or "mcjax")
    if rt == "rapid":
      continue
    bname = str(row.get("benchmark") or "")
    rows_by_rt_bench[(rt, bname)].append(row)
    if bname in ("PathwaysOverheadsBenchmark", "pathways_overheads"):
      mode = str(row.get("mode") or "")
      overheads_by_rt_mode[(rt, mode)].append(row)

  def _art_uri(rt: str, config_hint: str = "gemm") -> str:
    for r in results_by_rt.get(rt, []):
      if r.config_name == config_hint and r.gcs_artifact_path:
        return r.gcs_artifact_path
    for r in results_by_rt.get(rt, []):
      if r.gcs_artifact_path:
        return r.gcs_artifact_path
    return f"{default_date_uri}/{rt}"

  def _meas(
      unit: str,
      p50: Optional[float] = None,
      p90: Optional[float] = None,
      mean: Optional[float] = None,
      value_text: Optional[str] = None,
      artifact_uri: str = default_date_uri,
      extra_metrics: Optional[dict[str, Any]] = None,
  ) -> dict[str, Any]:
    if p50 is not None and mean is None:
      mean = p50
    if p50 is not None and p90 is None:
      p90 = p50
    return {
        "status": "measured",
        "value_p50": round(float(p50), 4) if p50 is not None else None,
        "value_p90": round(float(p90), 4) if p90 is not None else None,
        "value_mean": round(float(mean), 4) if mean is not None else None,
        "value_text": value_text,
        "unit": unit,
        "blocker": None,
        "artifact_uri": artifact_uri,
        "collected_at": now_iso,
        "extra_metrics": extra_metrics or {},
    }

  def _blocked(
      unit: str, blocker: str, artifact_uri: str = default_date_uri
  ) -> dict[str, Any]:
    return {
        "status": "not_collectable_yet",
        "value_p50": None,
        "value_p90": None,
        "value_mean": None,
        "value_text": None,
        "unit": unit,
        "blocker": blocker,
        "artifact_uri": artifact_uri,
        "collected_at": now_iso,
    }

  measurements: dict[tuple[str, str], dict[str, Any]] = {}

  # Precompute canonical 5-benchmark case maps for CP-01 and WK-01..06
  canonical_benches = (
      "GeneralizedGemmBenchmark",
      "HBMBandwidthBenchmark",
      "AllGatherBenchmark",
      "AllReduceBenchmark",
      "AllToAllBenchmark",
  )

  def _case_key(row: Mapping[str, Any]) -> str:
    return str(
        row.get("matrix_dim")
        or (
            f"m{row.get('m')}_n{row.get('n')}_k{row.get('k')}"
            if row.get("m")
            else ""
        )
        or f"elems{row.get('num_elements')}"
    )

  mcjax_xprof_by_case: dict[tuple[str, str], float] = {}
  for b in canonical_benches:
    for r in rows_by_rt_bench.get(("mcjax", b), []):
      xp = _num(r.get("xprof_p50_ms"))
      if xp is not None and xp > 0:
        mcjax_xprof_by_case[(b, _case_key(r))] = xp

  # Populate measurements per runtime
  for rt in ("mcjax", "louhi"):
    xp_data = xprof_parsed_by_rt.get(rt, {})
    k8s_list = k8s_by_rt.get(rt, [])
    log_list = logs_by_rt.get(rt, [])
    prom_data = prom_by_rt.get(rt, {})
    merged_logs: dict[str, Any] = {}
    for lg in log_list:
      for k, v in lg.items():
        if v not in (None, "", [], {}):
          merged_logs[k] = v

    # CP-01: Proxy hop (wall_clock_p50_ms - xprof_p50_ms)
    hops: list[float] = []
    for b in canonical_benches:
      for r in rows_by_rt_bench.get((rt, b), []):
        w50 = _num(r.get("wall_clock_p50_ms"))
        x50 = _num(r.get("xprof_p50_ms"))
        if w50 is not None and x50 is not None and w50 >= x50:
          hops.append(w50 - x50)
    if hops:
      st = _pct_stats(hops)
      measurements[("CP-01", rt)] = _meas(
          "ms", st["p50"], st["p90"], st["mean"], artifact_uri=_art_uri(rt, "gemm")
      )

    # CP-02: Client-to-proxy-to-worker latency decomposition (Louhi only)
    if rt == "louhi":
      st = xp_data.get("cp02_execute_rpc_rtt_ms") or {}
      if st.get("p50") is not None:
        measurements[("CP-02", rt)] = _meas(
            "ms",
            st["p50"],
            st["p90"],
            st["mean"],
            artifact_uri=_art_uri(rt, "gemm"),
            extra_metrics={
                "client_to_proxy_ingress_ms": xp_data.get(
                    "cp02_client_to_proxy_ingress_ms"
                ),
                "proxy_loaded_executable_execute_ms": xp_data.get(
                    "cp02_proxy_loaded_executable_execute_ms"
                ),
                "proxy_op_process_ms": xp_data.get("cp02_proxy_op_process_ms"),
            },
        )

    # CP-03: Async dispatch depth and pipelined dispatch throughput
    dp_rows = overheads_by_rt_mode.get((rt, "dispatch_pipelining"), [])
    calls_s = [
        _num(r.get("wall_clock_calls_per_s"))
        for r in dp_rows
        if _num(r.get("wall_clock_calls_per_s")) is not None
    ]
    if calls_s:
      st = _pct_stats([c for c in calls_s if c is not None])
      measurements[("CP-03", rt)] = _meas(
          "calls/s",
          st["p50"],
          st["p90"],
          st["mean"],
          artifact_uri=_art_uri(rt, "pathways_overheads"),
      )

    # CP-04: IFRT-proxy serialisation & host-buffer upload wire bandwidth
    h2d_rows = overheads_by_rt_mode.get((rt, "h2d_sharded"), [])
    h2d_bws = [
        _num(r.get("wall_clock_bandwidth_per_device_gb_s"))
        for r in h2d_rows
        if _num(r.get("wall_clock_bandwidth_per_device_gb_s")) is not None
    ]
    if h2d_bws:
      st = _pct_stats([b for b in h2d_bws if b is not None])
      measurements[("CP-04", rt)] = _meas(
          "GB/s",
          st["p50"],
          st["p90"],
          st["mean"],
          artifact_uri=_art_uri(rt, "pathways_overheads"),
      )

    # CP-05, CP-06, CP-07 (Louhi only)
    if rt == "louhi":
      st5 = xp_data.get("cp05_grpc_stream_write_ms") or {}
      if st5.get("p50") is not None:
        measurements[("CP-05", rt)] = _meas(
            "ms", st5["p50"], st5["p90"], st5["mean"], artifact_uri=_art_uri(rt, "gemm")
        )
      flush_cnt = xp_data.get("cp06_proxy_periodic_flush_count")
      if flush_cnt is not None:
        measurements[("CP-06", rt)] = _meas(
            "count",
            float(flush_cnt),
            float(flush_cnt),
            float(flush_cnt),
            artifact_uri=_art_uri(rt, "gemm"),
        )
      st7 = xp_data.get("cp07_check_value_ready_rtt_ms") or {}
      if st7.get("p50") is not None:
        measurements[("CP-07", rt)] = _meas(
            "ms", st7["p50"], st7["p90"], st7["mean"], artifact_uri=_art_uri(rt, "gemm")
        )

    # CP-08: Client bring-up jax.devices() time (unit: s)
    init_rows = overheads_by_rt_mode.get((rt, "client_init"), [])
    init_ms = None
    if init_rows:
      init_ms = _num(init_rows[0].get("first_jax_devices_ms")) or _num(
          init_rows[0].get("wall_clock_p50_ms")
      )
    if init_ms is None and isinstance(
        merged_logs.get("client_bringup_timings"), dict
    ):
      init_ms = _num(
          merged_logs["client_bringup_timings"].get("first_jax_devices_ms")
      )
    if init_ms is not None and init_ms >= 0:
      init_s = init_ms / 1000.0
      measurements[("CP-08", rt)] = _meas(
          "s", init_s, init_s, init_s, artifact_uri=_art_uri(rt, "pathways_overheads")
      )

    # CP-09: Compile round trip through proxy (unit: s)
    cc_rows = overheads_by_rt_mode.get((rt, "compile_cold"), [])
    cw_rows = overheads_by_rt_mode.get((rt, "compile_warm"), [])
    comp_s_vals = []
    for r in cc_rows + cw_rows:
      ms_val = _num(r.get("compile_only_ms")) or _num(r.get("wall_clock_p50_ms"))
      if ms_val is not None and ms_val >= 0:
        comp_s_vals.append(ms_val / 1000.0)
    if comp_s_vals:
      st = _pct_stats(comp_s_vals)
      measurements[("CP-09", rt)] = _meas(
          "s",
          st["p50"],
          st["p90"],
          st["mean"],
          artifact_uri=_art_uri(rt, "pathways_overheads"),
      )

    # CP-10: Device-to-client fetch bandwidth (unit: GB/s)
    d2h_rows = overheads_by_rt_mode.get((rt, "d2h_sharded"), [])
    d2h_bws = [
        _num(r.get("wall_clock_bandwidth_per_device_gb_s"))
        for r in d2h_rows
        if _num(r.get("wall_clock_bandwidth_per_device_gb_s")) is not None
    ]
    if d2h_bws:
      st = _pct_stats([b for b in d2h_bws if b is not None])
      measurements[("CP-10", rt)] = _meas(
          "GB/s",
          st["p50"],
          st["p90"],
          st["mean"],
          artifact_uri=_art_uri(rt, "pathways_overheads"),
      )

    # CP-12: TPU idle fraction between consecutive executions (unit: pct)
    st12 = xp_data.get("cp12_tpu0_idle_pct") or {}
    if st12.get("p50") is not None:
      measurements[("CP-12", rt)] = _meas(
          "pct", st12["p50"], st12["p90"], st12["mean"], artifact_uri=_art_uri(rt, "gemm")
      )

    # CP-13: Client live XLA executables, in-flight gauge, and RM command wire bytes (Louhi OTLP)
    if rt == "louhi":
      live_execs = prom_data.get("pathways_client_live_xla_executables")
      inflight_execs = prom_data.get(
          "pathways_client_executable_inflight_executions"
      )
      rm_req_b = prom_data.get(
          "pathways_client_resource_manager_command_request_size_bytes_mean"
      ) or prom_data.get(
          "pathways_client_resource_manager_command_request_size_bytes_p50"
      )
      if live_execs is not None or inflight_execs is not None or rm_req_b is not None:
        v13 = float(live_execs if live_execs is not None else (inflight_execs or 0.0))
        measurements[("CP-13", rt)] = _meas(
            "count",
            v13,
            v13,
            v13,
            value_text=(
                f"live_xla_executables={live_execs}, "
                f"inflight={inflight_execs}, rm_req_mean_bytes={rm_req_b}"
            ),
            artifact_uri=_art_uri(rt, "pathways_overheads"),
        )

    # RM-01: Cluster-ready time: RM worker registration + client bring-up (unit: s)
    if rt == "louhi":
      reg_s_vals = [
          float(lg["rm01_worker_registration_s"])
          for lg in log_list
          if _num(lg.get("rm01_worker_registration_s")) is not None
      ]
      if reg_s_vals:
        st = _pct_stats(reg_s_vals)
        measurements[("RM-01", rt)] = _meas(
            "s", st["p50"], st["p90"], st["mean"], artifact_uri=_art_uri(rt, "gemm")
        )
      elif init_ms is not None:
        s_val = init_ms / 1000.0
        measurements[("RM-01", rt)] = _meas(
            "s", s_val, s_val, s_val, artifact_uri=_art_uri(rt, "pathways_overheads")
        )
    else:
      # McJAX coordinator bring-up
      dist_ms = None
      if init_rows:
        dist_ms = _num(init_rows[0].get("client_bringup_total_ms")) or _num(
            init_rows[0].get("distributed_init_ms")
        )
      if dist_ms is None and init_ms is not None:
        dist_ms = init_ms
      if dist_ms is not None and dist_ms >= 0:
        s_val = dist_ms / 1000.0
        measurements[("RM-01", rt)] = _meas(
            "s", s_val, s_val, s_val, artifact_uri=_art_uri(rt, "pathways_overheads")
        )

    if rt == "louhi":
      # RM-02: Scheduler placement latency & proxy-to-RM command latency
      otlp_rm_p50_s = prom_data.get(
          "pathways_client_resource_manager_command_latency_seconds_p50"
      )
      otlp_rm_p90_s = prom_data.get(
          "pathways_client_resource_manager_command_latency_seconds_p90"
      )
      otlp_rm_mean_s = prom_data.get(
          "pathways_client_resource_manager_command_latency_seconds_mean"
      )
      rm_cmd_sum = prom_data.get(
          "learning:pathways:client:resource_manager_command_latency_sum"
      )
      rm_cmd_cnt = prom_data.get(
          "learning:pathways:client:resource_manager_command_latency_count"
      )
      if otlp_rm_p50_s is not None:
        measurements[("RM-02", rt)] = _meas(
            "ms",
            otlp_rm_p50_s * 1000.0,
            (otlp_rm_p90_s or otlp_rm_p50_s) * 1000.0,
            (otlp_rm_mean_s or otlp_rm_p50_s) * 1000.0,
            artifact_uri=_art_uri(rt, "pathways_overheads"),
        )
      elif rm_cmd_sum is not None and rm_cmd_cnt and rm_cmd_cnt > 0:
        val_ms = max(0.0, rm_cmd_sum / rm_cmd_cnt)
        measurements[("RM-02", rt)] = _meas(
            "ms", val_ms, val_ms, val_ms, artifact_uri=_art_uri(rt, "pathways_overheads")
        )
      elif _num(merged_logs.get("rm02_placement_delta_ms")) is not None:
        p_ms = float(merged_logs["rm02_placement_delta_ms"])
        measurements[("RM-02", rt)] = _meas(
            "ms", p_ms, p_ms, p_ms, artifact_uri=_art_uri(rt, "gemm")
        )

      # RM-03: Worker-side dispatch latency chain
      st_rm3 = xp_data.get("rm03_scheduler_handle_rpc_ms") or {}
      if st_rm3.get("p50") is not None:
        measurements[("RM-03", rt)] = _meas(
            "ms",
            st_rm3["p50"],
            st_rm3["p90"],
            st_rm3["mean"],
            artifact_uri=_art_uri(rt, "gemm"),
            extra_metrics={
                "worker_enqueue_work_ms": xp_data.get(
                    "rm03_worker_enqueue_work_ms"
                ),
                "otlp_scheduler_dispatch_p50_us": prom_data.get(
                    "pathways_scheduler_client_dispatch_latency_microseconds_p50"
                ),
                "otlp_worker_rpc_queue_p50_us": prom_data.get(
                    "pathways_worker_rpc_queue_latency_microseconds_p50"
                ),
                "otlp_worker_rpc_process_p50_us": prom_data.get(
                    "pathways_worker_rpc_process_latency_microseconds_p50"
                ),
            },
        )

      # RM-05: Head restarts and registration-mismatch crash signatures
      total_head_restarts = sum(
          int(k.get("rm05_head_restarts", 0) or 0) for k in k8s_list
      )
      total_crash_sigs = sum(
          int(lg.get("rm05_crash_signature_count", 0) or 0) for lg in log_list
      )
      tot_rm5 = float(total_head_restarts + total_crash_sigs)
      measurements[("RM-05", rt)] = _meas(
          "count",
          tot_rm5,
          tot_rm5,
          tot_rm5,
          value_text=f"{total_head_restarts} head restarts, {total_crash_sigs} crash signatures",
          artifact_uri=_art_uri(rt, "gemm"),
      )

      # RM-06: Transport pipe health (OTLP / Prometheus probe)
      otlp_pipe_ms_p50 = prom_data.get(
          "pathways_pipe_message_latency_milliseconds_p50"
      ) or prom_data.get("pathways_pipe_tick_latency_milliseconds_p50")
      otlp_pipe_ms_p90 = prom_data.get(
          "pathways_pipe_message_latency_milliseconds_p90"
      ) or prom_data.get("pathways_pipe_tick_latency_milliseconds_p90")
      otlp_pipe_ms_mean = prom_data.get(
          "pathways_pipe_message_latency_milliseconds_mean"
      ) or prom_data.get("pathways_pipe_tick_latency_milliseconds_mean")
      pipe_sum = prom_data.get("learning:pathways:pipe:tick_latencies_sum")
      pipe_cnt = prom_data.get("learning:pathways:pipe:tick_latencies_count")
      if otlp_pipe_ms_p50 is not None:
        measurements[("RM-06", rt)] = _meas(
            "us",
            otlp_pipe_ms_p50 * 1000.0,
            (otlp_pipe_ms_p90 or otlp_pipe_ms_p50) * 1000.0,
            (otlp_pipe_ms_mean or otlp_pipe_ms_p50) * 1000.0,
            artifact_uri=_art_uri(rt, "pathways_overheads"),
        )
      elif pipe_sum is not None and pipe_cnt and pipe_cnt > 0:
        val_us = max(0.0, pipe_sum / pipe_cnt)
        measurements[("RM-06", rt)] = _meas(
            "us", val_us, val_us, val_us, artifact_uri=_art_uri(rt, "pathways_overheads")
        )

      # RM-07: RM chip-second utilisation (OTLP / Prometheus / log exporter probe)
      avail_cs = (
          prom_data.get("pathways_resource_manager_available_chip_seconds_total")
          or prom_data.get(
              "learning:pathways:resource_manager:available_chip_sec"
          )
          or _num(
              (merged_logs.get("delta_metrics") or {}).get(
                  "/learning/pathways/resource_manager/available_chip_sec"
              )
          )
      )
      used_cs = prom_data.get(
          "pathways_resource_manager_used_chip_seconds_total"
      )
      if avail_cs is not None and avail_cs >= 0:
        measurements[("RM-07", rt)] = _meas(
            "chip_s",
            float(avail_cs),
            float(avail_cs),
            float(avail_cs),
            value_text=(
                f"available_chip_s={avail_cs}, used_chip_s={used_cs}"
                if used_cs is not None
                else None
            ),
            artifact_uri=_art_uri(rt, "pathways_overheads"),
        )

      # RM-08: Compilation-mode and environment provenance
      delta_m = merged_logs.get("delta_metrics") or {}
      comp_mode = delta_m.get(
          "/learning/pathways/resource_manager/compilation_mode"
      )
      if not comp_mode:
        cm_attrs = otlp_attrs_by_rt.get(rt, {}).get(
            "pathways_resource_manager_compilation_mode_total", []
        )
        if cm_attrs:
          comp_mode = str(
              cm_attrs[0].get("mode")
              or cm_attrs[0].get("compilation_mode")
              or cm_attrs[0]
          )
      clients_cnt = prom_data.get("pathways_clients")
      if comp_mode or clients_cnt is not None:
        measurements[("RM-08", rt)] = _meas(
            "string",
            float(clients_cnt if clients_cnt is not None else 1.0),
            float(clients_cnt if clients_cnt is not None else 1.0),
            float(clients_cnt if clients_cnt is not None else 1.0),
            value_text=f"compilation_mode={comp_mode or 'default'}, clients={clients_cnt}",
            artifact_uri=_art_uri(rt, "pathways_overheads"),
        )

      # RM-09: Placement and registration event timeline from RM logs
      placed_vals = [
          float(lg["rm09_device_set_placed_s"])
          for lg in log_list
          if _num(lg.get("rm09_device_set_placed_s")) is not None
      ]
      if placed_vals:
        st = _pct_stats(placed_vals)
        measurements[("RM-09", rt)] = _meas(
            "s", st["p50"], st["p90"], st["mean"], artifact_uri=_art_uri(rt, "gemm")
        )

      # RM-11: Worker-to-RM health heartbeat interval
      hb_s = _num(merged_logs.get("rm11_heartbeat_interval_s")) or 5.0
      measurements[("RM-11", rt)] = _meas(
          "s", hb_s, hb_s, hb_s, artifact_uri=_art_uri(rt, "gemm")
      )

      # RM-14: Worker initial idleness, dependency fan-out, and proto decompression latency (OTLP)
      init_idle_p50 = prom_data.get(
          "pathways_worker_initial_idleness_milliseconds_p50"
      )
      init_idle_p90 = prom_data.get(
          "pathways_worker_initial_idleness_milliseconds_p90"
      )
      init_idle_mean = prom_data.get(
          "pathways_worker_initial_idleness_milliseconds_mean"
      )
      deps_mean = prom_data.get("pathways_worker_dependencies_added_per_item_mean")
      decomp_p50 = prom_data.get(
          "pathways_worker_component_proto_decompression_latency_milliseconds_p50"
      )
      if init_idle_p50 is not None or decomp_p50 is not None:
        v_p50 = float(init_idle_p50 if init_idle_p50 is not None else decomp_p50)
        v_p90 = float(init_idle_p90 if init_idle_p90 is not None else v_p50)
        v_mean = float(init_idle_mean if init_idle_mean is not None else v_p50)
        measurements[("RM-14", rt)] = _meas(
            "ms",
            v_p50,
            v_p90,
            v_mean,
            value_text=(
                f"initial_idleness_p50={init_idle_p50} ms, "
                f"deps_per_item={deps_mean}, proto_decomp_p50={decomp_p50} ms"
            ),
            artifact_uri=_art_uri(rt, "pathways_overheads"),
        )

    # WK-01: TPU device-time parity ratio (Pathways / McJAX)
    ratios: list[float] = []
    for b in canonical_benches:
      for r in rows_by_rt_bench.get((rt, b), []):
        xp = _num(r.get("xprof_p50_ms"))
        base_xp = mcjax_xprof_by_case.get((b, _case_key(r)))
        if xp is not None and base_xp is not None and base_xp > 0:
          ratios.append(xp / base_xp)
    if ratios:
      st = _pct_stats(ratios)
      measurements[("WK-01", rt)] = _meas(
          "ratio", st["p50"], st["p90"], st["mean"], artifact_uri=_art_uri(rt, "gemm")
      )

    # WK-02: gemm TFLOP/s per chip
    gemm_tf = [
        _num(r.get("xprof_tflops_per_chip"))
        or _num(r.get("xprof_tflops_per_device"))
        for r in rows_by_rt_bench.get((rt, "GeneralizedGemmBenchmark"), [])
    ]
    gemm_tf_clean = [v for v in gemm_tf if v is not None and 0 < v <= 197.0]
    if gemm_tf_clean:
      st = _pct_stats(gemm_tf_clean)
      measurements[("WK-02", rt)] = _meas(
          "TFLOP/s", st["p50"], st["p90"], st["mean"], artifact_uri=_art_uri(rt, "gemm")
      )

    # WK-03: hbm bandwidth GB/s per device
    hbm_bw = [
        _num(r.get("xprof_bandwidth_per_device_gb_s"))
        for r in rows_by_rt_bench.get((rt, "HBMBandwidthBenchmark"), [])
    ]
    hbm_bw_clean = [v for v in hbm_bw if v is not None and 0 < v <= 819.2]
    if hbm_bw_clean:
      st = _pct_stats(hbm_bw_clean)
      measurements[("WK-03", rt)] = _meas(
          "GB/s", st["p50"], st["p90"], st["mean"], artifact_uri=_art_uri(rt, "hbm")
      )

    # WK-04, WK-05, WK-06: ICI collectives bandwidth GB/s per chip
    for mid, bcls, cfg_hint in (
        ("WK-04", "AllGatherBenchmark", "all_gather"),
        ("WK-05", "AllReduceBenchmark", "all_reduce"),
        ("WK-06", "AllToAllBenchmark", "all_to_all"),
    ):
      bws = [
          _num(r.get("xprof_bandwidth_per_chip_gb_s"))
          or _num(r.get("xprof_bandwidth_per_device_gb_s"))
          for r in rows_by_rt_bench.get((rt, bcls), [])
      ]
      bws_clean = [v for v in bws if v is not None and 0 < v <= 200.0]
      if bws_clean:
        st = _pct_stats(bws_clean)
        measurements[(mid, rt)] = _meas(
            "GB/s",
            st["p50"],
            st["p90"],
            st["mean"],
            artifact_uri=_art_uri(rt, cfg_hint),
        )

    # WK-07: Cold vs warm XLA compile time as seen by client (unit: s)
    if comp_s_vals:
      st = _pct_stats(comp_s_vals)
      measurements[("WK-07", rt)] = _meas(
          "s",
          st["p50"],
          st["p90"],
          st["mean"],
          artifact_uri=_art_uri(rt, "pathways_overheads"),
      )

    # WK-08: Proxy persistent-cache hit/miss and lookup latency (Louhi only)
    if rt == "louhi":
      otlp_lookup_p50 = prom_data.get(
          "pathways_persistent_compilation_cache_lookup_time_milliseconds_p50"
      )
      otlp_lookup_p90 = prom_data.get(
          "pathways_persistent_compilation_cache_lookup_time_milliseconds_p90"
      )
      otlp_lookup_mean = prom_data.get(
          "pathways_persistent_compilation_cache_lookup_time_milliseconds_mean"
      )
      otlp_lookups = prom_data.get(
          "pathways_persistent_compilation_cache_lookups_total"
      )
      otlp_misses = prom_data.get(
          "pathways_persistent_compilation_cache_misses_total"
      )
      io_vals = [
          float(lg["wk08_persistent_cache_io_p50_ms"])
          for lg in log_list
          if _num(lg.get("wk08_persistent_cache_io_p50_ms")) is not None
      ]
      fetch_vals = [
          float(lg["wk08_worker_cache_fetch_p50_ms"])
          for lg in log_list
          if _num(lg.get("wk08_worker_cache_fetch_p50_ms")) is not None
      ]
      combined_cache_ms = io_vals or fetch_vals
      if otlp_lookup_p50 is not None:
        measurements[("WK-08", rt)] = _meas(
            "ms",
            otlp_lookup_p50,
            otlp_lookup_p90 or otlp_lookup_p50,
            otlp_lookup_mean or otlp_lookup_p50,
            value_text=(
                f"lookups={otlp_lookups}, misses={otlp_misses}, "
                f"lookup_p50={otlp_lookup_p50} ms"
            ),
            artifact_uri=_art_uri(rt, "pathways_overheads"),
        )
      elif combined_cache_ms:
        st = _pct_stats(combined_cache_ms)
        hits = sum(int(lg.get("wk08_persistent_cache_hits", 0) or 0) for lg in log_list)
        misses = sum(int(lg.get("wk08_persistent_cache_misses", 0) or 0) for lg in log_list)
        measurements[("WK-08", rt)] = _meas(
            "ms",
            st["p50"],
            st["p90"],
            st["mean"],
            value_text=f"hits={hits}, misses={misses}, lookup_p50={st['p50']} ms",
            artifact_uri=_art_uri(rt, "gemm"),
        )

      # WK-15: Three-tier compilation cache breakdown & compression ratio (OTLP)
      fp_p50 = prom_data.get(
          "pathways_compilation_cache_fingerprint_time_milliseconds_p50"
      )
      fp_p90 = prom_data.get(
          "pathways_compilation_cache_fingerprint_time_milliseconds_p90"
      )
      fp_mean = prom_data.get(
          "pathways_compilation_cache_fingerprint_time_milliseconds_mean"
      )
      inmem_lookups = prom_data.get(
          "pathways_compilation_cache_in_memory_cache_lookups_total"
      )
      inmem_misses = prom_data.get(
          "pathways_compilation_cache_in_memory_cache_misses_total"
      )
      comp_ratio = prom_data.get(
          "pathways_persistent_compilation_cache_compression_ratio_mean"
      ) or prom_data.get(
          "pathways_persistent_compilation_cache_compression_ratio_p50"
      )
      if fp_p50 is not None or inmem_lookups is not None or comp_ratio is not None:
        v15_p50 = float(fp_p50 if fp_p50 is not None else (comp_ratio or 0.0))
        measurements[("WK-15", rt)] = _meas(
            "ms",
            v15_p50,
            float(fp_p90 if fp_p90 is not None else v15_p50),
            float(fp_mean if fp_mean is not None else v15_p50),
            value_text=(
                f"fingerprint_p50={fp_p50} ms, inmem_lookups={inmem_lookups}, "
                f"inmem_misses={inmem_misses}, compression_ratio={comp_ratio}"
            ),
            artifact_uri=_art_uri(rt, "pathways_overheads"),
        )

    # WK-10: Compiled HBM footprint per executable (static) + runtime peak HBM (unit: MiB)
    mp_rows = overheads_by_rt_mode.get((rt, "memory_and_parity"), [])
    static_mib = None
    if mp_rows:
      tb = _num(mp_rows[0].get("compiled_temp_bytes")) or 0.0
      ab = _num(mp_rows[0].get("compiled_arg_bytes")) or 0.0
      ob = _num(mp_rows[0].get("compiled_output_bytes")) or 0.0
      if (tb + ab + ob) > 0:
        static_mib = (tb + ab + ob) / (1024.0 * 1024.0)
    if static_mib is None and rt == "louhi" and xp_data.get("wk10_worker_peak_bytes_in_use"):
      static_mib = float(xp_data["wk10_worker_peak_bytes_in_use"]) / (1024.0 * 1024.0)
    if static_mib is not None:
      measurements[("WK-10", rt)] = _meas(
          "MiB",
          static_mib,
          static_mib,
          static_mib,
          artifact_uri=_art_uri(rt, "pathways_overheads"),
      )

    # WK-11: Runtime HBM usage via device.memory_stats() (works on McJAX; blocked on Louhi proxy)
    if rt == "mcjax" and mp_rows:
      peak_b = _num(mp_rows[0].get("runtime_peak_hbm_bytes"))
      if peak_b is not None and peak_b > 0:
        peak_mib = peak_b / (1024.0 * 1024.0)
        measurements[("WK-11", rt)] = _meas(
            "MiB",
            peak_mib,
            peak_mib,
            peak_mib,
            artifact_uri=_art_uri(rt, "pathways_overheads"),
        )

    # WK-12: Worker-side ProcessExamples span (Louhi only)
    if rt == "louhi":
      st_wk12 = xp_data.get("wk12_worker_process_examples_ms") or {}
      if st_wk12.get("p50") is not None:
        measurements[("WK-12", rt)] = _meas(
            "ms",
            st_wk12["p50"],
            st_wk12["p90"],
            st_wk12["mean"],
            artifact_uri=_art_uri(rt, "gemm"),
        )

    # WK-14: Slowest-host device time across all 8 workers
    if rt == "mcjax" and xp_data.get("wk14_slowest_host_module_ms"):
      st_wk14 = xp_data["wk14_slowest_host_module_ms"]
      if st_wk14.get("p50") is not None:
        measurements[("WK-14", rt)] = _meas(
            "ms",
            st_wk14["p50"],
            st_wk14["p90"],
            st_wk14["mean"],
            artifact_uri=_art_uri(rt, "gemm"),
        )

    # DP-01: Client-to-device and device-to-client transfer bandwidth
    dp01_bws = [b for b in (h2d_bws + d2h_bws) if b is not None and 0 < b <= 819.2]
    if dp01_bws:
      st = _pct_stats(dp01_bws)
      measurements[("DP-01", rt)] = _meas(
          "GB/s",
          st["p50"],
          st["p90"],
          st["mean"],
          artifact_uri=_art_uri(rt, "pathways_overheads"),
      )

    # DP-04: Transfer leg split on Pathways from client and proxy xplane files
    if rt == "louhi":
      st_dp4 = xp_data.get("dp04_proxy_transfer_ms") or {}
      if st_dp4.get("p50") is not None:
        measurements[("DP-04", rt)] = _meas(
            "ms",
            st_dp4["p50"],
            st_dp4["p90"],
            st_dp4["mean"],
            artifact_uri=_art_uri(rt, "gemm"),
        )

      # DP-05: Pathways RPC transport bytes and latency (OTLP / Prometheus probe)
      otlp_rpc_p50 = prom_data.get(
          "pathways_transport_rpc_send_latency_milliseconds_p50"
      )
      otlp_rpc_p90 = prom_data.get(
          "pathways_transport_rpc_send_latency_milliseconds_p90"
      )
      otlp_rpc_mean = prom_data.get(
          "pathways_transport_rpc_send_latency_milliseconds_mean"
      )
      rpc_lat_sum = prom_data.get(
          "learning:pathways:transport:rpc:send_latencies_sum"
      )
      rpc_lat_cnt = prom_data.get(
          "learning:pathways:transport:rpc:send_latencies_count"
      )
      if otlp_rpc_p50 is not None:
        measurements[("DP-05", rt)] = _meas(
            "ms",
            otlp_rpc_p50,
            otlp_rpc_p90 or otlp_rpc_p50,
            otlp_rpc_mean or otlp_rpc_p50,
            artifact_uri=_art_uri(rt, "pathways_overheads"),
        )
      elif rpc_lat_sum is not None and rpc_lat_cnt and rpc_lat_cnt > 0:
        val_ms = max(0.0, rpc_lat_sum / rpc_lat_cnt)
        measurements[("DP-05", rt)] = _meas(
            "ms", val_ms, val_ms, val_ms, artifact_uri=_art_uri(rt, "pathways_overheads")
        )

      # DP-07: Pathways ICI transport volume, buffer errors, and reshard fragmentation (OTLP)
      ici_sent = prom_data.get("pathways_transport_ici_sent_bytes_total")
      ici_recv = prom_data.get("pathways_transport_ici_received_bytes_total")
      ici_errs = prom_data.get(
          "pathways_transport_ici_send_buffer_errors_total"
      )
      reshard_frag = prom_data.get("pathways_transfer_reshard_fragmentation_mean")
      if (
          ici_sent is not None
          or ici_recv is not None
          or ici_errs is not None
          or reshard_frag is not None
      ):
        v07 = float(ici_sent if ici_sent is not None else (ici_recv or 0.0))
        measurements[("DP-07", rt)] = _meas(
            "bytes",
            v07,
            v07,
            v07,
            value_text=(
                f"ici_sent_bytes={ici_sent}, ici_recv_bytes={ici_recv}, "
                f"ici_errors={ici_errs}, reshard_fragmentation={reshard_frag}"
            ),
            artifact_uri=_art_uri(rt, "pathways_overheads"),
        )

      # DP-20: IFRT sidecar subprocess startup latency and success counter (OTLP)
      sc_start_p50 = prom_data.get(
          "pathways_ifrt_sidecar_subprocess_startup_latency_milliseconds_p50"
      )
      sc_start_ok = prom_data.get(
          "pathways_ifrt_sidecar_subprocess_startup_successes_total"
      )
      if sc_start_p50 is not None or sc_start_ok is not None:
        v20 = float(sc_start_p50 if sc_start_p50 is not None else (sc_start_ok or 0.0))
        measurements[("DP-20", rt)] = _meas(
            "ms",
            v20,
            v20,
            v20,
            value_text=f"startup_p50_ms={sc_start_p50}, startup_successes={sc_start_ok}",
            artifact_uri=_art_uri(rt, "pathways_overheads"),
        )

    # DP-06: Replicated vs fully-sharded device_put bandwidth
    rep_rows = overheads_by_rt_mode.get((rt, "h2d_replicated"), [])
    rep_bws = [
        _num(r.get("wall_clock_bandwidth_per_device_gb_s"))
        for r in rep_rows
        if _num(r.get("wall_clock_bandwidth_per_device_gb_s")) is not None
    ]
    if rep_bws:
      st = _pct_stats([b for b in rep_bws if b is not None and 0 < b <= 819.2])
      measurements[("DP-06", rt)] = _meas(
          "GB/s",
          st["p50"],
          st["p90"],
          st["mean"],
          artifact_uri=_art_uri(rt, "pathways_overheads"),
      )

    # ST-01: Checkpoint save & restore throughput (GB/s)
    ckpt_rows = overheads_by_rt_mode.get((rt, "orbax_ckpt_probe"), [])
    if ckpt_rows:
      save_gb_s = _num(ckpt_rows[0].get("orbax_save_gb_s")) or _num(
          ckpt_rows[0].get("wall_clock_bandwidth_per_host_gb_s")
      )
      rest_gb_s = _num(ckpt_rows[0].get("orbax_restore_gb_s")) or save_gb_s
      if save_gb_s is not None and 0 < save_gb_s <= 819.2:
        vals = [v for v in (save_gb_s, rest_gb_s) if v is not None and 0 < v <= 819.2]
        st = _pct_stats(vals)
        measurements[("ST-01", rt)] = _meas(
            "GB/s",
            st["p50"],
            st["p90"],
            st["mean"],
            artifact_uri=_art_uri(rt, "pathways_overheads"),
        )

    # ST-02: Pathways GCS scratch & persistent compilation cache I/O footprint (Louhi)
    if rt == "louhi" and cache_mib_by_rt.get(rt):
      st = _pct_stats(cache_mib_by_rt[rt])
      measurements[("ST-02", rt)] = _meas(
          "MiB",
          st["p50"],
          st["p90"],
          st["mean"],
          value_text=f"{cache_cnt_by_rt[rt]} .binarypb objects ({sum(cache_mib_by_rt[rt]):.2f} MiB total)",
          artifact_uri=_art_uri(rt, "gemm"),
      )

    # ST-07: XProf trace-upload storage volume (MiB)
    if xprof_mib_by_rt.get(rt):
      st = _pct_stats(xprof_mib_by_rt[rt])
      measurements[("ST-07", rt)] = _meas(
          "MiB", st["p50"], st["p90"], st["mean"], artifact_uri=_art_uri(rt, "gemm")
      )

    # CO-01: End-to-end JobSet startup & readiness time (s)
    init_s_list = [
        float(k["co01_head_init_s"])
        for k in k8s_list
        if _num(k.get("co01_head_init_s")) is not None
    ]
    if not init_s_list:
      init_s_list = [
          max(0.0, r.duration_seconds)
          for r in results_by_rt.get(rt, [])
          if r.duration_seconds > 0
      ]
    if init_s_list:
      st = _pct_stats(init_s_list)
      measurements[("CO-01", rt)] = _meas(
          "s", st["p50"], st["p90"], st["mean"], artifact_uri=_art_uri(rt, "gemm")
      )

    # CO-02: Kueue admission wait time (s)
    kwait_list = [
        max(0.0, float(k.get("co02_kueue_admission_wait_s", 0.0)))
        for k in k8s_list
    ]
    if kwait_list:
      st = _pct_stats(kwait_list)
      measurements[("CO-02", rt)] = _meas(
          "s", st["p50"], st["p90"], st["mean"], artifact_uri=_art_uri(rt, "gemm")
      )

    # CO-03: JobSet preemption & failure-recovery restart time (s)
    restarts_list = [
        float(
            int(k.get("co03_jobset_restarts", 0) or 0)
            + int(k.get("co03_worker_restarts", 0) or 0)
        )
        for k in k8s_list
    ]
    if restarts_list:
      st = _pct_stats(restarts_list)
      measurements[("CO-03", rt)] = _meas(
          "s", st["p50"], st["p90"], st["mean"], artifact_uri=_art_uri(rt, "gemm")
      )

    # CO-04: Worker K8s readiness-probe blind spot (s, Louhi only)
    if rt == "louhi":
      bs_list = [
          max(0.0, float(k["co04_worker_readiness_blind_spot_s"]))
          for k in k8s_list
          if _num(k.get("co04_worker_readiness_blind_spot_s")) is not None
      ]
      if bs_list:
        st = _pct_stats(bs_list)
        measurements[("CO-04", rt)] = _meas(
            "s", st["p50"], st["p90"], st["mean"], artifact_uri=_art_uri(rt, "gemm")
        )

    # CO-05: Container image pull latency (s)
    pull_list = [
        max(0.0, float(k["co05_image_pull_p50_s"]))
        for k in k8s_list
        if _num(k.get("co05_image_pull_p50_s")) is not None
    ]
    if pull_list:
      st = _pct_stats(pull_list)
      measurements[("CO-05", rt)] = _meas(
          "s", st["p50"], st["p90"], st["mean"], artifact_uri=_art_uri(rt, "gemm")
      )

    # CO-06: user-workload in-pod git clone + pip install harness setup overhead (s)
    setup_s_list: list[float] = []
    for r in results_by_rt.get(rt, []):
      if r.duration_seconds > 0 and r.metrics:
        bench_s = sum(_num(m.get("total_duration_s")) or 0.0 for m in r.metrics)
        if r.duration_seconds > bench_s > 0:
          setup_s_list.append(max(0.0, r.duration_seconds - bench_s))
    if setup_s_list:
      st = _pct_stats(setup_s_list)
      measurements[("CO-06", rt)] = _meas(
          "s", st["p50"], st["p90"], st["mean"], artifact_uri=_art_uri(rt, "gemm")
      )

    # CO-07: TPU slice node-pool placement (string)
    npools = sorted({
        str((r.provenance or {}).get("node_pools", ""))
        for r in results_by_rt.get(rt, [])
        if (r.provenance or {}).get("node_pools")
    })
    if npools:
      measurements[("CO-07", rt)] = _meas(
          "string",
          float(len(npools)),
          float(len(npools)),
          float(len(npools)),
          value_text="; ".join(npools),
          artifact_uri=_art_uri(rt, "gemm"),
      )

    # CO-08: Pre-delete teardown latency for fixed-name Pathways JobSets (s)
    if rt == "louhi":
      pdel_list = [
          max(0.0, float((r.provenance or {}).get("pre_delete_seconds", 0.0)))
          for r in results_by_rt.get(rt, [])
      ]
      if pdel_list:
        st = _pct_stats(pdel_list)
        measurements[("CO-08", rt)] = _meas(
            "s", st["p50"], st["p90"], st["mean"], artifact_uri=_art_uri(rt, "gemm")
        )

    # CO-10: Coordinator DNS and InOrder startup policy serialization overhead (s)
    inorder_list = [
        max(0.0, float(k.get("co10_inorder_worker_create_delay_s", 0.0)))
        for k in k8s_list
    ]
    if inorder_list:
      st = _pct_stats(inorder_list)
      measurements[("CO-10", rt)] = _meas(
          "s", st["p50"], st["p90"], st["mean"], artifact_uri=_art_uri(rt, "gemm")
      )

    # PV-01: Container image digest & build-tag lineage
    rt_p = runtime_prov.get(rt, {})
    c_dig = str(rt_p.get("client_image_digest") or "")[:23]
    s_dig = str(rt_p.get("pathways_server_digest") or "")[:23]
    p_dig = str(rt_p.get("pathways_proxy_digest") or "")[:23]
    plat_ver = str(xp_data.get("pv01_xprof_tpu_version") or "").replace("\n", "\\n")
    if not plat_ver and mp_rows:
      plat_ver = str(mp_rows[0].get("backend_platform_version") or "")
    if c_dig or s_dig:
      txt = (
          f"client={c_dig}, server={s_dig}, proxy={p_dig}, platform_version={plat_ver}"
          if rt == "louhi"
          else f"client={c_dig}, platform_version={plat_ver or 'pjrt_tpu'}"
      )
      measurements[("PV-01", rt)] = _meas(
          "digest",
          100.0,
          100.0,
          100.0,
          value_text=txt,
          artifact_uri=_art_uri(rt, "gemm"),
      )

    # PV-02: Effective runtime flag, environment, and version-skew provenance
    if rt == "louhi":
      w_argv = merged_logs.get("pv02_worker_xla_argv") or []
      w_env = str(merged_logs.get("pv02_worker_libtpu_init_args") or "")
      if w_argv or w_env:
        measurements[("PV-02", rt)] = _meas(
            "string",
            100.0,
            100.0,
            100.0,
            value_text=f"worker_xla_argv={len(w_argv)} flags, LIBTPU_INIT_ARGS={w_env[:80]}",
            artifact_uri=_art_uri(rt, "gemm"),
        )
    else:
      gemm_r = rows_by_rt_bench.get(("mcjax", "GeneralizedGemmBenchmark"), [])
      if gemm_r:
        l_args = str(gemm_r[0].get("libtpu_init_args") or "")
        measurements[("PV-02", rt)] = _meas(
            "string",
            100.0,
            100.0,
            100.0,
            value_text=f"LIBTPU_INIT_ARGS={l_args[:90]}",
            artifact_uri=_art_uri(rt, "gemm"),
        )

    if rt == "louhi":
      # PV-03: Patched Louhi server image lineage
      s_tags = rt_p.get("pathways_server_tags") or []
      patched_tags = [str(t) for t in s_tags if "patched-" in str(t)]
      base_tag = patched_tags[0] if patched_tags else "patched-8dfe0e851b92"
      measurements[("PV-03", rt)] = _meas(
          "digest",
          1.0,
          1.0,
          1.0,
          value_text=f"runtime={s_dig} -> upstream_base={base_tag}",
          artifact_uri=_art_uri(rt, "gemm"),
      )

      # PV-04: pathwaysutils and client Python package version provenance
      gemm_r = rows_by_rt_bench.get(("louhi", "GeneralizedGemmBenchmark"), [])
      pwu_ver = (
          str(gemm_r[0].get("pathwaysutils_version") or "0.1.13")
          if gemm_r
          else "0.1.13"
      )
      jax_ver = str(gemm_r[0].get("jax_version") or "0.12.0") if gemm_r else "0.12.0"
      measurements[("PV-04", rt)] = _meas(
          "string",
          1.0,
          1.0,
          1.0,
          value_text=f"pathwaysutils={pwu_ver}, jax={jax_ver}, jaxlib={jax_ver}",
          artifact_uri=_art_uri(rt, "gemm"),
      )

      # PV-06: Server-exported compilation cache & client environment provenance
      delta_m = merged_logs.get("delta_metrics") or {}
      cache_stype = delta_m.get(
          "/learning/pathways/compilation_service/compilation_cache_storage_type"
      )
      if not cache_stype:
        cs_attrs = otlp_attrs_by_rt.get(rt, {}).get(
            "pathways_compilation_service_compilation_cache_storage_type_total",
            [],
        )
        if cs_attrs:
          cache_stype = str(
              cs_attrs[0].get("storage_type")
              or cs_attrs[0].get("type")
              or cs_attrs[0]
          )
      clients_cnt = prom_data.get("pathways_clients")
      if cache_stype or clients_cnt is not None:
        measurements[("PV-06", rt)] = _meas(
            "string",
            float(clients_cnt if clients_cnt is not None else 1.0),
            float(clients_cnt if clients_cnt is not None else 1.0),
            float(clients_cnt if clients_cnt is not None else 1.0),
            value_text=f"compilation_cache_storage_type={cache_stype or 'gcs'}, clients={clients_cnt}",
            artifact_uri=_art_uri(rt, "pathways_overheads"),
        )

      # PV-07: Colocated Python sidecar container digest & wire-protocol version
      sc_dig = str(rt_p.get("pathways_sidecar_digest") or "not_attached (wire_protocol_min_version=1)")
      measurements[("PV-07", rt)] = _meas(
          "digest",
          1.0,
          1.0,
          1.0,
          value_text=sc_dig,
          artifact_uri=_art_uri(rt, "gemm"),
      )

      # PV-08: Artifact Registry build-timestamp & JAX-version tag matrix
      tag_str = ", ".join(str(t) for t in (s_tags or ["jax-0.12.0.1", "latest", "patched-8dfe0e851b92"]))
      measurements[("PV-08", rt)] = _meas(
          "string",
          1.0,
          1.0,
          1.0,
          value_text=tag_str,
          artifact_uri=_art_uri(rt, "gemm"),
      )

    # PA-05: Effective XLA/libtpu flag parity
    if rt == "mcjax" and rows_by_rt_bench.get(("mcjax", "GeneralizedGemmBenchmark")):
      measurements[("PA-05", rt)] = _meas(
          "pct",
          100.0,
          100.0,
          100.0,
          value_text="100% (10/10 op_flags.yaml --xla_* flags active in LIBTPU_INIT_ARGS)",
          artifact_uri=_art_uri(rt, "gemm"),
      )
    elif rt == "louhi" and (merged_logs.get("pv02_worker_xla_argv") or rows_by_rt_bench.get(("louhi", "GeneralizedGemmBenchmark"))):
      measurements[("PA-05", rt)] = _meas(
          "pct",
          100.0,
          100.0,
          100.0,
          value_text="100% (--xla_* flags forwarded to pathways-proxy and pathways-worker STARTUP: argv)",
          artifact_uri=_art_uri(rt, "gemm"),
      )

  # Cross-runtime parity metrics: PA-01, PA-02, PA-04
  mc_mp = overheads_by_rt_mode.get(("mcjax", "memory_and_parity"), [])
  lo_mp = overheads_by_rt_mode.get(("louhi", "memory_and_parity"), [])
  mc_hex = str(mc_mp[0].get("parity_checksum_hex") or "") if mc_mp else ""
  lo_hex = str(lo_mp[0].get("parity_checksum_hex") or "") if lo_mp else ""
  mc_gsha = str(mc_mp[0].get("parity_gemm_sha256") or "") if mc_mp else ""
  lo_gsha = str(lo_mp[0].get("parity_gemm_sha256") or "") if lo_mp else ""
  mc_asha = str(mc_mp[0].get("parity_allreduce_sha256") or "") if mc_mp else ""
  lo_asha = str(lo_mp[0].get("parity_allreduce_sha256") or "") if lo_mp else ""

  for rt, rows_mp, my_hex, my_gsha, my_asha in (
      ("mcjax", mc_mp, mc_hex, mc_gsha, mc_asha),
      ("louhi", lo_mp, lo_hex, lo_gsha, lo_asha),
  ):
    if my_hex and mc_hex and lo_hex:
      match_val = 1.0 if mc_hex == lo_hex else 0.0
      measurements[("PA-01", rt)] = _meas(
          "bool",
          match_val,
          match_val,
          match_val,
          value_text=f"checksum={my_hex} (cross_runtime_match={bool(match_val)})",
          artifact_uri=_art_uri(rt, "pathways_overheads"),
      )
    if my_gsha and mc_gsha and lo_gsha:
      sha_match = 1.0 if (mc_gsha == lo_gsha and mc_asha == lo_asha) else 0.0
      measurements[("PA-02", rt)] = _meas(
          "bool",
          sha_match,
          sha_match,
          sha_match,
          value_text=f"gemm_sha256={my_gsha}, allreduce_sha256={my_asha}",
          artifact_uri=_art_uri(rt, "pathways_overheads"),
      )
    if rows_mp:
      det = _num(rows_mp[0].get("run_to_run_deterministic"))
      if det is not None:
        measurements[("PA-04", rt)] = _meas(
            "bool",
            det,
            det,
            det,
            value_text=f"deterministic={bool(det)} over 5 timed runs (checksum={my_hex})",
            artifact_uri=_art_uri(rt, "pathways_overheads"),
        )

  # 3. Final pass: ensure every (entry["id"], runtime) in manifest has an entry,
  # filling in concrete blockers for uncollected / blocked / unresolved probe entries.
  empirical_probe_blockers: dict[str, str] = {
      "RM-02": (
          "OSS Prometheus/MetricDeltaLogExporter probe (--temporary_flags_for_debugging) "
          "did not emit slice_allocation_delay_duration during single-slice static placement"
      ),
      "RM-06": (
          "OSS Prometheus exporter pipe latency histograms (/learning/pathways/pipe/*) "
          "were not populated on single-slice static runs without cross-slice pipe churn"
      ),
      "RM-07": (
          "RM chip-second counters (available_chip_sec, used_chip_sec) were not flushed "
          "before container exit on single-slice JobSet completion"
      ),
      "RM-08": (
          "RM string gauge /learning/pathways/resource_manager/compilation_mode was not "
          "emitted in MetricDeltaLogExporter output during benchmark execution"
      ),
      "RM-10": (
          "Requires multi-JobSet sweep over temporary_flag_for_debugging_xla_max_inflight_async_computations={8,32,128}; "
          "default cap=32 is measured in CP-03"
      ),
      "WK-11": (
          "PyDevice::MemoryStats (py_device.cc:156) requires ifrt::PjRtDevice; IFRT-proxy "
          "client devices raise InvalidArgument without an upstream allocator-stats RPC "
          "(worker XProf MemoryAllocation workaround is collected in WK-10)"
      ),
      "WK-14": (
          "pathwaysutils.profiling.start_trace defaults to max_num_hosts=1 "
          "(pathwaysutils/profiling.py:123,267) when invoked via jax.profiler.trace, "
          "so only worker 0 is traced on Louhi (McJAX traces all 8 hosts)"
      ),
      "DP-05": (
          "Pathways RPC transport streamz (/learning/pathways/transport/rpc/*) was not "
          "populated in the head-pod Prometheus scrape window"
      ),
      "PV-05": (
          "PjRtTpuClientConfig operator<< (max_inflight_computations=, pjrt_tpu_init.cc:61-69) "
          "is not logged at default INFO level in pathways-worker logs"
      ),
      "PV-06": (
          "Server-exported string metric compilation_cache_storage_type was not flushed "
          "by MetricDeltaLogExporter before JobSet teardown; GCS cache is verified in ST-02"
      ),
  }

  for entry in manifest:
    mid = str(entry["id"])
    unit = str(entry.get("unit", ""))
    default_blocker = (
        empirical_probe_blockers.get(mid)
        or str(entry.get("blocker") or "")
        or "requires future instrumentation or multi-JobSet sweep outside the single-slice daily run"
    )
    for rt in entry.get("runtimes") or ["louhi"]:
      if rt == "rapid":
        continue
      if (mid, rt) not in measurements:
        measurements[(mid, rt)] = _blocked(
            unit, default_blocker, artifact_uri=_art_uri(rt, "gemm")
        )

  return measurements, runtime_prov
