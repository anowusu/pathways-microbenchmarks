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
"""Persistent HTTP server for TPUMS nightly/hourly dashboards and reports.

Syncs `gs://cloud-pathways-staging/tpums-nightly/reports/` into a local cache
directory in the background and serves `latest_dashboard.html` at `/` so the
browser URL never redirects to an expiring `*-apidata.googleusercontent.com`
signed token.
"""

import argparse
from http import server
import logging
import os
import pathlib
import subprocess
import threading
import time

DEFAULT_GCS_REPORTS = "gs://cloud-pathways-staging/tpums-nightly/reports"
DEFAULT_CACHE_DIR = os.path.expanduser("~/.cache/tpums-dashboard")
DEFAULT_PORT = 8788
SYNC_INTERVAL_SECONDS = 60


def sync_reports(gcs_reports: str, cache_dir: pathlib.Path) -> None:
  """Rsyncs GCS report artifacts into `cache_dir`."""
  cache_dir.mkdir(parents=True, exist_ok=True)
  try:
    subprocess.run(
        ["gcloud", "storage", "rsync", gcs_reports, str(cache_dir)],
        capture_output=True,
        text=True,
        check=False,
        stdin=subprocess.DEVNULL,
        timeout=45,
    )
  except (OSError, subprocess.TimeoutExpired) as e:
    logging.warning("GCS sync warning: %s", e)


def _sync_loop(gcs_reports: str, cache_dir: pathlib.Path, interval: int) -> None:
  while True:
    sync_reports(gcs_reports, cache_dir)
    time.sleep(interval)


class DashboardHandler(server.SimpleHTTPRequestHandler):
  """Serves cached GCS reports with `/` mapped to `latest_dashboard.html`."""

  cache_dir: pathlib.Path = pathlib.Path(DEFAULT_CACHE_DIR)
  gcs_reports: str = DEFAULT_GCS_REPORTS

  def __init__(self, *args, **kwargs):
    super().__init__(*args, directory=str(self.cache_dir), **kwargs)

  def _rewrite_root(self) -> None:
    path = self.path.split("?", 1)[0].split("#", 1)[0]
    if path in ("", "/", "/index.html"):
      self.path = "/latest_dashboard.html"

  def do_HEAD(self) -> None:  # pylint: disable=invalid-name
    self._rewrite_root()
    return super().do_HEAD()

  def do_GET(self) -> None:  # pylint: disable=invalid-name
    self._rewrite_root()
    path = self.path.split("?", 1)[0].split("#", 1)[0]
    if path == "/sync":
      sync_reports(self.gcs_reports, self.cache_dir)
      self.send_response(302)
      self.send_header("Location", "/latest_dashboard.html")
      self.end_headers()
      return
    return super().do_GET()

  def end_headers(self) -> None:
    self.send_header("Cache-Control", "no-cache, max-age=0")
    super().end_headers()


def main() -> None:
  parser = argparse.ArgumentParser(description="Serve TPUMS dashboards.")
  parser.add_argument("--port", type=int, default=DEFAULT_PORT)
  parser.add_argument("--gcs-reports", default=DEFAULT_GCS_REPORTS)
  parser.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
  parser.add_argument("--sync-interval", type=int, default=SYNC_INTERVAL_SECONDS)
  args = parser.parse_args()

  logging.basicConfig(
      level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
  )
  cache_dir = pathlib.Path(args.cache_dir)
  DashboardHandler.cache_dir = cache_dir
  DashboardHandler.gcs_reports = args.gcs_reports

  sync_reports(args.gcs_reports, cache_dir)
  t = threading.Thread(
      target=_sync_loop,
      args=(args.gcs_reports, cache_dir, args.sync_interval),
      daemon=True,
  )
  t.start()

  httpd = server.ThreadingHTTPServer(("0.0.0.0", args.port), DashboardHandler)
  logging.info(
      "Serving TPUMS dashboards on http://0.0.0.0:%d/ (syncing %s -> %s)",
      args.port,
      args.gcs_reports,
      cache_dir,
  )
  httpd.serve_forever()


if __name__ == "__main__":
  main()
