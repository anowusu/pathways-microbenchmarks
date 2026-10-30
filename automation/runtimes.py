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
"""Runtimes a TPUMS config can be benchmarked on, and their image provenance.

A runtime is how the benchmark process reaches the TPUs:

* `mcjax`: one JAX process per TPU host (`automation/jobset_template.yaml`).
* Pathways runtimes (`louhi`, `rapid`, or a custom one): one JAX process in
  the Pathways head drives the slice through the Pathways proxy
  (`templates/pathways_jobset.py`). They differ only in the Pathways
  server/proxy images: `louhi` is the open-source build, `rapid` the internal
  release. Both publish one image per build that carries every `-jax_<x.y.z>`
  tag as an alias, so `latest` works with any supported client JAX.
"""

from collections.abc import Iterable, Sequence
import dataclasses
import json
import logging
import os
import subprocess
from typing import Any, Optional

from automation import utils

_PATHWAYS_REGISTRY = "us-docker.pkg.dev/cloud-tpu-v2-images"


@dataclasses.dataclass(frozen=True)
class Runtime:
  """A benchmark runtime.

  Attributes:
    name: Short name used in workload names, GCS paths and reports.
    pathways_server_image: Pathways resource manager / worker image (without
      tag), or None for McJAX.
    pathways_proxy_image: Pathways proxy image (without tag), or None for
      McJAX.
    pathways_version: Image tag of both Pathways images.
  """

  name: str
  pathways_server_image: Optional[str] = None
  pathways_proxy_image: Optional[str] = None
  pathways_version: str = "latest"

  @property
  def is_pathways(self) -> bool:
    return self.pathways_server_image is not None

  def image_refs(self) -> dict[str, str]:
    """Returns the tagged Pathways image references, keyed by role."""
    if not self.is_pathways:
      return {}
    return {
        "server": f"{self.pathways_server_image}:{self.pathways_version}",
        "proxy": f"{self.pathways_proxy_image}:{self.pathways_version}",
    }


MCJAX = Runtime(name="mcjax")
LOUHI = Runtime(
    name="louhi",
    pathways_server_image=os.environ.get(
        "TPUMS_LOUHI_SERVER_IMAGE",
        f"{_PATHWAYS_REGISTRY}/cloud-pathways/server",
    ),
    pathways_proxy_image=os.environ.get(
        "TPUMS_LOUHI_PROXY_IMAGE",
        f"{_PATHWAYS_REGISTRY}/cloud-pathways/proxy_server",
    ),
    pathways_version=os.environ.get("TPUMS_LOUHI_VERSION", "latest"),
)
RAPID = Runtime(
    name="rapid",
    pathways_server_image=f"{_PATHWAYS_REGISTRY}/pathways/server",
    pathways_proxy_image=f"{_PATHWAYS_REGISTRY}/pathways/proxy_server",
)

RUNTIMES: dict[str, Runtime] = {r.name: r for r in (MCJAX, LOUHI, RAPID)}
# Order runtimes appear in reports; McJAX first because it is the baseline.
RUNTIME_ORDER: tuple[str, ...] = tuple(RUNTIMES)


def parse_runtimes(tokens: Optional[Iterable[str]]) -> list[Runtime]:
  """Resolves runtime names (comma-separated or listed) to `Runtime`s.

  Args:
    tokens: Names such as ['mcjax', 'louhi,rapid']; None or empty means
      `[MCJAX]`.

  Returns:
    The runtimes in the given order, de-duplicated.

  Raises:
    ValueError: If a name is unknown.
  """
  names = utils.split_csv_tokens(tokens) if tokens else []
  if not names:
    return [MCJAX]
  unknown = [n for n in names if n.lower() not in RUNTIMES]
  if unknown:
    raise ValueError(
        f"Unknown runtime(s) {unknown}; choose from {list(RUNTIMES)}."
    )
  return list(dict.fromkeys(RUNTIMES[n.lower()] for n in names))


def runtime_sort_key(name: str) -> tuple[int, str]:
  """Sorts known runtimes in `RUNTIME_ORDER`, unknown ones after them."""
  try:
    return (RUNTIME_ORDER.index(name), name)
  except ValueError:
    return (len(RUNTIME_ORDER), name)


def _gcloud_json(args: Sequence[str]) -> Any:
  proc = subprocess.run(
      ["gcloud", *args, "--format=json"],
      capture_output=True,
      text=True,
      check=False,
      stdin=subprocess.DEVNULL,
      timeout=utils.CLI_TIMEOUT_SECONDS,
  )
  if proc.returncode != 0:
    raise RuntimeError(
        f"gcloud {' '.join(args)} failed (exit {proc.returncode}):"
        f" {proc.stderr.strip()}"
    )
  return json.loads(proc.stdout or "null")


def resolve_image_provenance(image_ref: str) -> dict[str, Any]:
  """Resolves a tagged image to its digest and the tags sharing that digest.

  Args:
    image_ref: `<registry>/<repo>/<image>:<tag>`.

  Returns:
    `{"image": image_ref, "digest": "sha256:…", "tags": [...]}`; `digest` is
    None and `tags` empty if the lookup fails, which is logged but not fatal
    so a registry hiccup does not fail the night.
  """
  provenance: dict[str, Any] = {"image": image_ref, "digest": None, "tags": []}
  try:
    described = _gcloud_json(
        ["artifacts", "docker", "images", "describe", image_ref]
    )
    digest = (described.get("image_summary") or {}).get("digest")
    provenance["digest"] = digest
    if digest:
      repo_image = image_ref.rsplit(":", 1)[0]
      listed = _gcloud_json([
          "artifacts",
          "docker",
          "images",
          "list",
          repo_image,
          "--include-tags",
          f"--filter=DIGEST={digest}",
      ])
      tags: list[str] = []
      for item in listed or []:
        raw = item.get("tags") or []
        tags.extend(raw if isinstance(raw, list) else str(raw).split(","))
      provenance["tags"] = sorted(t for t in tags if t)
  except (RuntimeError, ValueError, subprocess.TimeoutExpired) as e:
    logging.warning("Could not resolve provenance of %s: %s", image_ref, e)
  return provenance


def resolve_git_commit(repo_url: str, ref: str) -> Optional[str]:
  """Returns the commit `ref` points at in `repo_url`, or None if unknown."""
  try:
    proc = subprocess.run(
        ["git", "ls-remote", repo_url, ref, f"refs/heads/{ref}"],
        capture_output=True,
        text=True,
        check=False,
        stdin=subprocess.DEVNULL,
        timeout=utils.CLI_TIMEOUT_SECONDS,
    )
  except (OSError, subprocess.TimeoutExpired) as e:
    logging.warning("git ls-remote %s %s failed: %s", repo_url, ref, e)
    return None
  for line in proc.stdout.splitlines():
    parts = line.split()
    if len(parts) == 2:
      return parts[0]
  return None


def runtime_provenance(
    runtime: Runtime, git_repo: str, git_branch: str
) -> dict[str, Any]:
  """Collects what identifies a runtime's build for a night's report.

  Args:
    runtime: The runtime.
    git_repo: Repository the workloads clone.
    git_branch: Branch or tag they check out.

  Returns:
    A JSON-serializable dict with `runtime`, `git_repo`, `git_branch`,
    `git_commit` and, for Pathways runtimes, `pathways_server_image`,
    `pathways_proxy_image` with their digests and tag aliases.
  """
  provenance: dict[str, Any] = {
      "runtime": runtime.name,
      "git_repo": git_repo,
      "git_branch": git_branch,
      "git_commit": resolve_git_commit(git_repo, git_branch),
  }
  for role, ref in runtime.image_refs().items():
    image = resolve_image_provenance(ref)
    provenance[f"pathways_{role}_image"] = ref
    provenance[f"pathways_{role}_digest"] = image["digest"]
    provenance[f"pathways_{role}_tags"] = image["tags"]
  return provenance
