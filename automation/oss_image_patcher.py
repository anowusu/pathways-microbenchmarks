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
"""On-the-fly OCI image patcher for OSS Pathways (`cloud-pathways/server`).

Until cl/1010067962 (which registers the PJRT TPU profiler extension with
`tsl::ProfilerSession` in `third_party/pathways`) is merged and rolled out to
the public Louhi image (`us-docker.pkg.dev/cloud-tpu-v2-images/cloud-pathways/server:latest`),
unpatched OSS `pathways_server` worker containers only record `/host:CPU`
XPlanes during XProf sessions and omit `/device:TPU:*` device traces.

This module patches the live upstream `cloud-pathways/server:latest` image on
the fly at the start of each CronJob run without requiring a Docker daemon or
root privileges:
  1. Compiles (or loads pre-compiled) `libpjrt_profiler_patch.so` from
     `automation/oss_patch/pjrt_profiler_patch.c`.
  2. Fetches the current `cloud-pathways/server:latest` manifest and config
     blob via the Artifact Registry OCI HTTP v2 API.
  3. Appends a single deterministic tar.gz layer containing
     `/opt/pathways/libpjrt_profiler_patch.so` and sets
     `LD_PRELOAD=/opt/pathways/libpjrt_profiler_patch.so` in the image `Env`.
     (If the upstream binary later incorporates native PJRT profiler
     registration, `libpjrt_profiler_patch.so` detects it at startup and
     becomes a no-op.)
  4. Pushes the patched manifest to the target Artifact Registry repository
     (`us-docker.pkg.dev/cloud-tpu-multipod-dev/pathways-dev/anowusu/server`)
     under `patched-<upstream_digest_short12>`, `latest`, and all upstream
     tag aliases (`sha-...`, `<YYYYMMDD_HHMM>`, etc.), and returns an updated
     `Runtime` for `louhi` paired with the matching upstream `proxy_server`.
"""

from collections.abc import Sequence
import copy
import dataclasses
import gzip
import hashlib
import io
import json
import logging
import os
import pathlib
import re
import subprocess
import tarfile
import tempfile
from typing import Any, Optional
import urllib.error
import urllib.parse
import urllib.request

from automation import runtimes as runtimes_lib
from automation import utils

DEFAULT_OSS_SOURCE_SERVER_IMAGE = (
    "us-docker.pkg.dev/cloud-tpu-v2-images/cloud-pathways/server"
)
DEFAULT_OSS_SOURCE_PROXY_IMAGE = (
    "us-docker.pkg.dev/cloud-tpu-v2-images/cloud-pathways/proxy_server"
)
DEFAULT_OSS_SOURCE_TAG = "latest"
DEFAULT_PATCHED_SERVER_IMAGE = (
    "us-docker.pkg.dev/cloud-tpu-multipod-dev/pathways-dev/anowusu/server"
)

PRELOAD_TARGET_RELPATH = "opt/pathways/libpjrt_profiler_patch.so"
PRELOAD_CONTAINER_PATH = "/" + PRELOAD_TARGET_RELPATH
PREBUILT_SO_PATH = pathlib.Path("/opt/tpums/libpjrt_profiler_patch.so")
PATCH_C_SOURCE = (
    pathlib.Path(__file__).resolve().parent
    / "oss_patch"
    / "pjrt_profiler_patch.c"
)

_MANIFEST_ACCEPT = ", ".join([
    "application/vnd.docker.distribution.manifest.v2+json",
    "application/vnd.oci.image.manifest.v1+json",
])
_STREAM_CHUNK_BYTES = 16 * 1024 * 1024
_HTTP_TIMEOUT_SECONDS = 120


@dataclasses.dataclass(frozen=True)
class PatchedOssRuntimeResult:
  """Result of patching the upstream OSS Pathways server image on the fly."""

  runtime: runtimes_lib.Runtime
  source_server_image: str
  source_digest: str
  patched_server_image: str
  patched_digest: str
  patched_tag: str
  mirrored_tags: tuple[str, ...]
  already_existed: bool


def parse_image_repo(image_without_tag: str) -> tuple[str, str]:
  """Splits `<registry_host>/<repo_path>` into `(registry_host, repo_path)`."""
  cleaned = image_without_tag.strip()
  if "://" in cleaned:
    cleaned = cleaned.split("://", 1)[1]
  parts = cleaned.split("/", 1)
  if len(parts) != 2 or not parts[0] or not parts[1]:
    raise ValueError(
        f"Invalid container image repository {image_without_tag!r}; expected"
        " '<registry>/<project>/<repo>/<image>'."
    )
  return parts[0], parts[1]


def get_access_token() -> str:
  """Obtains an OAuth2 bearer token from GCE metadata server or gcloud."""
  meta_url = (
      "http://metadata.google.internal/computeMetadata/v1/"
      "instance/service-accounts/default/token"
  )
  req = urllib.request.Request(
      meta_url, headers={"Metadata-Flavor": "Google"}, method="GET"
  )
  try:
    with urllib.request.urlopen(req, timeout=3) as resp:
      payload = json.loads(resp.read().decode("utf-8"))
      token = payload.get("access_token")
      if token:
        return str(token)
  except Exception:  # pylint: disable=broad-exception-caught
    pass

  proc = subprocess.run(
      ["gcloud", "auth", "print-access-token"],
      capture_output=True,
      text=True,
      check=False,
      stdin=subprocess.DEVNULL,
      timeout=utils.CLI_TIMEOUT_SECONDS,
  )
  if proc.returncode != 0 or not proc.stdout.strip():
    raise RuntimeError(
        "Failed to obtain GCP access token from metadata server or gcloud:"
        f" {(proc.stderr or '').strip()}"
    )
  return proc.stdout.strip()


def compile_patch_library(
    c_source: pathlib.Path = PATCH_C_SOURCE,
    prebuilt_so: pathlib.Path = PREBUILT_SO_PATH,
) -> bytes:
  """Returns the compiled `libpjrt_profiler_patch.so` ELF bytes."""
  if prebuilt_so.is_file():
    return prebuilt_so.read_bytes()
  if not c_source.is_file():
    raise FileNotFoundError(
        f"PJRT profiler patch C source not found at {c_source}"
    )
  with tempfile.TemporaryDirectory() as tmpdir:
    out_so = pathlib.Path(tmpdir) / "libpjrt_profiler_patch.so"
    cmd = [
        "gcc",
        "-shared",
        "-fPIC",
        "-O2",
        "-Wall",
        "-Wextra",
        str(c_source),
        "-ldl",
        "-lpthread",
        "-o",
        str(out_so),
    ]
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=False,
        stdin=subprocess.DEVNULL,
        timeout=utils.CLI_TIMEOUT_SECONDS,
    )
    if proc.returncode != 0 or not out_so.is_file():
      raise RuntimeError(
          f"Failed to compile {c_source} (exit {proc.returncode}):"
          f" {(proc.stderr or '').strip()}"
      )
    return out_so.read_bytes()


def build_patch_tar_layer(
    so_bytes: bytes,
    target_relpath: str = PRELOAD_TARGET_RELPATH,
) -> tuple[bytes, str, str]:
  """Builds a deterministic gzipped tar layer containing `so_bytes`.

  Args:
    so_bytes: Compiled shared library bytes.
    target_relpath: Path inside the tar archive (without leading slash).

  Returns:
    `(compressed_targz_bytes, layer_digest, uncompressed_diff_id)`.
  """
  relpath = target_relpath.lstrip("/")
  tar_buf = io.BytesIO()
  with tarfile.open(fileobj=tar_buf, mode="w", format=tarfile.USTAR_FORMAT) as tf:
    parts = relpath.split("/")
    for i in range(1, len(parts)):
      dir_info = tarfile.TarInfo(name="/".join(parts[:i]) + "/")
      dir_info.type = tarfile.DIRTYPE
      dir_info.mode = 0o755
      dir_info.uid = 0
      dir_info.gid = 0
      dir_info.uname = "root"
      dir_info.gname = "root"
      dir_info.mtime = 0
      tf.addfile(dir_info)

    info = tarfile.TarInfo(name=relpath)
    info.size = len(so_bytes)
    info.mode = 0o755
    info.uid = 0
    info.gid = 0
    info.uname = "root"
    info.gname = "root"
    info.mtime = 0
    tf.addfile(info, io.BytesIO(so_bytes))

  uncompressed_bytes = tar_buf.getvalue()
  diff_id = "sha256:" + hashlib.sha256(uncompressed_bytes).hexdigest()
  compressed_bytes = gzip.compress(uncompressed_bytes, compresslevel=9, mtime=0)
  layer_digest = "sha256:" + hashlib.sha256(compressed_bytes).hexdigest()
  return compressed_bytes, layer_digest, diff_id


def _merge_ld_preload(
    env_list: Sequence[str], preload_path: str
) -> list[str]:
  """Adds `preload_path` to `LD_PRELOAD` in an OCI `Env` list."""
  updated: list[str] = []
  found = False
  for entry in env_list:
    if entry.startswith("LD_PRELOAD="):
      found = True
      existing = entry.split("=", 1)[1].strip()
      tokens = [t for t in re.split(r"[:\s]+", existing) if t]
      if preload_path not in tokens:
        tokens.insert(0, preload_path)
      updated.append("LD_PRELOAD=" + ":".join(tokens))
    else:
      updated.append(entry)
  if not found:
    updated.append(f"LD_PRELOAD={preload_path}")
  return updated


def patch_oci_config_and_manifest(
    manifest: dict[str, Any],
    config: dict[str, Any],
    *,
    layer_gz_size: int,
    layer_digest: str,
    diff_id: str,
    preload_path: str = PRELOAD_CONTAINER_PATH,
    source_image_ref: str = "",
    source_digest: str = "",
) -> tuple[bytes, str, bytes, str]:
  """Appends the patch layer and `LD_PRELOAD` env to `config` and `manifest`.

  Args:
    manifest: Parsed upstream OCI / Docker v2 image manifest dict.
    config: Parsed upstream image config dict.
    layer_gz_size: Byte size of the compressed patch layer tar.gz.
    layer_digest: `sha256:...` digest of the compressed patch layer tar.gz.
    diff_id: `sha256:...` digest of the uncompressed patch layer tar.
    preload_path: Container path of `libpjrt_profiler_patch.so`.
    source_image_ref: Upstream image reference for provenance labels.
    source_digest: Upstream manifest digest for provenance labels.

  Returns:
    `(new_config_bytes, new_config_digest, new_manifest_bytes, new_manifest_digest)`.
  """
  new_config = copy.deepcopy(config)
  rootfs = new_config.setdefault("rootfs", {"type": "layers", "diff_ids": []})
  diff_ids = list(rootfs.get("diff_ids") or [])
  diff_ids.append(diff_id)
  rootfs["diff_ids"] = diff_ids

  cfg_section = new_config.setdefault("config", {})
  cfg_section["Env"] = _merge_ld_preload(
      cfg_section.get("Env") or [], preload_path
  )
  labels = dict(cfg_section.get("Labels") or {})
  labels["tpums.pjrt_profiler_patched"] = "true"
  if source_image_ref:
    labels["tpums.oss_source_image"] = source_image_ref
  if source_digest:
    labels["tpums.oss_source_digest"] = source_digest
  cfg_section["Labels"] = labels

  history = list(new_config.get("history") or [])
  history.append({
      "created": "1970-01-01T00:00:00Z",
      "created_by": (
          "tpums oss_image_patcher: inject libpjrt_profiler_patch.so"
          f" ({preload_path})"
      ),
  })
  new_config["history"] = history

  new_config_bytes = json.dumps(
      new_config, separators=(",", ":"), sort_keys=True
  ).encode("utf-8")
  new_config_digest = "sha256:" + hashlib.sha256(new_config_bytes).hexdigest()

  new_manifest = copy.deepcopy(manifest)
  layers = list(new_manifest.get("layers") or [])
  base_layer_media_type = (
      layers[-1].get("mediaType")
      if layers
      else "application/vnd.docker.image.rootfs.diff.tar.gzip"
  )
  layers.append({
      "mediaType": base_layer_media_type,
      "size": layer_gz_size,
      "digest": layer_digest,
  })
  new_manifest["layers"] = layers
  new_manifest["config"] = {
      "mediaType": (
          (new_manifest.get("config") or {}).get("mediaType")
          or "application/vnd.docker.container.image.v1+json"
      ),
      "size": len(new_config_bytes),
      "digest": new_config_digest,
  }
  new_manifest_bytes = json.dumps(
      new_manifest, separators=(",", ":"), sort_keys=True
  ).encode("utf-8")
  new_manifest_digest = (
      "sha256:" + hashlib.sha256(new_manifest_bytes).hexdigest()
  )
  return (
      new_config_bytes,
      new_config_digest,
      new_manifest_bytes,
      new_manifest_digest,
  )


def select_preferred_version_tag(
    server_tags: Sequence[str],
    proxy_tags: Sequence[str],
    fallback_tag: str = "latest",
) -> str:
  """Selects the most specific immutable tag shared by `server` and `proxy`.

  Prefers:
    1. `sha-<commit>` tags (sorted descending)
    2. `<YYYYMMDD>_<HHMM>` date build tags (sorted descending)
    3. `<YYYYMMDD>_<HHMM>-jax_*` tags (sorted descending)
    4. Any other non-`latest` shared tag
    5. `fallback_tag`
  """
  candidates = set(server_tags)
  if proxy_tags:
    shared = candidates & set(proxy_tags)
    if shared:
      candidates = shared

  sha_tags = sorted(
      (t for t in candidates if t.startswith("sha-")), reverse=True
  )
  if sha_tags:
    return sha_tags[0]

  date_only_tags = sorted(
      (t for t in candidates if re.fullmatch(r"\d{8}_\d{4}", t)), reverse=True
  )
  if date_only_tags:
    return date_only_tags[0]

  date_jax_tags = sorted(
      (t for t in candidates if re.match(r"^\d{8}_\d{4}-jax_", t)), reverse=True
  )
  if date_jax_tags:
    return date_jax_tags[0]

  other_tags = sorted((t for t in candidates if t != "latest"), reverse=True)
  if other_tags:
    return other_tags[0]

  return fallback_tag


class OciRegistryClient:
  """Minimal OCI Distribution Spec v2 HTTP client for Artifact Registry."""

  def __init__(self, registry_host: str, token: Optional[str] = None):
    self.registry_host = registry_host
    self._token = token or get_access_token()

  def _request(
      self,
      method: str,
      path_or_url: str,
      *,
      headers: Optional[dict[str, str]] = None,
      data: Optional[bytes] = None,
  ) -> urllib.response.addinfourl:
    if path_or_url.startswith("http://") or path_or_url.startswith("https://"):
      url = path_or_url
    else:
      url = f"https://{self.registry_host}{path_or_url}"
    req_headers = {"Authorization": f"Bearer {self._token}"}
    if headers:
      req_headers.update(headers)
    req = urllib.request.Request(
        url, data=data, headers=req_headers, method=method
    )
    return urllib.request.urlopen(req, timeout=_HTTP_TIMEOUT_SECONDS)

  def get_manifest(self, repo: str, ref: str) -> tuple[dict[str, Any], str]:
    """Fetches a manifest by tag or digest and returns `(manifest, digest)`."""
    with self._request(
        "GET",
        f"/v2/{repo}/manifests/{ref}",
        headers={"Accept": _MANIFEST_ACCEPT},
    ) as resp:
      raw = resp.read()
      digest = resp.headers.get("Docker-Content-Digest") or (
          "sha256:" + hashlib.sha256(raw).hexdigest()
      )
      return json.loads(raw.decode("utf-8")), digest

  def head_manifest(self, repo: str, ref: str) -> Optional[str]:
    """Returns the manifest digest if `ref` exists in `repo`, else None."""
    try:
      with self._request(
          "HEAD",
          f"/v2/{repo}/manifests/{ref}",
          headers={"Accept": _MANIFEST_ACCEPT},
      ) as resp:
        return resp.headers.get("Docker-Content-Digest")
    except urllib.error.HTTPError as e:
      if e.code == 404:
        return None
      raise

  def get_blob(self, repo: str, digest: str) -> bytes:
    """Downloads a blob by digest."""
    with self._request("GET", f"/v2/{repo}/blobs/{digest}") as resp:
      return resp.read()

  def blob_exists(self, repo: str, digest: str) -> bool:
    """Returns True if `digest` already exists in `repo`."""
    try:
      with self._request("HEAD", f"/v2/{repo}/blobs/{digest}"):
        return True
    except urllib.error.HTTPError as e:
      if e.code == 404:
        return False
      raise

  def upload_blob(self, repo: str, data: bytes, digest: str) -> None:
    """Uploads an in-memory blob to `repo` if not already present."""
    if self.blob_exists(repo, digest):
      return
    with self._request("POST", f"/v2/{repo}/blobs/uploads/") as resp:
      loc = resp.headers.get("Location")
    if not loc:
      raise RuntimeError(f"Missing Location header starting upload to {repo}")
    sep = "&" if "?" in loc else "?"
    put_url = f"{loc}{sep}digest={urllib.parse.quote(digest)}"
    with self._request(
        "PUT",
        put_url,
        headers={
            "Content-Type": "application/octet-stream",
            "Content-Length": str(len(data)),
        },
        data=data,
    ):
      pass

  def ensure_blob_from_repo(
      self, src_repo: str, dst_repo: str, digest: str
  ) -> None:
    """Ensures `digest` exists in `dst_repo`, mounting or streaming from `src_repo`."""
    if self.blob_exists(dst_repo, digest):
      return

    mount_path = (
        f"/v2/{dst_repo}/blobs/uploads/"
        f"?mount={urllib.parse.quote(digest)}&from={urllib.parse.quote(src_repo)}"
    )
    with self._request("POST", mount_path) as resp:
      if resp.status == 201:
        return
      loc = resp.headers.get("Location")

    if not loc:
      raise RuntimeError(
          f"Blob mount for {digest} did not complete and returned no upload URL"
      )

    # Cross-project mount returned 202 with an upload session URL; stream chunks.
    offset = 0
    with self._request("GET", f"/v2/{src_repo}/blobs/{digest}") as src_resp:
      while True:
        chunk = src_resp.read(_STREAM_CHUNK_BYTES)
        if not chunk:
          break
        end = offset + len(chunk) - 1
        with self._request(
            "PATCH",
            loc,
            headers={
                "Content-Type": "application/octet-stream",
                "Content-Length": str(len(chunk)),
                "Content-Range": f"{offset}-{end}",
            },
            data=chunk,
        ) as patch_resp:
          loc = patch_resp.headers.get("Location") or loc
        offset += len(chunk)

    sep = "&" if "?" in loc else "?"
    put_url = f"{loc}{sep}digest={urllib.parse.quote(digest)}"
    with self._request(
        "PUT",
        put_url,
        headers={
            "Content-Type": "application/octet-stream",
            "Content-Length": "0",
        },
        data=b"",
    ):
      pass

  def put_manifest(
      self, repo: str, ref: str, manifest_bytes: bytes, media_type: str
  ) -> str:
    """Uploads `manifest_bytes` to `repo` under tag `ref`."""
    with self._request(
        "PUT",
        f"/v2/{repo}/manifests/{ref}",
        headers={
            "Content-Type": media_type,
            "Content-Length": str(len(manifest_bytes)),
        },
        data=manifest_bytes,
    ) as resp:
      return resp.headers.get("Docker-Content-Digest") or (
          "sha256:" + hashlib.sha256(manifest_bytes).hexdigest()
      )

  def list_tags_for_digest(self, repo: str, digest: str) -> list[str]:
    """Returns all tags pointing to `digest` in `repo` via `/v2/<repo>/tags/list`."""
    try:
      with self._request("GET", f"/v2/{repo}/tags/list") as resp:
        data = json.loads(resp.read().decode("utf-8"))
      manifest_map = data.get("manifest") or {}
      entry = manifest_map.get(digest) or {}
      tags = entry.get("tag") or []
      return sorted(str(t) for t in tags if t)
    except Exception:  # pylint: disable=broad-exception-caught
      return []


def patch_oss_server_image(
    *,
    source_server_image: str = DEFAULT_OSS_SOURCE_SERVER_IMAGE,
    source_proxy_image: str = DEFAULT_OSS_SOURCE_PROXY_IMAGE,
    source_tag: str = DEFAULT_OSS_SOURCE_TAG,
    patched_server_image: str = DEFAULT_PATCHED_SERVER_IMAGE,
    runtime_name: str = "louhi",
    client: Optional[OciRegistryClient] = None,
    so_bytes: Optional[bytes] = None,
) -> PatchedOssRuntimeResult:
  """Patches the live OSS `pathways_server` image on the fly and pushes it.

  Args:
    source_server_image: Upstream OSS server image repository (without tag).
    source_proxy_image: Upstream OSS proxy_server image repository (without tag).
    source_tag: Upstream tag to track (default: `'latest'`).
    patched_server_image: Destination Artifact Registry image repository
      (without tag) where the patched server image is pushed.
    runtime_name: Runtime name to assign (default: `'louhi'`).
    client: Optional pre-initialized `OciRegistryClient` (for testing).
    so_bytes: Optional pre-compiled `libpjrt_profiler_patch.so` bytes.

  Returns:
    `PatchedOssRuntimeResult` with the updated `Runtime` and image provenance.
  """
  src_host, src_repo = parse_image_repo(source_server_image)
  dst_host, dst_repo = parse_image_repo(patched_server_image)
  if src_host != dst_host:
    raise ValueError(
        f"Source registry {src_host!r} and destination registry {dst_host!r}"
        " must match."
    )

  if client is None:
    client = OciRegistryClient(src_host)
  if so_bytes is None:
    so_bytes = compile_patch_library()

  layer_gz, layer_digest, diff_id = build_patch_tar_layer(so_bytes)

  src_ref = f"{source_server_image}:{source_tag}"
  proxy_ref = f"{source_proxy_image}:{source_tag}"
  logging.info("Inspecting upstream OSS Pathways image %s ...", src_ref)
  manifest, src_digest = client.get_manifest(src_repo, source_tag)
  config_digest = (manifest.get("config") or {}).get("digest")
  if not config_digest:
    raise RuntimeError(f"Upstream manifest for {src_ref} has no config.digest")
  config_bytes = client.get_blob(src_repo, config_digest)
  config = json.loads(config_bytes.decode("utf-8"))

  (
      new_config_bytes,
      new_config_digest,
      new_manifest_bytes,
      new_manifest_digest,
  ) = patch_oci_config_and_manifest(
      manifest,
      config,
      layer_gz_size=len(layer_gz),
      layer_digest=layer_digest,
      diff_id=diff_id,
      source_image_ref=src_ref,
      source_digest=src_digest,
  )

  short_src = src_digest.split(":", 1)[-1][:12]
  patched_tag = f"patched-{short_src}"

  # Discover upstream tag aliases on server and proxy_server so we can mirror
  # them onto the patched server image and pick an immutable shared tag.
  _, proxy_repo = parse_image_repo(source_proxy_image)
  upstream_server_tags = client.list_tags_for_digest(src_repo, src_digest)
  if not upstream_server_tags:
    src_prov = runtimes_lib.resolve_image_provenance(src_ref)
    upstream_server_tags = src_prov.get("tags") or [source_tag]

  proxy_digest = client.head_manifest(proxy_repo, source_tag)
  upstream_proxy_tags = (
      client.list_tags_for_digest(proxy_repo, proxy_digest)
      if proxy_digest
      else []
  )
  if not upstream_proxy_tags:
    upstream_proxy_tags = list(upstream_server_tags)

  tags_to_publish = sorted(
      set(upstream_server_tags) | {source_tag, "latest", patched_tag}
  )

  existing_digest = client.head_manifest(dst_repo, patched_tag)
  already_existed = existing_digest == new_manifest_digest

  if already_existed:
    logging.info(
        "Patched OSS server image %s:%s (%s) already up-to-date for upstream"
        " %s.",
        patched_server_image,
        patched_tag,
        new_manifest_digest,
        src_digest,
    )
  else:
    base_layers = manifest.get("layers") or []
    logging.info(
        "Syncing %d base layer(s) + PJRT profiler patch layer (%d bytes) to"
        " %s:%s ...",
        len(base_layers),
        len(layer_gz),
        patched_server_image,
        patched_tag,
    )
    for layer in base_layers:
      ldigest = layer.get("digest")
      if ldigest:
        client.ensure_blob_from_repo(src_repo, dst_repo, ldigest)

    client.upload_blob(dst_repo, layer_gz, layer_digest)
    client.upload_blob(dst_repo, new_config_bytes, new_config_digest)

  manifest_media_type = (
      manifest.get("mediaType")
      or "application/vnd.docker.distribution.manifest.v2+json"
  )
  existing_dst_tags = (
      set(client.list_tags_for_digest(dst_repo, new_manifest_digest))
      if already_existed
      else set()
  )
  for tag in tags_to_publish:
    if tag in existing_dst_tags:
      continue
    client.put_manifest(dst_repo, tag, new_manifest_bytes, manifest_media_type)

  preferred_version = select_preferred_version_tag(
      tags_to_publish, upstream_proxy_tags, fallback_tag=source_tag
  )
  logging.info(
      "On-the-fly OSS Pathways patch ready: server=%s:%s (%s, upstream=%s),"
      " proxy=%s:%s",
      patched_server_image,
      preferred_version,
      new_manifest_digest,
      src_digest,
      source_proxy_image,
      preferred_version,
  )

  runtime = runtimes_lib.Runtime(
      name=runtime_name,
      pathways_server_image=patched_server_image,
      pathways_proxy_image=source_proxy_image,
      pathways_version=preferred_version,
  )
  return PatchedOssRuntimeResult(
      runtime=runtime,
      source_server_image=f"{source_server_image}:{source_tag}",
      source_digest=src_digest,
      patched_server_image=patched_server_image,
      patched_digest=new_manifest_digest,
      patched_tag=patched_tag,
      mirrored_tags=tuple(tags_to_publish),
      already_existed=already_existed,
  )


def maybe_patch_runtimes_on_the_fly(
    runtimes: Sequence[runtimes_lib.Runtime],
    *,
    enabled: bool = False,
    dry_run: bool = False,
    source_server_image: str = DEFAULT_OSS_SOURCE_SERVER_IMAGE,
    source_proxy_image: str = DEFAULT_OSS_SOURCE_PROXY_IMAGE,
    source_tag: str = DEFAULT_OSS_SOURCE_TAG,
    patched_server_image: str = DEFAULT_PATCHED_SERVER_IMAGE,
) -> list[runtimes_lib.Runtime]:
  """Patches any `louhi` runtime in `runtimes` on the fly when `enabled`."""
  if not enabled:
    return list(runtimes)
  has_louhi = any(r.name == "louhi" for r in runtimes)
  if not has_louhi:
    return list(runtimes)
  if dry_run:
    logging.info(
        "[Dry-Run] Would patch %s:%s on the fly -> %s",
        source_server_image,
        source_tag,
        patched_server_image,
    )
    return [
        runtimes_lib.Runtime(
            name="louhi",
            pathways_server_image=patched_server_image,
            pathways_proxy_image=source_proxy_image,
            pathways_version=source_tag,
        )
        if r.name == "louhi"
        else r
        for r in runtimes
    ]

  patched = patch_oss_server_image(
      source_server_image=source_server_image,
      source_proxy_image=source_proxy_image,
      source_tag=source_tag,
      patched_server_image=patched_server_image,
  )
  return [patched.runtime if r.name == "louhi" else r for r in runtimes]
