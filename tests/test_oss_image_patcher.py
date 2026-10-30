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
"""Unit tests for automation.oss_image_patcher."""

import gzip
import hashlib
import io
import json
import tarfile
import unittest

from automation import oss_image_patcher
from automation import runtimes


class _FakeOciClient:
  """In-memory fake OciRegistryClient for unit testing."""

  def __init__(self):
    self.manifests: dict[tuple[str, str], tuple[bytes, str]] = {}
    self.blobs: dict[tuple[str, str], bytes] = {}
    self.tags_by_digest: dict[tuple[str, str], list[str]] = {}
    self.ensured_blobs: list[tuple[str, str, str]] = []

  def seed_image(
      self,
      repo: str,
      tag: str,
      manifest: dict[str, object],
      config: dict[str, object],
      extra_tags: list[str],
  ) -> str:
    cfg_bytes = json.dumps(config, separators=(",", ":"), sort_keys=True).encode(
        "utf-8"
    )
    cfg_digest = "sha256:" + hashlib.sha256(cfg_bytes).hexdigest()
    self.blobs[(repo, cfg_digest)] = cfg_bytes

    manifest = dict(manifest)
    manifest["config"] = {
        "mediaType": "application/vnd.oci.image.config.v1+json",
        "size": len(cfg_bytes),
        "digest": cfg_digest,
    }
    m_bytes = json.dumps(
        manifest, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    m_digest = "sha256:" + hashlib.sha256(m_bytes).hexdigest()
    for t in [tag, *extra_tags]:
      self.manifests[(repo, t)] = (m_bytes, m_digest)
    self.tags_by_digest[(repo, m_digest)] = sorted({tag, *extra_tags})
    return m_digest

  def get_manifest(self, repo: str, ref: str) -> tuple[dict[str, object], str]:
    raw, digest = self.manifests[(repo, ref)]
    return json.loads(raw.decode("utf-8")), digest

  def head_manifest(self, repo: str, ref: str):
    item = self.manifests.get((repo, ref))
    return item[1] if item else None

  def get_blob(self, repo: str, digest: str) -> bytes:
    return self.blobs[(repo, digest)]

  def blob_exists(self, repo: str, digest: str) -> bool:
    return (repo, digest) in self.blobs

  def upload_blob(self, repo: str, data: bytes, digest: str) -> None:
    self.blobs[(repo, digest)] = data

  def ensure_blob_from_repo(
      self, src_repo: str, dst_repo: str, digest: str
  ) -> None:
    self.ensured_blobs.append((src_repo, dst_repo, digest))
    self.blobs[(dst_repo, digest)] = self.blobs.get(
        (src_repo, digest), b"layer-bytes"
    )

  def put_manifest(
      self, repo: str, ref: str, manifest_bytes: bytes, media_type: str
  ) -> str:
    del media_type
    digest = "sha256:" + hashlib.sha256(manifest_bytes).hexdigest()
    self.manifests[(repo, ref)] = (manifest_bytes, digest)
    tags = set(self.tags_by_digest.get((repo, digest), []))
    tags.add(ref)
    self.tags_by_digest[(repo, digest)] = sorted(tags)
    return digest

  def list_tags_for_digest(self, repo: str, digest: str) -> list[str]:
    return list(self.tags_by_digest.get((repo, digest), []))


class OssImagePatcherTest(unittest.TestCase):

  def test_parse_image_repo(self):
    host, repo = oss_image_patcher.parse_image_repo(
        "us-docker.pkg.dev/cloud-tpu-v2-images/cloud-pathways/server"
    )
    self.assertEqual(host, "us-docker.pkg.dev")
    self.assertEqual(repo, "cloud-tpu-v2-images/cloud-pathways/server")
    with self.assertRaises(ValueError):
      oss_image_patcher.parse_image_repo("invalid_repo_without_slash")

  def test_build_patch_tar_layer_is_deterministic(self):
    payload = b"\x7fELF-mock-so-payload"
    gz1, ldig1, diff1 = oss_image_patcher.build_patch_tar_layer(payload)
    gz2, ldig2, diff2 = oss_image_patcher.build_patch_tar_layer(payload)
    self.assertEqual(gz1, gz2)
    self.assertEqual(ldig1, ldig2)
    self.assertEqual(diff1, diff2)

    raw_tar = gzip.decompress(gz1)
    self.assertEqual(
        diff1, "sha256:" + hashlib.sha256(raw_tar).hexdigest()
    )
    with tarfile.open(fileobj=io.BytesIO(raw_tar), mode="r") as tf:
      names = tf.getnames()
      self.assertIn("opt/pathways/libpjrt_profiler_patch.so", names)
      extracted = tf.extractfile("opt/pathways/libpjrt_profiler_patch.so").read()
      self.assertEqual(extracted, payload)

  def test_patch_oci_config_and_manifest_injects_ld_preload_and_layer(self):
    manifest = {
        "schemaVersion": 2,
        "mediaType": "application/vnd.oci.image.manifest.v1+json",
        "config": {
            "mediaType": "application/vnd.oci.image.config.v1+json",
            "size": 100,
            "digest": "sha256:oldcfg",
        },
        "layers": [{
            "mediaType": "application/vnd.oci.image.layer.v1.tar+gzip",
            "size": 1234,
            "digest": "sha256:baselayer",
        }],
    }
    config = {
        "architecture": "amd64",
        "os": "linux",
        "config": {
            "Env": ["PATH=/usr/bin", "TPU_LIBRARY_PATH=/lib/libtpu.so"],
            "Entrypoint": ["/opt/pathways/pathways_server"],
        },
        "rootfs": {"type": "layers", "diff_ids": ["sha256:basediff"]},
    }
    cfg_bytes, cfg_digest, m_bytes, m_digest = (
        oss_image_patcher.patch_oci_config_and_manifest(
            manifest,
            config,
            layer_gz_size=555,
            layer_digest="sha256:patchlayer",
            diff_id="sha256:patchdiff",
            source_image_ref="us-docker.pkg.dev/src/server:latest",
            source_digest="sha256:srcdigest",
        )
    )
    new_cfg = json.loads(cfg_bytes.decode("utf-8"))
    new_m = json.loads(m_bytes.decode("utf-8"))

    self.assertIn(
        "LD_PRELOAD=/opt/pathways/libpjrt_profiler_patch.so",
        new_cfg["config"]["Env"],
    )
    self.assertEqual(
        new_cfg["rootfs"]["diff_ids"], ["sha256:basediff", "sha256:patchdiff"]
    )
    self.assertEqual(
        new_cfg["config"]["Labels"]["tpums.oss_source_digest"],
        "sha256:srcdigest",
    )
    self.assertEqual(len(new_m["layers"]), 2)
    self.assertEqual(new_m["layers"][-1]["digest"], "sha256:patchlayer")
    self.assertEqual(new_m["config"]["digest"], cfg_digest)
    self.assertTrue(m_digest.startswith("sha256:"))

  def test_select_preferred_version_tag(self):
    server_tags = ["latest", "20261029_1817", "sha-abc1234"]
    proxy_tags = ["latest", "20261029_1817", "sha-abc1234"]
    self.assertEqual(
        oss_image_patcher.select_preferred_version_tag(server_tags, proxy_tags),
        "sha-abc1234",
    )
    self.assertEqual(
        oss_image_patcher.select_preferred_version_tag(
            ["latest", "20261029_1817"], ["latest", "20261029_1817"]
        ),
        "20261029_1817",
    )

  def test_patch_oss_server_image_and_idempotency(self):
    fake = _FakeOciClient()
    src_repo = "cloud-tpu-v2-images/cloud-pathways/server"
    proxy_repo = "cloud-tpu-v2-images/cloud-pathways/proxy_server"
    dst_repo = "cloud-tpu-multipod-dev/pathways-dev/anowusu/server"

    base_manifest = {
        "schemaVersion": 2,
        "mediaType": "application/vnd.oci.image.manifest.v1+json",
        "layers": [{
            "mediaType": "application/vnd.oci.image.layer.v1.tar+gzip",
            "size": 42,
            "digest": "sha256:base1",
        }],
    }
    base_config = {
        "config": {"Env": ["PATH=/usr/bin"]},
        "rootfs": {"type": "layers", "diff_ids": ["sha256:diff1"]},
    }
    src_digest = fake.seed_image(
        src_repo,
        "latest",
        base_manifest,
        base_config,
        extra_tags=["20261029_1817", "sha-79dcecd"],
    )
    fake.seed_image(
        proxy_repo,
        "latest",
        base_manifest,
        base_config,
        extra_tags=["20261029_1817", "sha-79dcecd"],
    )

    res1 = oss_image_patcher.patch_oss_server_image(
        client=fake, so_bytes=b"mock-so"
    )
    self.assertFalse(res1.already_existed)
    self.assertEqual(res1.source_digest, src_digest)
    self.assertEqual(res1.runtime.name, "louhi")
    self.assertEqual(
        res1.runtime.pathways_server_image,
        f"us-docker.pkg.dev/{dst_repo}",
    )
    self.assertEqual(
        res1.runtime.pathways_proxy_image,
        f"us-docker.pkg.dev/{proxy_repo}",
    )
    self.assertEqual(res1.runtime.pathways_version, "sha-79dcecd")

    # Second run against unchanged upstream image is a fast idempotent hit.
    fake.ensured_blobs.clear()
    res2 = oss_image_patcher.patch_oss_server_image(
        client=fake, so_bytes=b"mock-so"
    )
    self.assertTrue(res2.already_existed)
    self.assertEqual(fake.ensured_blobs, [])
    self.assertEqual(res2.patched_digest, res1.patched_digest)

  def test_maybe_patch_runtimes_on_the_fly_dry_run(self):
    rts = [runtimes.MCJAX, runtimes.LOUHI, runtimes.RAPID]
    updated = oss_image_patcher.maybe_patch_runtimes_on_the_fly(
        rts,
        enabled=True,
        dry_run=True,
        patched_server_image="us-docker.pkg.dev/proj/repo/server",
    )
    self.assertEqual(updated[0], runtimes.MCJAX)
    self.assertEqual(updated[2], runtimes.RAPID)
    self.assertEqual(
        updated[1].pathways_server_image,
        "us-docker.pkg.dev/proj/repo/server",
    )

  def test_compile_patch_library_compiles_real_c_source(self):
    so_bytes = oss_image_patcher.compile_patch_library()
    self.assertTrue(so_bytes.startswith(b"\x7fELF"))


if __name__ == "__main__":
  unittest.main()
