#!/usr/bin/env python3
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

"""Unit tests for generate_workload.py.

Needs the launcher dependencies (requirements.txt) but no cluster access:
  python -m pytest cloud_launch/pathways/generate_workload_test.py
"""

import os
import pathlib
import sys
import tempfile
import unittest

import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import generate_workload  # pylint: disable=g-import-not-at-top,wrong-import-position

_BASE_ARGS = [
    "--name=tpums-pw-test",
    "--config=configs/v5e/4x8/all_gather.yaml",
    "--image=us-docker.pkg.dev/p/r/tpums-pathways:test",
    "--tpu_type=v5e",
    "--topology=4x8",
    "--pathways_version=20261013-jax_0.11.1",
    "--pathways_dir=gs://bucket/run/pathways_scratch",
    "--gcs_output_dir=gs://bucket/run",
    "--xprof_dir=gs://bucket/run/xprof",
]


def _containers(pod_spec: dict) -> dict:
  return {
      c["name"]: c
      for c in pod_spec.get("initContainers", []) + pod_spec["containers"]
  }


def _env(container: dict) -> dict:
  return {e["name"]: e.get("value") for e in container.get("env", [])}


class GenerateWorkloadTest(unittest.TestCase):

  def _generate(self, extra_args=()) -> dict:
    args = generate_workload._parse_args(_BASE_ARGS + list(extra_args))
    return generate_workload.build_workload(args).jobset.to_dict()

  def _jobs(self, config: dict) -> dict:
    return {j["name"]: j for j in config["spec"]["replicatedJobs"]}

  def test_jobset_layout(self):
    config = self._generate()
    self.assertEqual(config["kind"], "JobSet")
    self.assertEqual(config["metadata"]["name"], "tpums-pw-test")
    self.assertEqual(
        config["metadata"]["labels"]["kueue.x-k8s.io/queue-name"],
        "multislice-queue",
    )
    self.assertEqual(
        config["spec"]["successPolicy"]["targetReplicatedJobs"],
        ["pathways-head"],
    )
    jobs = self._jobs(config)
    self.assertCountEqual(jobs, ["pathways-head", "pathways-worker"])
    # A v5e 4x8 slice has 8 TPU VMs with 4 chips each.
    worker_job = jobs["pathways-worker"]["template"]["spec"]
    self.assertEqual(worker_job["parallelism"], 8)
    worker_pod = worker_job["template"]["spec"]
    self.assertEqual(
        worker_pod["nodeSelector"],
        {
            "cloud.google.com/gke-tpu-accelerator": "tpu-v5-lite-podslice",
            "cloud.google.com/gke-tpu-topology": "4x8",
        },
    )

  def test_user_workload_container(self):
    jobs = self._jobs(self._generate())
    head_job = jobs["pathways-head"]["template"]["spec"]
    self.assertEqual(head_job["activeDeadlineSeconds"], 1800)
    head_pod = head_job["template"]["spec"]
    self.assertEqual(
        head_pod["nodeSelector"], {"cloud.google.com/gke-nodepool": "cpu-np"}
    )
    self.assertEqual(head_pod["priorityClassName"], "medium")

    init_names = [c["name"] for c in head_pod["initContainers"]]
    self.assertCountEqual(init_names, ["pathways-rm", "pathways-proxy"])
    (user,) = head_pod["containers"]
    self.assertEqual(user["name"], "user-workload")
    self.assertEqual(user["image"], "us-docker.pkg.dev/p/r/tpums-pathways:test")
    self.assertEqual(user["command"], ["bash", generate_workload.RUN_SCRIPT])
    env = _env(user)
    self.assertEqual(env["JAX_PLATFORMS"], "proxy")
    self.assertEqual(env["JAX_BACKEND_TARGET"], "grpc://localhost:29000")
    self.assertEqual(
        env["TPUMS_CONFIG"], "/opt/tpums/configs/v5e/4x8/all_gather.yaml"
    )
    self.assertEqual(env["TPUMS_GCS_OUTPUT_DIR"], "gs://bucket/run")
    self.assertEqual(env["TPUMS_XPROF_DIR"], "gs://bucket/run/xprof")
    self.assertEqual(env["TPUMS_PATHWAYS_VERSION"], "20261013-jax_0.11.1")
    self.assertEqual(user["resources"]["limits"], {"cpu": "8", "memory": "32G"})

  def test_pathways_images_use_version_tag(self):
    jobs = self._jobs(self._generate())
    head = _containers(jobs["pathways-head"]["template"]["spec"]["template"]["spec"])
    self.assertEqual(
        head["pathways-proxy"]["image"],
        "us-docker.pkg.dev/cloud-tpu-v2-images/pathways/proxy_server:20261013-jax_0.11.1",
    )
    self.assertEqual(
        head["pathways-rm"]["image"],
        "us-docker.pkg.dev/cloud-tpu-v2-images/pathways/server:20261013-jax_0.11.1",
    )

  def test_op_flags_go_to_proxy(self):
    jobs = self._jobs(self._generate())
    head = _containers(jobs["pathways-head"]["template"]["spec"]["template"]["spec"])
    worker = _containers(
        jobs["pathways-worker"]["template"]["spec"]["template"]["spec"]
    )["pathways-worker"]
    expected, _ = generate_workload.resolve_op_flags(
        "all_gather", generate_workload.DEFAULT_OP_FLAGS_PATH
    )
    self.assertNotEmpty(expected)
    proxy_args = head["pathways-proxy"]["args"]
    self.assertEqual(proxy_args[-len(expected):], expected)
    self.assertIn("--xla_tpu_enable_all_gather_offload_tracing=true", proxy_args)
    # Rapid parses proxy.args; Louhi OSS parses LIBTPU_INIT_ARGS inside
    # dynamically loaded /lib/libtpu.so and needs --undefok on proxy.args.
    undefok = proxy_args[-len(expected) - 1]
    self.assertTrue(undefok.startswith("--undefok="))
    self.assertIn("xla_tpu_enable_all_gather_offload_tracing", undefok)
    expected_init_args = " ".join(expected)
    self.assertEqual(
        _env(head["pathways-proxy"])["LIBTPU_INIT_ARGS"], expected_init_args
    )
    self.assertEqual(
        _env(head["pathways-rm"])["LIBTPU_INIT_ARGS"], expected_init_args
    )
    self.assertEqual(_env(worker)["LIBTPU_INIT_ARGS"], expected_init_args)
    for c in (head["pathways-rm"], head["pathways-proxy"], worker):
      self.assertEqual(_env(c)["TEST_UNDECLARED_OUTPUTS_DIR"], "/tmp")

  def test_louhi_oss_images(self):
    jobs = self._jobs(
        self._generate([
            "--pathways_version=20261013_0917-jax_0.11.1",
            "--pathways_server_image=us-docker.pkg.dev/cloud-tpu-multipod-dev/pathways-prod/server",
            "--pathways_proxy_image=us-docker.pkg.dev/cloud-tpu-multipod-dev/pathways-prod/proxy_server",
        ])
    )
    head = _containers(jobs["pathways-head"]["template"]["spec"]["template"]["spec"])
    worker = _containers(
        jobs["pathways-worker"]["template"]["spec"]["template"]["spec"]
    )["pathways-worker"]
    self.assertEqual(
        head["pathways-proxy"]["image"],
        "us-docker.pkg.dev/cloud-tpu-multipod-dev/pathways-prod/proxy_server:20261013_0917-jax_0.11.1",
    )
    self.assertEqual(
        head["pathways-rm"]["image"],
        "us-docker.pkg.dev/cloud-tpu-multipod-dev/pathways-prod/server:20261013_0917-jax_0.11.1",
    )
    self.assertEqual(
        worker["image"],
        "us-docker.pkg.dev/cloud-tpu-multipod-dev/pathways-prod/server:20261013_0917-jax_0.11.1",
    )
    self.assertEqual(
        _env(head["user-workload"])["TPUMS_PATHWAYS_VERSION"],
        "20261013_0917-jax_0.11.1",
    )

  def test_op_flags_none_and_extra_proxy_args(self):
    jobs = self._jobs(
        self._generate([
            "--op_flags=none",
            "--extra_proxy_args=--xla_tpu_a=1 --xla_tpu_b=2",
        ])
    )
    head = _containers(jobs["pathways-head"]["template"]["spec"]["template"]["spec"])
    proxy_args = head["pathways-proxy"]["args"]
    self.assertFalse(any(a.startswith("--xla_jf") for a in proxy_args))
    self.assertEqual(proxy_args[-2:], ["--xla_tpu_a=1", "--xla_tpu_b=2"])

  def test_op_flags_env_goes_to_workers(self):
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
      yaml.safe_dump(
          {"all_gather": {"flags": ["--xla_x=1"], "env": {"TPU_FOO": "7"}}}, f
      )
    self.addCleanup(os.remove, f.name)
    jobs = self._jobs(self._generate([f"--op_flags={f.name}"]))
    worker = _containers(
        jobs["pathways-worker"]["template"]["spec"]["template"]["spec"]
    )["pathways-worker"]
    self.assertEqual(_env(worker)["TPU_FOO"], "7")
    head = _containers(jobs["pathways-head"]["template"]["spec"]["template"]["spec"])
    self.assertEqual(head["pathways-proxy"]["args"][-1], "--xla_x=1")

  def test_scheduling_can_be_disabled(self):
    config = self._generate(
        ["--queue=", "--priority_class=", "--active_deadline_seconds=0"]
    )
    self.assertNotIn("kueue.x-k8s.io/queue-name", config["metadata"]["labels"])
    head_job = self._jobs(config)["pathways-head"]["template"]["spec"]
    self.assertNotIn("activeDeadlineSeconds", head_job)
    self.assertNotIn("priorityClassName", head_job["template"]["spec"])

  def test_export_yaml_round_trip(self):
    args = generate_workload._parse_args(_BASE_ARGS)
    js = generate_workload.build_workload(args).jobset
    with tempfile.TemporaryDirectory() as tmp:
      path = pathlib.Path(tmp) / "jobset.yaml"
      js.export_yaml(str(path))
      with open(path) as f:
        self.assertEqual(yaml.safe_load(f), js.to_dict())

  def test_invalid_arguments(self):
    for bad in (
        ["--name=Bad_Name"],
        ["--name=" + "a" * 23],
        ["--gcs_output_dir=/tmp/results"],
        ["--xprof_dir=/tmp/xprof"],
    ):
      with self.assertRaises(SystemExit, msg=bad):
        generate_workload._parse_args(_BASE_ARGS + bad)

  def assertNotEmpty(self, container):  # pylint: disable=invalid-name
    self.assertTrue(container, "expected a non-empty container")


if __name__ == "__main__":
  unittest.main()
