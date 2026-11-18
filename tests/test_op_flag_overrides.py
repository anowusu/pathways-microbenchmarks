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
"""Tests for the per-accelerator op flags in automation/op_flags/."""

from absl.testing import absltest
from absl.testing import parameterized
from automation import utils
import yaml

_UPSTREAM = (
    utils.PACKAGE_ROOT / "src" / "accelerator_microbenchmarks" / "op_flags.yaml"
)

# The only flags an accelerator's file may give another value than upstream.
_ALLOWED_DIFFERENCES = {
    ("v5e", "dcn_all_reduce"): {"--megascale_grpc_premap_memory_bytes"},
}


def _flag_values(entry):
  """Maps the flag names of an op flags entry to their values."""
  return dict(str(flag).split("=", 1) for flag in entry.get("flags") or [])


class OpFlagOverridesTest(parameterized.TestCase):

  def test_overrides_only_change_the_documented_flags(self):
    upstream = yaml.safe_load(_UPSTREAM.read_text())
    seen = set()
    for path in sorted(utils.OP_FLAGS_OVERRIDE_DIR.glob("*.yaml")):
      accelerator = path.stem
      self.assertIn(accelerator, utils.SUPPORTED_ACCELERATORS)
      for benchmark, entry in yaml.safe_load(path.read_text()).items():
        seen.add((accelerator, benchmark))
        self.assertIn(benchmark, upstream, f"{path.name}: unknown benchmark")
        ours, theirs = _flag_values(entry), _flag_values(upstream[benchmark])
        # Same flags in the same order, and the same environment.
        self.assertEqual(list(ours), list(theirs), f"{path.name}: {benchmark}")
        self.assertEqual(entry.get("env"), upstream[benchmark].get("env"))
        changed = {name for name in ours if ours[name] != theirs[name]}
        self.assertEqual(
            changed,
            _ALLOWED_DIFFERENCES.get((accelerator, benchmark), set()),
            f"{path.name}: {benchmark}",
        )
    self.assertEqual(seen, set(_ALLOWED_DIFFERENCES))

  def test_v5e_dcn_all_reduce_premap_fits_the_host(self):
    entry = yaml.safe_load((utils.OP_FLAGS_OVERRIDE_DIR / "v5e.yaml").read_text())
    premap = int(
        _flag_values(entry["dcn_all_reduce"])["--megascale_grpc_premap_memory_bytes"]
    )
    self.assertEqual(premap, 64 * 1024**3)

  @parameterized.named_parameters(
      (
          "v5e_dcn_all_reduce",
          "v5e",
          "configs/v5e/4x8/dcn_all_reduce.yaml",
          "automation/op_flags/v5e.yaml",
      ),
      ("v5e_dcn_all_to_all", "v5e", "configs/v5e/4x8/dcn_all_to_all.yaml", ""),
      ("v5e_all_reduce", "v5e", "configs/v5e/4x8/all_reduce.yaml", ""),
      ("tpu7x_dcn_all_reduce", "tpu7x", "configs/tpu7x/2x2x1/dcn_all_reduce.yaml", ""),
      ("no_accelerator", None, "configs/v5e/4x8/dcn_all_reduce.yaml", ""),
  )
  def test_op_flags_override_path(self, accelerator, config, expected):
    self.assertEqual(utils.op_flags_override_path(accelerator, config), expected)


if __name__ == "__main__":
  absltest.main()
