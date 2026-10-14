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

"""Copies a local results directory to a gs:// directory.

Used by run_benchmark.sh inside the TPUMS Pathways image. It uses
`tf.io.gfile` because TensorFlow is already a TPUMS dependency and has a
built-in GCS filesystem that authenticates with the pod's credentials, which
avoids adding the gcloud CLI to the image.

Usage: upload_results.py <local_dir> <gs://dest_dir>
"""

import os
import sys


def upload(local_dir: str, gcs_dir: str) -> int:
  """Recursively copies `local_dir` to `gcs_dir`; returns the file count."""
  import tensorflow as tf  # pylint: disable=g-import-not-at-top

  count = 0
  for root, _, files in os.walk(local_dir):
    for file_name in sorted(files):
      src = os.path.join(root, file_name)
      rel = os.path.relpath(src, local_dir)
      dst = f"{gcs_dir.rstrip('/')}/{rel}"
      tf.io.gfile.makedirs(os.path.dirname(dst))
      tf.io.gfile.copy(src, dst, overwrite=True)
      print(f"Uploaded {src} -> {dst}")
      count += 1
  return count


def main(argv: list[str]) -> int:
  if len(argv) != 3 or not argv[2].startswith("gs://"):
    print(__doc__, file=sys.stderr)
    return 2
  count = upload(argv[1], argv[2])
  print(f"Uploaded {count} file(s) to {argv[2]}")
  return 0


if __name__ == "__main__":
  sys.exit(main(sys.argv))
