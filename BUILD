# Copyright 2019 The SEED Authors
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

# SEED: Scalable, Efficient Deep-RL

licenses(["notice"])

exports_files(["LICENSE"])

py_test(
    name = "vtrace_test",
    srcs = [
        "tests/vtrace_test.py",
    ],
    python_version = "PY3",
    deps = [
        "//tensorflow:tensorflow_py",
        "//third_party/py/numpy",
        "//tensorflow/cc/seed_rl/common",
    ],
)

py_test(
    name = "utils_test",
    srcs = [
        "tests/utils_test.py",
    ],
    python_version = "PY3",
    deps = [
        "//tensorflow:tensorflow_py",
        "//tensorflow/cc/seed_rl/common",
    ],
)

py_test(
    name = "agents_test",
    srcs = [
        "tests/agents_test.py",
    ],
    python_version = "PY3",
    deps = [
        "//tensorflow:tensorflow_py",
        "//tensorflow/cc/seed_rl/common",
        "//tensorflow/cc/seed_rl/dmlab:lib",
    ],
)
