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

licenses(["notice"])

py_library(
    name = "lib",
    srcs = glob(
        ["**/*.py"],
        exclude = [
            "*_test.py",
            "observation.py",
        ],
    ),
    visibility = ["//visibility:public"],
    deps = [
        ":observation",
        "//third_party/py/numpy",
        "//tensorflow/cc/seed_rl:utils",
        "//tensorflow/cc/seed_rl/grpc",
    ],
)

py_binary(
    name = "vtrace_main",
    srcs = ["vtrace_main.py"],
    python_version = "PY3",
    deps = [
        ":lib",
        "//tensorflow/cc/seed_rl/agents/vtrace:learner",
        "//tensorflow/cc/seed_rl/common:actor",
    ],
)

py_library(
    name = "observation",
    srcs = [
        "observation.py",
    ],
    deps = ["//third_party/py/numpy"],
)

py_test(
    name = "environment_test",
    srcs = ["environment_test.py"],
    python_version = "PY3",
    deps = [
        ":lib",
        "//tensorflow:tensorflow_py",
    ],
)

py_test(
    name = "observation_test",
    srcs = [
        "observation_test.py",
    ],
    python_version = "PY3",
    deps = [
        ":observation",
        "//tensorflow:tensorflow_py",
        "//third_party/py/numpy",
        "//tensorflow/cc/seed_rl:utils",
    ],
)
