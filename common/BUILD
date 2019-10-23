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

exports_files(["LICENSE"])

py_library(
    name = "actor",
    srcs = ["actor.py"],
    srcs_version = "PY3",
    visibility = ["//tensorflow/cc/seed_rl:__subpackages__"],
    deps = [
        ":common_flags",
        "//third_party/py/absl/flags",
        "//third_party/py/absl/logging",
        "//third_party/py/numpy",
        "//tensorflow/cc/seed_rl:utils",
        "//tensorflow/cc/seed_rl/grpc",
        "//third_party/py/tensorflow:tensorflow_google",
    ],
)

py_library(
    name = "common_flags",
    srcs = ["common_flags.py"],
    srcs_version = "PY3",
    visibility = ["//tensorflow/cc/seed_rl:__subpackages__"],
    deps = [
        "//third_party/py/absl/flags",
    ],
)
