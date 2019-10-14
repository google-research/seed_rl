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

load("//tensorflow:tensorflow.bzl", "tf_gen_op_wrapper_py")
load("//tensorflow/core/platform:default/build_config.bzl", "tf_proto_library")

licenses(["notice"])

tf_custom_op_library(
    name = "grpc_cc.so",
    srcs = ["ops/grpc.cc"],
    deps = [
        ":service_proto_cc",
        "//tensorflow:grpc",
        "//tensorflow:grpc++",
        "//tensorflow/core:core_cpu_headers_lib",
        "//tensorflow/core:protos_all_cc",
        "@com_google_absl//absl/container:flat_hash_map",
    ],
)

py_library(
    name = "grpc_wrapper",
    srcs = ["python/ops_wrapper.py"],
    data = [":grpc_cc.so"],
    srcs_version = "PY2AND3",
    deps = ["//tensorflow:tensorflow_py"],
)

tf_proto_library(
    name = "service_proto",
    srcs = ["service.proto"],
    has_services = 1,
    cc_api_version = 2,
    cc_grpc_version = 1,
    protodeps = [
        "//tensorflow/core:protos_all",
    ],
)

py_library(
    name = "grpc",
    srcs = [
        "__init__.py",
        "python/__init__.py",
        "python/ops.py",
    ],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = [
        ":grpc_wrapper",
        ":service_proto_py",
        "//tensorflow:tensorflow_py",
        "//third_party/py/numpy",
        "@six_archive//:six",
    ],
)

# tf_py_test for Bazel.
py_test(
    name = "ops_test",
    size = "medium",
    srcs = [
        "python/ops_test.py",
    ],
    shard_count = 3,
    srcs_version = "PY2AND3",
    deps = [
        ":grpc",
        "//tensorflow:tensorflow_py",
        "@six_archive//:six",
    ],
)
