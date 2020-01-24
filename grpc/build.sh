#!/bin/bash
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


# Compiles the TF gRPC using Docker.
# Usage: build.sh [Directory to grpc/ for grpc_cc.so and service_pb2.py]
set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd $DIR

CONFIG=grpc ../docker/build.sh

id=$(docker create seed_rl:grpc)
docker cp $id:/seed_rl/grpc/grpc_cc.so $1/grpc_cc.so
docker cp $id:/seed_rl/grpc/service_pb2.py $1/service_pb2.py
docker rm -v $id
