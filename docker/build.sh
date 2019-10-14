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


set -e
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd $DIR/..
if "${USE_PREBUILD_GRPC_DOCKER_IMAGE:-true}"; then
  GRPC_IMAGE=gcr.io/seedimages/seed:grpc
else
  docker build -t seed_rl:grpc -f docker/Dockerfile.grpc .
  GRPC_IMAGE=seed_rl:grpc
fi
docker build --build-arg grpc_image=${GRPC_IMAGE} -t seed_rl:${CONFIG} -f docker/Dockerfile.${CONFIG} .
