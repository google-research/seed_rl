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
die () {
    echo >&2 "$@"
    exit 1
}

CONFIGS="atari|dmlab|football"
[ "$#" -ne 0 ] || die "Usage: run_local.sh [$CONFIGS]"
echo $1 | grep -E -q $CONFIGS || die "Supported config: $CONFIGS"
export CONFIG=$1

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd $DIR
docker/build.sh
docker_version=$(docker version --format '{{.Server.Version}}')
if [[ "19.03" > $docker_version ]]; then
  docker run --entrypoint ./docker/run.sh -ti -it --name seed --rm seed_rl:$CONFIG $CONFIG
else
  docker run --gpus all --entrypoint ./docker/run.sh -ti -it -e HOST_PERMS="$(id -u):$(id -g)" --name seed --rm seed_rl:$CONFIG $CONFIG
fi
