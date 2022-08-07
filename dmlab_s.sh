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

ENVIRONMENTS="atari|dmlab|football|mujoco|procgen"
AGENTS="r2d2|vtrace|sac|ppo"
[ "$#" -ne 0 ] || die "Usage: run_local.sh [$ENVIRONMENTS] [$AGENTS] [Num. actors]"
echo $1 | grep -E -q $ENVIRONMENTS || die "Supported games: $ENVIRONMENTS"
echo $2 | grep -E -q $AGENTS || die "Supported agents: $AGENTS"
echo $3 | grep -E -q "^((0|([1-9][0-9]*))|(0x[0-9a-fA-F]+))$" || die "Number of actors should be a non-negative integer without leading zeros"
export ENVIRONMENT=$1
export AGENT=$2
export NUM_ACTORS=$3
export ENV_PER_ACTOR=$4
export SUB_TASK=$5
export PORT=$6
export CUDA=$7
export RUN_ID=$8

RUN_NAME="sampler_${ENVIRONMENT}_${AGENT}_${NUM_ACTORS}_${ENV_PER_ACTOR}_${SUB_TASK}_${PORT}_${CUDA}_${RUN_ID}"
LOG_DIR="/outdata/logs/seed_rl/sampler_${ENVIRONMENT}_${AGENT}/${SUB_TASK}/${NUM_ACTORS}_${ENV_PER_ACTOR}_${PORT}_${CUDA}_${RUN_ID}"
# shift 4
shift 3
if [[ $1 ]]; then
  echo $1 | grep -E -q "^((0|([1-9][0-9]*))|(0x[0-9a-fA-F]+))$" || die "Number of environments per actor should be a non-negative integer without leading zeros"
  export ENV_BATCH_SIZE=$1
  shift 1
else
  export ENV_BATCH_SIZE=1
fi
export CONFIG=$ENVIRONMENT
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd $DIR
docker/build.sh
docker_version=$(docker version --format '{{.Server.Version}}')
if [[ "19.03" > $docker_version ]]; then
  docker run -v ~/:/outdata --entrypoint ./docker/run.sh -ti -it -p 6006-6015:6006-6015 --name seed --rm seed_rl:$ENVIRONMENT $ENVIRONMENT $AGENT $NUM_ACTORS $ENV_BATCH_SIZE $@
else
  # docker run --gpus '"device=5"' -v ~/:/outdata --entrypoint ./docker/run.sh -ti -it -p 6026-6035:6026-6035 -e HOST_PERMS="$(id -u):$(id -g)" --name ${RUN_NAME} --rm seed_rl:$ENVIRONMENT $ENVIRONMENT $AGENT $NUM_ACTORS $ENV_BATCH_SIZE $LOG_DIR $@
  docker run --gpus '"device='"${CUDA}"'"' -v ~/:/outdata --entrypoint ./docker/sample_${ENVIRONMENT}.sh -ti -it -p ${PORT}-${PORT}:${PORT}-${PORT} -e HOST_PERMS="$(id -u):$(id -g)" --name ${RUN_NAME} --rm seed_rl:$ENVIRONMENT $ENVIRONMENT $AGENT $NUM_ACTORS $ENV_BATCH_SIZE $LOG_DIR $PORT $SUB_TASK $@
fi
