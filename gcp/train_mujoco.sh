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
source $DIR/setup.sh

export ENVIRONMENT=mujoco
export CONFIG=mujoco
export AGENT=ppo
export WORKERS=32
export ACTORS_PER_WORKER=20

cat > /tmp/config.yaml <<EOF
trainingInput:
  scaleTier: CUSTOM
  masterType: standard_p100
  masterConfig:
    imageUri: ${IMAGE_URI}:${CONFIG}
  workerCount: ${WORKERS}
  workerType: standard
  workerConfig:
    imageUri: ${IMAGE_URI}:${CONFIG}
  parameterServerCount: 0
  hyperparameters:
    goal: MAXIMIZE
    hyperparameterMetricTag: episode_return
    maxTrials: 1
    maxParallelTrials: 1
    enableTrialEarlyStopping: True
    params:
    - parameterName: env_name
      type: CATEGORICAL
      categoricalValues:
      - HalfCheetah-v2
    - parameterName: gin_config
      type: CATEGORICAL
      categoricalValues:
      - /seed_rl/mujoco/gin/ppo.gin
    - parameterName: total_environment_frames
      type: INTEGER
      minValue: 2000000
      maxValue: 2000000
      scaleType: UNIT_LOG_SCALE
EOF

start_training
