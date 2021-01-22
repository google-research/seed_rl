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
export WORKERS=4
export ACTORS_PER_WORKER=1

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
      scaleType: NONE
    - parameterName: batch_mode
      type: CATEGORICAL
      categoricalValues:
      - split
    - parameterName: epochs_per_step
      type: INTEGER
      minValue: 10
      maxValue: 10
      scaleType: NONE
    - parameterName: lr_decay_multiplier
      type: DOUBLE
      minValue: 0
      maxValue: 0
      scaleType: NONE
    - parameterName: step_size_transitions
      type: INTEGER
      minValue: 2048
      maxValue: 2048
      scaleType: NONE
    - parameterName: batch_size_transitions
      type: INTEGER
      minValue: 64
      maxValue: 64
      scaleType: NONE
    - parameterName: unroll_length
      type: INTEGER
      minValue: 16
      maxValue: 16
      scaleType: NONE
    - parameterName: learning_rate
      type: DOUBLE
      minValue: 0.0003
      maxValue: 0.0003
      scaleType: NONE
    - parameterName: clip_norm
      type: DOUBLE
      minValue: 0.5
      maxValue: 0.5
      scaleType: NONE
    - parameterName: batch_size
      type: INTEGER
      minValue: 0
      maxValue: 0
      scaleType: NONE
    - parameterName: batches_per_step
      type: INTEGER
      minValue: 0
      maxValue: 0
      scaleType: NONE
EOF

start_training
