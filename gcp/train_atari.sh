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

export ENVIRONMENT=atari
export CONFIG=atari
export AGENT=r2d2
export WORKERS=32
export ACTORS_PER_WORKER=20

cat > /tmp/config.yaml <<EOF
trainingInput:
  scaleTier: CUSTOM
  # n1-highmem-32 provides 208GBs of RAM, n1-highmem-16 provides 104GBS.
  # Training on ATARI requires a bit more than 104GBs due to the large replay
  # buffer, so we need n1-highmem-32, which requires 2 GPUs (see
  # https://cloud.google.com/ml-engine/docs/using-gpus).
  masterType: n1-highmem-32
  masterConfig:
    imageUri: ${IMAGE_URI}:${CONFIG}
    acceleratorConfig:
      count: 2
      type: NVIDIA_TESLA_P100
  workerCount: ${WORKERS}
  workerType: n1-standard-4
  workerConfig:
    imageUri: ${IMAGE_URI}:${CONFIG}
  parameterServerCount: 0
  hyperparameters:
    goal: MAXIMIZE
    hyperparameterMetricTag: eval/episode_return
    maxTrials: 10
    maxParallelTrials: 1
    enableTrialEarlyStopping: True
    params:
    - parameterName: game
      type: CATEGORICAL
      categoricalValues:
      - Pong
    - parameterName: inference_batch_size
      type: INTEGER
      minValue: 32
      maxValue: 32
      scaleType: UNIT_LOG_SCALE
    - parameterName: batch_size
      type: INTEGER
      minValue: 64
      maxValue: 64
      scaleType: UNIT_LOG_SCALE
    - parameterName: replay_buffer_min_size
      type: INTEGER
      minValue: 5000
      maxValue: 5000
      scaleType: UNIT_LOG_SCALE
    - parameterName: replay_buffer_size
      type: INTEGER
      minValue: 100000
      maxValue: 100000
      scaleType: UNIT_LOG_SCALE
    - parameterName: total_environment_frames
      type: DOUBLE
      minValue: 50.e9
      maxValue: 50.e9
      scaleType: UNIT_LOG_SCALE
    - parameterName: replay_ratio
      type: DOUBLE
      minValue: 0.75
      maxValue: 0.75
      scaleType: UNIT_LINEAR_SCALE
    - parameterName: unroll_length
      type: INTEGER
      minValue: 80
      maxValue: 80
      scaleType: UNIT_LINEAR_SCALE
    - parameterName: learning_rate
      type: DOUBLE
      minValue: 1.e-4
      maxValue: 1.e-4
      scaleType: UNIT_LOG_SCALE
    - parameterName: num_eval_actors
      type: INTEGER
      minValue: 30
      maxValue: 30
      scaleType: UNIT_LINEAR_SCALE
    - parameterName: burn_in
      type: INTEGER
      minValue: 40
      maxValue: 40
      scaleType: UNIT_LINEAR_SCALE
    - parameterName: clip_norm
      type: INTEGER
      minValue: 80
      maxValue: 80
      scaleType: UNIT_LINEAR_SCALE
EOF

start_training
