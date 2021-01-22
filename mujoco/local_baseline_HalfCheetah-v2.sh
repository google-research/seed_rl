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
./run_local.sh mujoco ppo 8 32 \
 --env_name=HalfCheetah-v2 \
 --inference_batch_size=256 \
 --gin_config=/seed_rl/mujoco/gin/ppo.gin \
 --total_environment_frames=2000000 \
 --batch_mode=split \
 --epochs_per_step=10 \
 --lr_decay_multiplier=0 \
 --step_size_transitions=2048 \
 --batch_size_transitions=64 \
 --unroll_length=16 \
 --learning_rate=3e-4 \
 --clip_norm=0.5 \
 --batch_size=0 \
 --batches_per_step=0
