# coding=utf-8
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

"""Atari env factory."""

import tempfile
from mlagents_envs.environment import UnityEnvironment
from absl import flags
from absl import logging
import gym
from gym import wrappers
from gym_unity.envs import UnityToGymWrapper
from seed_rl.unity import unity_preprocessing

import os
os.environ["DISPLAY"]=":100"


FLAGS = flags.FLAGS

# Environment settings.
flags.DEFINE_string('game', 'GridWorldLinux', 'Game name.')
flags.DEFINE_integer('num_action_repeats', 1, 'Number of action repeats.')
flags.DEFINE_integer('max_random_noops', 30,
                     'Maximal number of random no-ops at the beginning of each '
                     'episode.')


def create_environment(task):  
  logging.info('Creating environment: %s', FLAGS.game)

  #print(FLAGS)
  #print(task)
  #print(FLAGS.run_mode)

  full_game_name = '{}'.format(FLAGS.game)
  import os
  modeOffset = FLAGS.run_mode == 'actor'
  unity_env = UnityEnvironment('../unity/env/{}/{}'.format(FLAGS.game,FLAGS.game), base_port=5005+task+int(modeOffset))
  env = UnityToGymWrapper(unity_env, flatten_branched = True, use_visual=True, uint8_visual=True)
  env.seed(task)


  return unity_preprocessing.UnityPreprocessing(
      env, max_random_noops=FLAGS.max_random_noops)