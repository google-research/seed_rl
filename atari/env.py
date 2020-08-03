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

from absl import flags
from absl import logging
import atari_py  
import gym
from seed_rl.atari import atari_preprocessing
from seed_rl.common import common_flags  


FLAGS = flags.FLAGS

# Environment settings.
flags.DEFINE_string('game', 'Pong', 'Game name.')
flags.DEFINE_integer('max_random_noops', 30,
                     'Maximal number of random no-ops at the beginning of each '
                     'episode.')
flags.DEFINE_boolean('sticky_actions', False,
                     'When sticky actions are enabled, the environment repeats '
                     'the previous action with probability 0.25, instead of '
                     'playing the action given by the agent. Used to introduce '
                     'stochasticity in ATARI-57 environments, see '
                     'Machado et al. (2017).')


def create_environment(task):  
  logging.info('Creating environment: %s', FLAGS.game)


  game_version = 'v0' if FLAGS.sticky_actions else 'v4'
  full_game_name = '{}NoFrameskip-{}'.format(FLAGS.game, game_version)
  env = gym.make(full_game_name, full_action_space=True)
  env.seed(task)

  if not isinstance(env, gym.wrappers.TimeLimit):
    raise ValueError('We expected gym.make to wrap the env with TimeLimit. '
                     'Got {}.'.format(type(env)))
  # Change TimeLimit wrapper to 108,000 steps (30 min) as default in the
  # litterature instead of OpenAI Gym's default of 100,000 steps.
  env = gym.wrappers.TimeLimit(env.env, max_episode_steps=108000)
  return atari_preprocessing.AtariPreprocessing(
      env, frame_skip=FLAGS.num_action_repeats,
      max_random_noops=FLAGS.max_random_noops)
