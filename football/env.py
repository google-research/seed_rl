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

"""Football env factory."""

from absl import flags
from absl import logging

import gym
from seed_rl.football import observation

FLAGS = flags.FLAGS

# Environment settings.
flags.DEFINE_string('game', '11_vs_11_easy_stochastic', 'Game/scenario name.')
flags.DEFINE_enum('reward_experiment', 'scoring',
                  ['scoring', 'scoring,checkpoints'],
                  'Reward to be used for training.')
flags.DEFINE_enum('smm_size', 'default', ['default', 'medium', 'large'],
                  'Size of the Super Mini Map.')
flags.DEFINE_integer('num_action_repeats', 1, 'Number of action repeats.')


def create_environment(_):
  """Returns a gym Football environment."""
  logging.info('Creating environment: %s', FLAGS.game)
  assert FLAGS.num_action_repeats == 1, 'Only action repeat of 1 is supported.'
  channel_dimensions = {
      'default': (96, 72),
      'medium': (120, 90),
      'large': (144, 108),
  }[FLAGS.smm_size]
  env = gym.make(
      'gfootball:GFootball-%s-SMM-v0' % FLAGS.game,
      stacked=True,
      rewards=FLAGS.reward_experiment,
      channel_dimensions=channel_dimensions)
  return observation.PackedBitsObservation(env)
