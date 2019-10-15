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

"""Flags and configuration."""

from absl import flags
from absl import logging

import gym
from seed_rl.football import agents
from seed_rl.football import observation
import tensorflow as tf

FLAGS = flags.FLAGS

# COMMON FLAGS

flags.DEFINE_string('logdir', '/tmp/agent', 'TensorFlow log directory.')
flags.DEFINE_alias('job-dir', 'logdir')
flags.DEFINE_string('server_address', 'localhost:8686', 'Server address.')

# LEARNER

# Training.
flags.DEFINE_integer('save_checkpoint_secs', 1800,
                     'Checkpoint save period in seconds.')
flags.DEFINE_integer('total_environment_frames', int(1e9),
                     'Total environment frames to train for.')
flags.DEFINE_integer('batch_size', 2, 'Batch size for training.')
flags.DEFINE_integer('inference_batch_size', 2, 'Batch size for inference.')
flags.DEFINE_integer('unroll_length', 100, 'Unroll length in agent steps.')
flags.DEFINE_integer('num_training_tpus', 1, 'Number of TPUs for training.')

# Loss settings.
flags.DEFINE_float('entropy_cost', 0.00025, 'Entropy cost/multiplier.')
flags.DEFINE_float('baseline_cost', .5, 'Baseline cost/multiplier.')
flags.DEFINE_float('discounting', .99, 'Discounting factor.')
flags.DEFINE_float('lambda_', 1., 'Lambda.')
flags.DEFINE_float('max_abs_reward', 0.,
                   'Maximum absolute reward when calculating loss.'
                   'Use 0. to disable clipping.')

# Optimizer settings.
flags.DEFINE_float('learning_rate', 0.00048, 'Learning rate.')

# ACTOR

flags.DEFINE_integer('task', 0, 'Task id.')
flags.DEFINE_integer('num_actors', 4, 'Number of actors.')
flags.DEFINE_integer('num_actors_with_summaries', 4,
                     'Number of actors that will log debug/profiling TF '
                     'summaries.')

# Environment settings.
flags.DEFINE_string('game', '11_vs_11_easy_stochastic', 'Game/scenario name.')
flags.DEFINE_enum('reward_experiment', 'scoring',
                  ['scoring', 'scoring,checkpoints'],
                  'Reward to be used for training.')
flags.DEFINE_enum('smm_size', 'default', ['default', 'medium', 'large'],
                  'Size of the Super Mini Map.')
flags.DEFINE_integer('num_action_repeats', 1, 'Number of action repeats.')

flags.DEFINE_integer('seed', None, 'Unused seed.')


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


def create_optimizer(unused_final_iteration):
  learning_rate_fn = lambda iteration: FLAGS.learning_rate
  optimizer = tf.keras.optimizers.Adam(FLAGS.learning_rate)
  return optimizer, learning_rate_fn


def create_agent(unused_env_output_specs, num_actions):
  return agents.GFootball(num_actions)
