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

from seed_rl.dmlab import agents
from seed_rl.dmlab import env
import tensorflow as tf

FLAGS = flags.FLAGS

# COMMON FLAGS

flags.DEFINE_string('logdir', '/tmp/agent', 'TensorFlow log directory.')
flags.DEFINE_alias('job-dir', 'logdir')
flags.DEFINE_string('server_address', 'localhost:8686', 'Server address.')
flags.DEFINE_string('level_cache_dir', None, 'Global level cache directory.')

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
flags.DEFINE_float('max_abs_reward', 1.,
                   'Maximum absolute reward when calculating loss.'
                   'Use 0. to disable clipping.')

# Optimizer settings.
flags.DEFINE_float('learning_rate', 0.00048, 'Learning rate.')
flags.DEFINE_float('adam_epsilon', 1e-3, 'Adam epsilon.')


# ACTOR

flags.DEFINE_integer('task', 0, 'Task id.')
flags.DEFINE_integer('num_actors', 4, 'Number of actors.')

# Environment settings.
flags.DEFINE_string('game', 'explore_goal_locations_small', 'Game/level name.')
flags.DEFINE_integer('width', 96, 'Width of observation.')
flags.DEFINE_integer('height', 72, 'Height of observation.')
flags.DEFINE_integer('num_action_repeats', 4, 'Number of action repeats.')


def create_environment(task):
  logging.info('Creating environment: %s', FLAGS.game)
  return env.DmLab(
      FLAGS.game,
      FLAGS.num_action_repeats,
      seed=task + 1,
      is_test=False,
      level_cache_dir=FLAGS.level_cache_dir,
      config={
          'width': FLAGS.width,
          'height': FLAGS.height,
          'logLevel': 'WARN',
      })


def create_optimizer(final_iteration):
  learning_rate_fn = tf.keras.optimizers.schedules.PolynomialDecay(
      FLAGS.learning_rate, final_iteration, 0)
  optimizer = tf.keras.optimizers.Adam(learning_rate_fn, beta_1=0,
                                       epsilon=FLAGS.adam_epsilon)
  return optimizer, learning_rate_fn


def create_agent(unused_env_output_specs, num_actions):
  return agents.ImpalaDeep(num_actions)
