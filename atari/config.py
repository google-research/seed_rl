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

import tempfile

from absl import flags
from absl import logging
import atari_py  
import gym
from seed_rl.atari import agents
from seed_rl.atari import atari_preprocessing
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
flags.DEFINE_float('replay_ratio', 1.5,
                   'Average number of times each observation is replayed and '
                   'used for training. '
                   'The default of 1.5 corresponds to an interpretation of the '
                   'R2D2 paper using the end of section 2.3.')
flags.DEFINE_integer('inference_batch_size', 2, 'Batch size for inference.')
flags.DEFINE_integer('unroll_length', 100, 'Unroll length in agent steps.')
flags.DEFINE_integer('num_training_tpus', 1, 'Number of TPUs for training.')
flags.DEFINE_integer('update_target_every_n_step',
                     2500,
                     'Update the target network at this frequency (expressed '
                     'in number of training steps)')
flags.DEFINE_integer('replay_buffer_size', 100,
                     'Size of the replay buffer (in number of unrolls stored).')
flags.DEFINE_integer('replay_buffer_min_size', 10,
                     'Learning only starts when there is at least this number '
                     'of unrolls in the replay buffer')
flags.DEFINE_float('priority_exponent', 0.9,
                   'Priority exponent used when sampling in the replay buffer. '
                   '0.9 comes from R2D2 paper, table 2.')
flags.DEFINE_integer('unroll_queue_max_size', 100,
                     'Max size of the unroll queue')
flags.DEFINE_integer('burn_in', 40,
                     'Length of the RNN burn-in prefix. This is the number of '
                     'time steps on which we update each stored RNN state '
                     'before computing the actual loss. The effective length '
                     'of unrolls will be burn_in + unroll_length, and two '
                     'consecutive unrolls will overlap on burn_in steps.')
flags.DEFINE_float('importance_sampling_exponent', 0.6,
                   'Exponent used when computing the importance sampling '
                   'correction. 0 means no importance sampling correction. '
                   '1 means full importance sampling correction.')
flags.DEFINE_integer('stack_size', 4, 'Number of frames to stack.')
flags.DEFINE_float('clip_norm', 40, 'We clip gradient norm to this value.')
flags.DEFINE_float('value_function_rescaling_epsilon', 1e-3,
                   'Epsilon used for value function rescaling.')
flags.DEFINE_integer('n_steps', 5,
                     'n-step returns: how far ahead we look for computing the '
                     'Bellman targets.')

# Loss settings.
flags.DEFINE_float('discounting', .997, 'Discounting factor.')

# Optimizer settings.
flags.DEFINE_float('learning_rate', 0.00048, 'Learning rate.')
flags.DEFINE_float('adam_epsilon', 1e-3, 'Adam epsilon.')

# ACTOR
flags.DEFINE_integer('task', 0, 'Task id.')
flags.DEFINE_integer('num_actors', 4,
                     'Total number of actors. The last --num_eval_actors will '
                     'be reserved for evaluation and not used for training.')
flags.DEFINE_integer('num_actors_with_summaries', 4,
                     'Number of actors that will log debug/profiling TF '
                     'summaries.')

# Environment settings.
flags.DEFINE_string('game', 'Pong', 'Game name.')
flags.DEFINE_integer('num_action_repeats', 4, 'Number of action repeats.')
flags.DEFINE_integer('max_random_noops', 30,
                     'Maximal number of random no-ops at the beginning of each '
                     'episode.')
flags.DEFINE_boolean('sticky_actions', False,
                     'When sticky actions are enabled, the environment repeats '
                     'the previous action with probability 0.25, instead of '
                     'playing the action given by the agent. Used to introduce '
                     'stochasticity in ATARI-57 environments, see '
                     'Machado et al. (2017).')

# Eval settings
flags.DEFINE_float('eval_epsilon', 1e-3,
                   'Epsilon (as in epsilon-greedy) used for evaluation.')
flags.DEFINE_integer('num_eval_actors', 2,
                     'Number of actors whose transitions will be used for '
                     'eval.')


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


def create_optimizer(unused_final_iteration):
  learning_rate_fn = lambda iteration: FLAGS.learning_rate
  optimizer = tf.keras.optimizers.Adam(FLAGS.learning_rate,
                                       epsilon=FLAGS.adam_epsilon)
  return optimizer, learning_rate_fn


def create_agent(env_output_specs, num_actions):
  return agents.DuelingLSTMDQNNet(
      num_actions, env_output_specs.observation.shape, FLAGS.stack_size)
