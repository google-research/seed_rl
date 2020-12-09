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


"""SAC example for Mujoco.

Warning!!! This code uses DeepMind wrappers which differ from OpenAI gym
wrappers and the results may not be comparable.
"""


from absl import app
from absl import flags

from seed_rl.agents.sac import learner
from seed_rl.agents.sac import networks
from seed_rl.common import actor
from seed_rl.common import common_flags  
from seed_rl.common import google_utils
from seed_rl.common import normalizer
from seed_rl.mujoco import env
import tensorflow as tf


# Optimizer settings.
flags.DEFINE_float('learning_rate', 3e-4, 'Learning rate.')
# Network settings.
flags.DEFINE_integer('n_critics', 2, 'Number of Q networks.')
flags.DEFINE_integer('n_mlp_layers', 2, 'Number of MLP hidden layers.')
flags.DEFINE_integer('mlp_size', 256, 'Sizes of each of MLP hidden layer.')
flags.DEFINE_integer(
    'n_lstm_layers', 0,
    'Number of LSTM layers. LSTM layers afre applied after MLP layers.')
flags.DEFINE_integer('lstm_size', 256, 'Sizes of each LSTM layer.')
flags.DEFINE_bool('normalize_observations', False, 'Whether to normalize'
                  'observations by subtracting mean and dividing by stddev.')
# Environment settings.
flags.DEFINE_string('env_name', 'HalfCheetah-v2',
                    'Name of the environment from OpenAI Gym.')
flags.DEFINE_enum(
    'discretization', 'none', ['none', 'lin', 'log'], 'Values other than '
    '"none" cause action coordinates to be discretized into n_actions_per_dim '
    'buckets. Buckets are spaced linearly between the bounds if "lin" mode is '
    'used and logarithmically for "log" mode.')
flags.DEFINE_integer(
    'n_actions_per_dim', 11, 'The number of buckets per action coordinate if '
    'discretization is used.')
flags.DEFINE_float(
    'action_ratio', 30.,
    'The ratio between the highest and the lowest positive '
    'action for logarithmic action discretization.')

FLAGS = flags.FLAGS


def create_agent(unused_action_space, unused_env_observation_space,
                 parametric_action_distribution):
  policy = networks.ActorCriticMLP(
      parametric_action_distribution,
      n_critics=FLAGS.n_critics,
      mlp_sizes=[FLAGS.mlp_size] * FLAGS.n_mlp_layers)
  if FLAGS.normalize_observations:
    policy = normalizer.NormalizeObservationsWrapper(policy,
                                                     normalizer.Normalizer())
  return policy


def create_optimizer(unused_final_iteration):
  learning_rate_fn = lambda iteration: FLAGS.learning_rate
  optimizer = tf.keras.optimizers.Adam(FLAGS.learning_rate)
  return optimizer, learning_rate_fn


def main(argv):
  create_environment = lambda task, config: env.create_environment(  
      env_name=config.env_name,
      discretization=config.discretization,
      n_actions_per_dim=config.n_actions_per_dim,
      action_ratio=config.action_ratio)

  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
  if FLAGS.run_mode == 'actor':
    actor.actor_loop(create_environment)
  elif FLAGS.run_mode == 'learner':
    learner.learner_loop(create_environment,
                         create_agent,
                         create_optimizer)
  else:
    raise ValueError('Unsupported run mode {}'.format(FLAGS.run_mode))


if __name__ == '__main__':
  app.run(main)
