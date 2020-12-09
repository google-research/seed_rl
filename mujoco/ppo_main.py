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


"""PPO learner for Mujoco."""


import os
import tempfile
from absl import app
from absl import flags
from absl import logging
import gin
import gin.tf.external_configurables
from seed_rl.agents.policy_gradient import learner
from seed_rl.agents.policy_gradient import learner_flags
from seed_rl.agents.policy_gradient.modules import continuous_control_agent
from seed_rl.agents.policy_gradient.modules import popart  
from seed_rl.common import actor
from seed_rl.common import common_flags  
from seed_rl.common import parametric_distribution
from seed_rl.common import utils
from seed_rl.mujoco import env
import tensorflow as tf

gin.external_configurable(tf.keras.initializers.Orthogonal,
                          name='Orthogonal', module='tf.keras.initializers')

# Enable configuring parametric action distribution before Gin configs are read
# and locked.
continuous_action_config = gin.external_configurable(
    parametric_distribution.continuous_action_config)

FLAGS = flags.FLAGS

flags.DEFINE_float('learning_rate', 0.0003, 'Learning rate.')
flags.DEFINE_float('lr_decay_multiplier', 0.,
                   'Franction of original learning rate to decay to.')
flags.DEFINE_string('env_name', 'HalfCheetah-v2',
                    'Name of the environment from OpenAI Gym.')
flags.DEFINE_string('mujoco_model',
                    None,
                    'Optional path to the xml mujoco model to use. It must be '
                    'compatible with the environment class selected with '
                    '--env_name.')

flags.DEFINE_string('gin_config', '', 'A path to a config file.')
flags.DEFINE_multi_string('gin_bindings', [],
                          'Newline separated list of Gin parameter bindings.')

gin.external_configurable(tf.exp, name='exp', module='tf')
gin.external_configurable(tf.keras.layers.LayerNormalization,
                          'LayerNormalization', module='tf.keras.layers')
gin.external_configurable(tf.keras.initializers.VarianceScaling)
gin.external_configurable(tf.keras.initializers.GlorotUniform)
gin.external_configurable(tf.keras.initializers.GlorotNormal)
gin.external_configurable(tf.keras.initializers.lecun_normal, 'lecun_normal')
gin.external_configurable(tf.keras.initializers.lecun_uniform, 'lecun_uniform')
gin.external_configurable(tf.keras.initializers.he_normal, 'he_normal')
gin.external_configurable(tf.keras.initializers.he_uniform, 'he_uniform')
gin.external_configurable(tf.keras.initializers.TruncatedNormal)


@gin.configurable
def orthogonal_gain_sqrt2():
  return tf.keras.initializers.Orthogonal(1.41421356237)


@gin.configurable
def orthogonal_gain_0dot01():
  return tf.keras.initializers.Orthogonal(0.01)


@gin.configurable
def create_optimizer(final_iteration, optimizer_fn):
  learning_rate_fn = tf.keras.optimizers.schedules.PolynomialDecay(
      FLAGS.learning_rate, final_iteration,
      FLAGS.lr_decay_multiplier * FLAGS.learning_rate)
  optimizer = optimizer_fn(learning_rate_fn)
  return optimizer, learning_rate_fn


def create_agent(unused_action_space, unused_observation_space,
                 parametric_action_distribution):
  return continuous_control_agent.ContinuousControlAgent(
      parametric_action_distribution=parametric_action_distribution)


def main(unused_argv):
  # Save the string flags now as we modify them later.
  string_flags = FLAGS.flags_into_string()
  gin.parse_config_files_and_bindings(
      [FLAGS.gin_config] if FLAGS.gin_config else [],
      # Gin uses slashes to denote scopes but XM doesn't allow slashes in
      # parameter names so we use __ instead and convert it to slashes here.
      [s.replace('__', '/') for s in FLAGS.gin_bindings])
  gym_kwargs = {}
  if FLAGS.mujoco_model:
    local_mujoco_model = tempfile.mkstemp(
        prefix='mujoco_model', suffix='.xml')[1]
    logging.info('Copying remote model %s to local file %s', FLAGS.mujoco_model,
                 local_mujoco_model)
    tf.io.gfile.copy(FLAGS.mujoco_model, local_mujoco_model, overwrite=True)
    gym_kwargs['model_path'] = local_mujoco_model

  create_environment = lambda task, config: env.create_environment(  
      env_name=config.env_name,
      discretization='none',
      n_actions_per_dim=11,
      action_ratio=30,
      gym_kwargs=gym_kwargs)

  if FLAGS.run_mode == 'actor':
    actor.actor_loop(create_environment)
  elif FLAGS.run_mode == 'learner':
    logdir = FLAGS.logdir
    settings = utils.init_learner_multi_host(FLAGS.num_training_tpus)
    learner.learner_loop(
        create_environment,
        create_agent,
        create_optimizer,
        learner_flags.training_config_from_flags(),
        settings,
        action_distribution_config=continuous_action_config())
    with tf.io.gfile.GFile(os.path.join(logdir, 'learner_flags.txt'), 'w') as f:
      f.write(string_flags)
    with tf.io.gfile.GFile(os.path.join(logdir, 'learner.gin'), 'w') as f:
      f.write(gin.operative_config_str())
  else:
    raise ValueError('Unsupported run mode {}'.format(FLAGS.run_mode))


if __name__ == '__main__':
  app.run(main)
