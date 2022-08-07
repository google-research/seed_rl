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


"""R2D2 binary for ATARI-57.

Actor and learner are in the same binary so that all flags are shared.
"""


from absl import app
from absl import flags
from seed_rl.agents.r2d2 import learner
from seed_rl.procgen import env
from seed_rl.procgen import networks
from seed_rl.common import actor
from seed_rl.common import common_flags  
import tensorflow as tf
import os


FLAGS = flags.FLAGS

# Optimizer settings.
flags.DEFINE_float('learning_rate', 1e-4, 'Learning rate.')
flags.DEFINE_float('adam_epsilon', 1e-3, 'Adam epsilon.')
flags.DEFINE_string('sub_task', 'all', 'sub tasks')
flags.DEFINE_list('task_names', [], 'names of tasks')
flags.DEFINE_float('reward_threshold', 0., 'reward threshold for sampling')

def create_agent(env_output_specs, num_actions):
  return networks.DuelingLSTMDQNNet(
      num_actions, env_output_specs.observation.shape)


def create_optimizer(final_iteration):
  learning_rate_fn = lambda iteration: FLAGS.learning_rate
  # learning_rate_fn = tf.keras.optimizers.schedules.PolynomialDecay(
  #   FLAGS.learning_rate, final_iteration, 0)
  optimizer = tf.keras.optimizers.Adam(FLAGS.learning_rate,
                                       epsilon=FLAGS.adam_epsilon)
  return optimizer, learning_rate_fn

def main(argv):
  if FLAGS.sub_task == 'all':
    FLAGS.task_names = env.games.keys()
  else:
    FLAGS.task_names = [FLAGS.sub_task]
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
  print('task')
  print(FLAGS.sub_task)
  print('subtask names')
  print(FLAGS.task_names)
  FLAGS.reward_threshold = env.games[FLAGS.sub_task][2]
  if FLAGS.run_mode == 'actor':
    actor.actor_loop(env.create_environment)
  elif FLAGS.run_mode == 'learner':
    # for i in range(len(FLAGS.task_names)):
    #   cur_path = FLAGS.logdir + '/' + FLAGS.task_names[i] + '_dataset'
    #   if not os.path.exists(cur_path):
    #     os.makedirs(cur_path)
    learner.learner_loop(env.create_environment,
                         create_agent,
                         create_optimizer)
  else:
    raise ValueError('Unsupported run mode {}'.format(FLAGS.run_mode))


if __name__ == '__main__':
  app.run(main)
