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
from seed_rl.agents.r2d2 import sampler
from seed_rl.procgen import env
from seed_rl.procgen import networks
from seed_rl.common import procgen_sampler
from seed_rl.common import common_flags  
import tensorflow as tf
import os


FLAGS = flags.FLAGS

# Optimizer settings.
flags.DEFINE_float('learning_rate', 1e-4, 'Learning rate.')
flags.DEFINE_float('adam_epsilon', 1e-3, 'Adam epsilon.')
flags.DEFINE_list('task_names', [], 'names of tasks')
flags.DEFINE_float('reward_threshold', 0., 'reward threshold for sampling')
flags.DEFINE_string('sub_task', 'all', 'sub tasks, i.e. dmlab30, dmlab26, all, others')
flags.DEFINE_string('init_checkpoint', None,
                    'Path to the checkpoint used to initialize the agent.')

flags.DEFINE_integer('save_checkpoint_secs', 900,
                     'Checkpoint save period in seconds.')
flags.DEFINE_integer('total_environment_frames', int(1e9),
                     'Total environment frames to train for.')
flags.DEFINE_integer('batch_size', 64, 'Batch size for training.')
flags.DEFINE_float('replay_ratio', 1.5,
                   'Average number of times each observation is replayed and '
                   'used for training. '
                   'The default of 1.5 corresponds to an interpretation of the '
                   'R2D2 paper using the end of section 2.3.')
flags.DEFINE_integer('inference_batch_size', 8,
                     'Batch size for inference, -1 for auto-tune.')
flags.DEFINE_integer('unroll_length', 100, 'Unroll length in agent steps.')
flags.DEFINE_integer('num_training_tpus', 1, 'Number of TPUs for training.')
flags.DEFINE_integer('update_target_every_n_step',
                     200,
                     'Update the target network at this frequency (expressed '
                     'in number of training steps)')
flags.DEFINE_integer('replay_buffer_size', int(1e3),
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
flags.DEFINE_float('clip_norm', 40, 'We clip gradient norm to this value.')
flags.DEFINE_float('value_function_rescaling_epsilon', 1e-3,
                   'Epsilon used for value function rescaling.')
flags.DEFINE_integer('n_steps', 5,
                     'n-step returns: how far ahead we look for computing the '
                     'Bellman targets.')
flags.DEFINE_float('discounting', .997, 'Discounting factor.')

# Eval settings
flags.DEFINE_float('eval_epsilon', 1e-3,
                   'Epsilon (as in epsilon-greedy) used for evaluation.')


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
    # procgen_sampler.actor_loop(env.create_environment)
    procgen_sampler.actor_loop(env.create_gym3_environment)
  elif FLAGS.run_mode == 'learner':
    # sampler.learner_loop(env.create_environment,
    #                      create_agent,
    #                      create_optimizer)
    sampler.learner_loop(env.create_gym3_environment,
                         create_agent,
                         create_optimizer)
  else:
    raise ValueError('Unsupported run mode {}'.format(FLAGS.run_mode))


if __name__ == '__main__':
  gpus = tf.config.experimental.list_physical_devices('GPU')
  if gpus:
      tf.config.experimental.set_memory_growth(gpus[0], True)
  app.run(main)
