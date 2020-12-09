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

# python3
"""Configurable onpolicy learner."""

from absl import flags

from seed_rl.agents.policy_gradient import learner_config
from seed_rl.common import common_flags  

# Training.
flags.DEFINE_integer('total_environment_frames', int(1e9),
                     'Total environment frames to train for.')
flags.DEFINE_integer('unroll_length', 100, 'Unroll length in agent steps.')
flags.DEFINE_integer('num_training_tpus', 1, 'Number of TPUs for training.')
flags.DEFINE_string('init_checkpoint', None,
                    'Path to the checkpoint used to initialize the agent.')
flags.DEFINE_integer(
    'num_checkpoints', 0,
    'How many checkpoints to generate besides the one created when the training'
    ' is done.')
flags.DEFINE_integer(
    'num_saved_models', 0,
    'How many times to save a model to disk periodically during training. Can'
    'be used for offline evaluation.')
flags.DEFINE_enum(
    'batch_mode', 'split',
    ['repeat', 'shuffle', 'split', 'split_with_advantage_recomputation'],
    'Defines how to handle virtual minibatches. We can go over exactly the '
    'same minibatches multiple times (repeat), shuffle unrolls in each epoch '
    '(shuffle) or compute the advantages once and then split the unrolls into '
    'individual transitions which are then shuffled (split). '
    'split_with_advantages_recomputation works like split but the advantages '
    'are recomputed at the beginning of each pass over the data.')
flags.DEFINE_bool(
    'prefetch_batches', True,
    'Whether to use tf.Dataset to prefetch a few (~4) virtual batches.'
    'Prefetching increases the training throughput but might hurt sample '
    'complexity.')

# Loss settings.
flags.DEFINE_float('clip_norm', None,
                   'We clip gradient global norm to this value.')

# Logging
flags.DEFINE_multi_string(
    'log_key', [], 'If set, only specified log keys are going to be logged.')
flags.DEFINE_integer('log_frequency', 100000,
                     'After how many steps to save logs.')
flags.DEFINE_integer(
    'log_episode_frequency', 10, 'We average at least that many episodes'
    ' before logging average episode return and length.')
flags.DEFINE_integer('summary_flush_seconds', 20,
                     'Time interval between flushes.')
flags.DEFINE_integer(
    'summary_max_queue_size', 1000,
    'Maximum number of summaries to keep in the queue before flushing.')
flags.DEFINE_integer(
    'number_of_sub_batches', 1,
    'In how many sub batches to split unroll batches in the '
    'unroll queue. Is used to decrease the wait time on grpc '
    'layer.')
flags.DEFINE_boolean(
    'block_inference_on_training', True,
    'Should inference block until current training batch is processed.')
flags.DEFINE_integer('inference_batch_size', -1,
                     'Batch size for inference, -1 for auto-tune.')
flags.DEFINE_boolean('print_episode_summaries', False,
                     'If true episode summaries will be added to task logs.')

# Flags to set size of the mini-batches which will be used for gradient steps.
# Only set one of the two options.
flags.DEFINE_integer('batch_size', 2,
                     'Batch size for training in terms of unrolls.')
flags.DEFINE_integer(
    'batch_size_transitions', 0,
    'Batch size for training in number of transitions (needs to be divisible '
    'by unroll_length).')


def unroll_length_from_flags() -> int:
  """Returns the unroll length from the flags."""
  if FLAGS.unroll_length > 0:
    return FLAGS.unroll_length

  step_size_transitions = None
  if FLAGS.step_size_transitions > 0:
    step_size_transitions = FLAGS.step_size_transitions
  elif FLAGS.batch_size_transitions > 0 and FLAGS.batches_per_step > 0:
    step_size_transitions = (
        FLAGS.batch_size_transitions * FLAGS.batches_per_step)

  if step_size_transitions is None:
    raise ValueError('Either flag `step_size_transitions` should be set or both'
                     ' flags `batch_size_transitions` and `batches_per_step` '
                     'should be set.')

  if step_size_transitions % FLAGS.num_envs != 0:
    raise ValueError('Flag `step_size_transitions` needs to be divisible by'
                     ' the flag `num_envs`.')
  return step_size_transitions // FLAGS.num_envs


def batch_size_from_flags() -> int:
  """Returns the batch size from flags."""
  if FLAGS.batch_size > 0 and FLAGS.batch_size_transitions == 0:
    return FLAGS.batch_size
  if FLAGS.batch_size == 0 and FLAGS.batch_size_transitions > 0:
    unroll_length = unroll_length_from_flags()
    if FLAGS.batch_size_transitions % unroll_length != 0:
      raise ValueError('Flag `batch_size_transitions` needs to be divisible by'
                       ' the flag `unroll_length`.')
    return FLAGS.batch_size_transitions // unroll_length
  raise ValueError('Exactly one of the flags `batch_size_transitions` and '
                   '`batch_size` needs to be non-zero.')


# Flags to set size of the experience used in one training step. Only set one of
# the options.
flags.DEFINE_integer('batches_per_step', 1,
                     'How many mini batches to pass over in a training step.')
flags.DEFINE_integer(
    'step_size_unroll', 0,
    'How many unrolls to pass over in a training step (needs to be divisible '
    'by batch_size).')
flags.DEFINE_integer(
    'step_size_transitions', 0,
    'How many transitions to pass over in a training step. (needs to be '
    'divisible by `batch_size`x`unroll_length`)')


def batches_per_step_from_flags() -> int:
  """Returns the number of batches per step from flags."""
  batch_size = batch_size_from_flags()
  if (FLAGS.batches_per_step > 0 and FLAGS.step_size_unroll == 0 and
      FLAGS.step_size_transitions == 0):
    return FLAGS.batches_per_step
  if (FLAGS.batches_per_step == 0 and FLAGS.step_size_unroll > 0 and
      FLAGS.step_size_transitions == 0):
    if FLAGS.step_size_unroll % batch_size != 0:
      raise ValueError('Flag `step_size_unroll` needs to be divisible by'
                       ' the `batch_size`.')
    return FLAGS.step_size_unroll // batch_size
  if (FLAGS.batches_per_step == 0 and FLAGS.step_size_unroll == 0 and
      FLAGS.step_size_transitions > 0):
    unroll_length = unroll_length_from_flags()
    if FLAGS.step_size_transitions % (batch_size * unroll_length) != 0:
      raise ValueError('Flag `step_size_transitions` needs to be divisible by'
                       ' the `batch_size` x `unroll_length`.')
    return FLAGS.step_size_transitions // (batch_size * unroll_length)
  raise ValueError('Exactly one of the flags `batches_per_step`, '
                   '`step_size_unroll` and `step_size_transitions` needs to be '
                   'non-zero.')


flags.DEFINE_integer(
    'epochs_per_step', 1, 'How many times to pass over all the'
    ' batches during a training step.')

# Profiling.
flags.DEFINE_enum_class(
    'profile_inference_return', learner_config.InferenceReturn.END,
    learner_config.InferenceReturn,
    'Allows early returns in the inference function to profile performance.')

FLAGS = flags.FLAGS


def training_config_from_flags() -> learner_config.TrainingConfig:
  """Returns training config from the command line flags."""
  return learner_config.TrainingConfig(
      batch_mode=FLAGS.batch_mode,
      batch_size=batch_size_from_flags(),
      batches_per_step=batches_per_step_from_flags(),
      block_inference_on_training=FLAGS.block_inference_on_training,
      clip_norm=FLAGS.clip_norm,
      env_batch_size=FLAGS.env_batch_size,
      env_name=FLAGS.env_name,
      epochs_per_step=FLAGS.epochs_per_step,
      inference_batch_size=FLAGS.inference_batch_size,
      log_episode_frequency=FLAGS.log_episode_frequency,
      num_action_repeats=FLAGS.num_action_repeats,
      num_envs=FLAGS.num_envs,
      num_eval_envs=FLAGS.num_eval_envs,
      print_episode_summaries=FLAGS.print_episode_summaries,
      profile_inference_return=FLAGS.profile_inference_return,
      summary_flush_seconds=FLAGS.summary_flush_seconds,
      summary_max_queue_size=FLAGS.summary_max_queue_size,
      total_environment_frames=FLAGS.total_environment_frames,
      unroll_length=unroll_length_from_flags(),
      num_checkpoints=FLAGS.num_checkpoints,
      num_saved_models=FLAGS.num_saved_models,
      prefetch_batches=FLAGS.prefetch_batches,
      server_address=FLAGS.server_address)
