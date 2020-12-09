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


"""Module to compute advantages and calculate a single training epoch."""
from seed_rl.agents.policy_gradient.modules import logging_module
import tensorflow as tf


def ppo_training_step(
    epochs_per_step, loss_fn, args, batch_mode, training_strategy,
    virtual_batch_size, unroll_length, batches_per_step, clip_norm,
    optimizer, logger):
  """Performs a PPO training step, running epochs_per_step times over the data.

  Args:
    epochs_per_step: The number of epochs to perform over the virtual batch.
    loss_fn: Policy loss class.
    args: Unroll struct with tensors whose leading dimensions are time and
      batch_size * batches_per_step (except for the core state which does not
      have a time leading dimension).
    batch_mode: How to handle minibatches: go over the same minibatches
      multiple times (repeat), shuffle unrolls in each epoch (shuffle) or
      compute the advantages once and then split the unrolls into individual
      transitions which are then shuffled (split).
      split_with_advantages_recomputation works like split but the advantages
      are recomputed at the beginning of each pass over the data.'
    training_strategy: TF distribute strategy.
    virtual_batch_size: batch_size * batches_per_step.
    unroll_length: length of one unroll in agent steps.
    batches_per_step: How many mini batches to pass over in a training step.
    clip_norm: value to clip the gradient global norm to.
    optimizer: tf.keras.optimizers.Optimizer.
    logger: SEED logger.

  Returns:
    Loss and log (SEED log session) of the training step.
  """
  orig_args = args
  if batch_mode in ['split', 'split_with_advantage_recomputation']:
    args = compute_advantages_and_split(loss_fn, orig_args)

  # tf.range loops can not create new variables so we first run a single
  # epoch to create the logs.
  loss, logs = single_epoch(loss_fn, args, batch_mode, training_strategy,
                            virtual_batch_size, unroll_length,
                            batches_per_step, clip_norm, optimizer,
                            logger)
  for _ in tf.range(epochs_per_step - 1):
    if batch_mode == 'split_with_advantage_recomputation':
      args = compute_advantages_and_split(loss_fn, orig_args)
    loss, logs = single_epoch(loss_fn, args, batch_mode, training_strategy,
                              virtual_batch_size, unroll_length,
                              batches_per_step, clip_norm, optimizer,
                              logger)
  return loss, logs



def compute_advantages_and_split(loss_fn, input_args):
  """Computes advantages and split tensors into transitions.

  Args:
    loss_fn: Policy loss providing a compute_advantages method.
    input_args: Unroll struct with tensors whose leading dimensions are time and
      batch_size * batches_per_step.

  Returns:
    output_args: Tuple containing (agent_state, prev_actions, env_outputs,
          agent_outputs, normalized_targets, normalized_advantages). It
          corresponds to input_args with the last step removed from each tensor,
          and the advantages, split into individual transitions (i.e., each
          tensor from input_args [T,B] becomes [1, (T-1)*B])
  """
  if input_args.agent_state != ():  
    raise ValueError('Agent state is not supported for split_* batch modes. '
                     'Use shuffle or repeat batch modes.')
  advantages = loss_fn.compute_advantages(*input_args)

  # Last timestep is only used for bootstrapping for advantage estimation
  # so we can remove it here.
  output_args = tf.nest.map_structure(lambda t: t[:-1], input_args)
  output_args += advantages

  # Split into individual transitions.
  def unroll2transitions(t):  # [T, B] -> [1, T * B]
    return tf.reshape(t, [1, t.shape[0] * t.shape[1]] + t.shape[2:])

  output_args = tf.nest.map_structure(unroll2transitions, output_args)
  return output_args



def single_epoch(loss_fn, args, batch_mode, training_strategy,
                 virtual_batch_size, unroll_length, batches_per_step, clip_norm,
                 optimizer, logger):
  """Computes a single training epoch in PPO.

  Args:
    loss_fn: Policy loss class.
    args: Unroll struct (or unroll and advantages) with tensors whose leading
      dimensions are time and batch_size * batches_per_step, unless batch_mode
      is `split` or `split_with_advantage_recomputation`. In that case, it is
      the output of compute_advantages_and_split (with leading dimensions
      [1, batch_size * batches_per_step * unroll_length]).
    batch_mode: How to handle minibatches: go over the same minibatches
      multiple times (repeat), shuffle unrolls in each epoch (shuffle) or
      compute the advantages once and then split the unrolls into individual
      transitions which are then shuffled (split).
      split_with_advantages_recomputation works like split but the advantages
      are recomputed at the beginning of each pass over the data.'
    training_strategy: TF distribute strategy.
    virtual_batch_size: batch_size * batches_per_step.
    unroll_length: length of one unroll in agent steps.
    batches_per_step: How many mini batches to pass over in a training step.
    clip_norm: value to clip the gradient global norm to.
    optimizer: tf.keras.optimizers.Optimizer.
    logger: SEED logger.

  Returns:
    Loss and log (SEED log session) of the epoch.
  """

  # Check if shapes are correct.
  front_shapes = tf.nest.map_structure(lambda t: list(t.shape[:2]),
                                       tf.nest.flatten(args))
  # All tensors apart from agent_state have leading dimensions [time, batch]
  # while agent_state has no time dimension.
  timesteps, batch_size = front_shapes[-1]
  for i, shape in enumerate(front_shapes):
    if shape[0] != batch_size and shape != [timesteps, batch_size]:
      raise ValueError(
          f'Loss argument {i} expected to have leading dimensions '
          f'[time, batch] ([{timesteps}, {batch_size}]) or [batch] '
          f'(in the case of memory state) but encountered a tensor with '
          f'leading dimensions {shape}')
  if batch_mode in ['split', 'split_with_advantage_recomputation']:
    if timesteps != 1:
      raise ValueError(f'Unexpected timesteps {timesteps}')
    if (training_strategy.num_replicas_in_sync * batch_size !=
        virtual_batch_size * unroll_length):
      raise ValueError(
          f'Batch size mismatch '
          f'{training_strategy.num_replicas_in_sync * batch_size} vs '
          f'{virtual_batch_size * unroll_length}')
  else:
    if timesteps != unroll_length + 1:
      raise ValueError(f'Timesteps mismatch {timesteps} vs {unroll_length + 1}')
    if (training_strategy.num_replicas_in_sync * batch_size !=
        virtual_batch_size):
      raise ValueError(
          f'Batch size mismatch '
          f'{training_strategy.num_replicas_in_sync * batch_size} vs '
          f'{virtual_batch_size}')
  # Split into minibatches.
  indices = tf.range(batch_size)
  if batch_mode != 'repeat':
    indices = tf.random.shuffle(indices)

  @tf.function
  def train_on_minibatch(indices, logger):
    if batch_mode.startswith('split'):
      agent_state = args[0]
      if agent_state:
        raise ValueError(
            'Split batch mode is not supported for models with memory. '
            'Use --batch_mode=shuffle.')
      minibatch = tf.nest.map_structure(lambda t: tf.gather(t, indices, axis=1),
                                        args)
    else:
      # All tensors apart from agent_state have leading dimensions [time, batch]
      # while agent_state has no time dimension.
      # We add a dummy time dimension to it here...
      expanded_args = args._replace(
          agent_state=tf.nest.map_structure(lambda t: t[tf.newaxis, ...],
                                            args.agent_state))
      minibatch = tf.nest.map_structure(lambda t: tf.gather(t, indices, axis=1),
                                        expanded_args)
      # ...and remove it here.
      minibatch = minibatch._replace(
          agent_state=tf.nest.map_structure(lambda t: tf.squeeze(t, axis=0),
                                            minibatch.agent_state))

    with tf.GradientTape() as tape:
      with logging_module.LoggingTape(loss_fn) as logs_dict:
        loss = loss_fn(*minibatch)
      session = logger.log_session_from_dict(logs_dict)
    # Normalizers may have unused compensation variables so we need
    # unconnected_gradients=tf.UnconnectedGradients.ZERO to avoid getting Nones
    # for them.
    grads = tape.gradient(
        loss, loss_fn.trainable_variables,
        unconnected_gradients=tf.UnconnectedGradients.ZERO)

    gradient_norm_before_clip = tf.linalg.global_norm(grads)
    if clip_norm is not None:
      grads, _ = tf.clip_by_global_norm(
          grads, clip_norm, use_norm=gradient_norm_before_clip)
    logger.log(session, 'grad/norm', gradient_norm_before_clip)
    logger.log(session, 'grad/trainable_variables',
               tf.linalg.global_norm(loss_fn.trainable_variables))
    optimizer.apply_gradients(zip(grads, loss_fn.trainable_variables))
    return loss, session

  # Train.
  for minibatch_indices in tf.split(indices, batches_per_step):
    loss, session = train_on_minibatch(minibatch_indices, logger)

  return loss, session
