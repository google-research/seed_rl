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
"""SEED agent using Keras."""

import collections
from seed_rl.common import utils
import tensorflow as tf

AgentOutput = collections.namedtuple('AgentOutput', 'action q_values')
AgentState = collections.namedtuple(
    # core_state: the opaque state of the recurrent core of the agent.
    # frame_stacking_state: a list of the last (stack_size - 1) observations
    #   with shapes typically [batch_size, height, width, 1].
    'AgentState', 'core_state frame_stacking_state')

STACKING_STATE_DTYPE = tf.int32


def initial_frame_stacking_state(stack_size, batch_size, observation_shape):
  """Returns the initial frame stacking state.

  It should match what stack_frames accepts and produces.

  Args:
    stack_size: int, the number of frames that should be stacked to form the
      observation provided to the neural network. stack_size=1 corresponds to no
      stacking.
    batch_size: int tensor.
    observation_shape: list, shape of a single observation, e.g.
      [height, width, 1].

  Returns:
    <STACKING_STATE_DTYPE>[batch_size, prod(observation_shape)] or an empty
    tuple if stack_size=1.
  """
  if stack_size == 1:
    return ()
  return tf.zeros(
      tf.concat([[batch_size], [tf.math.reduce_prod(observation_shape)]],
                axis=0),
      dtype=STACKING_STATE_DTYPE)


def stack_frames(frames, frame_stacking_state, done, stack_size):
  """Stacks frames.

  The [height, width] center dimensions of the tensors below are typical, but
  could be anything else consistent between tensors.

  Args:
    frames: <float32>[time, batch_size, height, width, channels]. These should
      be un-normalized frames in range [0, 255]. channels must be equal to 1
      when we actually stack frames (stack_size > 1).
    frame_stacking_state: If stack_size > 1, <int32>[batch_size, height*width].
      () if stack_size=1.
      Frame are bit-packed. The LSBs correspond to the oldest frames, MSBs to
      newest. Frame stacking state contains un-normalized frames in range
      [0, 256). We use [height*width] for the observation shape instead of
      [height, width] because it speeds up transfers to/from TPU by a factor ~2.
    done: <bool>[time, batch_size]
    stack_size: int, the number of frames to stack.
  Returns:
    A pair:
      - stacked frames, <float32>[time, batch_size, height, width, stack_size]
        tensor (range [0, 255]). Along the stack dimensions, frames go from
        newest to oldest.
      - New stacking state with the last (stack_size-1) frames.
  """
  if frames.shape[0:2] != done.shape[0:2]:
    raise ValueError(
        'Expected same first 2 dims for frames and dones. Got {} vs {}.'.format(
            frames.shape[0:2], done.shape[0:2]))
  batch_size = frames.shape[1]
  obs_shape = frames.shape[2:-1]
  if stack_size > 4:
    raise ValueError('Only up to stack size 4 is supported due to bit-packing.')
  if stack_size > 1 and frames.shape[-1] != 1:
    raise ValueError('Due to frame stacking, we require last observation '
                     'dimension to be 1. Got {}'.format(frames.shape[-1]))
  if stack_size == 1:
    return frames, ()
  if frame_stacking_state[0].dtype != STACKING_STATE_DTYPE:
    raise ValueError('Expected dtype {} got {}'.format(
        STACKING_STATE_DTYPE, frame_stacking_state[0].dtype))

  frame_stacking_state = tf.reshape(
      frame_stacking_state, [batch_size] + obs_shape)

  # Unpacked 'frame_stacking_state'. Ordered from oldest to most recent.
  unstacked_state = []
  for i in range(stack_size - 1):
    # [batch_size, height, width]
    unstacked_state.append(tf.cast(tf.bitwise.bitwise_and(
        tf.bitwise.right_shift(frame_stacking_state, i * 8), 0xFF),
                                   tf.float32))

  # Same as 'frames', but with the previous (stack_size - 1) frames from
  # frame_stacking_state prepended.
  # [time+stack_size-1, batch_size, height, width, 1]
  extended_frames = tf.concat(
      [tf.reshape(frame, [1] + frame.shape + [1])
       for frame in unstacked_state] +
      [frames],
      axis=0)

  # [time, batch_size, height, width, stack_size].
  # Stacked frames, but does not take 'done' into account. We need to zero-out
  # the frames that cross episode boundaries.
  # Along the stack dimensions, frames go from newest to oldest.
  stacked_frames = tf.concat(
      [extended_frames[stack_size - 1 - i:extended_frames.shape[0] - i]
       for i in range(stack_size)],
      axis=-1)

  # We create a mask that contains true when the frame should be zeroed out.
  # Setting the final shape of the mask early actually makes computing
  # stacked_done_masks a few times faster.
  done_mask_row_shape = done.shape[0:2] + [1] * (frames.shape.rank - 2)
  done_masks = [
      tf.zeros(done_mask_row_shape, dtype=tf.bool),
      tf.reshape(done, done_mask_row_shape)
  ]
  while len(done_masks) < stack_size:
    previous_row = done_masks[-1]
    # Add 1 zero in front (time dimension).
    done_masks.append(
        tf.math.logical_or(
            previous_row,
            tf.pad(previous_row[:-1],
                   [[1, 0]] + [[0, 0]] * (previous_row.shape.rank - 1))))

  # This contains true when the frame crosses an episode boundary and should
  # therefore be zeroed out.
  # Example: ignoring batch_size, if done is [0, 1, 0, 0, 1, 0], stack_size=4,
  # this will be:
  # [[0 0, 0, 0, 0, 0],
  #  [0 1, 0, 0, 1, 0],
  #  [0 1, 1, 0, 1, 1],
  #  [0 1, 1, 1, 1, 1]].T
  # <bool>[time, batch_size, 1, 1, stack_size].
  stacked_done_masks = tf.concat(done_masks, axis=-1)
  stacked_frames = tf.where(
      stacked_done_masks,
      tf.zeros_like(stacked_frames), stacked_frames)

  # Build the new bit-packed state.
  # We construct the new state from 'stacked_frames', to make sure frames
  # before done is true are zeroed out.
  # This shifts the stack_size-1 items of the last dimension of
  # 'stacked_frames[-1, ..., :-1]'.
  shifted = tf.bitwise.left_shift(
      tf.cast(stacked_frames[-1, ..., :-1], tf.int32),
      # We want to shift so that MSBs are newest frames.
      [8 * i for i in range(stack_size - 2, -1, -1)])
  # This is really a reduce_or, because bits don't overlap.
  new_state = tf.reduce_sum(shifted, axis=-1)

  new_state = tf.reshape(new_state, [batch_size, obs_shape.num_elements()])

  return stacked_frames, new_state


def _unroll_cell(inputs, done, start_state, zero_state, recurrent_cell):
  """Applies a recurrent cell on inputs, taking care of managing state.

  Args:
    inputs: A tensor of shape [time, batch_size, <remaining dims>]. These are
      the inputs passed to the recurrent cell.
    done: <bool>[time, batch_size].
    start_state: Recurrent cell state at the beginning of the input sequence.
      Opaque tf.nest structure of tensors with batch front dimension.
    zero_state: Blank recurrent cell state. The current recurrent state will be
      replaced by this blank state whenever 'done' is true. Same shape as
      'start_state'.
    recurrent_cell: Function that will be applied at each time-step. Takes
      (input_t: [batch_size, <remaining dims>], current_state) as input, and
      returns (output_t: [<cell output dims>], new_state).

  Returns:
    A pair:
      - The time-stacked outputs of the recurrent cell. Shape [time,
        <cell output dims>].
      - The last state output by the recurrent cell.
  """
  stacked_outputs = []
  state = start_state
  inputs_list = tf.unstack(inputs)
  done_list = tf.unstack(done)
  assert len(inputs_list) == len(done_list), (
      "Inputs and done tensors don't have same time dim {} vs {}".format(
          len(inputs_list), len(done_list)))
  # Loop over time dimension.
  # input_t: [batch_size, batch_size, <remaining dims>].
  # done_t: [batch_size].
  for input_t, done_t in zip(inputs_list, done_list):
    # If the episode ended, the frame state should be reset before the next.
    state = tf.nest.map_structure(
        lambda x, y, done_t=done_t: tf.where(  
            tf.reshape(done_t, [done_t.shape[0]] + [1] *
                       (x.shape.rank - 1)), x, y),
        zero_state,
        state)
    output_t, state = recurrent_cell(input_t, state)
    stacked_outputs.append(output_t)
  return tf.stack(stacked_outputs), state


class _Stack(tf.Module):
  """Stack of pooling and convolutional blocks with residual connections."""

  def __init__(self, num_ch, num_blocks):
    super(_Stack, self).__init__(name='stack')
    self._conv = tf.keras.layers.Conv2D(num_ch, 3, strides=1, padding='same')
    self._max_pool = tf.keras.layers.MaxPool2D(
        pool_size=3, padding='same', strides=2)

    self._res_convs0 = [
        tf.keras.layers.Conv2D(
            num_ch, 3, strides=1, padding='same', name='res_%d/conv2d_0' % i)
        for i in range(num_blocks)
    ]
    self._res_convs1 = [
        tf.keras.layers.Conv2D(
            num_ch, 3, strides=1, padding='same', name='res_%d/conv2d_1' % i)
        for i in range(num_blocks)
    ]

  def __call__(self, conv_out):
    # Downscale.
    conv_out = self._conv(conv_out)
    conv_out = self._max_pool(conv_out)

    # Residual block(s).
    for (res_conv0, res_conv1) in zip(self._res_convs0, self._res_convs1):
      block_input = conv_out
      conv_out = tf.nn.relu(conv_out)
      conv_out = res_conv0(conv_out)
      conv_out = tf.nn.relu(conv_out)
      conv_out = res_conv1(conv_out)
      conv_out += block_input

    return conv_out

class DuelingLSTMDQNNet(tf.Module):
  """The recurrent network used to compute the agent's Q values.

  This is the dueling LSTM net similar to the one described in
  https://openreview.net/pdf?id=rkHVZWZAZ (only the Q(s, a) part), with the
  layer sizes mentioned in the R2D2 paper
  (https://openreview.net/pdf?id=r1lyTjAqYX), section Hyper parameters.
  """

  def __init__(self, num_actions, observation_shape, stack_size=1):
    super(DuelingLSTMDQNNet, self).__init__(name='dueling_lstm_dqn_net')
    self._num_actions = num_actions
    # self._body = tf.keras.Sequential([
    #     tf.keras.layers.Conv2D(32, [8, 8], 4,
    #                            padding='valid', activation='relu'),
    #     tf.keras.layers.Conv2D(64, [4, 4], 2,
    #                            padding='valid', activation='relu'),
    #     tf.keras.layers.Conv2D(64, [3, 3], 1,
    #                            padding='valid', activation='relu'),
    #     tf.keras.layers.Flatten(),
    #     tf.keras.layers.Dense(512, activation='relu'),
    # ])

    self._stacks = [
        _Stack(num_ch, num_blocks)
        for num_ch, num_blocks in [(16, 2), (32, 2), (32, 2)]
    ]
    self._conv_to_linear = tf.keras.layers.Dense(256)

    self._value = tf.keras.Sequential([
        tf.keras.layers.Dense(256, activation='relu', name='hidden_value'),
        tf.keras.layers.Dense(1, name='value_head'),
    ])
    self._advantage = tf.keras.Sequential([
        tf.keras.layers.Dense(256, activation='relu', name='hidden_advantage'),
        tf.keras.layers.Dense(self._num_actions, use_bias=False,
                              name='advantage_head'),
    ])
    self._core = tf.keras.layers.LSTMCell(256)
    self._observation_shape = observation_shape
    self._stack_size = stack_size

  def initial_state(self, batch_size):
    return AgentState(
        core_state=self._core.get_initial_state(
            batch_size=batch_size, dtype=tf.float32),
        frame_stacking_state=initial_frame_stacking_state(
            self._stack_size, batch_size, self._observation_shape))

  def _torso(self, prev_action, env_output):
    # [batch_size, output_units]
    # conv_out = self._body(env_output.observation)

    conv_out = env_output.observation
    for stack in self._stacks:
      conv_out = stack(conv_out)
    conv_out = tf.nn.relu(conv_out)
    conv_out = tf.keras.layers.Flatten()(conv_out)
    conv_out = self._conv_to_linear(conv_out)
    conv_out = tf.nn.relu(conv_out)


    # clipped_reward = tf.expand_dims(tf.clip_by_value(env_output.reward, -1, 1), -1)
    squeezed = tf.tanh(env_output.reward / 5.0)
    # Negative rewards are given less weight than positive rewards.
    clipped_rewards = tf.expand_dims(tf.where(env_output.reward < 0, .3 * squeezed, squeezed) * 5., -1)
    # [batch_size, num_actions]
    one_hot_prev_action = tf.one_hot(prev_action, self._num_actions)
    # [batch_size, torso_output_size]
    # return tf.concat(
    #     [conv_out, tf.expand_dims(env_output.reward, -1), one_hot_prev_action],
    #     axis=1)
    return tf.concat(
        [conv_out, clipped_rewards, one_hot_prev_action],
        axis=1)

  def _head(self, core_output):
    # [batch_size, 1]
    value = self._value(core_output)

    # [batch_size, num_actions]
    advantage = self._advantage(core_output)
    advantage -= tf.reduce_mean(advantage, axis=-1, keepdims=True)

    # [batch_size, num_actions]
    q_values = value + advantage

    action = tf.cast(tf.argmax(q_values, axis=1), tf.int32)
    return AgentOutput(action, q_values)

  def __call__(self, input_, agent_state, unroll=False):
    """Applies a network mapping observations to actions.

    Args:
      input_: A pair of:
        - previous actions, <int32>[batch_size] tensor if unroll is False,
          otherwise <int32>[time, batch_size].
        - EnvOutput, where each field is a tensor with added front
          dimensions [batch_size] if unroll is False and [time, batch_size]
          otherwise.
      agent_state: AgentState with batched tensors, corresponding to the
        beginning of each unroll.
      unroll: should unrolling be aplied.

    Returns:
      A pair of:
        - outputs: AgentOutput, where action is a tensor <int32>[time,
            batch_size], q_values is a tensor <float32>[time, batch_size,
            num_actions]. The time dimension is not present if unroll=False.
        - agent_state: Output AgentState with batched tensors.
    """
    if not unroll:
      # Add time dimension.
      input_ = tf.nest.map_structure(lambda t: tf.expand_dims(t, 0),
                                     input_)
    prev_actions, env_outputs = input_
    outputs, agent_state = self._unroll(prev_actions, env_outputs, agent_state)
    if not unroll:
      # Remove time dimension.
      outputs = tf.nest.map_structure(lambda t: tf.squeeze(t, 0), outputs)

    return outputs, agent_state

  def _unroll(self, prev_actions, env_outputs, agent_state):
    # [time, batch_size, <field shape>]
    unused_reward, done, observation, _, _ = env_outputs
    observation = tf.cast(observation, tf.float32)

    initial_agent_state = self.initial_state(batch_size=tf.shape(done)[1])

    stacked_frames, frame_state = stack_frames(
        observation, agent_state.frame_stacking_state, done, self._stack_size)

    env_outputs = env_outputs._replace(observation=stacked_frames / 255)
    # [time, batch_size, torso_output_size]
    torso_outputs = utils.batch_apply(self._torso, (prev_actions, env_outputs))

    core_outputs, core_state = _unroll_cell(
        torso_outputs, done, agent_state.core_state,
        initial_agent_state.core_state,
        self._core)

    agent_output = utils.batch_apply(self._head, (core_outputs,))
    return agent_output, AgentState(core_state, frame_state)
