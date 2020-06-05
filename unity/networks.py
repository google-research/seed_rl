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
    'AgentState', 'core_state')

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


class DuelingLSTMDQNNet(tf.Module):
  """The recurrent network used to compute the agent's Q values.

  This is the dueling LSTM net similar to the one described in
  https://openreview.net/pdf?id=rkHVZWZAZ (only the Q(s, a) part), with the
  layer sizes mentioned in the R2D2 paper
  (https://openreview.net/pdf?id=r1lyTjAqYX), section Hyper parameters.
  """

  def __init__(self, num_actions, observation_shape):
    super(DuelingLSTMDQNNet, self).__init__(name='dueling_lstm_dqn_net')
    self._num_actions = num_actions
    self._body = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, [8, 8], 4,
                               padding='valid', activation='relu'),
        tf.keras.layers.Conv2D(64, [4, 4], 2,
                               padding='valid', activation='relu'),
        tf.keras.layers.Conv2D(64, [3, 3], 1,
                               padding='valid', activation='relu'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
    ])
    self._value = tf.keras.Sequential([
        tf.keras.layers.Dense(512, activation='relu', name='hidden_value'),
        tf.keras.layers.Dense(1, name='value_head'),
    ])
    self._advantage = tf.keras.Sequential([
        tf.keras.layers.Dense(512, activation='relu', name='hidden_advantage'),
        tf.keras.layers.Dense(self._num_actions, use_bias=False,
                              name='advantage_head'),
    ])
    self._core = tf.keras.layers.LSTMCell(512)
    self._observation_shape = observation_shape

  def initial_state(self, batch_size):
    return AgentState(
        core_state=self._core.get_initial_state(
            batch_size=batch_size, dtype=tf.float32))

  def _torso(self, prev_action, env_output):
    # [batch_size, output_units]
    conv_out = self._body(env_output.observation)
    # [batch_size, num_actions]
    one_hot_prev_action = tf.one_hot(prev_action, self._num_actions)
    # [batch_size, torso_output_size]
    return tf.concat(
        [conv_out, tf.expand_dims(env_output.reward, -1), one_hot_prev_action],
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
    unused_reward, done, observation = env_outputs
    observation = tf.cast(observation, tf.float32)

    initial_agent_state = self.initial_state(batch_size=tf.shape(done)[1])

    env_outputs = env_outputs._replace(observation=observation / 255)
    # [time, batch_size, torso_output_size]
    torso_outputs = utils.batch_apply(self._torso, (prev_actions, env_outputs))

    core_outputs, core_state = _unroll_cell(
        torso_outputs, done, agent_state.core_state,
        initial_agent_state.core_state,
        self._core)

    agent_output = utils.batch_apply(self._head, (core_outputs,))
    return agent_output, AgentState(core_state)
