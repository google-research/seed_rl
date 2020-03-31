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

"""MLP+LSTM network for use with V-trace."""

import collections
from seed_rl.common import utils
import tensorflow as tf

AgentOutput = collections.namedtuple('AgentOutput',
                                     'action policy_logits baseline')


class MLPandLSTM(tf.Module):
  """MLP+stacked LSTM Agent."""

  def __init__(self, parametric_action_distribution, mlp_sizes, lstm_sizes):
    """Creates an MLP followed by a stacked LSTM agent.

    Args:
      parametric_action_distribution: an object of ParametricDistribution class
        specifing a parametric distribution over actions to be used
      mlp_sizes: list of integers with sizes of hidden MLP layers
      lstm_sizes: list of integers with sizes of LSTM layers
    """
    super(MLPandLSTM, self).__init__(name='MLPandLSTM')
    self._parametric_action_distribution = parametric_action_distribution

    # MLP
    mlp_layers = [tf.keras.layers.Dense(size, 'relu') for size in mlp_sizes]
    self._mlp = tf.keras.Sequential(mlp_layers)
    # stacked LSTM
    lstm_cells = [tf.keras.layers.LSTMCell(size) for size in lstm_sizes]
    self._core = tf.keras.layers.StackedRNNCells(lstm_cells)
    # Layers for _head.
    self._policy_logits = tf.keras.layers.Dense(
        parametric_action_distribution.param_size, name='policy_logits')
    self._baseline = tf.keras.layers.Dense(1, name='baseline')

  @tf.function
  def initial_state(self, batch_size):
    return self._core.get_initial_state(batch_size=batch_size, dtype=tf.float32)

  def _head(self, core_output):
    policy_logits = self._policy_logits(core_output)
    baseline = tf.squeeze(self._baseline(core_output), axis=-1)

    # Sample an action from the policy.
    action = self._parametric_action_distribution.sample(policy_logits)

    return AgentOutput(action, policy_logits, baseline)

  # Not clear why, but if "@tf.function" declarator is placed directly onto
  # __call__, training fails with "uninitialized variable *baseline".
  # when running on multiple learning tpu cores.


  @tf.function
  def get_action(self, *args, **kwargs):
    return self.__call__(*args, **kwargs)

  def __call__(self, prev_actions, env_outputs, core_state, unroll=False,
               is_training=False, postprocess_action=True):
    """Runs the agent.

    Args:
      prev_actions: Previous action (after postprocessing). Not used by this
        agent.
      env_outputs: Structure with reward, done and observation fields. Only
        observation field is used by this agent. It should have the shape
        [time, batch_size, observation_size].
      core_state: Agent state.
      unroll: Should be True if inputs contain the time dimension and False
        otherwise.
      is_training: Whether we are in the loss computation. Not used by this
        agent.
      postprocess_action: Whether the action should be postprocessed (e.g. by
        applying tanh).
    Returns:
      A structure with action, policy_logits and baseline.
    """
    if not unroll:
      # Add time dimension.
      prev_actions, env_outputs = tf.nest.map_structure(
          lambda t: tf.expand_dims(t, 0), (prev_actions, env_outputs))

    outputs, core_state = self._unroll(prev_actions, env_outputs, core_state)

    if not unroll:
      # Remove time dimension.
      outputs = tf.nest.map_structure(lambda t: tf.squeeze(t, 0), outputs)

    if postprocess_action:
      outputs = outputs._replace(
          action=self._parametric_action_distribution.postprocess(
              outputs.action))

    return outputs, core_state

  def _unroll(self, unused_prev_actions, env_outputs, core_state):
    unused_reward, done, observation, _, _ = env_outputs
    observation = self._mlp(observation)

    initial_core_state = self._core.get_initial_state(
        batch_size=tf.shape(observation)[1], dtype=tf.float32)
    core_output_list = []
    for input_, d in zip(tf.unstack(observation), tf.unstack(done)):
      # If the episode ended, the core state should be reset before the next.
      core_state = tf.nest.map_structure(
          lambda x, y, d=d: tf.where(  
              tf.reshape(d, [d.shape[0]] + [1] * (x.shape.rank - 1)), x, y),
          initial_core_state,
          core_state)
      core_output, core_state = self._core(input_, core_state)
      core_output_list.append(core_output)
    outputs = tf.stack(core_output_list)

    return utils.batch_apply(self._head, (outputs,)), core_state
