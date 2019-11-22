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

import tensorflow as tf


def create_mlp(hidden_sizes, hidden_activation='relu',
               last_layer_activation=None):
  model = tf.keras.Sequential()
  for i, size in enumerate(hidden_sizes):
    if i == len(hidden_sizes) - 1:
      activation = last_layer_activation
    else:
      activation = hidden_activation
    model.add(tf.keras.layers.Dense(size, activation))
  return model



class ActorCriticMLP(tf.Module):
  """MLP agent."""

  def __init__(self, parametric_action_distribution, n_critics, mlp_sizes):
    """Initializes the agent.

    Args:
      parametric_action_distribution: An object of ParametricDistribution class
        specifing a parametric distribution over actions to be used.
      n_critics: Number of critic networks to use, usually 1 or 2.
      mlp_sizes: List of integers with sizes of hidden MLP layers.
    """
    super(ActorCriticMLP, self).__init__()
    self._action_distribution = parametric_action_distribution

    self._actor_mlp = create_mlp(mlp_sizes +
                                 [self._action_distribution.param_size])
    self._q_mlp = [create_mlp(mlp_sizes + [1]) for _ in range(n_critics)]
    self._v_mlp = create_mlp(mlp_sizes + [1])

  @tf.function
  def initial_state(self, batch_size):
    return ()

  @tf.function
  def get_Q(self, env_output, prev_action, state, action):
    """Computes state-action values.

    Args:
      env_output: Structure with reward, done and observation fields. Only
        observation field is used by this agent. It should have the shape
        [time, batch_size, observation_size].
      prev_action: [time, batch_size, action_size] tensor with previous actions
        taken in the environment (before postprocessing). Not used by this
        agent.
      state: Agent state at the first timestep. Not used by this agent.
      action: [time, batch_size, action_size] tensor with actions for which we
        compute Q-values (before postprocessing).
    Returns:
      [time, batch_size, n_critics] tensor with state-action values.
    """
    input_ = tf.concat(
        values=[
            env_output.observation,
            tf.cast(self._action_distribution.postprocess(action), tf.float32)
        ],
        axis=-1)
    return tf.concat(values=[critic(input_) for critic in self._q_mlp],
                     axis=-1)

  @tf.function
  def get_V(self, env_output, prev_action, state):
    """Returns state values.

    Args: See get_Q above.
    Returns: [time, batch_size] tensor with state values.
    """
    return tf.squeeze(self._v_mlp(env_output.observation), axis=-1)

  @tf.function
  def get_action_params(self, env_output, prev_action, state):
    """Returns action distribution parameters (i.e. actor network outputs).

    Args: See get_Q above.
    Returns: [time, batch_size, *] tensor with action distribution parameters.
    """
    return self._actor_mlp(env_output.observation)

  @tf.function
  def __call__(self, env_output, prev_action, state, unroll=False,
               is_training=False):
    """Runs the agent.

    Args:
      env_output: See get_Q above.
      prev_action: Not used. See get_Q above.
      state: Not used. See get_Q above.
      unroll: Should be True if inputs contain the time dimension and False
        otherwise.
      is_training: If True, the actions are not going to be postprocessed.
    Returns:
      action taken and new agent state.
    """
    action_params = self.get_action_params(env_output, prev_action, state)
    action = self._action_distribution.sample(action_params)
    if not is_training:
      action = self._action_distribution.postprocess(action)
    return action, ()
