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

  def get_Q(self, prev_action, env_output, state, action):
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

  def get_V(self, prev_action, env_output, state):
    """Returns state values.

    Args: See get_Q above.
    Returns: [time, batch_size] tensor with state values.
    """
    return tf.squeeze(self._v_mlp(env_output.observation), axis=-1)

  def get_action_params(self, prev_action, env_output, state):
    """Returns action distribution parameters (i.e. actor network outputs).

    Args: See get_Q above.
    Returns: [time, batch_size, *] tensor with action distribution parameters.
    """
    return self._actor_mlp(env_output.observation)

  # Not clear why, but if "@tf.function" declarator is placed directly onto
  # __call__, training fails with "uninitialized variable *baseline".
  # when running on multiple learning tpu cores.


  @tf.function
  def get_action(self, *args, **kwargs):
    return self.__call__(*args, **kwargs)

  def __call__(self, prev_action, env_output, state, unroll=False,
               is_training=False, postprocess_action=True):
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
    action_params = self.get_action_params(prev_action, env_output, state)
    action = self._action_distribution.sample(action_params)
    if postprocess_action:
      action = self._action_distribution.postprocess(action)
    return action, ()


class LSTMwithFeedForwardBranch(tf.Module):
  """MLP+LSTM+MLP with a parallel feed-forward branch.

  Based on https://arxiv.org/pdf/1710.06537.pdf.
  """

  def __init__(self, lstm_sizes, pre_mlp_sizes, post_mlp_sizes, ff_mlp_sizes):
    """Initialize the network.

    Args:
      lstm_sizes: List of integers with sizes of LSTM layers.
      pre_mlp_sizes: Hidden sizes of MLP layers applied before LSTM.
      post_mlp_sizes: Hidden sizes of MLP layers applied after LSTM.
      ff_mlp_sizes: Hidden sizes of MLP layers applied in parallel to LSTM.
    """
    super(LSTMwithFeedForwardBranch, self).__init__(name='MLPandLSTM')
    self._pre_mlp = create_mlp(pre_mlp_sizes)
    self._post_mlp = create_mlp(post_mlp_sizes)
    self._ff_mlp = create_mlp(ff_mlp_sizes)
    self._core = tf.keras.layers.StackedRNNCells([tf.keras.layers.LSTMCell(size)
                                                  for size in lstm_sizes])

  def initial_state(self, batch_size):
    return self._core.get_initial_state(batch_size=batch_size, dtype=tf.float32)

  def __call__(self, ff_input, recurrent_input, core_state, done,
               only_return_new_state=False):
    """Run the model.

    Args:
      ff_input: [time, batch_size, *] tensor with inputs to the feed-forward
        branch.
      recurrent_input: [time, batch_size, *] tensor with inputs to the recurrent
        part of the network.
      core_state: Initial hidden state of the network.
      done: [time, batch_size] bool tensor denoting *before* which timesteps
        to reset the hidden state.
      only_return_new_state: if True, only the final hidden state is returned.
    Returns:
      a tuple with [time, batch_size, *] tensor with outputs and the final
      hidden state. If only_return_new_state is True, then only the output
      tensor is returned.
    """
    lstm_input = self._pre_mlp(recurrent_input)  # MLP before LSTM

    initial_core_state = self.initial_state(tf.shape(lstm_input)[1])
    core_output_list = []
    for input_, d in zip(tf.unstack(lstm_input), tf.unstack(done)):
      # If the episode ended, the core state should be reset before the next.
      core_state = tf.nest.map_structure(
          lambda x, y, d=d: tf.where(  
              tf.reshape(d, [d.shape[0]] + [1] * (x.shape.rank - 1)), x, y),
          initial_core_state,
          core_state)
      core_output, core_state = self._core(input_, core_state)
      core_output_list.append(core_output)
    lstm_output = tf.stack(core_output_list)

    if only_return_new_state:
      return core_state

    ff_output = self._ff_mlp(ff_input)  # FF branch
    post_mlp_input = tf.concat(values=[ff_output, lstm_output], axis=-1)
    output = self._post_mlp(post_mlp_input)  # final MLP

    return output, core_state


class ActorCriticLSTM(tf.Module):
  """Actor-Critic architecture based on LSTM and MLP.

  Based on https://arxiv.org/pdf/1710.06537.pdf.
  """

  def __init__(self, parametric_action_distribution, n_critics, lstm_sizes,
               pre_mlp_sizes, post_mlp_sizes, ff_mlp_sizes):
    """Create an Actor Critic network.

    Args:
      parametric_action_distribution: An object of ParametricDistribution class
        specifing a parametric distribution over actions to be used.
      n_critics: Number of Q-networks (usually 1 or 2).
      lstm_sizes: List of integers with sizes of LSTM layers.
      pre_mlp_sizes: Hidden sizes of MLP layers applied before LSTM.
      post_mlp_sizes: Hidden sizes of MLP layers applied after LSTM.
      ff_mlp_sizes: Hidden sizes of MLP layers applied in parallel to LSTM.
    """
    super(ActorCriticLSTM, self).__init__(name='MLPandLSTM')
    self._action_distribution = parametric_action_distribution

    def create_net(output_size):
      return LSTMwithFeedForwardBranch(
          lstm_sizes=lstm_sizes,
          pre_mlp_sizes=pre_mlp_sizes,
          ff_mlp_sizes=ff_mlp_sizes,
          post_mlp_sizes=post_mlp_sizes + [output_size])

    self._actor_net = create_net(self._action_distribution.param_size)
    self._v_net = create_net(1)
    self._q_nets = [create_net(1) for _ in range(n_critics)]
    self._networks = [self._actor_net, self._v_net] + self._q_nets

  @tf.function
  def initial_state(self, batch_size):
    return [net.initial_state(batch_size) for net in self._networks]

  def _prepare_action_input(self, action):
    return tf.cast(self._action_distribution.postprocess(action), tf.float32)

  def _run_net(self, net, prev_action, env_output, state, ff_input,
               only_return_new_state=False):
    action_input = self._prepare_action_input(prev_action)
    recurrent_input = tf.concat(values=[env_output.observation, action_input],
                                axis=-1)
    return net(ff_input=ff_input,
               recurrent_input=recurrent_input,
               core_state=state,
               done=env_output.done,
               only_return_new_state=only_return_new_state)

  def get_Q(self, prev_action, env_output, state, action):
    """Computes state-action values.

    Args:
      env_output: Structure with reward, done and observation fields. All fields
        shapes start with [time, batch_size]. The done field denotes *before*
        which timesteps to reset the hidden state.
      prev_action: [time, batch_size, action_size] tensor with previous actions
        taken in the environment (before postprocessing).
      state: Agent state at the first timestep.
      action: [time, batch_size, action_size] tensor with actions for which we
        compute Q-values (before postprocessing).
    Returns:
      [time, batch_size, n_critics] tensor with state-action values.
    """
    ff_input = tf.concat(values=[env_output.observation,
                                 self._prepare_action_input(action)], axis=-1)
    q_values = [self._run_net(net, prev_action, env_output, state=net_state,
                              ff_input=ff_input)[0]
                for (net, net_state) in zip(self._q_nets, state[2:])]
    return tf.concat(values=q_values, axis=-1)

  def get_V(self, prev_action, env_output, state):
    """Returns state values.

    Args: See get_Q above.
    Returns: [time, batch_size] tensor with state values.
    """
    v = self._run_net(self._v_net, prev_action, env_output, state=state[1],
                      ff_input=env_output.observation)[0]
    return tf.squeeze(v, axis=-1)

  def get_action_params(self, prev_action, env_output, state):
    """Returns action distribution parameters (i.e. actor network outputs).

    Args: See get_Q above.
    Returns: [time, batch_size, *] tensor with action distribution parameters.
    """
    return self._run_net(self._actor_net, prev_action, env_output,
                         state=state[0],
                         ff_input=env_output.observation)[0]

  # Not clear why, but if "@tf.function" declarator is placed directly onto
  # __call__, training fails with "uninitialized variable *baseline".
  # when running on multiple learning tpu cores.


  @tf.function
  def get_action(self, *args, **kwargs):
    return self.__call__(*args, **kwargs)

  def __call__(self, prev_action, env_output, state, unroll=False,
               is_training=False, postprocess_action=True):
    """Runs the agent.

    Args:
      env_output: See get_Q above.
      prev_action: See get_Q above.
      state: See get_Q above.
      unroll: Should be True if inputs contain the time dimension and False
        otherwise.
      is_training: If True, the actions are not going to be postprocessed.
    Returns:
      action taken and new agent state.
    """
    if not unroll:
      # Add time dimension.
      env_output = tf.nest.map_structure(lambda t: tf.expand_dims(t, 0),
                                         env_output)
      prev_action = tf.expand_dims(prev_action, 0)
    action, state = self._unroll(prev_action, env_output, state)
    if not unroll:
      # Remove time dimension.
      action = tf.nest.map_structure(lambda t: tf.squeeze(t, 0), action)

    if postprocess_action:
      action = self._action_distribution.postprocess(action)

    return action, state

  def _unroll(self, prev_action, env_output, state):
    action_params = self.get_action_params(prev_action, env_output, state)
    action = self._action_distribution.sample(action_params)

    new_states = [self._run_net(net, prev_action, env_output, net_state,
                                ff_input=None, only_return_new_state=True)
                  for (net, net_state) in zip(self._networks, state)]

    return action, new_states

