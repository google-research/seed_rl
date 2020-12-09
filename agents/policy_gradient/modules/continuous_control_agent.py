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


"""SEED agent using Keras for continuous control tasks."""

import collections
import gin
from seed_rl.agents.policy_gradient.modules import input_normalization
from seed_rl.agents.policy_gradient.modules import running_statistics
from seed_rl.common import utils
import tensorflow as tf

AgentOutput = collections.namedtuple('AgentOutput',
                                     'action policy_logits baseline')




@gin.configurable
class ContinuousControlAgent(tf.Module):
  """Agent for continuous control tasks."""

  def __init__(self,
               parametric_action_distribution,
               observation_normalizer=None,
               num_layers_policy=3,
               num_layers_value=3,
               num_layers_rnn=0,
               num_units_policy=256,
               num_units_value=256,
               num_units_rnn=256,
               layer_normalizer=None,
               shared=False,
               residual_connections=False,
               activation=None,
               kernel_init='glorot_uniform',
               last_kernel_init_value=None,
               last_kernel_init_value_scaling=None,
               last_kernel_init_policy=None,
               last_kernel_init_policy_scaling=None,
               correct_observations=False,
               std_independent_of_input=False,
               input_clipping=None):
    """Creates the ContinuousControlAgent.

    Args:
      parametric_action_distribution: SEED distribution used for the actions.
      observation_normalizer: InputNormalization instance used to normalize
        observations or None for no normalization.
      num_layers_policy: Integer with the number of hidden layers in the policy
        MLP. Needs to be the same as `num_layers_value` if shared=True.
      num_layers_value: Integer with the number of hidden layers in the value
        MLP. If None, the number of layers is the same as in the policy.
        Needs to be the same as `num_layers_policy` if shared=True.
      num_layers_rnn: Number of RNN layers.
      num_units_policy: Integer with the number of hidden units in the policy
        MLP. Needs to be the same as `num_units_value` if shared=True.
      num_units_value: Integer with the number of hidden units in the value
        MLP. If None, the number of units is the same as in the policy.
        Needs to be the same as `num_units_policy` if shared=True.
      num_units_rnn: Integer with the number of hidden units in the RNN.
      layer_normalizer: Function that returns a tf.keras.Layer instance used to
        normalize observations or None for no layer normalization.
      shared: Boolean indicating whether the MLPs (except the heads) should be
        shared for the value and the policy networks.
      residual_connections: Boolean indicating whether residual connections
        should be added to all the layers except the first and last ones in the
        MLPs.
      activation: Activation function to be passed to the dense layers in the
        MLPs or None (in which case the swish activation function is used).
      kernel_init: tf.keras.initializers.Initializer instance used to initialize
        the dense layers of the MLPs.
      last_kernel_init_value: tf.keras.initializers.Initializer instance used to
        initialize the last dense layers of the value MLP or None (in which case
        `kernel_init` is used).
      last_kernel_init_value_scaling: None or a float that is used to rescale
        the initial weights of the value network.
      last_kernel_init_policy: tf.keras.initializers.Initializer instance used
        to initialize the last dense layers of the policy MLP or None (in which
        case `kernel_init` is used).
      last_kernel_init_policy_scaling: None or a float that is used to rescale
        the initial weights of the policy network.
      correct_observations: Boolean indicating if changes in the
        `observation_normalizer` due to updates should be compensated in
        trainable compensation variables.
      std_independent_of_input: If a Gaussian action distribution is used,
        this parameter makes the standard deviation trainable but independent
        of the policy input.
      input_clipping: None or float that is used to clip input values to range
        [-input_clipping, input_clipping] after (potential) input normalization.
    """
    super(ContinuousControlAgent, self).__init__(name='continuous_control')

    # Default values.
    if observation_normalizer is None:
      # No input normalization.
      observation_normalizer = input_normalization.InputNormalization(
          running_statistics.FixedMeanStd())

    if activation is None:
      activation = swish

    if last_kernel_init_value is None:
      last_kernel_init_value = kernel_init
    last_kernel_init_value = _rescale_initializer(
        last_kernel_init_value, last_kernel_init_value_scaling)

    if last_kernel_init_policy is None:
      last_kernel_init_policy = kernel_init
    last_kernel_init_policy = _rescale_initializer(
        last_kernel_init_policy, last_kernel_init_policy_scaling)

    if layer_normalizer is None:
      layer_normalizer = lambda: (lambda x: x)

    # Parameters and layers for unroll.
    self._parametric_action_distribution = parametric_action_distribution
    self.observation_normalizer = observation_normalizer
    self._correct_observations = correct_observations

    # Build the required submodules.
    self._shared = tf.keras.Sequential()
    self._policy = tf.keras.Sequential()
    self._value = tf.keras.Sequential()

    # Build the torso(s).
    num_layers_value = num_layers_value or num_layers_policy
    num_units_value = num_units_value or num_units_policy
    if shared:
      if num_layers_value != num_layers_policy:
        raise ValueError('If shared=True, num_layers_value needs to be equal to'
                         ' num_layers_policy')
      if num_units_value != num_units_policy:
        raise ValueError('If shared=True, num_units_value needs to be equal to'
                         ' num_units_policy')
      _add_layers(self._shared, num_layers_value, num_units_value, kernel_init,
                  activation, layer_normalizer, residual_connections)
    else:
      _add_layers(self._policy,
                  num_layers_policy, num_units_policy, kernel_init, activation,
                  layer_normalizer, residual_connections)
      _add_layers(self._value, num_layers_value, num_units_value, kernel_init,
                  activation, layer_normalizer, residual_connections)

    # Build the recurrent layers (if needed).
    if num_layers_rnn:
      lstm_sizes = [num_units_rnn] * num_layers_rnn
      lstm_cells = [tf.keras.layers.LSTMCell(size) for size in lstm_sizes]
      self._rnn = tf.keras.layers.StackedRNNCells(lstm_cells)
    else:
      self._rnn = None

    # Build the policy head.
    normalizer_policy = layer_normalizer()
    policy_output_size = self._parametric_action_distribution.param_size
    if std_independent_of_input:
      policy_output_size //= 2
    self._policy.add(
        _Layer(policy_output_size,
               last_kernel_init_policy, lambda x: x, normalizer_policy, False))
    if std_independent_of_input:
      self._policy.add(_ConcatTrainableTensor(tf.zeros(policy_output_size,
                                                       tf.float32)))

    # Build the value head.
    normalizer_value = normalizer_policy if shared else layer_normalizer()
    self._value.add(
        _Layer(1, last_kernel_init_value, lambda x: x, normalizer_value, False))

    self._input_clipping = input_clipping

  @tf.function
  def initial_state(self, batch_size):
    if self._rnn is None:
      return ()
    return self._rnn.get_initial_state(batch_size=batch_size, dtype=tf.float32)

  # Not clear why, but if "@tf.function" declarator is placed directly onto
  # __call__, training fails with "uninitialized variable *baseline".
  # when running on multiple learning tpu cores.
  @tf.function
  def get_action(self, input_, core_state):
    return self.__call__(input_, core_state)

  def update_observation_normalization_statistics(self, observations):
    """Updates the observation normalization statistics.

    Args:
      observations: a batch of observations with shape [time, batch_size,
      obs_size].
    """
    self.observation_normalizer.update_normalization_statistics(observations)

  def __call__(self, input_, core_state, unroll=False, is_training=False):
    """Applies the network.

    Args:
      input_: A pair (prev_actions: <int32>[batch_size], env_outputs: EnvOutput
        structure where each tensor has a [batch_size] front dimension). When
        unroll is True, an unroll (sequence of transitions) is expected, and
        those tensors are expected to have [time, batch_size] front dimensions.
      core_state: Opaque (batched) recurrent state structure corresponding to
        the beginning of the input sequence of transitions.
      unroll: Whether the input is an unroll (sequence of transitions) or just a
        single (batched) transition.
      is_training: Enables normalization statistics updates (when unroll is
        True).

    Returns:
      A pair:
        - agent_output: AgentOutput structure. Tensors have front dimensions
          [batch_size] or [time, batch_size] depending on the value of 'unroll'.
        - core_state: Opaque (batched) recurrent state structure.
    """
    _, env_outputs = input_

    # We first handle initializing and (potentially) updating normalization
    # statistics.  We only update during the gradient update steps.
    # `is_training` is slightly misleading as it is also True during inference
    # steps in the training phase. We hence also require unroll=True which
    # indicates gradient updates.
    training_model_update = is_training and unroll
    data = env_outputs[2]
    if not self.observation_normalizer.initialized:
      if training_model_update:
        raise ValueError('It seems unlikely that stats should be updated in the'
                         ' same call where the stats are initialized.')
      self.observation_normalizer.init_normalization_stats(data.shape[-1])

    if self._rnn is not None:

      if unroll:
        representations = utils.batch_apply(self._flat_apply_pre_lstm,
                                            (env_outputs,))
        representations, core_state = self._apply_rnn(
            representations, core_state, env_outputs.done)
        outputs = utils.batch_apply(self._flat_apply_post_lstm,
                                    (representations,))
      else:
        representations = self._flat_apply_pre_lstm(env_outputs)
        representations, done = tf.nest.map_structure(
            lambda t: tf.expand_dims(t, 0),
            (representations, env_outputs.done))
        representations, core_state = self._apply_rnn(
            representations, core_state, done)
        representations = tf.nest.map_structure(
            lambda t: tf.squeeze(t, 0), representations)
        outputs = self._flat_apply_post_lstm(representations)
    else:
      # Simplify.
      if unroll:
        outputs = utils.batch_apply(self._flat_apply_no_lstm, (env_outputs,))
      else:
        outputs = self._flat_apply_no_lstm(env_outputs)

    return outputs, core_state

  def _apply_rnn(self, representations, core_state, done):
    """Apply the recurrent part of the network.

    Args:
      representations: The representations coming out of the non-recurrent
        part of the network, tensor of size [num_timesteps, batch_size, depth].
      core_state: The recurrent state, given as nested structure of
        sub-states. Each sub-states is of size [batch_size, substate_depth].
      done: Tensor of size [num_timesteps, batch_size] which indicates
        the end of a trajectory.

    Returns:
      A pair holding the representations coming out of the RNN (tensor of size
      [num_timesteps, batch_size, depth]) and the updated RNN state (same size
      as the input core_state.
    """
    batch_size = tf.shape(representations)[1]
    initial_core_state = self._rnn.get_initial_state(
        batch_size=batch_size, dtype=tf.float32)
    core_output_list = []
    for input_, d in zip(tf.unstack(representations), tf.unstack(done)):
      # If the episode ended, the core state should be reset before the next.
      core_state = tf.nest.map_structure(
          lambda x, y, d=d: tf.where(  
              tf.reshape(d, [d.shape[0]] + [1] * (x.shape.rank - 1)), x, y),
          initial_core_state,
          core_state)
      core_output, core_state = self._rnn(input_, core_state)
      core_output_list.append(core_output)
    outputs = tf.stack(core_output_list)
    return outputs, core_state

  def _flat_apply_pre_lstm(self, env_outputs):
    _, _, observations, _, _ = env_outputs

    # Input normalization.
    observations = self.observation_normalizer.normalize(observations)
    if self._input_clipping is not None:
      observations = tf.clip_by_value(
          observations,
          -self._input_clipping,  
          self._input_clipping)
    if self._correct_observations:
      observations = self.observation_normalizer.correct(observations)

    # The actual MLPs with the different heads.
    representations = self._shared(observations)
    return representations

  def _flat_apply_no_lstm(self, env_outputs):
    """Applies the modules."""
    representations = self._flat_apply_pre_lstm(env_outputs)
    return self._flat_apply_post_lstm(representations)

  def _flat_apply_post_lstm(self, representations):
    values = self._value(representations)
    logits = self._policy(representations)

    baselines = tf.squeeze(values, axis=-1)

    new_action = self._parametric_action_distribution(logits).sample(seed=None)

    return AgentOutput(new_action, logits, baselines)


@gin.configurable
def swish(input_activation):
  """Swish activation function."""
  return tf.multiply(input_activation, tf.nn.sigmoid(input_activation))


def _add_layers(sequential, num_layers, num_units, kernel_init, activation,
                normalizer, residual_connections):
  """Adds several layers to a tf.keras.Sequential instance."""
  for i in range(num_layers):
    sequential.add(
        _Layer(num_units, kernel_init, activation, normalizer(),
               False if i == 0 else residual_connections))


class _Layer(tf.keras.layers.Layer):
  """Custom layer for our MLPs."""

  def __init__(self, num_units, kernel_init, activation, normalizer,
               residual_connection):
    """Creates a _Layer."""
    super(_Layer, self).__init__()
    self.dense = tf.keras.layers.Dense(
        num_units, kernel_initializer=kernel_init, activation=activation)
    self.normalizer = normalizer
    self.residual_connection = residual_connection

  def call(self, tensor):
    new_tensor = self.dense(self.normalizer(tensor))
    return tensor + new_tensor if self.residual_connection else new_tensor


class _ConcatTrainableTensor(tf.keras.layers.Layer):
  """Layer which concatenates a trainable tensor to its input."""

  def __init__(self, init_value):
    """Creates a layer."""
    super(_ConcatTrainableTensor, self).__init__()
    assert init_value.ndim == 1
    self.init_value = init_value

  def build(self, shape):
    self.var = tf.Variable(self.init_value, trainable=True)

  def call(self, tensor):
    return tf.concat(values=[
        tensor,
        tf.broadcast_to(self.var, tensor.shape[:-1] + self.var.shape)
    ], axis=-1)


def _rescale_initializer(initializer, rescale):
  if rescale is None:
    return initializer
  if isinstance(initializer, str):
    initializer = tf.keras.initializers.get(initializer)
  def rescaled_initializer(*args, **kwargs):
    return rescale*initializer(*args, **kwargs)
  return rescaled_initializer
