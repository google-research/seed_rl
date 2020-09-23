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


"""Parametric distributions over action spaces."""

import abc
from absl import flags
import gym
import tensorflow as tf
import tensorflow_probability as tfp
tfb = tfp.bijectors
tfd = tfp.distributions

FLAGS = flags.FLAGS


class ParametricDistribution(abc.ABC):
  """Abstract class for parametric (action) distribution."""

  def __init__(self,
               param_size,
               postprocessor,
               event_ndims,
               reparametrizable,
               jacobian_event_ndims=0):
    """Abstract class for parametric (action) distribution.

    Specifies how to transform distribution parameters (i.e. actor output)
    into a distribution over actions.

    Args:
      param_size: size of the parameters for the distribution
      postprocessor: tfp.bijector which is applied after sampling
      (in practice, it's tanh or identity)
      event_ndims: rank of the distribution sample (i.e. action)
      reparametrizable: is the distribution reparametrizable
      jacobian_event_ndims: rank of the action for the jacobian computation.
    """
    self._param_size = param_size
    self._postprocessor = postprocessor  # tfp.bijector
    self._event_ndims = event_ndims  # rank of events
    self._jacobian_event_ndims = jacobian_event_ndims
    self._reparametrizable = reparametrizable
    assert event_ndims in [0, 1]

  @abc.abstractmethod
  def create_dist(self, parameters):
    """Creates tfp.distribution from parameters."""
    pass

  @property
  def param_size(self):
    return self._param_size

  @property
  def reparametrizable(self):
    return self._reparametrizable

  def postprocess(self, event):
    return self._postprocessor.forward(event)

  def inverse_postprocess(self, event):
    return self._postprocessor.inverse(event)

  def sample(self, parameters):
    return self.create_dist(parameters).sample()

  def log_prob(self, parameters, actions):
    """Compute the log probability of the actions.

    Args:
      parameters: Tensor of parameters for the probability function.
      actions: Tensor of actions before postprocessing.
    Returns:
      Tensor of log probabilities, or logs of the density function if the
        actions are continuous.
    """
    dist = self.create_dist(parameters)
    log_probs = dist.log_prob(actions)
    log_probs -= self._postprocessor.forward_log_det_jacobian(
        tf.cast(actions, tf.float32), event_ndims=self._jacobian_event_ndims)
    if self._event_ndims == 1:
      log_probs = tf.reduce_sum(log_probs, axis=-1)  # sum over action dimension
    return log_probs

  def entropy(self, parameters):
    """Return the entropy of the given distribution."""
    dist = self.create_dist(parameters)
    entropy = dist.entropy()
    entropy += self._postprocessor.forward_log_det_jacobian(
        tf.cast(dist.sample(), tf.float32),
        event_ndims=self._jacobian_event_ndims)
    if self._event_ndims == 1:
      entropy = tf.reduce_sum(entropy, axis=-1)
    return entropy

  def kl_divergence(self, parameters_a, parameters_b):
    """Return KL divergence between the two distributions."""
    dist_a = self.create_dist(parameters_a)
    dist_b = self.create_dist(parameters_b)
    kl = tfd.kl_divergence(dist_a, dist_b)
    if self._event_ndims == 1:
      kl = tf.reduce_sum(kl, axis=-1)
    return kl


class CategoricalDistribution(ParametricDistribution):
  """Categorical action distribution."""

  def __init__(self, n_actions, dtype):
    """Initialize the distribution.

    Args:
      n_actions: the number of actions available.
      dtype: dtype of actions, usually int32 or int64.
    """
    super().__init__(
        param_size=n_actions, postprocessor=tfb.Identity(), event_ndims=0,
        reparametrizable=False)
    self._dtype = dtype

  def create_dist(self, parameters):
    return tfd.Categorical(logits=parameters, dtype=self._dtype)


class MultiCategoricalDistribution(ParametricDistribution):
  """Multidimensional categorical distribution."""

  def __init__(self, n_dimensions, n_actions_per_dim, dtype):
    """Initialize multimodal categorical distribution.

    Args:
      n_dimensions: the dimensionality of actions.
      n_actions_per_dim: number of actions available per dimension.
      dtype: dtype of actions, usually int32 or int64.
    """
    super().__init__(
        param_size=n_dimensions * n_actions_per_dim,
        postprocessor=tfb.Identity(),
        event_ndims=1,
        reparametrizable=False)
    self._n_dimensions = n_dimensions
    self._n_actions_per_dim = n_actions_per_dim
    self._dtype = dtype

  def create_dist(self, parameters):
    batch_shape = parameters.shape[:-1]
    logits_shape = [self._n_dimensions, self._n_actions_per_dim]
    logits = tf.reshape(parameters, batch_shape + logits_shape)
    return tfd.Categorical(logits=logits, dtype=self._dtype)


class NormalTanhDistribution(ParametricDistribution):
  """Normal distribution followed by tanh."""

  def __init__(self, event_size, min_std=0.001):
    """Initialize the distribution.

    Args:
      event_size: the size of events (i.e. actions).
      min_std: minimum std for the gaussian.
    """
    # We apply tanh to gaussian actions to bound them.
    # Normally we would use tfd.TransformedDistribution to automatically
    # apply tanh to the distribution.
    # We can't do it here because of tanh saturation
    # which would make log_prob computations impossible. Instead, most
    # of the code operate on pre-tanh actions and we take the postprocessor
    # jacobian into account in log_prob computations.
    super().__init__(
        param_size=2 * event_size, postprocessor=tfb.Tanh(), event_ndims=1,
        reparametrizable=True)
    self._min_std = min_std

  def create_dist(self, parameters):
    loc, scale = tf.split(parameters, 2, axis=-1)
    scale = tf.math.softplus(scale) + self._min_std
    dist = tfd.Normal(loc=loc, scale=scale)
    return dist


class JointDistribution(ParametricDistribution):
  """Distribution which mixes NormalTanh and Multicategorical."""

  def __init__(self,
               continuous_event_size,
               n_discrete_dimensions,
               n_discrete_actions_per_dim,
               discrete_dtype,
               min_std=0.001):
    """Initialize the distribution.

    Args:
      continuous_event_size: the size of continuous events (i.e. actions).
      n_discrete_dimensions: the dimensionality of discrete actions.
      n_discrete_actions_per_dim: number of discrete actions available per
        dimension.
      discrete_dtype: dtype of discrete actions, usually int32 or int64.
      min_std: minimum std for the gaussian.
    """
    super().__init__(
        param_size=2 * continuous_event_size +
        n_discrete_dimensions * n_discrete_actions_per_dim,
        postprocessor=tfb.Blockwise(
            [tfb.Tanh(), tfb.Identity()],
            block_sizes=[continuous_event_size, n_discrete_dimensions]),
        event_ndims=0,
        reparametrizable=True,
        # We need to set the jacobian_event_ndims to 1 for the Blockwise
        # post-processor to split the last dimension.
        jacobian_event_ndims=1)
    self._continuous_event_size = continuous_event_size
    self._n_discrete_dimensions = n_discrete_dimensions
    self._n_discrete_actions_per_dim = n_discrete_actions_per_dim
    self._discrete_dtype = discrete_dtype
    self._min_std = min_std

  def create_dist(self, parameters):
    continuous_params, discrete_params = tf.split(
        parameters, [
            2 * self._continuous_event_size,
            self._n_discrete_dimensions * self._n_discrete_actions_per_dim
        ],
        axis=-1)

    loc, scale = tf.split(continuous_params, 2, axis=-1)
    scale = tf.math.softplus(scale) + self._min_std
    continuous_dist = tfd.Normal(loc=loc, scale=scale)
    continuous_dist = tfd.Independent(continuous_dist, 1)

    batch_shape = discrete_params.shape[:-1]
    discrete_logits_shape = [
        self._n_discrete_dimensions, self._n_discrete_actions_per_dim
    ]
    discrete_logits = tf.reshape(discrete_params,
                                 batch_shape + discrete_logits_shape)
    discrete_dist = tfd.Categorical(
        logits=discrete_logits, dtype=self._discrete_dtype)
    discrete_dist = tfd.Independent(discrete_dist, 1)

    return tfd.Blockwise([continuous_dist, discrete_dist],
                         dtype_override=tf.float32)


def check_multi_discrete_space(space):
  if min(space.nvec) != max(space.nvec):
    raise ValueError('space nvec must be constant: {}'.format(space.nvec))


def check_box_space(space):
  assert len(space.shape) == 1, space.shape
  if any(l != -1 for l in space.low):
    raise ValueError(
        f'Learner only supports actions bounded to [-1,1]: {space.low}')
  if any(h != 1 for h in space.high):
    raise ValueError(
        f'Learner only supports actions bounded to [-1,1]: {space.high}')


def get_parametric_distribution_for_action_space(action_space):
  """Returns an action distribution parametrization based on the action space.

  Args:
    action_space: action space of the environment
  """
  if isinstance(action_space, gym.spaces.Discrete):
    return CategoricalDistribution(action_space.n,
                                   dtype=action_space.dtype)
  elif isinstance(action_space, gym.spaces.MultiDiscrete):
    check_multi_discrete_space(action_space)
    return MultiCategoricalDistribution(
        n_dimensions=len(action_space.nvec),
        n_actions_per_dim=action_space.nvec[0],
        dtype=action_space.dtype)
  elif isinstance(action_space, gym.spaces.Box):  # continuous actions
    check_box_space(action_space)
    return NormalTanhDistribution(
        event_size=action_space.shape[0])
  elif isinstance(action_space, gym.spaces.Tuple):  # mixed actions
    assert len(action_space) == 2, action_space

    continuous_space, discrete_space = action_space
    assert isinstance(continuous_space, gym.spaces.Box), continuous_space
    check_box_space(continuous_space)
    assert isinstance(discrete_space, gym.spaces.MultiDiscrete), discrete_space
    check_multi_discrete_space(discrete_space)

    return JointDistribution(
        continuous_event_size=continuous_space.shape[0],
        n_discrete_dimensions=len(discrete_space.nvec),
        n_discrete_actions_per_dim=discrete_space.nvec[0],
        discrete_dtype=discrete_space.dtype)
  else:
    raise ValueError(f'Unsupported action space {action_space}')
