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
from typing import Callable, Optional

import dataclasses
import gym
import tensorflow as tf
import tensorflow_probability as tfp
tfb = tfp.bijectors
tfd = tfp.distributions


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


@tf.custom_gradient
def safe_exp(x):
  e = tf.exp(tf.clip_by_value(x, -15, 15))

  def grad(dy):
    return dy * e

  return e, grad


class ClippedIdentity(tfb.identity.Identity):
  """Compute Y = clip_by_value(X, -1, 1).

  Note that we do not override `is_injective` despite this bijector not being
  injective, to not disable Identity's `forward_log_det_jacobian`. See also
  tensorflow_probability.bijectors.identity.Identity.
  """

  def __init__(self, validate_args=False, name='clipped_identity'):
    with tf.name_scope(name) as name:
      super(ClippedIdentity, self).__init__(
          validate_args=validate_args, name=name)

  @classmethod
  def _is_increasing(cls):
    return False

  def _forward(self, x):
    return tf.clip_by_value(x, -1., 1.)


def softplus_default_std_fn(scale):
  return tf.nn.softplus(scale) + 1e-3


@dataclasses.dataclass
class ContinuousDistributionConfig(object):
  """Configuration for continuous distributions.

  Currently, only NormalSquashedDistribution is supported. The default
  configuration corresponds to a normal distribution (with standard deviation
  computed from params using an unshifted softplus offset by 1e-3),
  followed by tanh.
  """
  # Transforms parameters into non-negative values for standard deviation of the
  # gaussian.
  gaussian_std_fn: Callable[[tf.Tensor], tf.Tensor] = softplus_default_std_fn
  # The squashing postprocessor, e.g. ClippedIdentity or
  # tensorflow_probability.bijectors.Tanh.
  postprocessor: tfb.bijector.Bijector = tfb.Tanh()


class NormalSquashedDistribution(ParametricDistribution):
  """Normal distribution followed by squashing (tanh by default).

  We apply tanh (or clipping) to gaussian actions to bound them. Normally we
  would use tfd.TransformedDistribution to automatically apply squashing to the
  distribution. We do not do it here because of tanh saturation which would
  make log_prob computations impossible. Instead, most of the code operates on
  pre-squashed actions and we take the postprocessor jacobian into account in
  log_prob computations.
  """

  def __init__(self,
               event_size,
               config: Optional[ContinuousDistributionConfig] = None):
    """Initialize the distribution.

    Args:
      event_size: the size of events (i.e. actions).
      config: Configuration, default configuration when None is passed.
    """
    if config is None:
      config = ContinuousDistributionConfig()
    super().__init__(
        param_size=2 * event_size,
        postprocessor=config.postprocessor,
        event_ndims=1,
        reparametrizable=True)
    self._std_fn = config.gaussian_std_fn

  def create_dist(self, parameters):
    loc, scale = tf.split(parameters, 2, axis=-1)
    scale = self._std_fn(scale)
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


def get_parametric_distribution_for_action_space(
    action_space,
    continuous_config: Optional[ContinuousDistributionConfig] = None):
  """Returns an action distribution parametrization based on the action space.

  Args:
    action_space: action space of the environment
    continuous_config: Configuration for the continuous action distribution
      (used when needed by the action space).
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
    return NormalSquashedDistribution(
        event_size=action_space.shape[0], config=continuous_config)
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


def safe_exp_std_fn(std_for_zero_param: float, min_std):
  std_shift = tf.math.log(std_for_zero_param - min_std)
  fn = lambda scale: safe_exp(scale + std_shift) + min_std
  assert abs(fn(0) - std_for_zero_param) < 1e-3
  return fn


def softplus_std_fn(std_for_zero_param: float, min_std: float):
  std_shift = tfp.math.softplus_inverse(std_for_zero_param - min_std)
  fn = lambda scale: tf.nn.softplus(scale + std_shift) + min_std
  assert abs(fn(0) - std_for_zero_param) < 1e-3
  return fn


def continuous_action_config(
    action_min_gaussian_std: float = 1e-3,
    action_gaussian_std_fn: str = 'softplus',
    action_std_for_zero_param: float = 1,
    action_postprocessor: str = 'Tanh') -> ContinuousDistributionConfig:
  """Configures continuous distributions from numerical and string inputs.

  Currently, only NormalSquashedDistribution is supported. The default
  configuration corresponds to a normal distribution with standard deviation
  computed from params using an unshifted softplus, followed by tanh.
  Args:
    action_min_gaussian_std: minimal standard deviation.
    action_gaussian_std_fn: transform for standard deviation parameters.
    action_std_for_zero_param: shifts the transform to get this std when
      parameters are zero.
    action_postprocessor: the non-linearity applied to the sample from the
      gaussian.

  Returns:
    A continuous distribution setup, with the parameters transform
    to get the standard deviation applied with a shift, as configured.
  """
  config = ContinuousDistributionConfig()

  # Note: see cl/319488607, which introduced the cast.
  config.min_gaussian_std = float(action_min_gaussian_std)
  if action_gaussian_std_fn == 'safe_exp':
    config.gaussian_std_fn = safe_exp_std_fn(action_std_for_zero_param,
                                             config.min_gaussian_std)
  elif action_gaussian_std_fn == 'softplus':
    config.gaussian_std_fn = softplus_std_fn(action_std_for_zero_param,
                                             config.min_gaussian_std)
  else:
    raise ValueError('Flag `action_gaussian_std_fn` only supports safe_exp and'
                     f' softplus, got: {action_gaussian_std_fn}')

  if action_postprocessor == 'ClippedIdentity':
    config.postprocessor = ClippedIdentity()
  elif action_postprocessor != 'Tanh':
    raise ValueError('Flag `action_postprocessor` only supports Tanh and'
                     f' ClippedIdentity, got: {action_postprocessor}')
  return config
