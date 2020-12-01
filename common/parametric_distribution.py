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
from typing import Callable

import dataclasses
import gym
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability.python.distributions import kullback_leibler

tfb = tfp.bijectors
tfd = tfp.distributions


class ParametricDistribution(abc.ABC):
  """Abstract class for parametric (action) distribution."""

  def __init__(self,
               param_size,
               create_dist):
    """Abstract class for parametric (action) distribution.

    Specifies how to transform distribution parameters (i.e. actor output)
    into a distribution over actions.

    Args:
      param_size: Size of the parameters for the distribution
      create_dist: Function from parameters to tf Distribution.
    """
    self._param_size = param_size
    self._create_dist = create_dist

  @property
  def create_dist(self):
    return self._create_dist

  def __call__(self, params):
    return self.create_dist(params)

  @property
  def param_size(self):
    return self._param_size

  @property
  def reparametrizable(self):
    return self._create_dist(tf.zeros(
        (self._param_size,
        ))).reparameterization_type == tfd.FULLY_REPARAMETERIZED

  def sample(self, parameters):
    return self._create_dist(parameters).sample()

  def log_prob(self, parameters, actions):
    return self._create_dist(parameters).log_prob(actions)

  def entropy(self, parameters):
    """Return the entropy of the given distribution."""
    return self._create_dist(parameters).entropy()

  def kl_divergence(self, parameters_a, parameters_b):
    """Return KL divergence between the two distributions."""
    dist_a = self._create_dist(parameters_a)
    dist_b = self._create_dist(parameters_b)
    return tfd.kl_divergence(dist_a, dist_b)


def categorical_distribution(n_actions, dtype):
  """Initialize the categorical distribution.

  Args:
    n_actions: the number of actions available.
    dtype: dtype of actions, usually int32 or int64.

  Returns:
    A tuple (param size, fn(params) -> distribution)
  """

  def create_dist(parameters):
    return tfd.Categorical(logits=parameters, dtype=dtype)

  return ParametricDistribution(n_actions, create_dist)


def multi_categorical_distribution(n_dimensions, n_actions_per_dim, dtype):
  """Initialize the categorical distribution.

  Args:
    n_dimensions: the dimensionality of actions.
    n_actions_per_dim: number of actions available per dimension.
    dtype: dtype of actions, usually int32 or int64.

  Returns:
    A tuple (param size, fn(params) -> distribution)
  """

  def create_dist(parameters):
    batch_shape = parameters.shape[:-1]
    logits_shape = [n_dimensions, n_actions_per_dim]
    logits = tf.reshape(parameters, batch_shape + logits_shape)
    return tfd.Independent(
        tfd.Categorical(logits=logits, dtype=dtype),
        reinterpreted_batch_ndims=1)

  return ParametricDistribution(n_dimensions * n_actions_per_dim, create_dist)


# NB: This distribution has no gradient w.r.t the action close to boundaries.
class TanhTransformedDistribution(tfd.TransformedDistribution):
  """Distribution followed by tanh."""

  def __init__(self, distribution, threshold=.999, validate_args=False):
    """Initialize the distribution.

    Args:
      distribution: The distribution to transform.
      threshold: Clipping value of the action when computing the logprob.
      validate_args: Passed to super class.
    """
    super().__init__(
        distribution=distribution,
        bijector=tfp.bijectors.Tanh(),
        validate_args=validate_args)
    # Computes the log of the average probability distribution outside the
    # clipping range, i.e. on the interval [-inf, -atanh(threshold)] for
    # log_prob_left and [atanh(threshold), inf] for log_prob_right.
    self._threshold = threshold
    inverse_threshold = self.bijector.inverse(threshold)
    # Let epsilon = 1 - threshold
    # average(pdf) on [threshold, 1] = probability([threshold, 1])/epsilon
    # So log(average(pdf)) = log(probability) - log(epsilon)
    log_epsilon = tf.math.log(1. - threshold)
    # Those 2 values are differentiable w.r.t. model parameters, such that the
    # gradient is defined everywhere.
    # There won't be any gradient w.r.t the action though.
    self._log_prob_left = self.distribution.log_cdf(
        -inverse_threshold) - log_epsilon
    self._log_prob_right = self.distribution.log_survival_function(
        inverse_threshold) - log_epsilon

  def log_prob(self, event):
    # Without this clip there would be NaNs in the inner tf.where and that
    # causes issues for some reasons.
    event = tf.clip_by_value(event, -self._threshold, self._threshold)
    # The inverse image of {threshold} is the interval [atanh(threshold), inf]
    # which has a probability of "log_prob_right" under the given distribution.
    return tf.where(
        event <= -self._threshold, self._log_prob_left,
        tf.where(event >= self._threshold, self._log_prob_right,
                 super().log_prob(event)))

  def mode(self):
    return self.bijector.forward(self.distribution.mode())

  def mean(self):
    return self.bijector.forward(self.distribution.mean())

  def entropy(self, seed=None):
    # We return an estimation using a single sample of the log_det_jacobian.
    # We can still do some backpropagation with this estimate.
    return self.distribution.entropy() + self.bijector.forward_log_det_jacobian(
        self.distribution.sample(seed=seed), event_ndims=0)


@kullback_leibler.RegisterKL(TanhTransformedDistribution,
                             TanhTransformedDistribution)
def _kl_transformed(a, b, name='kl_transformed'):
  return kullback_leibler.kl_divergence(
      a.distribution, b.distribution, name=name)


def softplus_default_std_fn(scale):
  return tf.nn.softplus(scale) + 1e-3


def normal_tanh_distribution(num_actions,
                             gaussian_std_fn=softplus_default_std_fn):
  """Normal distribution postprocessed by a tanh."""

  def create_dist(parameters):
    loc, scale = tf.split(parameters, 2, axis=-1)
    scale = gaussian_std_fn(scale)
    normal_dist = tfd.Normal(loc=loc, scale=scale)
    return tfd.Independent(
        TanhTransformedDistribution(normal_dist), reinterpreted_batch_ndims=1)

  return ParametricDistribution(2 * num_actions, create_dist)


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


def normal_clipped_distribution(num_actions,
                                gaussian_std_fn=softplus_default_std_fn):
  """Normal distribution postprocessed by a clipped identity."""

  def create_dist(parameters):
    loc, scale = tf.split(parameters, 2, axis=-1)
    scale = gaussian_std_fn(scale)
    normal_dist = tfd.Normal(loc=loc, scale=scale)
    return tfd.Independent(
        ClippedIdentity(normal_dist), reinterpreted_batch_ndims=1)

  return ParametricDistribution(2 * num_actions, create_dist)


def deterministic_tanh_distribution(num_actions):

  def create_dist(parameters):
    return tfd.Independent(
        TanhTransformedDistribution(tfd.Deterministic(loc=parameters)),
        reinterpreted_batch_ndims=1)

  return ParametricDistribution(num_actions, create_dist)


def joint_distribution(parametric_distributions,
                       dtype_override=tf.float32):
  """Initialize the distribution.

  Args:
    parametric_distributions: A list of ParametricDistributions.
    dtype_override: The type to output the actions in.

  Returns:
    A tuple (param size, fn(params) -> distribution)
  """
  param_sizes = [
      dist.param_size for dist in parametric_distributions
  ]

  def create_dist(parameters):
    split_params = tf.split(parameters, param_sizes, axis=-1)
    dists = [
        dist(param)
        for (dist, param) in zip(parametric_distributions, split_params)
    ]
    return tfd.Blockwise(dists, dtype_override=dtype_override)

  return ParametricDistribution(sum(param_sizes), create_dist)


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


def get_parametric_distribution_for_action_space(action_space,
                                                 continuous_config=None):
  """Returns an action distribution parametrization based on the action space.

  Args:
    action_space: action space of the environment
    continuous_config: Configuration for the continuous action distribution
      (used when needed by the action space)..
  """
  if isinstance(action_space, gym.spaces.Discrete):
    return categorical_distribution(action_space.n, dtype=action_space.dtype)
  elif isinstance(action_space, gym.spaces.MultiDiscrete):
    check_multi_discrete_space(action_space)
    return multi_categorical_distribution(
        n_dimensions=len(action_space.nvec),
        n_actions_per_dim=action_space.nvec[0],
        dtype=action_space.dtype)
  elif isinstance(action_space, gym.spaces.Box):  # continuous actions
    check_box_space(action_space)
    if continuous_config is None:
      continuous_config = ContinuousDistributionConfig()
    if continuous_config.postprocessor == 'Tanh':
      return normal_tanh_distribution(
          num_actions=action_space.shape[0],
          gaussian_std_fn=continuous_config.gaussian_std_fn)
    elif continuous_config.postprocessor == 'ClippedIdentity':
      return normal_clipped_distribution(
          num_actions=action_space.shape[0],
          gaussian_std_fn=continuous_config.gaussian_std_fn)
    else:
      raise ValueError(
          f'Postprocessor {continuous_config.postprocessor} not supported.')
  elif isinstance(action_space, gym.spaces.Tuple):  # mixed actions
    return joint_distribution([
        get_parametric_distribution_for_action_space(subspace,
                                                     continuous_config)
        for subspace in action_space
    ])
  else:
    raise ValueError(f'Unsupported action space {action_space}')


@tf.custom_gradient
def safe_exp(x):
  e = tf.exp(tf.clip_by_value(x, -15, 15))

  def grad(dy):
    return dy * e

  return e, grad


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
  # The squashing postprocessor.
  # Accepted values are Tanh and ClippedIdentity.
  postprocessor: str = 'Tanh'


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

  config.postprocessor = action_postprocessor
  return config
