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


"""Constraints with Lagrange multipliers."""

import abc
import gin
import tensorflow as tf


class Coefficient(tf.Module, metaclass=abc.ABCMeta):
  """Abstract base class for coefficients which can be fixed or adaptive."""

  def init(self):
    """Initializes TF variables (if any)."""
    pass

  @abc.abstractmethod
  def __call__(self):
    """Returns the value of the coefficient."""
    raise NotImplementedError('__call__() not implemented!')

  def adjustment_loss(self, reference_value):
    """Loss for coefficient adjustment.

    Args:
       reference_value: Adaptive coefficients may depend on this value, e.g.
          for a Lagrange multiplier for an inequality constraint this would
          be the variable we are trying to constrain.
    Returns:
       Loss which should be minimized to adjust the coefficient.
    """
    raise NotImplementedError('adjustment_loss() not implemented')

  def scale_loss(self, unscaled_loss):
    """Scales the given loss by the coefficient."""
    return tf.stop_gradient(self()) * unscaled_loss


@gin.configurable
class FixedCoefficient(Coefficient):
  """Fixed coefficient."""

  def __init__(self, value):
    super().__init__()
    self.value = value

  def __call__(self):
    return tf.convert_to_tensor(self.value, tf.float32)

  def adjustment_loss(self, reference_value):
    return tf.constant(0.)


@gin.configurable
class LagrangeInequalityCoefficient(Coefficient):
  """Lagrange coefficient enforcing a soft inequality constraint.

  Suppose that you want to minimize f(x) s.t. x <= threshold.
  This can be approximated by introducing a new variable c and optimizing
  f(x) + sg(c)*x + c*sg(threshold-x) where sg is the stop_gradient operator.
  This works since if x > threshold, c will increase, which in turn will make x
  decrease.

  Example:

  multiplier = constraints.LagrangeInequalityCoefficient(threshold=2)
  x = tf.Variable(1.23)
  f = lambda x: -tf.square(x)
  def loss():
    return (f(x) + multiplier.scale_loss(x) + multiplier.adjustment_loss(x))
  opt = tf.keras.optimizers.Adam(0.01)
  for _ in range(1000):
    opt.minimize(loss, multiplier.trainable_variables + (x,))
  self.assertAllClose(x, 2., atol=0.01)
  """

  def __init__(self, threshold,
               init_alpha=1.,
               alpha_range=(1e-6, 1e6),
               adjustment_speed=1.,
               init_variables=True):
    """Creates a constraint.

    Args:
      threshold: Maximum value that the constrained variable should not exceed.
      init_alpha: Initial value of the penalty coefficient (i.e. Lagrange
        multiplier).
      alpha_range: A pair of floats with min and max allowed values for alpha.
      adjustment_speed: A float controlling how fast alpha is adjusted.
      init_variables: Whether to create the variables in the constructor.
        If False, than you will have to call init() manually before calling
        any other methods of this object.
    """
    super().__init__()
    self.threshold = threshold
    self.init_alpha = init_alpha
    self.alpha_range = alpha_range
    self.adjustment_speed = adjustment_speed
    assert alpha_range[0] >= 0
    if init_variables:
      self.init()

  def init(self):
    if hasattr(self, 'param'):
      return
    mul = self.adjustment_speed
    def constraint(v):
      return tf.clip_by_value(
          v,
          *[tf.math.log(c) / mul for c in self.alpha_range])
    self.param = tf.Variable(
        tf.math.log(self.init_alpha) / mul,
        constraint=constraint,
        trainable=True,
        dtype=tf.float32)

  def __call__(self):
    return tf.exp(self.adjustment_speed * self.param)

  def adjustment_loss(self, reference_value):
    return self() * tf.stop_gradient(self.threshold -
                                     tf.reduce_mean(reference_value))
