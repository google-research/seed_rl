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


"""Implements different input normalization strategies for RL."""

import gin
from seed_rl.agents.policy_gradient.modules import running_statistics
import tensorflow as tf


@gin.configurable
class InputNormalization(tf.Module):
  """TensorFlow module with that normalizes input with tracked mean and std.

  The input normalization works by normalizing value targets to zero mean
  and unit variance by rescaling with the tracked mean m and standard deviation
  s, i.e.,
      normalized_input = (input - m)/s

  Whenever m and s are updated to new values m' and s', compensation variables
  a and b are updated to a' and b' to guarantee that for all values x:
      (x-m)/s * a + b = (x-m')/s' * a' + b'

  This guarantees that the update of the statistics does not change the policy
  or the value function.

  The module should be used as follows:
  - Initially, `init_normalization_stats()` should be called to initialize the
    variables.
  - The input tensors during the training step should be used to update the
    module parameters via `update_normalization_statistics(...)`.
  - The method `normalize(...)` should be applied to the input tensors.
  - The method `correct(...)` should be applied to the normalized,
    (potentially clipped) input tensors and the result can be fed to the model.
  """

  def __init__(self, mean_std_tracker):
    """Creates a InputNormalization.

    Args:
      mean_std_tracker: Instance of running_statistics.MeanStd used for tracking
        the mean and the standard deviation.
    """
    if not isinstance(mean_std_tracker, running_statistics.MeanStd):
      raise ValueError(
          f'`mean_std_tracker` needs to be an instance of MeanStd, '
          f'got {type(mean_std_tracker)}')
    self.mean_std_tracker = mean_std_tracker
    self.compensation_mean = None
    self.compensation_std = None

  @property
  def initialized(self):
    """Boolean indicating if the module is initialized."""
    return self.compensation_mean is not None

  def init_normalization_stats(self, input_size):
    """Initializes normalization variables.

    This is done explicitly in a manual step since variables need to be created
    under the correct distribution strategy scope.

    Args:
      input_size: Integer with dimensionality of input.
    """
    self.mean_std_tracker.init(input_size)
    self.compensation_mean = tf.Variable(
        name='running_mean',
        shape=[input_size],
        trainable=True,
        dtype=tf.float32,
        initial_value=tf.zeros(shape=[input_size]),
        aggregation=tf.VariableAggregation.MEAN)
    self.compensation_std = tf.Variable(
        name='running_std',
        shape=[input_size],
        trainable=True,
        dtype=tf.float32,
        initial_value=tf.ones(shape=[input_size]),
        aggregation=tf.VariableAggregation.MEAN)

  def normalize(self, x):
    """Normalizes input values x using past target statistics.

    Args:
      x: <float32>[(...), size] tensor.

    Returns:
      <float32>[(...), size] normalized tensor.
    """
    return self.mean_std_tracker.normalize(x)

  def correct(self, x):
    """Corrects a normalized input x using compensation parameters.

    Args:
      x: <float32>[(...), size] tensor.

    Returns:
      <float32>[(...), size] corrected tensor.
    """
    return self.compensation_std * x + self.compensation_mean

  def update_normalization_statistics(self, data):
    """Updates normalization statistics.

    Args:
      data: <float32>[(...), size].
    """
    # Update the running statistics.
    mean1, std1 = self.mean_std_tracker.get_mean_std()
    self.mean_std_tracker.update(data)
    mean2, std2 = self.mean_std_tracker.get_mean_std()

    # Update the compensation parameters based on means and standard deviations
    # before and after the statistics update.
    new_compensation_std = std2 / std1 * self.compensation_std
    new_compensation_mean = (self.compensation_mean +
                             self.compensation_std / std1 * (mean2 - mean1))
    self.compensation_mean.assign(new_compensation_mean)
    self.compensation_std.assign(new_compensation_std)

  def get_mean_std(self):
    return self.mean_std_tracker.get_mean_std()
