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

# lint as python3
"""Tests for seed_rl.common.parametric_distribution."""

from gym import spaces
import numpy as np
from seed_rl.common import parametric_distribution
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions


class ParametricDistributionTest(tf.test.TestCase):

  def create_box_space(self):
    return spaces.Box(np.array([-1, -1, -1]), np.array([1, 1, 1]))

  def create_multidiscrete_space(self):
    return spaces.MultiDiscrete([4, 4, 4])

  def create_tuple_space(self):
    return spaces.Tuple(
        (self.create_box_space(), self.create_multidiscrete_space()))

  def test_joint_distribution_shape(self):
    joint_distribution = parametric_distribution.get_parametric_distribution_for_action_space(
        self.create_tuple_space())

    batch_shape = [3, 2]
    parameters_shape = [3 * 2 + 3 * 4]

    parameters = tf.zeros(batch_shape + parameters_shape)
    self.assertFalse(joint_distribution.reparametrizable)
    self.assertEqual(
        joint_distribution(parameters).entropy().shape, batch_shape)

  def test_joint_distribution_logprob(self):
    joint_distribution = parametric_distribution.get_parametric_distribution_for_action_space(
        self.create_tuple_space())
    parameters = np.array([0., 0., 0.,   # Normal locs
                           .1, .2, .3,   # Normal scales
                           1, 0, 0, 0,   # Discrete action 1
                           0, 1, 0, 0,   # Discrete action 2
                           0, 0, 1, 0],  # Discrete action 3
                          np.float32)
    actions = np.array([[0, 0, 0, 0, 1, 2],
                        [0, 0, .99, 0, 1, 2],
                        [0, .99, 0, 0, 1, 2],
                        [.99, 0, 0, 0, 1, 2],
                        [0, 0, 0, 0, 1, 3],
                        [0, 0, 0, 0, 2, 2],
                        [0, 0, 0, 0, 2, 3],
                        [0, 0, 0, 1, 2, 3]], np.float32)
    continuous_actions = actions[:, :3]
    discrete_actions = actions[:, 3:]

    log_probs = joint_distribution(parameters).log_prob(actions)

    normaltanh_dist = parametric_distribution.get_parametric_distribution_for_action_space(
        self.create_box_space())
    continuous_parameters = parameters[:6]
    continuous_log_probs = normaltanh_dist(continuous_parameters).log_prob(
        continuous_actions)

    multidiscrete_dist = parametric_distribution.get_parametric_distribution_for_action_space(
        self.create_multidiscrete_space())
    discrete_parameters = tf.convert_to_tensor(parameters[6:])
    discrete_log_probs = multidiscrete_dist(discrete_parameters).log_prob(
        discrete_actions)

    self.assertAllClose(log_probs, continuous_log_probs + discrete_log_probs)

  def test_clipped_distribution(self):
    clipped_distribution = parametric_distribution.get_parametric_distribution_for_action_space(
        self.create_box_space(),
        continuous_config=parametric_distribution.continuous_action_config(
            action_postprocessor='ClippedIdentity'))

    clipped_distribution(np.zeros((6,), np.float32)).sample()


if __name__ == '__main__':
  tf.test.main()
