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
    parameters_shape = [3*2 + 3*4]

    parameters = tf.zeros(batch_shape + parameters_shape)
    self.assertEqual(joint_distribution.entropy(parameters).shape, batch_shape)

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

    actions = joint_distribution.inverse_postprocess(actions)
    log_probs = joint_distribution.log_prob(
        parameters, actions)

    create_normaltanh_dist = parametric_distribution.get_parametric_distribution_for_action_space(
        self.create_box_space())
    continuous_parameters = parameters[:6]
    continuous_actions = create_normaltanh_dist.inverse_postprocess(
        continuous_actions)
    continuous_log_probs = create_normaltanh_dist.log_prob(
        continuous_parameters, continuous_actions)

    create_multidiscrete_dist = parametric_distribution.get_parametric_distribution_for_action_space(
        self.create_multidiscrete_space())
    discrete_parameters = tf.convert_to_tensor(parameters[6:])
    discrete_actions = create_multidiscrete_dist.inverse_postprocess(
        discrete_actions)
    discrete_log_probs = create_multidiscrete_dist.log_prob(
        discrete_parameters, discrete_actions)

    self.assertAllClose(log_probs, continuous_log_probs + discrete_log_probs)

  def test_normal_clipped_distribution_logprob(self):
    config = parametric_distribution.ContinuousDistributionConfig(
        postprocessor=parametric_distribution.ClippedIdentity())
    distribution = (
        parametric_distribution.get_parametric_distribution_for_action_space(
            self.create_box_space(), continuous_config=config))
    locs = [0., 0., 0.]
    scale = [.1, .2, .3]
    parameters = np.array(locs + scale, np.float32)
    actions = np.array([[0, 0, 0], [0, 0, .99], [0, .99, 0], [.99, 0, 0]],
                       np.float32)

    actions = distribution.inverse_postprocess(actions)
    log_probs = distribution.log_prob(parameters, actions)

    min_std = 1e-3
    # The default is softplus, see argument continuous_safe_exp_std_fn.
    scale = tf.math.softplus(scale) + min_std
    continuous_dist = tfd.Normal(loc=locs, scale=scale)
    continuous_dist = tfd.Independent(continuous_dist, 1)

    self.assertAllClose(log_probs, continuous_dist.log_prob(actions))


if __name__ == '__main__':
  tf.test.main()
