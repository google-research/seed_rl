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


"""Tests for policy_regularizers."""

from seed_rl.agents.policy_gradient.modules import constraints
from seed_rl.agents.policy_gradient.modules import policy_regularizers
from seed_rl.agents.policy_gradient.modules import test_utils
from seed_rl.common import parametric_distribution
import tensorflow as tf


class PolicyRegularizer(test_utils.TestCase):

  def test_continuous_action(self):
    my_log_dict = {}
    regularizer = policy_regularizers.KLPolicyRegularizer(
        kl_pi_mu=1.2,
        kl_mu_pi=constraints.LagrangeInequalityCoefficient(threshold=1e-3),
        entropy=0.3,
        kl_ref_pi=0.2)
    regularizer.set_logging_dict(my_log_dict)
    distribution = parametric_distribution.normal_tanh_distribution(7)
    pi_logits = tf.random.normal((10, 20, 14))
    mu_logits = tf.random.normal((10, 20, 14))
    actions = tf.random.normal((10, 20, 7))
    regularizer(distribution, pi_logits, mu_logits, actions)
    regularizer.unset_logging_dict()
    self.assertIn('KLPolicyRegularizer/kl_ref_pi', my_log_dict)

  def test_discrete_actions(self):
    my_log_dict = {}
    regularizer = policy_regularizers.KLPolicyRegularizer(
        kl_pi_mu=1.2,
        kl_mu_pi=1.0,
        entropy=constraints.LagrangeInequalityCoefficient(threshold=1e-3),
        kl_ref_pi=1.3)
    regularizer.set_logging_dict(my_log_dict)
    distribution = parametric_distribution.multi_categorical_distribution(
        7, 11, tf.int32)
    pi_logits = tf.random.normal((10, 20, 7 * 11))
    mu_logits = tf.random.normal((10, 20, 7 * 11))
    actions = tf.zeros((10, 20, 7), tf.int32)
    regularizer(distribution, pi_logits, mu_logits, actions)
    regularizer.unset_logging_dict()
    self.assertNotIn('KLPolicyRegularizer/kl_mu_pi_mean', my_log_dict)


if __name__ == '__main__':
  tf.enable_v2_behavior()
  tf.test.main()
