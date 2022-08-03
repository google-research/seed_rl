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

import gin
from seed_rl.agents.policy_gradient.modules import constraints
from seed_rl.agents.policy_gradient.modules import logging_module
import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions


@gin.configurable
class KLPolicyRegularizer(tf.Module, logging_module.LoggingModule):
  """Policy regularizer covering KL(mu||pi), K(pi||mu) and entropy."""

  
  def __init__(self, **coefficients):
    """Creates the regularizer.

    Args:
       kl_pi_mu: Coefficient for KL(current_policy||behaviour_policy) loss. It
         can be a float or a Coefficient object. Defaults to 0.
       kl_mu_pi: Coefficient for KL(behaviour_policy||current_policy) loss.
       entropy: Coefficient for entropy bonus. If a constraint is used, than
         the constraint is -entropy < threshold so the higher the threshold,
         the lower the entropy.
       kl_ref_pi: Coefficient for the KL between a reference distribution and
         the current one. The reference distribution is defined as one which is
         parametrized by zeros, i.e. uniform for categorical distributions
         and tanh(N(0, softplus(0))) for NormalSquashedDistribution.
    """
    self.coefficients = coefficients
    for key, coe in coefficients.items():
      assert key in ['kl_pi_mu', 'kl_mu_pi', 'entropy', 'kl_ref_pi']
      if not isinstance(coe, constraints.Coefficient):
        self.coefficients[key] = constraints.FixedCoefficient(coe)

  def __call__(self, parametric_action_distribution, pi_logits, mu_logits,
               actions, with_logging=True):
    assert pi_logits.shape == mu_logits.shape
    dist = parametric_action_distribution

    losses = {
        'kl_pi_mu':
            dist(pi_logits).kl_divergence(dist(mu_logits)),
        'kl_mu_pi':
            dist(mu_logits).kl_divergence(dist(pi_logits)),
        'kl_ref_pi':
            dist(tf.zeros_like(pi_logits)).kl_divergence(dist(pi_logits)),
        'entropy':
            -dist(pi_logits).entropy()
    }

    if with_logging:
      for key, val in losses.items():
        self.log('KLPolicyRegularizer/%s' % key,
                 val * (-1 if key == 'entropy' else 1))

    per_step_loss = tf.zeros(pi_logits.shape[:-1], tf.float32)
    global_loss = tf.constant(0., tf.float32)
    for key, coe in self.coefficients.items():
      loss = losses[key]
      assert loss.shape == per_step_loss.shape
      if with_logging:
        self.log('KLPolicyRegularizer/%s/coefficient' % key, coe())
      per_step_loss += coe.scale_loss(loss)
      global_loss += coe.adjustment_loss(tf.reduce_mean(loss))
    if with_logging:
      self.log('KLPolicyRegularizer/per_step_loss',
               tf.reduce_mean(per_step_loss))
      self.log('KLPolicyRegularizer/global_loss', global_loss)
    assert per_step_loss.shape == pi_logits.shape[:-1]
    assert not global_loss.shape.as_list()
    return per_step_loss, global_loss
