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


"""Different policy losses."""

import gin
from seed_rl.agents.policy_gradient.modules import constraints
from seed_rl.agents.policy_gradient.modules import logging_module
import tensorflow as tf


@gin.configurable
class AdvantagePreprocessor(tf.Module):
  """Advantages preprocessor."""

  def __init__(self, normalize=False, only_positive=False, only_top_half=False,
               offset=None):
    """Creates the advantage preprocessor.

    Args:
      normalize: Whether to normalize adventages to have mean 0 and std 1 in
        each batch.
      only_positive: Whether to only take positive advantages.
      only_top_half: Whether to only take top half of advantages in each batch.
      offset: A value added to advantages (after normalization).
    """
    self.normalize = normalize
    self.only_positive = only_positive
    self.only_top_half = only_top_half
    self.offset = offset

  def __call__(self, advantages):
    """Processes the advantages.

    Args:
       advantages: A tensor with advantages.

    Returns:
       Processed advantages and a tf.float32 tensor of the same shape
       with 0s and 1s indicating which advantages should be used.
    """
    mask = tf.ones_like(advantages)
    if self.normalize:
      advantages -= tf.reduce_mean(advantages)
      advantages /= tf.math.reduce_std(advantages) + 1e-8
    if self.only_top_half:
      flat = tf.reshape(advantages, [-1])
      median = tf.math.reduce_min(tf.math.top_k(flat, k=flat.shape[0] // 2,
                                                sorted=False)[0])
      mask *= tf.cast(advantages >= median, tf.float32)
    if self.only_positive:
      mask *= tf.cast(advantages > 0., tf.float32)
    if self.offset is not None:
      advantages += self.offset
    return mask * advantages, mask


@gin.configurable
class GeneralizedAdvantagePolicyLoss(tf.Module, logging_module.LoggingModule):
  """Generalized advantage-based policy loss.

  Covers typical cases like PG, PPO, V-trace, AWR and V-MPO.
  """

  def __init__(self,
               advantage_preprocessor=None,
               use_importance_weights=False,
               max_importance_weight=None,
               ppo_epsilon=None,
               max_advantage=None,
               advantage_transformation=None,
               temperature=None):
    """Creates the loss.

    Args:
       advantage_preprocessor: An object (of AdvantagePreprocessor class) used
         to process the advantages.
       use_importance_weights: Whether to use importance sampling weights.
       max_importance_weight: Bigger importance weights are clipped.
       ppo_epsilon: If not None, than PPO-style pessimistic clipping is used.
       max_advantage: Bigger advantages are clipped. Clipping happens
         between scaling and applying the transformation.
       advantage_transformation: A function applied to advantages.
       temperature: If not None, than MPO/AWR-style advantages exponentiation
         is performed. This argument should be a Coefficient object which
         provides the temperature for the exponentiation.
    """
    self.advantage_preprocessor = (advantage_preprocessor or
                                   AdvantagePreprocessor())
    self.use_importance_weights = use_importance_weights
    self.max_importance_weight = max_importance_weight
    self.max_advantage = max_advantage
    self.advantage_transformation = advantage_transformation
    self.ppo_epsilon = ppo_epsilon
    self.temperature = temperature

  def __call__(self, advantages, target_action_log_probs,
               behaviour_action_log_probs, actions, target_logits,
               behaviour_logits, parametric_action_distribution=None):
    self.log('GeneralizedAdvantagePolicyLoss/advantages', advantages)
    self.log('GeneralizedAdvantagePolicyLoss/abs_advantages',
             tf.abs(advantages))
    self.log('GeneralizedAdvantagePolicyLoss/log_pi', target_action_log_probs)
    self.log('GeneralizedAdvantagePolicyLoss/log_mu',
             behaviour_action_log_probs)
    advantages, mask = self.advantage_preprocessor(advantages)

    # advantage transformation (e.g. AWR/V-MPO)
    if self.advantage_transformation is not None:
      assert self.temperature is not None
      self.log('GeneralizedAdvantagePolicyLoss/temperature', self.temperature())
      advantages = advantages / tf.stop_gradient(self.temperature())
      if self.max_advantage is not None:
        advantages = tf.minimum(advantages, self.max_advantage)
        self.log('GeneralizedAdvantagePolicyLoss/p_clipped_advantage',
                 tf.cast(advantages == self.max_advantage, tf.float32))
      advantages_before_transformation = advantages
      advantages = mask * self.advantage_transformation(advantages)
      self.log('GeneralizedAdvantagePolicyLoss/transformed_advantages',
               advantages)
    else:
      if self.max_advantage is not None:
        advantages = tf.minimum(advantages, self.max_advantage)
      advantages *= mask

    self.log('GeneralizedAdvantagePolicyLoss/processed_advantages', advantages)
    max_adv = tf.reduce_max(mask * advantages + (1. - mask) * -1e9)
    min_adv = tf.reduce_min(mask * advantages + (1. - mask) * 1e9)
    self.log('GeneralizedAdvantagePolicyLoss/processed_advantages_min', min_adv)
    self.log('GeneralizedAdvantagePolicyLoss/processed_advantages_max', max_adv)
    self.log('GeneralizedAdvantagePolicyLoss/processed_advantages_range',
             max_adv - min_adv)

    # PG loss
    loss = -target_action_log_probs * tf.stop_gradient(advantages)

    # importance sampling weights
    log_rho = target_action_log_probs - behaviour_action_log_probs
    log_rho = tf.stop_gradient(log_rho)
    if self.ppo_epsilon is not None:
      # This is written differently that the standard PPO loss but should give
      # the same gradient.
      clip_pos_mask = ((advantages > 0) &
                       (log_rho > tf.math.log(1 + self.ppo_epsilon)))
      clip_neg_mask = ((advantages < 0) &
                       (log_rho < -tf.math.log(1 + self.ppo_epsilon)))
      loss_mask = tf.cast(~(clip_pos_mask | clip_neg_mask), tf.float32)
      loss *= loss_mask
      log_rho *= loss_mask  # to avoid overflow in exp
    if self.max_importance_weight is not None:
      log_rho = tf.minimum(log_rho, tf.math.log(self.max_importance_weight))
      self.log('GeneralizedAdvantagePolicyLoss/p_clipped_iw',
               tf.cast(log_rho == tf.math.log(self.max_importance_weight),
                       tf.float32))
    self.log('GeneralizedAdvantagePolicyLoss/log_rho', log_rho)
    if self.use_importance_weights:
      loss *= tf.exp(log_rho)

    loss = tf.reduce_mean(loss)

    if self.advantage_transformation is not None:  # temperature adjustment
      # This is KL between nonparametric target distribution and behavioral one.
      # Eq. (4) in V-MPO paper.
      advantages = advantages_before_transformation
      advantages *= mask
      advantages -= (1. - mask) * 1e3  # will be 0 after exp
      kl = tf.math.reduce_logsumexp(advantages) - tf.math.log(
          tf.reduce_sum(mask) + 1e-3)
      self.log('GeneralizedAdvantagePolicyLoss/mpo_kl', kl)
      loss += self.temperature.adjustment_loss(kl)

    return loss


@gin.configurable
def pg():
  return GeneralizedAdvantagePolicyLoss()


@gin.configurable
def vtrace(max_importance_weight=1.):
  return GeneralizedAdvantagePolicyLoss(
      use_importance_weights=True,
      max_importance_weight=max_importance_weight)


@gin.configurable
def ppo(epsilon, normalize_advantages=False, advantage_offset=None):
  return GeneralizedAdvantagePolicyLoss(
      use_importance_weights=True,
      ppo_epsilon=epsilon,
      advantage_preprocessor=AdvantagePreprocessor(
          normalize=normalize_advantages,
          offset=advantage_offset))


@gin.configurable
def awr(beta, w_max):
  return GeneralizedAdvantagePolicyLoss(
      advantage_transformation=tf.exp,
      temperature=constraints.FixedCoefficient(beta),
      max_advantage=tf.math.log(w_max))


def bc_logp():
  return GeneralizedAdvantagePolicyLoss(
      advantage_transformation=lambda x: tf.constant(  
          1, dtype=x.dtype, shape=x.shape),
      temperature=constraints.FixedCoefficient(1))


@gin.configurable
def softmax_all_dims(t):
  # Softmax with reduction across all axes.
  flat = tf.reshape(t, [-1])
  return tf.reshape(tf.nn.softmax(flat), t.shape)


@gin.configurable
def vmpo(e_n):
  # Backward KL regularizer needs to be added separately to get full V-MPO.
  return GeneralizedAdvantagePolicyLoss(
      advantage_transformation=softmax_all_dims,
      advantage_preprocessor=AdvantagePreprocessor(only_top_half=True),
      temperature=constraints.LagrangeInequalityCoefficient(
          threshold=e_n,
          adjustment_speed=10,
          init_variables=False))


@gin.configurable
def repeat_positive_advantages():
  # This is supervised learning on actions with positive advantages.
  # Both, AWR and V-MPO have this behaviour in the limit (beta->0 or e_n->0).
  return awr(beta=1e-6, w_max=1.)
