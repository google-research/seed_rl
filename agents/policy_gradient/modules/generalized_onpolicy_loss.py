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


"""Implements a generalized onpolicy loss."""

import abc
import inspect
import gin
from seed_rl.agents.policy_gradient.modules import logging_module
import tensorflow as tf


@gin.configurable
class GeneralizedOnPolicyLoss(tf.Module, logging_module.LoggingModule):
  """TensorFlow module implementing the generalized onpolicy loss."""

  def __init__(self, agent, reward_normalizer, parametric_action_distribution,
               advantage_estimator, policy_loss, discount_factor,
               regularizer=None, max_abs_reward=None,
               handle_abandoned_episodes_properly=True,
               huber_delta=None, value_ppo_style_clip_eps=None,
               baseline_cost=1., include_regularization_in_returns=False,
               frame_skip=1, reward_scaling=1.0):
    """Creates a GeneralizedOnPolicyLoss."""
    self._agent = agent
    self._reward_normalizer = reward_normalizer
    self._parametric_action_distribution = parametric_action_distribution
    self._advantage_estimator = advantage_estimator
    self._policy_loss = policy_loss
    self._regularizer = regularizer
    self._max_abs_reward = max_abs_reward
    self._reward_scaling = reward_scaling
    self._baseline_cost = baseline_cost
    # Provided here so that it is shared.
    self._discount_factor = discount_factor
    self._frame_skip = frame_skip
    self._handle_abandoned_episodes_properly = handle_abandoned_episodes_properly
    self._value_ppo_style_clip_eps = value_ppo_style_clip_eps
    self._include_regularization_in_returns = include_regularization_in_returns
    if huber_delta is not None:
      self.v_loss_fn = tf.keras.losses.Huber(
          delta=huber_delta, reduction=tf.keras.losses.Reduction.NONE)
    else:
      self.v_loss_fn = tf.keras.losses.MeanSquaredError(
          reduction=tf.keras.losses.Reduction.NONE)

  def init(self):
    for module in self.submodules:
      if hasattr(module, 'init'):
        if not inspect.signature(module.init).parameters:
          module.init()

  def compute_advantages(self, agent_state, prev_actions, env_outputs,
                         agent_outputs, return_learner_outputs=False):
    # Extract rewards and done information.
    rewards, done, _, abandoned, _ = tf.nest.map_structure(lambda t: t[1:],
                                                           env_outputs)
    if self._max_abs_reward is not None:
      rewards = tf.clip_by_value(rewards, -self._max_abs_reward,  
                                 self._max_abs_reward)
    rewards *= self._reward_scaling

    # Compute the outputs of the neural networks on the learner.
    learner_outputs, _ = self._agent((prev_actions, env_outputs),
                                     agent_state,
                                     unroll=True,
                                     is_training=True)

    # At this point, we have unroll length + 1 steps. The last step is only used
    # as bootstrap value, so it's removed.
    agent_outputs = tf.nest.map_structure(lambda t: t[:-1], agent_outputs)
    learner_v = learner_outputs.baseline  # current value function
    learner_outputs = tf.nest.map_structure(lambda t: t[:-1], learner_outputs)

    target_action_log_probs = self._parametric_action_distribution(
        learner_outputs.policy_logits).log_prob(agent_outputs.action)
    behaviour_action_log_probs = self._parametric_action_distribution(
        agent_outputs.policy_logits).log_prob(agent_outputs.action)

    # Compute the advantages.

    if self._reward_normalizer:
      corrected_predictions = self._reward_normalizer.correct_prediction(
          learner_v)
      unnormalized_predictions = self._reward_normalizer.unnormalize_prediction(
          corrected_predictions)
    else:
      corrected_predictions = learner_v
      unnormalized_predictions = learner_v
    if not self._handle_abandoned_episodes_properly:
      abandoned = tf.zeros_like(abandoned)
    done_terminated = tf.logical_and(done, ~abandoned)
    done_abandoned = tf.logical_and(done, abandoned)

    if self._include_regularization_in_returns and self._regularizer:
      additional_rewards, _ = self._regularizer(
          self._parametric_action_distribution,
          learner_outputs.policy_logits,
          agent_outputs.policy_logits,
          agent_outputs.action, with_logging=False)
      assert rewards.shape == additional_rewards.shape
      rewards += additional_rewards

    # tf.math.pow does not work on TPU so we compute it manually.
    adjusted_discount_factor = 1.
    for _ in range(self._frame_skip):
      adjusted_discount_factor *= self._discount_factor

    vs, advantages = self._advantage_estimator(
        unnormalized_predictions,
        rewards, done_terminated,
        done_abandoned,
        adjusted_discount_factor,
        target_action_log_probs,
        behaviour_action_log_probs)

    if self._reward_normalizer:
      normalized_targets = self._reward_normalizer.normalize_target(vs)
      normalized_advantages = self._reward_normalizer.normalize_advantage(
          advantages)
      self._reward_normalizer.update_normalization_statistics(vs)
    else:
      normalized_targets = vs
      normalized_advantages = advantages

    outputs = (normalized_targets, normalized_advantages)
    if return_learner_outputs:
      outputs += (learner_outputs,)
    return outputs

  def __call__(self, agent_state, prev_actions, env_outputs, agent_outputs,
               normalized_targets=None, normalized_advantages=None):
    """Computes the loss."""
    if normalized_targets is None:
      normalized_targets, normalized_advantages, learner_outputs = \
          self.compute_advantages(  
              agent_state, prev_actions, env_outputs, agent_outputs,
              return_learner_outputs=True)
      # The last timestep is only used for computing advantages so we
      # remove it here.
      agent_state, prev_actions, env_outputs, agent_outputs = \
          tf.nest.map_structure(
              lambda t: t[:-1],
              (agent_state, prev_actions, env_outputs, agent_outputs))
    else:  # Advantages are already precomputed.
      learner_outputs, _ = self._agent((prev_actions, env_outputs),
                                       agent_state,
                                       unroll=True,
                                       is_training=True)

    target_action_log_probs = self._parametric_action_distribution(
        learner_outputs.policy_logits).log_prob(agent_outputs.action)
    behaviour_action_log_probs = self._parametric_action_distribution(
        agent_outputs.policy_logits).log_prob(agent_outputs.action)

    # Compute the advantages.
    if self._reward_normalizer:
      corrected_predictions = self._reward_normalizer.correct_prediction(
          learner_outputs.baseline)
      old_corrected_predictions = self._reward_normalizer.correct_prediction(
          agent_outputs.baseline)
    else:
      corrected_predictions = learner_outputs.baseline
      old_corrected_predictions = agent_outputs.baseline

    # Compute the advantage-based loss.
    policy_loss = tf.reduce_mean(
        self._policy_loss(
            normalized_advantages,
            target_action_log_probs,
            behaviour_action_log_probs,
            actions=agent_outputs.action,
            target_logits=learner_outputs.policy_logits,
            behaviour_logits=agent_outputs.policy_logits,
            parametric_action_distribution=self._parametric_action_distribution)
    )

    # Value function loss
    v_error = normalized_targets - corrected_predictions
    self.log('GeneralizedOnPolicyLoss/V_error', v_error)
    self.log('GeneralizedOnPolicyLoss/abs_V_error', tf.abs(v_error))
    self.log('GeneralizedOnPolicyLoss/corrected_predictions',
             corrected_predictions)
    # Huber loss reduces the last dimension so we add a dummy one here.
    normalized_targets = normalized_targets[..., tf.newaxis]
    corrected_predictions = corrected_predictions[..., tf.newaxis]
    v_loss = self.v_loss_fn(normalized_targets, corrected_predictions)

    # PPO-style value loss clipping
    if self._value_ppo_style_clip_eps is not None:
      old_corrected_predictions = old_corrected_predictions[..., tf.newaxis]
      clipped_corrected_predictions = tf.clip_by_value(
          corrected_predictions,
          old_corrected_predictions - self._value_ppo_style_clip_eps,
          old_corrected_predictions + self._value_ppo_style_clip_eps)
      clipped_v_loss = self.v_loss_fn(normalized_targets,
                                      clipped_corrected_predictions)
      v_loss = tf.maximum(v_loss, clipped_v_loss)
    v_loss = tf.reduce_mean(v_loss)

    # Compute the regularization loss.
    if self._regularizer:
      per_step_regularization, regularization_loss = self._regularizer(
          self._parametric_action_distribution,
          learner_outputs.policy_logits,
          agent_outputs.policy_logits,
          agent_outputs.action)
      if not self._include_regularization_in_returns:
        regularization_loss += tf.reduce_mean(per_step_regularization)
    else:
      regularization_loss = 0.

    total_loss = policy_loss + self._baseline_cost*v_loss + regularization_loss
    return total_loss


class PolicyLoss(tf.Module, metaclass=abc.ABCMeta):
  """Abstract base class for policy losses."""

  @abc.abstractmethod
  def __call__(self, advantages, target_action_log_probs,
               behaviour_action_log_probs):
    r"""Computes policy loss.

    Args:
      advantages: A float32 tensor of shape [T, B] of advantages.
      target_action_log_probs: A float32 tensor of shape [T, B] with
        log-probabilities of taking the action by the current policy
      behaviour_action_log_probs: A float32 tensor of shape [T, B] with
        log-probabilities of taking the action by the behavioural policy


    Returns:
      A float32 tensor of shape [T, B] with the policy loss.
    """
    raise NotImplementedError('`__call__()` is not implemented!')


class RegularizationLoss(tf.Module, metaclass=abc.ABCMeta):
  """Abstract base class for policy losses."""

  @abc.abstractmethod
  def __call__(self, parametric_action_distribution, target_action_logits,
               behaviour_action_logits, actions):
    r"""Computes regularization loss.

    Args:
      parametric_action_distribution: Parametric action distribution.
      target_action_logits: A float32 tensor of shape [T, B, A] with
        the logits of the target policy.
      behaviour_action_logits: A float32 tensor of shape [T, B, A] with
        the logits of the behavioural policy.
      actions: A float32 tensor of shape [T, B, A] with the actions taken by the
        behaviour policy.

    Returns:
      A float32 tensor of shape [T, B] with the regularization loss.
    """
    raise NotImplementedError('`__call__()` is not implemented!')
