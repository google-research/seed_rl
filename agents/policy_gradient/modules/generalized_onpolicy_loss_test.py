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


"""Tests for generalized_onpolicy_loss."""

import collections
from seed_rl.agents.policy_gradient.modules import advantages
from seed_rl.agents.policy_gradient.modules import generalized_onpolicy_loss
from seed_rl.agents.policy_gradient.modules import popart
from seed_rl.agents.policy_gradient.modules import running_statistics
from seed_rl.agents.policy_gradient.modules import test_utils
from seed_rl.common import parametric_distribution
import tensorflow as tf

_AgentOutput = collections.namedtuple('AgentOutput',
                                      'action policy_logits baseline')



_NUM_ACTIONS = 10
_NUM_BATCH = 15
_NUM_UNROLLS = 5
_NUM_CORE_STATE = 20


class _DummyAgent(tf.Module):

  def __call__(self, input_, core_state, unroll=False, is_training=False):
    # prev_actions, env_outputs = input_
    action = tf.ones((_NUM_UNROLLS, _NUM_BATCH, _NUM_ACTIONS),
                     dtype=tf.float32)
    policy_logits = tf.ones((_NUM_UNROLLS, _NUM_BATCH, 2 * _NUM_ACTIONS),
                            dtype=tf.float32)
    baseline = tf.ones((_NUM_UNROLLS, _NUM_BATCH), dtype=tf.float32)
    return _AgentOutput(action, policy_logits, baseline), core_state


class _DummyPolicyLoss(generalized_onpolicy_loss.PolicyLoss):

  def __call__(self, advantages_, target_action_log_probs,
               behaviour_action_log_probs, **kwargs):
    return tf.reduce_mean(advantages_ * tf.exp(target_action_log_probs))


class _DummyRegularizationLoss(generalized_onpolicy_loss.RegularizationLoss):

  def __call__(self, parametric_action_distribution, target_action_logits,
               behaviour_action_logits, actions):
    return parametric_action_distribution(target_action_logits).entropy(), 0.



class GeneralizedOnPolicyLossTest(test_utils.TestCase):

  def setUp(self):
    super().setUp()
    reward_normalizer = popart.PopArt(running_statistics.AverageMeanStd())
    reward_normalizer.init()
    self._loss = generalized_onpolicy_loss.GeneralizedOnPolicyLoss(
        _DummyAgent(), reward_normalizer,
        parametric_distribution.normal_tanh_distribution(
            _NUM_ACTIONS).create_dist, advantages.GAE(lambda_=0.95),
        _DummyPolicyLoss(), 0.97, _DummyRegularizationLoss(), 0.2, 0.5)

  def test_one(self):
    agent_state = tf.zeros((_NUM_UNROLLS, _NUM_BATCH, _NUM_CORE_STATE),
                           tf.float32)
    prev_actions = tf.zeros((_NUM_BATCH, _NUM_CORE_STATE), tf.float32)
    env_outputs = (tf.zeros((_NUM_UNROLLS, _NUM_BATCH), tf.float32),
                   tf.zeros((_NUM_UNROLLS, _NUM_BATCH), tf.bool),
                   tf.zeros((_NUM_UNROLLS, _NUM_BATCH, 17), tf.float32),
                   tf.zeros((_NUM_UNROLLS, _NUM_BATCH), tf.bool),
                   tf.zeros((_NUM_UNROLLS, _NUM_BATCH), tf.bool))
    action = tf.ones((_NUM_UNROLLS, _NUM_BATCH, _NUM_ACTIONS),
                     dtype=tf.float32)
    policy_logits = tf.ones((_NUM_UNROLLS, _NUM_BATCH, 2 * _NUM_ACTIONS),
                            dtype=tf.float32)
    baseline = tf.ones((_NUM_UNROLLS, _NUM_BATCH), dtype=tf.float32)
    agent_outputs = _AgentOutput(action, policy_logits, baseline)
    self._loss(agent_state, prev_actions, env_outputs, agent_outputs)


if __name__ == '__main__':
  tf.enable_v2_behavior()
  tf.test.main()
