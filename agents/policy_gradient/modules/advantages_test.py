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


"""Tests for advantages."""

from absl.testing import parameterized

from seed_rl.agents.policy_gradient.modules import advantages
from seed_rl.agents.policy_gradient.modules import test_utils
from seed_rl.common import vtrace
import tensorflow as tf
import tensorflow_probability as tfp


class GAETest(test_utils.TestCase, parameterized.TestCase):

  def test_discount_one_lambda_one(self):
    """Tests the case where lambda_ and discounts are 1 with zero rewards.

    In this case, the target should be just the bootstrap values for each time
    step.
    """
    values = tf.reshape(tf.range(210, dtype=tf.float32), (21, 10))
    rewards = tf.zeros((20, 10), dtype=tf.float32)
    done_terminated = tf.zeros_like(rewards, dtype=tf.bool)
    done_abandoned = tf.zeros_like(rewards, dtype=tf.bool)

    tested_targets, tested_advantages = advantages.gae(
        values, rewards, done_terminated, done_abandoned, discount_factor=1.,
        lambda_=1.)

    real_targets = tf.broadcast_to(tf.expand_dims(values[-1], 0), (20, 10))
    real_advantages = real_targets - values[:-1]

    self.assertAllClose(tested_targets, real_targets)
    self.assertAllClose(tested_advantages, real_advantages)

  def test_zero_discount(self):
    """Tests the case where all the value comes from the immediate rewards."""
    rewards = tf.reshape(tf.range(200, dtype=tf.float32), (20, 10))
    values = tf.zeros((21, 10), dtype=tf.float32)
    done_terminated = tf.zeros_like(rewards, dtype=tf.bool)
    done_abandoned = tf.zeros_like(rewards, dtype=tf.bool)

    tested_targets, tested_advantages = advantages.gae(
        values, rewards, done_terminated, done_abandoned, discount_factor=0.,
        lambda_=1.)

    self.assertAllClose(tested_targets, rewards)
    self.assertAllClose(tested_advantages, rewards)

  def test_lambda_zero(self):
    """Tests the case where lambda_ is zero.

    In this case the value is the sum of rewards and discounted next-step
    value/bootstrap value.
    """

    values = tf.reshape(tf.range(210, dtype=tf.float32), (21, 10))
    rewards = tf.ones((20, 10), dtype=tf.float32)
    done_terminated = tf.zeros_like(rewards, dtype=tf.bool)
    done_abandoned = tf.zeros_like(rewards, dtype=tf.bool)

    tested_targets, tested_advantages = advantages.gae(
        values, rewards, done_terminated, done_abandoned, discount_factor=.97,
        lambda_=0.)
    real_targets = rewards + 0.97 * values[1:]
    real_advantages = real_targets - values[:-1]

    self.assertAllClose(tested_targets, real_targets, rtol=1e-5, atol=1e-5)
    self.assertAllClose(tested_advantages, real_advantages, rtol=1e-5,
                        atol=1e-5)

  @parameterized.named_parameters(
      ('terminated', True),
      ('abandoned', False))
  def test_done(self, terminated):
    """Tests that done values are handled correctly.

    Args:
      terminated: Bool indicating if done_terminated should be set. Otherwise,
        done_abandoned is set.
    """
    # We generate a case where there is a standard transition, a reset (either
    # due to termination or abandonment), and then another standard transition.
    values = tf.convert_to_tensor([[1.], [2.], [3.], [4.]], dtype=tf.float32)
    rewards = tf.convert_to_tensor([[.1], [.2], [.3]], dtype=tf.float32)
    active_transition = [[False], [True], [False]]
    if terminated:
      done_terminated = tf.convert_to_tensor(active_transition, dtype=tf.bool)
      done_abandoned = tf.zeros_like(rewards, dtype=tf.bool)
    else:
      done_abandoned = tf.convert_to_tensor(active_transition, dtype=tf.bool)
      done_terminated = tf.zeros_like(rewards, dtype=tf.bool)

    # We compute the  results using the tested function.
    tested_targets, tested_advantages = advantages.gae(
        values, rewards, done_terminated, done_abandoned, discount_factor=.97,
        lambda_=.5)

    # We compute the real solution for this case by hand.
    delta1 = rewards[0][0] + .97 * values[1][0] - values[0][0]
    delta2 = (rewards[1][0]  - values[1][0]) if terminated else 0.
    delta3 = rewards[2][0] + .97 * values[3][0] - values[2][0]

    real_advantages = [[delta1 + .97 * .5 * delta2], [delta2], [delta3]]
    real_advantages = tf.convert_to_tensor(real_advantages, dtype=tf.float32)
    real_targets = real_advantages + values[:-1]

    self.assertAllClose(tested_targets, real_targets, rtol=1e-5, atol=1e-5)
    self.assertAllClose(tested_advantages, real_advantages, rtol=1e-5,
                        atol=1e-5)


class VTraceTest(test_utils.TestCase, parameterized.TestCase):

  def test_vtrace_vs_seed(self):
    values = tf.random.uniform((21, 10), maxval=3)
    rewards = tf.random.uniform((20, 10), maxval=3)
    target_action_log_probs = tf.random.uniform((20, 10), minval=-2, maxval=2)
    behaviour_action_log_probs = tf.random.uniform((20, 10), minval=-2,
                                                   maxval=2)
    done_terminated = tf.cast(
        tfp.distributions.Bernoulli(0.05).sample((20, 10)), tf.bool)
    done_abandoned = tf.zeros_like(rewards, dtype=tf.bool)

    tested_targets, unused_tested_advantages = advantages.vtrace(
        values, rewards,
        done_terminated, done_abandoned, 0.99,
        target_action_log_probs, behaviour_action_log_probs,
        lambda_=0.95)

    seed_output = vtrace.from_importance_weights(
        target_action_log_probs, behaviour_action_log_probs,
        0.99 * tf.cast(~done_terminated, tf.float32), rewards,
        values[:-1], values[-1], lambda_=0.95)

    self.assertAllClose(tested_targets, seed_output.vs)

    # Currently this implementation computes advantages in a slightly different
    # way than the one from SEED. The one from SEED follows formula from the
    # IMPALA paper: A_t = r_t + gamma * v_{t+1} - V_t where v_{t+1} is the
    # target for V_{t+1}. This target doesn't include V_{t+1} term and as
    # a result A_t doesn't take V_{t+1} into account.
    # On the other hand, this implementation follows GAE and uses
    # the following advantages:
    # A_t = r_t + gamma * (V_{t+1} + lambda * (v_{t+1} - V{t+1})) - V_t


class NStepTest(test_utils.TestCase, parameterized.TestCase):

  def test_nstep_vs_gae(self):
    # inf-step return should be the same as GAE with lambda=1.
    values = tf.random.uniform((21, 10), maxval=3)
    rewards = tf.random.uniform((20, 10), maxval=3)
    target_action_log_probs = tf.random.uniform((20, 10), minval=-2, maxval=2)
    behaviour_action_log_probs = tf.random.uniform((20, 10), minval=-2,
                                                   maxval=2)
    done_terminated = tf.cast(
        tfp.distributions.Bernoulli(probs=0.2).sample((20, 10)), tf.bool)
    done_abandoned = tf.cast(
        tfp.distributions.Bernoulli(probs=0.2).sample((20, 10)), tf.bool)

    nstep_targets, nstep_advantages = advantages.NStep(n=10000)(
        values, rewards,
        done_terminated, done_abandoned, 0.99,
        target_action_log_probs, behaviour_action_log_probs)

    gae_targets, gae_advantages = advantages.GAE(lambda_=1.)(
        values, rewards,
        done_terminated, done_abandoned, 0.99,
        target_action_log_probs, behaviour_action_log_probs)

    self.assertAllClose(nstep_targets, gae_targets)
    self.assertAllClose(nstep_advantages, gae_advantages)

  def test_1step_returns(self):
    values = tf.random.uniform((201, 10), maxval=3)
    rewards = tf.random.uniform((200, 10), maxval=3)
    target_action_log_probs = tf.random.uniform((200, 10), minval=-2, maxval=2)
    behaviour_action_log_probs = tf.random.uniform((200, 10), minval=-2,
                                                   maxval=2)
    done_terminated = tf.cast(
        tfp.distributions.Bernoulli(probs=0.2).sample((200, 10)), tf.bool)
    done_abandoned = tf.cast(
        tfp.distributions.Bernoulli(probs=0.2).sample((200, 10)), tf.bool)

    nstep_targets, nstep_advantages = advantages.NStep(n=1)(
        values, rewards,
        done_terminated, done_abandoned, 0.99,
        target_action_log_probs, behaviour_action_log_probs)

    correct_targets = rewards + 0.99 * tf.cast(
        ~done_terminated, tf.float32) * values[1:]
    correct_targets += tf.cast(done_abandoned, tf.float32) * (
        values[:-1] - correct_targets)
    correct_advantages = correct_targets - values[:-1]

    self.assertAllClose(nstep_targets, correct_targets, rtol=3e-5, atol=3e-5)
    self.assertAllClose(nstep_advantages, correct_advantages,
                        rtol=3e-5, atol=3e-5)

  def test_nstep_terminated(self):
    values = tf.cast(tf.reshape(1 + tf.range(11), (11, 1)), tf.float32)
    rewards = values[:-1] * 0.1
    target_action_log_probs = tf.zeros_like(rewards)
    behaviour_action_log_probs = tf.zeros_like(rewards)
    done_terminated = [0, 1, 0, 1, 0, 0, 0, 0, 1, 1]
    done_terminated = tf.reshape(tf.cast(done_terminated, tf.bool), (10, 1))
    done_abandoned = tf.zeros_like(done_terminated)

    nstep_targets, nstep_advantages = advantages.NStep(n=3)(
        values, rewards,
        done_terminated, done_abandoned, 0.99,
        target_action_log_probs, behaviour_action_log_probs)
    d = 0.99
    d2 = d * d
    d3 = d * d * d

    correct_targets = [0.1 + d * 0.2,
                       0.2,
                       0.3 + d * 0.4,
                       0.4,
                       0.5 + d * 0.6 + d2 * 0.7 + d3 * 8.,
                       0.6 + d * 0.7 + d2 * 0.8 + d3 * 9.,
                       0.7 + d * 0.8 + d2 * 0.9,
                       0.8 + d * 0.9,
                       0.9,
                       1.]
    correct_targets = tf.reshape(correct_targets, (10, 1))
    correct_advantages = correct_targets - values[:-1]

    self.assertAllClose(nstep_targets, correct_targets, rtol=3e-5, atol=3e-5)
    self.assertAllClose(nstep_advantages, correct_advantages,
                        rtol=3e-5, atol=3e-5)

  def test_nstep_abandoned(self):
    values = tf.cast(tf.reshape(1 + tf.range(11), (11, 1)), tf.float32)
    rewards = values[:-1] * 0.1
    target_action_log_probs = tf.zeros_like(rewards)
    behaviour_action_log_probs = tf.zeros_like(rewards)
    done_abandoned = [0, 1, 0, 1, 0, 0, 0, 0, 1, 1]
    done_abandoned = tf.reshape(tf.cast(done_abandoned, tf.bool), (10, 1))
    done_terminated = tf.zeros_like(done_abandoned)

    nstep_targets, nstep_advantages = advantages.NStep(n=3)(
        values, rewards,
        done_terminated, done_abandoned, 0.99,
        target_action_log_probs, behaviour_action_log_probs)
    d = 0.99
    d2 = d * d
    d3 = d * d * d

    correct_targets = [0.1 + d * 2,
                       2.,
                       0.3 + d * 4,
                       4.,
                       0.5 + d * 0.6 + d2 * 0.7 + d3 * 8.,
                       0.6 + d * 0.7 + d2 * 0.8 + d3 * 9.,
                       0.7 + d * 0.8 + d2 * 9.,
                       0.8 + d * 9.,
                       9.,
                       10.]
    correct_targets = tf.reshape(correct_targets, (10, 1))
    correct_advantages = correct_targets - values[:-1]

    self.assertAllClose(nstep_targets, correct_targets, rtol=3e-5, atol=3e-5)
    self.assertAllClose(nstep_advantages, correct_advantages,
                        rtol=3e-5, atol=3e-5)


if __name__ == '__main__':
  tf.enable_v2_behavior()
  tf.test.main()
