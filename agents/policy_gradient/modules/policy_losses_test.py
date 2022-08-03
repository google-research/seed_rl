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


"""Tests for policy_lisses."""
from seed_rl.agents.policy_gradient.modules import policy_losses
from seed_rl.agents.policy_gradient.modules import test_utils
import tensorflow as tf


class AdvantagePreprocessorTest(test_utils.TestCase):

  def test_normalization(self):
    adv = tf.constant([1.5, 4., 123., -3.])
    adv = policy_losses.AdvantagePreprocessor(normalize=True)(adv)[0]
    assert adv.shape == (4,)
    self.assertAllClose(tf.reduce_mean(adv), 0.)
    self.assertAllClose(tf.math.reduce_std(adv), 1, atol=0.01)

  def test_only_positive(self):
    adv = tf.constant([1.5, 4., 123., -3.])
    adv = policy_losses.AdvantagePreprocessor(only_positive=True)(adv)[0]
    self.assertAllClose(adv, [1.5, 4., 123., 0.])

  def test_only_top_half(self):
    adv = tf.constant([1.5, 4., 123., -3.])
    adv = policy_losses.AdvantagePreprocessor(only_top_half=True)(adv)[0]
    self.assertAllClose(adv, [0., 4., 123., 0.])


class GeneralizedAdvantagePolicyLossTest(tf.test.TestCase):

  def setUp(self):
    super().setUp()
    self.adv = tf.random.uniform([100], maxval=2)
    self.pi = tf.random.uniform([100], maxval=2)
    self.mu = tf.random.uniform([100], maxval=2)

  def _loss_grad(self, loss, feed_log_probs):
    with tf.GradientTape() as t:
      t.watch(self.pi)
      return t.gradient(
          loss(self.adv,
               tf.math.log(self.pi) if feed_log_probs else self.pi,
               tf.math.log(self.mu) if feed_log_probs else self.mu, None, None,
               None), self.pi)

  def test_pg(self):
    loss = lambda adv, pi, mu, d1, d2, d3: -tf.reduce_mean(  
        tf.math.log(pi) * adv)
    self.assertAllClose(self._loss_grad(policy_losses.pg(), True),
                        self._loss_grad(loss, False))

  def test_vtrace(self):
    loss = lambda adv, pi, mu, d1, d2, d3: -tf.reduce_mean(  
        pi / tf.stop_gradient(tf.maximum(mu, pi)) * adv)
    self.assertAllClose(self._loss_grad(policy_losses.vtrace(), True),
                        self._loss_grad(loss, False))

  def test_ppo(self):
    def loss(adv, pi, mu, *unused_args):
      rho = pi / mu
      rho_clipped = tf.clip_by_value(rho, 1/1.2, 1.2)
      return -tf.reduce_mean(tf.minimum(rho * adv, rho_clipped * adv))
    self.assertAllClose(self._loss_grad(policy_losses.ppo(epsilon=0.2), True),
                        self._loss_grad(loss, False))

  def test_awr(self):
    loss = lambda adv, pi, mu, d1, d2, d3: -tf.reduce_mean(  
        tf.math.log(pi) * tf.exp(adv / 5.))
    self.assertAllClose(self._loss_grad(policy_losses.awr(beta=5.,
                                                          w_max=1e9), True),
                        self._loss_grad(loss, False))


if __name__ == '__main__':
  tf.test.main()
