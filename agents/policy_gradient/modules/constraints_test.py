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


"""Tests constraints."""

from seed_rl.agents.policy_gradient.modules import constraints
import tensorflow as tf


class LagrangeInequalityCoefficientTest(tf.test.TestCase):

  def test_ineqaulity_constraint(self):
    multiplier = constraints.LagrangeInequalityCoefficient(threshold=2)
    x = tf.Variable(1.23)
    def loss():
      return (multiplier.scale_loss(tf.square(x)) - x
              + multiplier.adjustment_loss(x))
    opt = tf.keras.optimizers.Adam(0.01)
    for _ in range(1000):
      opt.minimize(loss, multiplier.trainable_variables + (x,))
    self.assertAllClose(x, 2., atol=0.01)


if __name__ == '__main__':
  tf.test.main()
