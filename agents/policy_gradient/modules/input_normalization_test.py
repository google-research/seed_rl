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


"""Tests for input_normalization."""
from seed_rl.agents.policy_gradient.modules import input_normalization
from seed_rl.agents.policy_gradient.modules import running_statistics
from seed_rl.agents.policy_gradient.modules import test_utils
import tensorflow as tf


def setUpModule():
  # We want our tests to run on several devices with a mirrored strategy.
  test_utils.simulate_two_devices()


class EMANormalizationTest(test_utils.TestCase):
  # We manually enter a strategy scope in our tests.
  ENTER_PRIMARY_DEVICE = False

  def _setup_input_normalization(self, beta, size):
    """Sets up the input normalizer and a distribution strategy to run."""
    mean_std_tracker = running_statistics.EMAMeanStd(beta=beta)
    input_normalizer = input_normalization.InputNormalization(mean_std_tracker)
    strategy = test_utils.create_distribution_strategy(
        use_tpu=self.primary_device == 'TPU')
    self.assertEqual(strategy.num_replicas_in_sync, 2)
    with strategy.scope():
      input_normalizer.init_normalization_stats(size)
    return input_normalizer, strategy

  def _update_normalization_statistics(self, input_normalizer, strategy,
                                       inputs):
    """Updates the normalization statistics via the strategy."""
    dataset = tf.data.Dataset.from_tensors(inputs)
    dataset_iterator = iter(strategy.experimental_distribute_dataset(dataset))
    distributed_values = next(dataset_iterator)
    f = tf.function(input_normalizer.update_normalization_statistics)
    strategy.run(f, (distributed_values,))

  def test_normalization(self):
    """Tests that the normalization works."""
    input_normalizer, strategy = self._setup_input_normalization(1., 10)

    # Apply update.
    inputs = tf.reshape(tf.range(200, dtype=tf.float32), (20, 10))
    self._update_normalization_statistics(input_normalizer, strategy, inputs)

    # Verify that normalization works correctly.
    with strategy.scope():
      normalized = input_normalizer.normalize(inputs)

    mean = tf.reduce_mean(normalized, 0)
    std = tf.math.reduce_std(normalized, 0)
    self.assertAllClose(mean, tf.zeros_like(mean))
    self.assertAllClose(std, tf.ones_like(std))

  def test_variables(self):
    """Tests that we have the correct number of variables."""
    input_normalizer, _ = self._setup_input_normalization(.5, 10)
    self.assertLen(input_normalizer.variables, 4)
    self.assertLen(input_normalizer.trainable_variables, 2)

  def test_invariance(self):
    """Tests that the update keeps the convolution invariant."""
    input_normalizer, strategy = self._setup_input_normalization(.5, 10)
    inputs = tf.reshape(tf.range(200, dtype=tf.float32), (20, 10))

    # Run the convolution of `normalize(...)` and `correct(...)` *before* the
    # update of the statistics.
    with strategy.scope():
      before = input_normalizer.normalize(inputs)
      before = input_normalizer.correct(before)

    self._update_normalization_statistics(input_normalizer, strategy, inputs)

    # Run the convolution of `normalize(...)` and `correct(...)` *after* the
    # update of the statistics.
    with strategy.scope():
      after = input_normalizer.normalize(inputs)
      after = input_normalizer.correct(after)

    # Verify that the update is correct.
    self.assertAllClose(before, after, rtol=1e-5, atol=1e-5)


if __name__ == '__main__':
  tf.enable_v2_behavior()
  tf.test.main()
