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

"""Tests for normalizer."""

import numpy as np
from seed_rl.common import normalizer
import tensorflow as tf


class NormalizerTest(tf.test.TestCase):

  def test_normalization(self):
    norm = normalizer.Normalizer(eps=0., clip_range=(-np.inf, np.inf))
    data = tf.random.uniform((100, 32))
    for _ in range(5):
      norm.update(data)
    normalized = norm(data)
    tf.debugging.assert_near(tf.reduce_mean(normalized, axis=0),
                             tf.zeros(32),
                             atol=1e-4)
    tf.debugging.assert_near(tf.math.reduce_std(normalized, axis=0),
                             tf.ones(32),
                             atol=1e-4)

  def test_normalizer_many_replicas(self):
    resolver = tf.distribute.cluster_resolver.TPUClusterResolver('')
    strategy = tf.distribute.experimental.TPUStrategy(resolver)

    strategy._enable_packed_variable_in_eager_mode = False  
    topology = tf.tpu.experimental.initialize_tpu_system(resolver)
    training_da = tf.tpu.experimental.DeviceAssignment.build(
        topology, num_replicas=1)
    training_strategy = tf.distribute.experimental.TPUStrategy(
        resolver, device_assignment=training_da)

    training_strategy._enable_packed_variable_in_eager_mode = False  

    data = tf.random.uniform((100, 32))

    with strategy.scope():
      norm = normalizer.Normalizer(eps=0., clip_range=(-np.inf, np.inf))
      norm(data)  # create the variables

    @tf.function
    def training_step():
      norm.update(data, only_accumulate=True)

    for _ in range(5):
      training_strategy.run(training_step)
      norm.finish_update()

    normalized = norm(data)
    tf.debugging.assert_near(tf.reduce_mean(normalized, axis=0),
                             tf.zeros(32),
                             atol=1e-4)
    tf.debugging.assert_near(tf.math.reduce_std(normalized, axis=0),
                             tf.ones(32),
                             atol=1e-4)

    for var in [norm.steps_acc, norm.sum_acc, norm.sumsq_acc]:
      var = training_strategy.experimental_local_results(var)
      tf.debugging.assert_equal(tf.norm(var), 0.)


if __name__ == '__main__':
  tf.test.main()
