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

"""Tests for utils."""

import collections

from absl.testing import parameterized

import numpy as np
from seed_rl.common import utils
import tensorflow as tf


class UnrollStoreTest(tf.test.TestCase):

  def test_duplicate_actor_id(self):
    store = utils.UnrollStore(
        num_actors=2,
        unroll_length=3,
        timestep_specs=tf.TensorSpec([], tf.int32))
    with self.assertRaises(tf.errors.InvalidArgumentError):
      store.append(
          tf.constant([2, 2], dtype=tf.int32),
          tf.constant([42, 43], dtype=tf.int32))

  def test_full(self):
    store = utils.UnrollStore(
        num_actors=4,
        unroll_length=3,
        timestep_specs=tf.TensorSpec([], tf.int32))

    def gen():
      yield False, 0, 10
      yield False, 2, 30
      yield False, 1, 20

      yield False, 0, 11
      yield False, 2, 31
      yield False, 3, 40

      yield False, 0, 12
      yield False, 2, 32
      yield False, 3, 41

      yield False, 0, 13  # Unroll: 10, 11, 12, 13
      yield False, 1, 21
      yield True, 2, 33  # No unroll because of reset

      yield False, 0, 14
      yield False, 2, 34
      yield False, 3, 42

      yield False, 0, 15
      yield False, 1, 22
      yield False, 2, 35

      yield False, 0, 16  # Unroll: 13, 14, 15, 16
      yield False, 1, 23  # Unroll: 20, 21, 22, 23
      yield False, 2, 36  # Unroll: 33, 34, 35, 36

    dataset = tf.data.Dataset.from_generator(gen, (tf.bool, tf.int32, tf.int32),
                                             ([], [], []))
    dataset = dataset.batch(3, drop_remainder=True)
    i = iter(dataset)

    should_reset, actor_ids, values = next(i)
    store.reset(actor_ids[should_reset])
    completed_ids, unrolls = store.append(actor_ids, values)
    self.assertAllEqual(tf.zeros([0]), completed_ids)
    self.assertAllEqual(tf.zeros([0, 4]), unrolls)

    should_reset, actor_ids, values = next(i)
    store.reset(actor_ids[should_reset])
    completed_ids, unrolls = store.append(actor_ids, values)
    self.assertAllEqual(tf.zeros([0]), completed_ids)
    self.assertAllEqual(tf.zeros([0, 4]), unrolls)

    should_reset, actor_ids, values = next(i)
    store.reset(actor_ids[should_reset])
    completed_ids, unrolls = store.append(actor_ids, values)
    self.assertAllEqual(tf.zeros([0]), completed_ids)
    self.assertAllEqual(tf.zeros([0, 4]), unrolls)

    should_reset, actor_ids, values = next(i)
    store.reset(actor_ids[should_reset])
    completed_ids, unrolls = store.append(actor_ids, values)
    self.assertAllEqual(tf.constant([0]), completed_ids)
    self.assertAllEqual(tf.constant([[10, 11, 12, 13]]), unrolls)

    should_reset, actor_ids, values = next(i)
    store.reset(actor_ids[should_reset])
    completed_ids, unrolls = store.append(actor_ids, values)
    self.assertAllEqual(tf.zeros([0]), completed_ids)
    self.assertAllEqual(tf.zeros([0, 4]), unrolls)

    should_reset, actor_ids, values = next(i)
    store.reset(actor_ids[should_reset])
    completed_ids, unrolls = store.append(actor_ids, values)
    self.assertAllEqual(tf.zeros([0]), completed_ids)
    self.assertAllEqual(tf.zeros([0, 4]), unrolls)

    should_reset, actor_ids, values = next(i)
    store.reset(actor_ids[should_reset])
    completed_ids, unrolls = store.append(actor_ids, values)
    self.assertAllEqual(tf.constant([0, 1, 2]), completed_ids)
    self.assertAllEqual(
        tf.constant([[13, 14, 15, 16], [20, 21, 22, 23], [33, 34, 35, 36]]),
        unrolls)

  def test_structure(self):
    named_tuple = collections.namedtuple('named_tuple', 'x y')
    num_actors = 2
    unroll_length = 10
    store = utils.UnrollStore(
        num_actors=num_actors,
        unroll_length=unroll_length,
        timestep_specs=named_tuple(
            x=tf.TensorSpec([], tf.int32), y=tf.TensorSpec([], tf.int32)))
    for _ in range(unroll_length):
      completed_ids, unrolls = store.append(
          tf.range(num_actors),
          named_tuple(
              tf.zeros([num_actors], tf.int32), tf.zeros([num_actors],
                                                         tf.int32)))
      self.assertAllEqual(tf.constant(()), completed_ids)
      self.assertAllEqual(
          named_tuple(
              tf.zeros([0, unroll_length + 1]),
              tf.zeros([0, unroll_length + 1])), unrolls)
    completed_ids, unrolls = store.append(
        tf.range(num_actors),
        named_tuple(
            tf.zeros([num_actors], tf.int32), tf.zeros([num_actors], tf.int32)))
    self.assertAllEqual(tf.range(num_actors), completed_ids)
    self.assertAllEqual(
        named_tuple(
            tf.zeros([num_actors, unroll_length + 1]),
            tf.zeros([num_actors, unroll_length + 1])), unrolls)

  def test_overlap_2(self):
    store = utils.UnrollStore(
        num_actors=2,
        unroll_length=2,
        timestep_specs=tf.TensorSpec([], tf.int32),
        num_overlapping_steps=2)

    def gen():
      yield False, 0, 10
      yield False, 1, 20

      yield False, 0, 11
      yield False, 1, 21

      yield False, 0, 12  # Unroll: 0, 0, 10, 11, 12
      yield True, 1, 22

      yield False, 0, 13
      yield False, 1, 23

      yield False, 0, 14  # Unroll: 10, 11, 12, 13, 14
      yield False, 1, 24  # Unroll: 0, 0, 22, 23, 24

      yield True, 0, 15
      yield False, 1, 25

      yield False, 0, 16
      yield False, 1, 26  # Unroll: 22, 23, 24, 25, 26

      yield False, 0, 17  # Unroll: 0, 0, 15, 16, 17
      yield False, 1, 27

    dataset = tf.data.Dataset.from_generator(gen, (tf.bool, tf.int32, tf.int32),
                                             ([], [], []))
    dataset = dataset.batch(2, drop_remainder=True)
    i = iter(dataset)

    should_reset, actor_ids, values = next(i)
    store.reset(actor_ids[should_reset])
    completed_ids, unrolls = store.append(actor_ids, values)
    self.assertAllEqual(tf.zeros([0]), completed_ids)

    should_reset, actor_ids, values = next(i)
    store.reset(actor_ids[should_reset])
    completed_ids, unrolls = store.append(actor_ids, values)
    self.assertAllEqual(tf.zeros([0]), completed_ids)

    should_reset, actor_ids, values = next(i)
    store.reset(actor_ids[should_reset])
    completed_ids, unrolls = store.append(actor_ids, values)
    self.assertAllEqual(tf.constant([0]), completed_ids)
    self.assertAllEqual(tf.constant([[0, 0, 10, 11, 12]]), unrolls)

    should_reset, actor_ids, values = next(i)
    store.reset(actor_ids[should_reset])
    completed_ids, unrolls = store.append(actor_ids, values)
    self.assertAllEqual(tf.zeros([0]), completed_ids)

    should_reset, actor_ids, values = next(i)
    store.reset(actor_ids[should_reset])
    completed_ids, unrolls = store.append(actor_ids, values)
    self.assertAllEqual(tf.constant([0, 1]), completed_ids)
    self.assertAllEqual(
        tf.constant([[10, 11, 12, 13, 14], [0, 0, 22, 23, 24]]), unrolls)

    should_reset, actor_ids, values = next(i)
    store.reset(actor_ids[should_reset])
    completed_ids, unrolls = store.append(actor_ids, values)
    self.assertAllEqual(tf.zeros([0]), completed_ids)

    should_reset, actor_ids, values = next(i)
    store.reset(actor_ids[should_reset])
    completed_ids, unrolls = store.append(actor_ids, values)
    self.assertAllEqual(tf.constant([1]), completed_ids)
    self.assertAllEqual(tf.constant([[22, 23, 24, 25, 26]]), unrolls)

    should_reset, actor_ids, values = next(i)
    store.reset(actor_ids[should_reset])
    completed_ids, unrolls = store.append(actor_ids, values)
    self.assertAllEqual(tf.constant([0]), completed_ids)
    self.assertAllEqual(tf.constant([[0, 0, 15, 16, 17]]), unrolls)


class AggregatorTest(tf.test.TestCase):

  def test_full(self):
    agg = utils.Aggregator(num_actors=4, specs=tf.TensorSpec([], tf.int32))

    self.assertAllEqual([0, 0, 0, 0], agg.read([0, 1, 2, 3]))
    agg.add([0, 1], tf.convert_to_tensor([42, 43]))
    self.assertAllEqual([42, 43], agg.read([0, 1]))
    self.assertAllEqual([42, 43, 0, 0], agg.read([0, 1, 2, 3]))
    agg.reset([0])
    self.assertAllEqual([0, 43, 0, 0], agg.read([0, 1, 2, 3]))
    agg.replace([0, 2], tf.convert_to_tensor([1, 2]))
    self.assertAllEqual([1, 43, 2, 0], agg.read([0, 1, 2, 3]))


class BatchApplyTest(tf.test.TestCase):

  def test_simple(self):

    def f(a, b):
      return tf.reduce_sum(a, axis=-1), tf.reduce_max(b, axis=-1)

    a_sum, b_max = utils.batch_apply(f, (
        tf.constant([[[0, 1], [2, 3]], [[4, 5], [6, 7]]]),
        tf.constant([[[8, 9], [10, 11]], [[12, 13], [14, 15]]]),
    ))
    self.assertAllEqual(tf.constant([[1, 5], [9, 13]]), a_sum)
    self.assertAllEqual(tf.constant([[9, 11], [13, 15]]), b_max)


class PrioritizedReplayTest(tf.test.TestCase):

  def test_simple(self):
    rb = utils.PrioritizedReplay(
        size=2,
        specs=tf.TensorSpec([], tf.int32),
        importance_sampling_exponent=.5)

    insert_indices = rb.insert(tf.constant([1, 2]), tf.constant([1., 1.]))
    self.assertAllEqual([0, 1], insert_indices)

    sampled_indices, weights, sampled_values = rb.sample(2, .5)

    self.assertAllEqual(sampled_indices + 1, sampled_values)
    self.assertAllEqual([1., 1.], weights)

    sampled_indices, weights, sampled_values = rb.sample(2, 0)

    self.assertAllEqual(sampled_indices + 1, sampled_values)
    self.assertAllEqual([1., 1.], weights)

  def test_nests(self):
    specs = (tf.TensorSpec([], tf.int32), [tf.TensorSpec([2], tf.int64)])
    zeros = tf.nest.map_structure(lambda ts: tf.zeros([1] + ts.shape, ts.dtype),
                                  specs)
    rb = utils.PrioritizedReplay(
        size=2, specs=specs, importance_sampling_exponent=.5)

    _ = rb.insert(zeros, tf.constant([1.]))
    _, _, sampled_values = rb.sample(1, .5)

    tf.nest.map_structure(self.assertAllEqual, zeros, sampled_values)

  def test_update_priorities(self):
    rb = utils.PrioritizedReplay(
        size=2,
        specs=tf.TensorSpec([], tf.int32),
        importance_sampling_exponent=.5)
    insert_indices = rb.insert(tf.constant([1, 2]), tf.constant([1., 1.]))
    self.assertAllEqual([0, 1], insert_indices)

    rb.update_priorities([0], [100])

    sampled_indices, weights, sampled_values = rb.sample(2, .5)

    self.assertAllEqual([0, 0], sampled_indices)
    self.assertAllEqual([1, 1], sampled_values)
    self.assertAllEqual([1., 1.], weights)

  def test_initial_priorities(self):
    tf.random.set_seed(5)
    rb = utils.PrioritizedReplay(
        size=2,
        specs=tf.TensorSpec([], tf.int32),
        importance_sampling_exponent=.5)
    rb.insert(tf.constant([1, 2]), tf.constant([0.1, 0.9]))

    num_sampled = 1000
    _, _, sampled_values = rb.sample(num_sampled, 1)
    counted_values = collections.Counter(sampled_values.numpy())
    self.assertGreater(counted_values[1], num_sampled * 0.1 * 0.7)
    self.assertLess(counted_values[1], num_sampled * 0.1 * 1.3)

  def _check_weights(self, sampled_weights, sampled_values, expected_weights):
    actual_weights = [None, None]
    for w, v in zip(sampled_weights, sampled_values):
      if actual_weights[v.numpy()] is None:
        actual_weights[v.numpy()] = w
      else:
        self.assertAllClose(actual_weights[v.numpy()], w,
                            msg='v={}'.format(v))
    self.assertAllClose(actual_weights, expected_weights)

  def test_importance_sampling_weights1(self):
    tf.random.set_seed(5)
    rb = utils.PrioritizedReplay(
        size=2,
        specs=tf.TensorSpec([], tf.int32),
        importance_sampling_exponent=1)
    rb.insert(tf.constant([0, 1]), tf.constant([0.3, 0.9]))
    _, weights, sampled_values = rb.sample(100, 1)
    expected_weights = np.array([
        (0.3 + 0.9) / 0.3,
        (0.3 + 0.9) / 0.9,
    ])
    expected_weights /= np.max(expected_weights)
    self._check_weights(weights, sampled_values, expected_weights)

  def test_importance_sampling_weights2(self):
    tf.random.set_seed(5)
    rb = utils.PrioritizedReplay(
        size=2,
        specs=tf.TensorSpec([], tf.int32),
        importance_sampling_exponent=.3)
    rb.insert(tf.constant([0, 1]), tf.constant([0.3, 0.9]))
    _, weights, sampled_values = rb.sample(100, .7)

    inv_sampling_probs = np.array([((0.3 ** .7 + 0.9 ** .7) / 0.3 ** .7),
                                   ((0.3 ** .7 + 0.9 ** .7) / 0.9 ** .7)])
    expected_weights = inv_sampling_probs ** .3
    expected_weights /= np.max(expected_weights)
    self._check_weights(weights, sampled_values, expected_weights)




class HindsightExperienceReplayTest(tf.test.TestCase):

  def wrap(self, x, y=None):
    unroll = collections.namedtuple('unroll', 'env_outputs')
    return unroll(
        env_outputs=utils.EnvOutput(
            observation={
                'achieved_goal': x,
                'desired_goal': y if (y is not None) else x
            },
            done=tf.zeros(x.shape[:-1], tf.bool),
            reward=tf.zeros(x.shape[:-1], tf.float32)))

  def compute_reward_fn(self, achieved_goal, desired_goal):
    return tf.norm(tf.cast(achieved_goal - desired_goal, tf.float32), axis=-1)

  def test_subsampling(self):
    rb = utils.HindsightExperienceReplay(
        size=2,
        specs=self.wrap(tf.TensorSpec([5, 1], tf.int32),
                        tf.TensorSpec([5, 1], tf.int32)),
        importance_sampling_exponent=1,
        unroll_length=2,
        compute_reward_fn=self.compute_reward_fn,
        substitution_probability=0.
        )

    rb.insert(self.wrap(tf.constant([[[10], [20], [30], [40], [50]]])),
              tf.constant([1.]))

    samples = rb.sample(1000, 1.)[-1].env_outputs.observation['achieved_goal']
    assert samples.shape == (1000, 3, 1)
    samples = tf.squeeze(samples, axis=-1)
    for i in range(samples.shape[0]):
      assert samples[i][0] in [10, 20, 30]
      assert samples[i][1] == samples[i][0] + 10
      assert samples[i][2] == samples[i][0] + 20
    for val in [10, 20, 30]:
      assert (samples[:, 0] == val).numpy().any()

  def test_goal_substitution(self):
    rb = utils.HindsightExperienceReplay(
        size=2,
        specs=self.wrap(tf.TensorSpec([5, 2], tf.int32),
                        tf.TensorSpec([5, 2], tf.int32)),
        importance_sampling_exponent=1,
        unroll_length=4,
        compute_reward_fn=self.compute_reward_fn,
        substitution_probability=1.
        )

    rb.insert(self.wrap(
        tf.constant([[[10, 10], [20, 20], [30, 30], [40, 40], [50, 50]]]),
        tf.constant([[[100, 100], [200, 200], [300, 300], [400, 400],
                      [500, 500]]]),
        ),
              tf.constant([1.]))

    samples = rb.sample(1000, 1.)[-1].env_outputs.observation
    for key in ['achieved_goal', 'desired_goal']:
      assert samples[key].shape == (1000, 5, 2)
      assert (samples[key][..., 0] == samples[key][..., 1]).numpy().all()
    samples = tf.nest.map_structure(lambda t: t[..., 0], samples)
    diffs = set()
    for i in range(samples['achieved_goal'].shape[0]):
      assert (samples['achieved_goal'][i] == [10, 20, 30, 40, 50]).numpy().all()
      for t in range(5):
        goal = samples['desired_goal'][i][t]
        assert goal in [10, 20, 30, 40, 50]
        goal //= 10
        assert goal > t + 1 or t == 4
        diffs.add(goal.numpy() - t - 1)
    assert len(diffs) == 5


class TPUEncodeTest(tf.test.TestCase, parameterized.TestCase):

  def setUp(self):
    super(TPUEncodeTest, self).setUp()
    self.data = (
        # Supported on TPU
        tf.random.uniform([128], maxval=100000, dtype=tf.int32),
        # Not supported on TPU
        tf.cast(
            tf.random.uniform([128], maxval=65535, dtype=tf.int32), tf.uint16),
        # Not supported on TPU
        tf.cast(
            tf.random.uniform([64, 84, 84, 4], maxval=256, dtype=tf.int32),
            tf.uint8),
        # Not supported on TPU
        tf.cast(tf.random.uniform([1], maxval=256, dtype=tf.int32), tf.uint8),
        # Not supported on TPU
        tf.cast(
            tf.random.uniform([100, 128, 1, 1, 1], maxval=256, dtype=tf.int32),
            tf.uint8),
        # Not supported on TPU
        tf.cast(
            tf.random.uniform([128, 100, 1, 1, 1], maxval=256, dtype=tf.int32),
            tf.uint8),
    )

  def test_simple(self):
    encoded = utils.tpu_encode(self.data)
    decoded = utils.tpu_decode(encoded)

    self.assertEqual(tf.int32, encoded[1].dtype)
    self.assertIsInstance(encoded[2], utils.TPUEncodedUInt8)
    self.assertEqual(tf.bfloat16, encoded[3].dtype)
    self.assertIsInstance(encoded[4], utils.TPUEncodedUInt8)
    self.assertIsInstance(encoded[5], utils.TPUEncodedUInt8)

    for a, b in zip(decoded, self.data):
      self.assertAllEqual(a, b)

  def test_dataset(self):
    def gen():
      yield 0

    dataset = tf.data.Dataset.from_generator(gen, tf.int64)
    dataset = dataset.map(lambda _: utils.tpu_encode(self.data))
    encoded = list(dataset)[0]
    decoded = utils.tpu_decode(encoded)

    for a, b in zip(decoded, self.data):
      self.assertAllEqual(a, b)

  @parameterized.parameters((1,), (2,))
  def test_strategy(self, num_cores):
    resolver = tf.distribute.cluster_resolver.TPUClusterResolver('')
    topology = tf.tpu.experimental.initialize_tpu_system(resolver)
    da = tf.tpu.experimental.DeviceAssignment.build(topology,
                                                    num_replicas=num_cores)
    strategy = tf.distribute.experimental.TPUStrategy(
        resolver, device_assignment=da)

    def dataset_fn(unused_ctx):
      def gen():
        yield 0
        yield 1

      dataset = tf.data.Dataset.from_generator(gen, (tf.int64))
      return dataset.map(lambda _: utils.tpu_encode(self.data))

    dataset = strategy.experimental_distribute_datasets_from_function(
        dataset_fn)
    encoded = next(iter(dataset))

    decoded = strategy.experimental_run_v2(
        tf.function(lambda args: utils.tpu_decode(args, encoded)), (encoded,))
    decoded = tf.nest.map_structure(
        lambda t: strategy.experimental_local_results(t)[0], decoded)

    for a, b in zip(decoded, self.data):
      self.assertAllEqual(a, b)


class SplitStructureTest(tf.test.TestCase):

  def test_basic(self):
    prefix, suffix = utils.split_structure(
        [tf.constant([1, 2, 3]),
         tf.constant([[4, 5], [6, 7], [8, 9]])], 1)
    self.assertAllEqual(prefix[0], tf.constant([1]))
    self.assertAllEqual(prefix[1], tf.constant([[4, 5]]))
    self.assertAllEqual(suffix[0], tf.constant([2, 3]))
    self.assertAllEqual(suffix[1], tf.constant([[6, 7], [8, 9]]))

  def test_zero_length_prefix(self):
    prefix, suffix = utils.split_structure(tf.constant([1, 2, 3]), 0)
    self.assertAllEqual(prefix, tf.constant([]))
    self.assertAllEqual(suffix, tf.constant([1, 2, 3]))


class MakeTimeMajorTest(tf.test.TestCase):

  def test_static(self):
    x = tf.constant([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    self.assertAllEqual(utils.make_time_major(x),
                        tf.constant([[[1, 2], [5, 6]], [[3, 4], [7, 8]]]))

  def test_dynamic(self):
    x, = tf.py_function(lambda: np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]),
                        [], [tf.int32])
    self.assertAllEqual(utils.make_time_major(x),
                        tf.constant([[[1, 2], [5, 6]], [[3, 4], [7, 8]]]))

  def test_uint16(self):
    x = tf.constant([[1, 2], [3, 4]], tf.uint16)
    self.assertAllEqual(utils.make_time_major(x), tf.constant([[1, 3], [2, 4]]))

  def test_nest(self):
    x = (tf.constant([[1, 2], [3, 4]]), tf.constant([[1], [2]]))
    a, b = utils.make_time_major(x)
    self.assertAllEqual(a, tf.constant([[1, 3], [2, 4]]))
    self.assertAllEqual(b, tf.constant([[1, 2]]))


class MinimizeTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.parameters((1,), (2,))
  def test_minimize(self, num_training_tpus):
    resolver = tf.distribute.cluster_resolver.TPUClusterResolver('')
    strategy = tf.distribute.experimental.TPUStrategy(resolver)
    topology = tf.tpu.experimental.initialize_tpu_system(resolver)
    training_da = tf.tpu.experimental.DeviceAssignment.build(
        topology, num_replicas=num_training_tpus)
    training_strategy = tf.distribute.experimental.TPUStrategy(
        resolver, device_assignment=training_da)

    with strategy.scope():
      a = tf.Variable(1., trainable=True)
      temp_grad = tf.Variable(
          tf.zeros_like(a),
          trainable=False,
          synchronization=tf.VariableSynchronization.ON_READ)

    @tf.function
    def compute_gradients():
      with tf.GradientTape() as tape:
        tape.watch(a)
        loss = a * 2
      g = tape.gradient(loss, a)
      temp_grad.assign(g)
      return loss

    loss = training_strategy.experimental_run_v2(compute_gradients, ())
    loss = training_strategy.experimental_local_results(loss)[0]

    optimizer = tf.keras.optimizers.SGD(.1)
    @tf.function
    def apply_gradients(_):
      optimizer.apply_gradients([(temp_grad, a)])

    strategy.experimental_run_v2(apply_gradients, (loss,))

    a_values = [v.read_value() for v in strategy.experimental_local_results(a)]

    expected_a = 1. - num_training_tpus * .2
    self.assertAllClose([expected_a, expected_a], a_values)


class ProgressLoggerTest(tf.test.TestCase):

  def test_logger(self):
    logger = utils.ProgressLogger()
    logger.start()
    logger._log()

    @tf.function(input_signature=(tf.TensorSpec([], tf.int32, 'value'),))
    def log_something(value):
      session = logger.log_session()
      logger.log(session, 'value_1', value)
      logger.log(session, 'value_2', value + 1)
      logger.step_end(session)

    log_something(tf.constant(10))
    logger._log()
    self.assertAllEqual(logger.ready_values.read_value(), tf.constant([10, 11]))
    log_something(tf.constant(15))
    self.assertAllEqual(logger.ready_values.read_value(), tf.constant([15, 16]))
    logger._log()
    logger.shutdown()


if __name__ == '__main__':
  tf.test.main()
