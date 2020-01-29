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

"""Utility functions/classes."""

import collections
import contextlib

from absl import logging

import numpy as np
import tensorflow as tf

from tensorflow.python.distribute import values as values_lib  
from tensorflow.python.framework import composite_tensor  
from tensorflow.python.framework import tensor_conversion_registry  


# `observation` is the observation *after* a transition. When `done` is True,
# `observation` will be the observation *after* the reset.
EnvOutput = collections.namedtuple('EnvOutput', 'reward done observation')


Settings = collections.namedtuple(
    'Settings', 'strategy inference_devices training_strategy encode decode')


def init_learner(num_training_tpus):
  """Performs common learner initialization."""
  if tf.config.experimental.list_logical_devices('TPU'):
    resolver = tf.distribute.cluster_resolver.TPUClusterResolver('')
    topology = tf.tpu.experimental.initialize_tpu_system(resolver)
    strategy = tf.distribute.experimental.TPUStrategy(resolver)
    training_da = tf.tpu.experimental.DeviceAssignment.build(
        topology, num_replicas=num_training_tpus)
    training_strategy = tf.distribute.experimental.TPUStrategy(
        resolver, device_assignment=training_da)
    inference_devices = list(set(strategy.extended.worker_devices) -
                             set(training_strategy.extended.worker_devices))
    return Settings(strategy, inference_devices, training_strategy, tpu_encode,
                    tpu_decode)
  else:
    tf.device('/cpu').__enter__()
    any_gpu = tf.config.experimental.list_logical_devices('GPU')
    device_name = '/device:GPU:0' if any_gpu else '/device:CPU:0'
    strategy = tf.distribute.OneDeviceStrategy(device=device_name)
    enc = lambda x: x
    dec = lambda x, s=None: x if s is None else tf.nest.pack_sequence_as(s, x)
    return Settings(strategy, [device_name], strategy, enc, dec)


class UnrollStore(tf.Module):
  """Utility module for combining individual actor steps into unrolls."""

  def __init__(self,
               num_actors,
               unroll_length,
               timestep_specs,
               num_overlapping_steps=0,
               name='UnrollStore'):
    super(UnrollStore, self).__init__(name=name)
    with self.name_scope:
      self._full_length = num_overlapping_steps + unroll_length + 1

      def create_unroll_variable(spec):
        z = tf.zeros(
            [num_actors, self._full_length] + spec.shape.dims, dtype=spec.dtype)
        return tf.Variable(z, trainable=False, name=spec.name)

      self._unroll_length = unroll_length
      self._num_overlapping_steps = num_overlapping_steps
      self._state = tf.nest.map_structure(create_unroll_variable,
                                          timestep_specs)
      # For each actor, the index into the actor dimension of the tensors in
      # self._state where we should add the next element.
      self._index = tf.Variable(
          tf.fill([num_actors], tf.constant(num_overlapping_steps, tf.int32)),
          trainable=False,
          name='index')

  @property
  def unroll_specs(self):
    return tf.nest.map_structure(lambda v: tf.TensorSpec(v.shape[1:], v.dtype),
                                 self._state)

  @tf.function
  @tf.Module.with_name_scope
  def append(self, actor_ids, values):
    """Appends values and returns completed unrolls.

    Args:
      actor_ids: 1D tensor with the list of actor IDs for which we append data.
        There must not be duplicates.
      values: Values to add for each actor. This is a structure (in the tf.nest
        sense) of tensors following "timestep_specs", with a batch front
        dimension which must be equal to the length of 'actor_ids'.

    Returns:
      A pair of:
        - 1D tensor of the actor IDs of the completed unrolls.
        - Completed unrolls. This is a structure of tensors following
          'timestep_specs', with added front dimensions: [num_completed_unrolls,
          num_overlapping_steps + unroll_length + 1].
    """
    tf.debugging.assert_equal(
        tf.shape(actor_ids),
        tf.shape(tf.unique(actor_ids)[0]),
        message='Duplicate actor ids')
    
    tf.nest.map_structure(
        lambda s: tf.debugging.assert_equal(
            tf.shape(actor_ids)[0],
            tf.shape(s)[0],
            message='Batch dimension must be same size as number of actors.'),
        values)
    

    curr_indices = self._index.sparse_read(actor_ids)
    unroll_indices = tf.stack([actor_ids, curr_indices], axis=-1)
    for s, v in zip(tf.nest.flatten(self._state), tf.nest.flatten(values)):
      s.scatter_nd_update(unroll_indices, v)

    # Intentionally not protecting against out-of-bounds to make it possible to
    # detect completed unrolls.
    self._index.scatter_add(tf.IndexedSlices(1, actor_ids))

    return self._complete_unrolls(actor_ids)

  @tf.function
  @tf.Module.with_name_scope
  def reset(self, actor_ids):
    """Resets state.

    Note, this is only intended to be called when actors need to be reset after
    preemptions. Not at episode boundaries.

    Args:
      actor_ids: The actors that need to have their state reset.
    """
    self._index.scatter_update(
        tf.IndexedSlices(self._num_overlapping_steps, actor_ids))

    # The following code is the equivalent of:
    # s[actor_ids, :j] = 0
    j = self._num_overlapping_steps
    repeated_actor_ids = tf.reshape(
        tf.tile(tf.expand_dims(tf.cast(actor_ids, tf.int64), -1), [1, j]), [-1])

    repeated_range = tf.tile(tf.range(j, dtype=tf.int64),
                             [tf.shape(actor_ids)[0]])
    indices = tf.stack([repeated_actor_ids, repeated_range], axis=-1)

    for s in tf.nest.flatten(self._state):
      z = tf.zeros(tf.concat([tf.shape(repeated_actor_ids),
                              s.shape[2:]], axis=0), s.dtype)
      s.scatter_nd_update(indices, z)

  def _complete_unrolls(self, actor_ids):
    # Actor with unrolls that are now complete and should be returned.
    actor_indices = self._index.sparse_read(actor_ids)
    actor_ids = tf.gather(
        actor_ids,
        tf.where(tf.equal(actor_indices, self._full_length))[:, 0])
    actor_ids = tf.cast(actor_ids, tf.int64)
    unrolls = tf.nest.map_structure(lambda s: s.sparse_read(actor_ids),
                                    self._state)

    # Store last transitions as the first in the next unroll.
    # The following code is the equivalent of:
    # s[actor_ids, :j] = s[actor_ids, -j:]
    j = self._num_overlapping_steps + 1
    repeated_start_range = tf.tile(tf.range(j, dtype=tf.int64),
                                   [tf.shape(actor_ids)[0]])
    repeated_end_range = tf.tile(
        tf.range(self._full_length - j, self._full_length, dtype=tf.int64),
        [tf.shape(actor_ids)[0]])
    repeated_actor_ids = tf.reshape(
        tf.tile(tf.expand_dims(actor_ids, -1), [1, j]), [-1])
    start_indices = tf.stack([repeated_actor_ids, repeated_start_range], -1)
    end_indices = tf.stack([repeated_actor_ids, repeated_end_range], -1)

    for s in tf.nest.flatten(self._state):
      s.scatter_nd_update(start_indices, s.gather_nd(end_indices))

    self._index.scatter_update(
        tf.IndexedSlices(1 + self._num_overlapping_steps, actor_ids))

    return actor_ids, unrolls


class PrioritizedReplay(tf.Module):
  """Prioritized Replay Buffer.

  This buffer is not threadsafe. Make sure you call insert() and sample() from a
  single thread.
  """

  def __init__(self, size, specs, importance_sampling_exponent,
               name='PrioritizedReplay'):
    super(PrioritizedReplay, self).__init__(name=name)
    self._priorities = tf.Variable(tf.zeros([size]), dtype=tf.float32)
    self._buffer = tf.nest.map_structure(
        lambda ts: tf.Variable(tf.zeros([size] + ts.shape, dtype=ts.dtype)),
        specs)
    self.num_inserted = tf.Variable(0, dtype=tf.int64)
    self._importance_sampling_exponent = importance_sampling_exponent

  @tf.function
  @tf.Module.with_name_scope
  def insert(self, values, priorities):
    """FIFO insertion/removal.

    Args:
      values: The batched values to insert. The tensors must be of the same
        shape and dtype as the `specs` provided in the constructor, except
        including a batch dimension.
      priorities: <float32>[batch_size] tensor with the priorities of the
        elements we insert.
    Returns:
      The indices of the inserted values.
    """
    tf.nest.assert_same_structure(values, self._buffer)
    values = tf.nest.map_structure(tf.convert_to_tensor, values)
    append_size = tf.nest.flatten(values)[0].shape[0]
    start_index = self.num_inserted
    end_index = start_index + append_size

    # Wrap around insertion.
    size = self._priorities.shape[0]
    insert_indices = tf.range(start_index, end_index) % size
    tf.nest.map_structure(
        lambda b, v: b.batch_scatter_update(  
            tf.IndexedSlices(v, insert_indices)),
        self._buffer,
        values)
    self.num_inserted.assign_add(append_size)

    self._priorities.batch_scatter_update(
        tf.IndexedSlices(priorities, insert_indices))

    return insert_indices

  @tf.function
  @tf.Module.with_name_scope
  def sample(self, num_samples, priority_exp):
    r"""Samples items from the replay buffer, using priorities.

    Args:
      num_samples: int, number of replay items to sample.
      priority_exp: Priority exponent. Every item i in the replay buffer will be
        sampled with probability:
         priority[i] ** priority_exp /
             sum(priority[j] ** priority_exp, j \in [0, num_items))
        Set this to 0 in order to get uniform sampling.

    Returns:
      Tuple of:
        - indices: An int64 tensor of shape [num_samples] with the indices in
          the replay buffer of the sampled items.
        - weights: A float32 tensor of shape [num_samples] with the normalized
          weights of the sampled items.
        - sampled_values: A nested structure following the spec passed in the
          contructor, where each tensor has an added front batch dimension equal
          to 'num_samples'.
    """
    tf.debugging.assert_greater_equal(
        self.num_inserted,
        tf.constant(0, tf.int64),
        message='Cannot sample if replay buffer is empty')
    size = self._priorities.shape[0]
    limit = tf.minimum(tf.cast(size, tf.int64), self.num_inserted)
    if priority_exp == 0:
      indices = tf.random.uniform([num_samples], maxval=limit, dtype=tf.int64)
      weights = tf.ones_like(indices, dtype=tf.float32)
    else:
      prob = self._priorities[:limit]**priority_exp
      prob /= tf.reduce_sum(prob)
      indices = tf.random.categorical([tf.math.log(prob)], num_samples)[0]
      # Importance weights.
      weights = (((1. / tf.cast(limit, tf.float32)) /
                  tf.gather(prob, indices)) **
                 self._importance_sampling_exponent)
      weights /= tf.reduce_max(weights)  # Normalize.

    sampled_values = tf.nest.map_structure(
        lambda b: b.sparse_read(indices), self._buffer)
    return indices, weights, sampled_values

  @tf.function
  @tf.Module.with_name_scope
  def update_priorities(self, indices, priorities):
    """Updates the priorities of the items with the given indices.

    Args:
      indices: <int64>[batch_size] tensor with the indices of the items to
        update. If duplicate indices are provided, the priority that will be set
        among possible ones is not specified.
      priorities: <float32>[batch_size] tensor with the new priorities.
    """

    self._priorities.batch_scatter_update(tf.IndexedSlices(priorities, indices))


class Aggregator(tf.Module):
  """Utility module for keeping state and statistics for individual actors."""

  def __init__(self, num_actors, specs, name='Aggregator'):
    """Inits an Aggregator.

    Args:
      num_actors: int, number of actors.
      specs: Structure (as defined by tf.nest) of tf.TensorSpecs that will be
        stored for each actor.
      name: Name of the scope for the operations.
    """
    super(Aggregator, self).__init__(name=name)
    def create_variable(spec):
      z = tf.zeros([num_actors] + spec.shape.dims, dtype=spec.dtype)
      return tf.Variable(z, trainable=False, name=spec.name)

    self._state = tf.nest.map_structure(create_variable, specs)

  @tf.Module.with_name_scope
  def reset(self, actor_ids):
    """Fills the tensors for the given actors with zeros."""
    with tf.name_scope('Aggregator_reset'):
      for s in tf.nest.flatten(self._state):
        s.scatter_update(tf.IndexedSlices(0, actor_ids))

  @tf.Module.with_name_scope
  def add(self, actor_ids, values):
    """In-place adds values to the state associated to the given actors.

    Args:
      actor_ids: 1D tensor with the list of actor IDs we want to add values to.
      values: A structure of tensors following the input spec, with an added
        first dimension that must either have the same size as 'actor_ids', or
        should not exist (in which case, the value is broadcasted to all actor
        ids).
    """
    tf.nest.assert_same_structure(values, self._state)
    for s, v in zip(tf.nest.flatten(self._state), tf.nest.flatten(values)):
      s.scatter_add(tf.IndexedSlices(v, actor_ids))

  @tf.Module.with_name_scope
  def read(self, actor_ids):
    """Reads the values corresponding to a list of actors.

    Args:
      actor_ids: 1D tensor with the list of actor IDs we want to read.

    Returns:
      A structure of tensors with the same shapes as the input specs. A
      dimension is added in front of each tensor, with size equal to the number
      of actor_ids provided.
    """
    return tf.nest.map_structure(lambda s: s.sparse_read(actor_ids),
                                 self._state)

  @tf.Module.with_name_scope
  def replace(self, actor_ids, values):
    """Replaces the state associated to the given actors.

    Args:
      actor_ids: 1D tensor with the list of actor IDs.
      values: A structure of tensors following the input spec, with an added
        first dimension that must either have the same size as 'actor_ids', or
        should not exist (in which case, the value is broadcasted to all actor
        ids).
    """
    tf.nest.assert_same_structure(values, self._state)
    for s, v in zip(tf.nest.flatten(self._state), tf.nest.flatten(values)):
      s.scatter_update(tf.IndexedSlices(v, actor_ids))


class StructuredFIFOQueue(tf.queue.FIFOQueue):
  """A tf.queue.FIFOQueue that supports nests and tf.TensorSpec."""

  def __init__(self,
               capacity,
               specs,
               shared_name=None,
               name='structured_fifo_queue'):
    self._specs = specs
    self._flattened_specs = tf.nest.flatten(specs)
    dtypes = [ts.dtype for ts in self._flattened_specs]
    shapes = [ts.shape for ts in self._flattened_specs]
    super(StructuredFIFOQueue, self).__init__(capacity, dtypes, shapes)

  def dequeue(self, name=None):
    result = super(StructuredFIFOQueue, self).dequeue(name=name)
    return tf.nest.pack_sequence_as(self._specs, result)

  def dequeue_many(self, batch_size, name=None):
    result = super(StructuredFIFOQueue, self).dequeue_many(
        batch_size, name=name)
    return tf.nest.pack_sequence_as(self._specs, result)

  def enqueue(self, vals, name=None):
    tf.nest.assert_same_structure(vals, self._specs)
    return super(StructuredFIFOQueue, self).enqueue(
        tf.nest.flatten(vals), name=name)

  def enqueue_many(self, vals, name=None):
    tf.nest.assert_same_structure(vals, self._specs)
    return super(StructuredFIFOQueue, self).enqueue_many(
        tf.nest.flatten(vals), name=name)


def batch_apply(fn, inputs):
  """Folds time into the batch dimension, runs fn() and unfolds the result.

  Args:
    fn: Function that takes as input the n tensors of the tf.nest structure,
      with shape [time*batch, <remaining shape>], and returns a tf.nest
      structure of batched tensors.
    inputs: tf.nest structure of n [time, batch, <remaining shape>] tensors.

  Returns:
    tf.nest structure of [time, batch, <fn output shape>]. Structure is
    determined by the output of fn.
  """
  time_to_batch_fn = lambda t: tf.reshape(t, [-1] + t.shape[2:].as_list())
  batched = tf.nest.map_structure(time_to_batch_fn, inputs)
  output = fn(*batched)
  prefix = [int(tf.nest.flatten(inputs)[0].shape[0]), -1]
  batch_to_time_fn = lambda t: tf.reshape(t, prefix + t.shape[1:].as_list())
  return tf.nest.map_structure(batch_to_time_fn, output)


def make_time_major(x):
  """Transposes the batch and time dimensions of a nest of Tensors.

  If an input tensor has rank < 2 it returns the original tensor. Retains as
  much of the static shape information as possible.

  Args:
    x: A nest of Tensors.

  Returns:
    x transposed along the first two dimensions.
  """

  def transpose(t):  
    t_static_shape = t.shape
    if t_static_shape.rank is not None and t_static_shape.rank < 2:
      return t

    t_rank = tf.rank(t)
    t_t = tf.transpose(t, tf.concat(([1, 0], tf.range(2, t_rank)), axis=0))
    t_t.set_shape(
        tf.TensorShape([t_static_shape[1],
                        t_static_shape[0]]).concatenate(t_static_shape[2:]))
    return t_t

  return tf.nest.map_structure(
      lambda t: tf.xla.experimental.compile(transpose, [t])[0], x)


class TPUEncodedUInt8Spec(tf.TypeSpec):
  """Type specification for composite tensor TPUEncodedUInt8."""

  def __init__(self, encoded_shape, original_shape):
    self._value_specs = (tf.TensorSpec(encoded_shape, tf.uint32),)
    self.original_shape = original_shape

  @property
  def _component_specs(self):
    return self._value_specs

  def _to_components(self, value):
    return (value.encoded,)

  def _from_components(self, components):
    return TPUEncodedUInt8(components[0], self.original_shape)

  def _serialize(self):
    return self._value_specs[0].shape, self.original_shape

  def _to_legacy_output_types(self):
    return self._value_specs[0].dtype

  def _to_legacy_output_shapes(self):
    return self._value_specs[0].shape

  @property
  def value_type(self):
    assert False


class TPUEncodedUInt8(composite_tensor.CompositeTensor):

  def __init__(self, encoded, shape):
    self.encoded = encoded
    self.original_shape = shape
    self._spec = TPUEncodedUInt8Spec(encoded.shape, tf.TensorShape(shape))

  @property
  def _type_spec(self):
    return self._spec


tensor_conversion_registry.register_tensor_conversion_function(
    TPUEncodedUInt8, lambda value, *unused_args, **unused_kwargs: value.encoded)


def tpu_encode(ts):
  """Encodes a nest of Tensors in a suitable way for TPUs.

  TPUs do not support tf.uint8, tf.uint16 and other data types. Furthermore,
  the speed of transfer and device reshapes depend on the shape of the data.
  This function tries to optimize the data encoding for a number of use cases.

  Should be used on CPU before sending data to TPU and in conjunction with
  `tpu_decode` after the data is transferred.

  Args:
    ts: A tf.nest of Tensors.

  Returns:
    A tf.nest of encoded Tensors.
  """

  def visit(t):  
    num_elements = t.shape.num_elements()
    # We need a multiple of 128 elements: encoding reduces the number of
    # elements by a factor 4 (packing uint8s into uint32s), and first thing
    # decode does is to reshape with a 32 minor-most dimension.
    if (t.dtype == tf.uint8 and num_elements is not None and
        num_elements % 128 == 0):
      # For details of these transformations, see b/137182262.
      x = tf.xla.experimental.compile(
          lambda x: tf.transpose(x, list(range(1, t.shape.rank)) + [0]), [t])[0]
      x = tf.reshape(x, [-1, 4])
      x = tf.bitcast(x, tf.uint32)
      x = tf.reshape(x, [-1])
      return TPUEncodedUInt8(x, t.shape)
    elif t.dtype == tf.uint8:
      logging.warning('Inefficient uint8 transfer with shape: %s', t.shape)
      return tf.cast(t, tf.bfloat16)
    elif t.dtype == tf.uint16:
      return tf.cast(t, tf.int32)
    else:
      return t

  return tf.nest.map_structure(visit, ts)


def tpu_decode(ts, structure=None):
  """Decodes a nest of Tensors encoded with tpu_encode.

  Args:
    ts: A nest of Tensors or TPUEncodedUInt8 composite tensors.
    structure: If not None, a nest of Tensors or TPUEncodedUInt8 composite
      tensors (possibly within PerReplica's) that are only used to recreate the
      structure of `ts` which then should be a list without composite tensors.

  Returns:
    A nest of decoded tensors packed as `structure` if available, otherwise
    packed as `ts`.
  """
  def visit(t, s):  
    s = s.primary if isinstance(s, values_lib.PerReplica) else s
    if isinstance(s, TPUEncodedUInt8):
      x = t.encoded if isinstance(t, TPUEncodedUInt8) else t
      x = tf.reshape(x, [-1, 32, 1])
      x = tf.broadcast_to(x, x.shape[:-1] + [4])
      x = tf.reshape(x, [-1, 128])
      x = tf.bitwise.bitwise_and(x, [0xFF, 0xFF00, 0xFF0000, 0xFF000000] * 32)
      x = tf.bitwise.right_shift(x, [0, 8, 16, 24] * 32)
      rank = s.original_shape.rank
      perm = [rank - 1] + list(range(rank - 1))
      inverted_shape = np.array(s.original_shape)[np.argsort(perm)]
      x = tf.reshape(x, inverted_shape)
      x = tf.transpose(x, perm)
      return x
    else:
      return t

  return tf.nest.map_structure(visit, ts, structure or ts)


def split_structure(structure, prefix_length):
  """Splits in two a tf.nest structure of tensors along the first axis."""
  flattened = tf.nest.flatten(structure)
  split = [tf.split(x, [prefix_length, tf.shape(x)[0] - prefix_length])
           for x in flattened]
  flattened_prefix = [pair[0] for pair in split]
  flattened_suffix = [pair[1] for pair in split]
  return (tf.nest.pack_sequence_as(structure, flattened_prefix),
          tf.nest.pack_sequence_as(structure, flattened_suffix))


@contextlib.contextmanager
def nullcontext(*args, **kwds):
  del args  # unused
  del kwds  # unused
  yield None
