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


"""Utilities to normalize a stream of tensors (typically observations)."""

import tensorflow as tf


class Normalizer(tf.Module):
  """Normalizes tensors by tracking their element-wise mean and variance."""

  def __init__(self, eps=0.001, clip_range=(-5, 5)):
    """Initialize the normalizer.

    Args:
      eps: A constant added to the standard deviation of data before
        normalization.
      clip_range: Normalized values are clipped to this range. [-5, 5] is the
        default.
    """
    super(Normalizer, self).__init__()
    self.eps = eps
    self.clip_range = clip_range
    self.initialized = False

  def build(self, input_shape):
    size = input_shape[-1]
    def get_variable(name, initial_value, local):  
      if local:
        # ON_READ causes the replicated variable to act as independent variables
        # for each replica. The variable only gets aggregated if it is read
        # in cross-replica context, which may happen e.g. when the normalizer
        # is checkpointed.
        return tf.Variable(name=name,
                           initial_value=initial_value,
                           trainable=False,
                           dtype=tf.float32,
                           synchronization=tf.VariableSynchronization.ON_READ,
                           aggregation=tf.VariableAggregation.MEAN)
      else:  # mirrored variable, same value on each replica
        return tf.Variable(name=name,
                           initial_value=initial_value,
                           trainable=False,
                           dtype=tf.float32)
    # local accumulators
    self.steps_acc = get_variable('steps_acc', 0, local=True)
    self.sum_acc = get_variable('sum_acc', tf.zeros(shape=[size]), local=True)
    self.sumsq_acc = get_variable('sumsq_acc', tf.zeros(shape=[size]),
                                  local=True)
    # mirrored variables
    self.steps = get_variable('steps', 0, local=False)
    self.sum = get_variable('sum', tf.zeros(shape=[size]), local=False)
    self.sumsq = get_variable('sumsq', tf.zeros(shape=[size]), local=False)
    self.mean = get_variable('mean', tf.zeros(shape=[size]), local=False)
    self.std = get_variable('std', tf.zeros(shape=[size]), local=False)
    self.initialized = True

  def update(self, input_, only_accumulate=False):
    """Update normalization statistics.

    Args:
      input_: A tensor. All dimensions apart from the last one are treated
        as batch dimensions.
      only_accumulate: If True, only local accumulators are updated and the
        normalization is not affected. Use this option if running on TPU.
        In this case, you need to call `finish_update` method in cross-replica
        context later to update the normalization.
    """
    if not self.initialized:
      self.build(input_.shape)

    # reshape to 2 dimensions
    shape = input_.shape
    input_ = tf.reshape(input_, [tf.reduce_prod(shape[:-1]), shape[-1]])
    assert len(input_.shape) == 2

    # update local accumulators
    self.steps_acc.assign_add(float(input_.shape[0]))
    self.sum_acc.assign_add(tf.reduce_sum(input_, axis=0))
    self.sumsq_acc.assign_add(tf.reduce_sum(tf.square(input_), axis=0))

    if not only_accumulate:
      self.finish_update()

  def finish_update(self):
    """Update the normalization (mean and std) based on local accumulators.

    You only need to call this method manually if `update` was called with
    `only_accumulate=True` (usually on a TPU). This method needs to be called
    in cross-replica context (i.e. not inside experimental_run_v2).
    """
    # sum the accumulators accross all replicas
    step_increment, sum_increment, sumsq_increment = (
        tf.distribute.get_replica_context().all_reduce(
            tf.distribute.ReduceOp.SUM,
            [self.steps_acc, self.sum_acc, self.sumsq_acc]))

    # zero the accumulators
    self.steps_acc.assign(tf.zeros_like(self.steps_acc))
    self.sum_acc.assign(tf.zeros_like(self.sum_acc))
    self.sumsq_acc.assign(tf.zeros_like(self.sumsq_acc))

    # update the normalization
    self.steps.assign_add(step_increment)
    self.sum.assign_add(sum_increment)
    self.sumsq.assign_add(sumsq_increment)
    self.mean.assign(self.sum / self.steps)
    self.std.assign(tf.sqrt((self.sumsq / self.steps) - tf.square(self.mean)))

  def __call__(self, input_):
    """Normalize the tensor.

    Args:
      input_: tensor to be normalizer. All dimensions apart from the last one
        are treated as batch dimensions.

    Returns:
      a tensor of the same shape and dtype as input_.
    """
    if not self.initialized:
      self.build(input_.shape)
    # reshape to 2 dimensions
    shape = input_.shape
    input_ = tf.reshape(input_, [tf.reduce_prod(shape[:-1]), shape[-1]])
    assert len(input_.shape) == 2
    # normalize
    input_ -= self.mean[tf.newaxis, :]
    input_ /= self.std[tf.newaxis, :] + self.eps
    input_ = tf.clip_by_value(input_, *self.clip_range)
    # reshape to the original shape
    return tf.reshape(input_, shape)

  def get_logs(self):
    logs = dict()
    for key, var in [('mean', self.mean), ('std', self.std)]:
      for i in range(var.shape[0]):
        logs['%s/%d' % (key, i)] = var[i]
    return logs
