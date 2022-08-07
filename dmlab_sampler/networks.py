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

"""SEED agent using Keras."""

import collections
import utils
import tensorflow as tf


AgentOutput = collections.namedtuple('AgentOutput',
                                     'action policy_logits baseline')


class _Stack(tf.Module):
  """Stack of pooling and convolutional blocks with residual connections."""

  def __init__(self, num_ch, num_blocks):
    super(_Stack, self).__init__(name='stack')
    self._conv = tf.keras.layers.Conv2D(num_ch, 3, strides=1, padding='same')
    self._max_pool = tf.keras.layers.MaxPool2D(
        pool_size=3, padding='same', strides=2)

    self._res_convs0 = [
        tf.keras.layers.Conv2D(
            num_ch, 3, strides=1, padding='same', name='res_%d/conv2d_0' % i)
        for i in range(num_blocks)
    ]
    self._res_convs1 = [
        tf.keras.layers.Conv2D(
            num_ch, 3, strides=1, padding='same', name='res_%d/conv2d_1' % i)
        for i in range(num_blocks)
    ]

  def __call__(self, conv_out):
    # Downscale.
    conv_out = self._conv(conv_out)
    conv_out = self._max_pool(conv_out)

    # Residual block(s).
    for (res_conv0, res_conv1) in zip(self._res_convs0, self._res_convs1):
      block_input = conv_out
      conv_out = tf.nn.relu(conv_out)
      conv_out = res_conv0(conv_out)
      conv_out = tf.nn.relu(conv_out)
      conv_out = res_conv1(conv_out)
      conv_out += block_input

    return conv_out


class ImpalaDeep(tf.Module):
  """Agent with ResNet.

  The deep model in
  "IMPALA: Scalable Distributed Deep-RL with
  Importance Weighted Actor-Learner Architectures"
  by Espeholt, Soyer, Munos et al.
  """

  def __init__(self, num_actions):
    super(ImpalaDeep, self).__init__(name='impala_deep')

    # Parameters and layers for unroll.
    self._num_actions = num_actions
    self._core = tf.keras.layers.LSTMCell(256)

    # Parameters and layers for _torso.
    self._stacks = [
        _Stack(num_ch, num_blocks)
        for num_ch, num_blocks in [(16, 2), (32, 2), (32, 2)]
    ]
    self._conv_to_linear = tf.keras.layers.Dense(256)

    # Layers for _head.
    self._policy_logits = tf.keras.layers.Dense(
        self._num_actions, name='policy_logits')
    self._baseline = tf.keras.layers.Dense(1, name='baseline')

  def initial_state(self, batch_size):
    return self._core.get_initial_state(batch_size=batch_size, dtype=tf.float32)

  def _torso(self, prev_action, env_output):
    reward, _, frame, _, _ = env_output

    # Convert to floats.
    frame = tf.cast(frame, tf.float32)

    frame /= 255
    conv_out = frame
    for stack in self._stacks:
      conv_out = stack(conv_out)

    conv_out = tf.nn.relu(conv_out)
    conv_out = tf.keras.layers.Flatten()(conv_out)

    conv_out = self._conv_to_linear(conv_out)
    conv_out = tf.nn.relu(conv_out)

    # Append clipped last reward and one hot last action.
    clipped_reward = tf.expand_dims(tf.clip_by_value(reward, -1, 1), -1)
    one_hot_prev_action = tf.one_hot(prev_action, self._num_actions)
    return tf.concat([conv_out, clipped_reward, one_hot_prev_action], axis=1)

  def _head(self, core_output):
    policy_logits = self._policy_logits(core_output)
    baseline = tf.squeeze(self._baseline(core_output), axis=-1)

    # Sample an action from the policy.
    new_action = tf.random.categorical(policy_logits, 1, dtype=tf.int64)
    new_action = tf.squeeze(new_action, 1, name='action')

    return AgentOutput(new_action, policy_logits, baseline)

  # Not clear why, but if "@tf.function" declarator is placed directly onto
  # __call__, training fails with "uninitialized variable *baseline".
  # when running on multiple learning tpu cores.


  @tf.function
  def get_action(self, *args, **kwargs):
    return self.__call__(*args, **kwargs)

  def __call__(self,
               prev_actions,
               env_outputs,
               core_state,
               unroll=False,
               is_training=False):
    if not unroll:
      # Add time dimension.
      prev_actions, env_outputs = tf.nest.map_structure(
          lambda t: tf.expand_dims(t, 0), (prev_actions, env_outputs))
    outputs, core_state = self._unroll(prev_actions, env_outputs, core_state)
    if not unroll:
      # Remove time dimension.
      outputs = tf.nest.map_structure(lambda t: tf.squeeze(t, 0), outputs)

    return outputs, core_state

  def _unroll(self, prev_actions, env_outputs, core_state):
    unused_reward, done, unused_observation, _, _ = env_outputs

    torso_outputs = utils.batch_apply(self._torso, (prev_actions, env_outputs))

    initial_core_state = self._core.get_initial_state(
        batch_size=tf.shape(prev_actions)[1], dtype=tf.float32)
    core_output_list = []
    for input_, d in zip(tf.unstack(torso_outputs), tf.unstack(done)):
      # If the episode ended, the core state should be reset before the next.
      core_state = tf.nest.map_structure(
          lambda x, y, d=d: tf.where(  
              tf.reshape(d, [d.shape[0]] + [1] * (x.shape.rank - 1)), x, y),
          initial_core_state,
          core_state)
      core_output, core_state = self._core(input_, core_state)
      core_output_list.append(core_output)
    core_outputs = tf.stack(core_output_list)

    return utils.batch_apply(self._head, (core_outputs,)), core_state
