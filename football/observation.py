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

"""Observation utilities for Google Research Football."""

import gym
import numpy as np
import tensorflow as tf


class PackedBitsObservation(gym.ObservationWrapper):
  """Wrapper that encodes a frame as packed bits instead of booleans.

  8x less to be transferred across the wire (16 booleans stored as uint16
  instead of 16 uint8) and 8x less to be transferred from CPU to TPU (16
  booleans stored as uint32 instead of 16 bfloat16).

  """

  def __init__(self, env):
    super(PackedBitsObservation, self).__init__(env)
    self.observation_space = gym.spaces.Box(
        low=0, high=np.iinfo(np.uint16).max,
        shape=env.observation_space.shape[:-1] + \
        ((env.observation_space.shape[-1] + 15) // 16,),
        dtype=np.uint16)

  def observation(self, observation):
    data = np.packbits(observation, axis=-1)  # This packs to uint8
    # Now we want to pack pairs of uint8 into uint16's.
    # We first need to ensure that the last dimention has even size.
    if data.shape[-1] % 2 == 1:
      data = np.pad(data, [(0, 0)] * (data.ndim - 1) + [(0, 1)], 'constant')
    return data.view(np.uint16)


def unpackbits(frame):
  def _(frame):
    # Unpack each uint16 into 16 bits
    bit_patterns = [
        2**7, 2**6, 2**5, 2**4, 2**3, 2**2, 2**1, 2**0, 2**15, 2**14, 2**13,
        2**12, 2**11, 2**10, 2**9, 2**8
    ]
    frame = tf.bitwise.bitwise_and(frame[..., tf.newaxis], bit_patterns)
    frame = tf.cast(tf.cast(frame, tf.bool), tf.float32) * 255
    # Reshape to the right size.
    frame = tf.reshape(frame, frame.shape[:-2] + \
                             (frame.shape[-2] * frame.shape[-1],))
    return frame
  if tf.test.is_gpu_available():
    return tf.xla.experimental.compile(_, [frame])[0]
  return _(frame)

