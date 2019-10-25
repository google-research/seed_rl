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

"""Tests for observation.py."""

from absl.testing import parameterized
import gym
import numpy as np
from seed_rl.football import observation
from seed_rl.utils import utils
import tensorflow as tf


class UtilsTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.parameters(
      (False),
      (True),
  )
  def test_packed_bits(self, stacked):
    env = gym.make(
        'gfootball:GFootball-11_vs_11_easy_stochastic-SMM-v0', stacked=stacked)
    env.reset()
    for _ in range(10):
      obs, _, done, _ = env.step(env.action_space.sample())

      baseline_obs = tf.cast(np.array(obs), tf.float32)

      packed_obs = observation.PackedBitsObservation.observation(env, obs)
      packed_obs = tf.convert_to_tensor(packed_obs)
      tpu_obs = observation.unpackbits(utils.tpu_encode(packed_obs))
      non_tpu_obs = observation.unpackbits(packed_obs)
      # baseline_obs has less than 16 channels, so first channels should
      # correspond to baseline_obs and then all the rest should be 0
      self.assertAllEqual(baseline_obs, tpu_obs[..., :obs.shape[-1]])
      self.assertAllEqual(baseline_obs, non_tpu_obs[..., :obs.shape[-1]])
      self.assertAllEqual(tf.math.reduce_sum(tpu_obs[..., obs.shape[-1]:]), 0)
      self.assertAllEqual(tf.math.reduce_sum(non_tpu_obs[..., obs.shape[-1]:]),
                          0)

      if done:
        env.reset()
    env.close()


if __name__ == '__main__':
  tf.test.main()
