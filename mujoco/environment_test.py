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

"""Environment test."""

from absl import flags
from seed_rl.mujoco import env
import tensorflow as tf

FLAGS = flags.FLAGS


class EnvironmentTest(tf.test.TestCase):

  def run_environment(self, environment):
    environment.reset()
    for _ in range(100):
      _, _, done, _ = environment.step(environment.action_space.sample())
      if done:
        environment.reset()

  def test_mujoco_env(self):
    for discretization in ['none', 'log', 'lin']:
      self.run_environment(
          env.create_environment(
              'HalfCheetah-v2', discretization=discretization))

  def test_toy_envs(self):
    self.run_environment(env.create_environment('toy_env'))
    self.run_environment(env.create_environment('toy_memory_env'))


if __name__ == '__main__':
  tf.test.main()
