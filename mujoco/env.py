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

"""Mujoco environment from OpenAI gym."""

from absl import flags

import gym
from gym import spaces
import numpy as np
from seed_rl.common import common_flags  
from seed_rl.common import env_wrappers
from seed_rl.mujoco import toy_env

FLAGS = flags.FLAGS


class SinglePrecisionWrapper(gym.Wrapper):
  """Single precision Wrapper for Mujoco environments."""

  def __init__(self, env):
    """Initialize the wrapper.

    Args:
      env: MujocoEnv to be wrapped.
    """
    super().__init__(env)
    self.observation_space = spaces.Box(
        self.observation_space.low,
        self.observation_space.high,
        dtype=np.float32)
    self.num_steps = 0
    self.max_episode_steps = self.env.spec.max_episode_steps

  def reset(self):
    self.num_steps = 0
    return self.env.reset().astype(np.float32)

  def step(self, action):
    self.num_steps += 1
    obs, reward, done, info = self.env.step(action)
    if self.num_steps >= self.max_episode_steps:
      done = True
    if isinstance(reward, np.ndarray):
      reward = reward.astype(np.float32)
    else:
      reward = float(reward)
    return obs.astype(np.float32), reward, done, info


def create_environment(env_name,
                       discretization='none',
                       n_actions_per_dim=11,
                       action_ratio=30.,
                       gym_kwargs=None):
  """Create environment from OpenAI Gym.

  Actions are rescaled to the range [-1, 1] and optionally discretized.

  Args:
    env_name: environment name from OpenAI Gym. You can also use 'toy_env' or
      'toy_memory_env' to get very simple environments which can be used for
      sanity testing RL algorithms.
    discretization: 'none', 'lin' or 'log'. Values other than 'none' cause
      action coordinates to be discretized into n_actions_per_dim buckets.
      Buckets are spaced linearly between the bounds if 'lin' mode is used and
      logarithmically for 'log' mode.
    n_actions_per_dim: the number of buckets per action coordinate if
      discretization is used.
    action_ratio: the ratio between the highest and the lowest positive action
      for logarithmic action discretization.
    gym_kwargs: Kwargs to pass to the gym environment contructor.

  Returns:
    wrapped environment
  """

  assert FLAGS.num_action_repeats == 1, 'Only action repeat of 1 is supported.'

  if env_name == 'toy_env':
    env = toy_env.ToyEnv()
  elif env_name == 'toy_memory_env':
    env = toy_env.ToyMemoryEnv()
  elif env_name == 'bit_flip':
    return toy_env.BitFlippingEnv()
  else:  # mujoco
    gym_kwargs = gym_kwargs if gym_kwargs else {}
    gym_spec = gym.spec(env_name)
    env = gym_spec.make(**gym_kwargs)
    env = SinglePrecisionWrapper(env)

  # rescale actions so that all bounds are [-1, 1]
  env = env_wrappers.UniformBoundActionSpaceWrapper(env)
  # optionally discretize actions
  if discretization != 'none':
    env = env_wrappers.DiscretizeEnvWrapper(env, n_actions_per_dim,
                                            discretization, action_ratio)

  return env
