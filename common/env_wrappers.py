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

"""Environment wrappers."""

from absl import flags

import gym
from matplotlib import pyplot as plt
import numpy as np

FLAGS = flags.FLAGS


class TFAgents2GymWrapper(gym.Env):
  """Transform TFAgents environment into an OpenAI Gym environment."""

  def __init__(self, env):
    self.env = env
    self.observation_space = env.observation_space
    self.action_space = env.action_space

  def step(self, action):
    assert self.action_space.contains(action)
    env_output = self.env.step(action)
    reward = env_output.reward
    done = env_output.is_last()
    try:
      info = self.env.get_info()
    except NotImplementedError:
      info = {}
    return env_output.observation, reward, done, info

  def reset(self):
    return self.env.reset().observation

  def render(self):
    frame = self.env.render()
    plt.figure(1)
    plt.clf()
    plt.imshow(frame)
    plt.pause(0.001)


class UniformBoundActionSpaceWrapper(gym.Env):
  """Rescale actions so that action space bounds are [-1, 1]."""

  def __init__(self, env):
    """Initialize the wrapper.

    Args:
      env: Environment to be wrapped. It must have an action space of type
        gym.spaces.Box.
    """
    self.env = env
    self.observation_space = env.observation_space
    assert isinstance(env.action_space, gym.spaces.Box)
    assert env.action_space.dtype == np.float32
    n_action_dim = env.action_space.shape[0]
    self.low = env.action_space.low
    self.high = env.action_space.high
    self.center = (self.low + self.high) / 2.
    self.action_space = gym.spaces.Box(low=-np.ones(n_action_dim),
                                       high=np.ones(n_action_dim),
                                       dtype=np.float32)

  def step(self, action):
    assert np.abs(action).max() < 1.00001, 'Action: %s' % action
    action = np.clip(action, -1, 1)
    assert self.action_space.contains(action)
    action = self.center + action * (self.high - self.center)
    assert self.env.action_space.contains(action)
    obs, rew, done, info = self.env.step(action)
    return obs, rew, done, info

  def reset(self):
    return self.env.reset()

  def render(self, *args, **kwargs):
    return self.env.render(*args, **kwargs)


class DiscretizeEnvWrapper(gym.Env):
  """Wrapper for discretizing actions."""

  def __init__(self, env, n_actions_per_dim, discretization='lin',
               action_ratio=None):
    """"Discretize actions.

    Args:
      env: Environment to be wrapped.
      n_actions_per_dim: The number of buckets per action dimension.
      discretization: Discretization mode, can be 'lin' or 'log',
        'lin' spaces buckets linearly between low and high while 'log'
        spaces them logarithmically.
      action_ratio: The ratio of the highest and lowest positive action
        for logarithim discretization.
    """

    self.env = env
    assert len(env.action_space.shape) == 1
    dim_action = env.action_space.shape[0]
    self.action_space = gym.spaces.MultiDiscrete([n_actions_per_dim] *
                                                 dim_action)
    self.observation_space = env.observation_space
    high = env.action_space.high
    if isinstance(high, float):
      assert env.action_space.low == -high
    else:
      high = high[0]
      assert (env.action_space.high == [high] * dim_action).all()
      assert (env.action_space.low == -env.action_space.high).all()
    if discretization == 'log':
      assert n_actions_per_dim % 2 == 1, (
          'The number of actions per dimension '
          'has to be odd for logarithmic discretization.')
      assert action_ratio is not None
      log_range = np.linspace(np.log(high / action_ratio),
                              np.log(high),
                              n_actions_per_dim // 2)
      self.action_set = np.concatenate([-np.exp(np.flip(log_range)),
                                        [0.],
                                        np.exp(log_range)])
    elif discretization == 'lin':
      self.action_set = np.linspace(-high, high, n_actions_per_dim)

  def step(self, action):
    assert self.action_space.contains(action)
    action = np.take(self.action_set, action)
    assert self.env.action_space.contains(action)
    obs, rew, done, info = self.env.step(action)
    return obs, rew, done, info

  def reset(self):
    return self.env.reset()

  def render(self, *args, **kwargs):
    return self.env.render(*args, **kwargs)
