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

"""Toy environments for sanity testing algorithms."""

import collections
import gym
import numpy as np
import tensorflow as tf


class ToyEnv(gym.Env):
  """Environment in which we need to output observations."""

  def __init__(self, horizon=3, n_actions=3):
    """Initialize environment.

    Args:
      horizon: number of timesteps the observations have to be remembered.
      n_actions: dimensionality of actions.
    """
    self.horizon = horizon
    self.observation_space = gym.spaces.Box(-np.inf, np.inf, [n_actions+1])
    self.action_space = gym.spaces.Box(-1, 1, [n_actions])

  def _get_obs(self):
    self._obs = np.random.uniform(
        -1, 1, size=self.observation_space.shape[0] - 1).astype(np.float32)
    return np.concatenate([self._obs, [0.]], axis=0).astype(np.float32)

  def step(self, action):
    assert self.action_space.contains(np.clip(action, -1, 1))
    self.t += 1
    reward = -float(sum((action-self._obs)**2))
    return self._get_obs(), reward, self.t >= self.horizon, None

  def reset(self):
    self.t = 0
    return self._get_obs()

  def render(self):
    pass


class ToyMemoryEnv(gym.Env):
  """Environment in which we need to output observations from previous steps."""

  def __init__(self, horizon=3, n_actions=3):
    """Initialize environment.

    Args:
      horizon: number of timesteps we need to retain observations in memory.
      n_actions: dimensionality of actions.
    """
    self.horizon = horizon
    self.n_actions = n_actions
    self.observation_space = gym.spaces.Box(-np.inf, np.inf, [n_actions+1])
    self.action_space = gym.spaces.Box(-1, 1, [n_actions])

  def _get_obs(self):
    if self.t < self.horizon:
      return np.concatenate([self.memory[self.t], [0.]],
                            axis=0).astype(np.float32)
    else:
      return np.zeros(self.n_actions+1, dtype=np.float32)

  def step(self, action):
    assert self.action_space.contains(action)
    if self.t == 2*self.horizon:
      return np.zeros(self.n_actions+1), 0., True, None
    if self.t < self.horizon:
      reward = float(0.)
    else:
      reward = -float(sum((action-self.memory[self.t - self.horizon])**2))
    self.t += 1
    return self._get_obs(), reward, False, None

  def reset(self):
    self.memory = np.random.uniform(
        -1, 1, size=(self.horizon, self.n_actions)).astype(np.float32)
    self.t = 0
    return self._get_obs()

  def render(self):
    pass


class BitFlippingEnv(gym.GoalEnv):
  """Goal-based environment in which actions correspond to switching bits.

  Based on https://arxiv.org/pdf/1707.01495.pdf.
  """

  def __init__(self, n_bits=10, horizon=20):
    self._n_bits = n_bits
    self._horizon = horizon
    self.observation_space = gym.spaces.Dict(
        achieved_goal=gym.spaces.Box(low=0, high=1, shape=[n_bits]),
        desired_goal=gym.spaces.Box(low=0, high=1, shape=[n_bits]),
        observation=gym.spaces.Box(low=0, high=1, shape=[horizon + 1]))
    self.action_space = gym.spaces.Discrete(n_bits + 1)

  def reset(self):
    self.state = np.random.randint(2, size=self._n_bits).astype(np.float32)
    self.goal = np.random.randint(2, size=self._n_bits).astype(np.float32)
    self.t = 0
    return self._get_obs()

  def _get_obs(self):
    obs = {'achieved_goal': self.state.copy(),
           'desired_goal': self.goal.copy(),
           'observation': tf.one_hot(self.t, self._horizon + 1).numpy()}
    assert self.observation_space.contains(obs)
    return collections.OrderedDict(sorted(obs.items()))

  def step(self, action):
    assert self.action_space.contains(action)
    if action != self._n_bits:
      self.state[action] = 1 - self.state[action]
    self.t += 1
    reward = self.compute_reward(self.state, self.goal)
    return self._get_obs(), reward, self.t >= self._horizon, {}

  def compute_reward(self, achieved_goal, desired_goal, info=None):
    return tf.clip_by_value(tf.reduce_sum(
        -tf.cast(achieved_goal != desired_goal, tf.float32), axis=-1), -1, 0)
