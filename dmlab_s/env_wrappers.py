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
import numpy as np
import tensorflow as tf

FLAGS = flags.FLAGS


def spec_to_box(spec):
  minimum, maximum = -np.inf, np.inf
  if hasattr(spec, 'minimum'):
    if not hasattr(spec, 'maximum'):
      raise ValueError('spec has minimum but no maximum: {}'.format(spec))
    minimum = np.array(spec.minimum, np.float32)
    maximum = np.array(spec.maximum, np.float32)
    return gym.spaces.Box(minimum, maximum)

  return gym.spaces.Box(-np.inf, np.inf, shape=spec.shape)


def flatten_and_concatenate_obs(obs_dict):
  return np.concatenate(
      [obs.astype(np.float32).flatten() for obs in obs_dict.values()])




class UniformBoundActionSpaceWrapper(gym.Wrapper):
  """Rescale actions so that action space bounds are [-1, 1]."""

  def __init__(self, env):
    """Initialize the wrapper.

    Args:
      env: Environment to be wrapped. It must have an action space of type
        gym.spaces.Box.
    """
    super().__init__(env)
    assert isinstance(env.action_space, gym.spaces.Box)
    assert env.action_space.dtype == np.float32
    n_action_dim = env.action_space.shape[0]
    self.half_range = (env.action_space.high - env.action_space.low) / 2.
    self.center = env.action_space.low + self.half_range
    self.action_space = gym.spaces.Box(low=-np.ones(n_action_dim),
                                       high=np.ones(n_action_dim),
                                       dtype=np.float32)

  def step(self, action):
    assert np.abs(action).max() < 1.00001, 'Action: %s' % action
    action = np.clip(action, -1, 1)
    action = self.center + action * self.half_range
    return self.env.step(action)


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


class BatchedEnvironment:
  """A wrapper that batches several environment instances."""

  def __init__(self, create_env_fn, batch_size, id_offset, config):
    """Initialize the wrapper.

    Args:
      create_env_fn: A function to create environment instances.
      batch_size: The number of environment instances to create.
      id_offset: The offset for environment ids. Environments receive sequential
        ids starting from this offset.
      config: Config object defining configuration of the environment
    """
    self._batch_size = batch_size
    # Note: some environments require an argument to be of a native Python
    # numeric type. If we create env_ids as a numpy array, its elements will
    # be of type np.int32. So we create it as a plain Python array first.
    env_ids = [id_offset + i for i in range(batch_size)]
    self._envs = [create_env_fn(id, config) for id in env_ids]
    self._env_ids = np.array(env_ids, np.int32)
    self._obs = None

  @property
  def env_ids(self):
    return self._env_ids

  @property
  def envs(self):
    return self._envs

  @property
  def _mapped_obs(self):
    """Maps observations to preserve the original structure.

    This is needed to support environments that return structured observations.
    For example, gym.GoalEnv has `observation`, `desired_goal`, and
    `achieved_goal` elements in its observations. In this case the batched
    observations would contain the same three elements batched by element.

    Returns:
      Mapped observations.
    """
    return tf.nest.map_structure(lambda *args: np.array(args), *self._obs)

  def step(self, action_batch):
    """Does one step for all batched environments sequentially."""
    num_envs = self._batch_size
    rewards = np.zeros(num_envs, np.float32)
    dones = np.zeros(num_envs, np.bool)
    infos = [None] * num_envs
    for i in range(num_envs):
      self._obs[i], rewards[i], dones[i], infos[i] = self._envs[i].step(
          action_batch[i])
    return self._mapped_obs, rewards, dones, infos

  def reset(self):
    """Reset all environments."""
    observations = [env.reset() for env in self._envs]
    self._obs = observations
    return self._mapped_obs

  def reset_if_done(self, done):
    """Reset the environments for which 'done' is True.

    Args:
      done: An array that specifies which environments are 'done', meaning their
        episode is terminated.

    Returns:
      Observations for all environments.
    """
    assert self._obs is not None, 'reset_if_done() called before reset()'
    for i in range(len(self._envs)):
      if done[i]:
        self._obs[i] = self.envs[i].reset()

    return self._mapped_obs

  def render(self, mode='human', **kwargs):
    # Render only the first one
    self._envs[0].render(mode, **kwargs)

  def close(self):
    for env in self._envs:
      env.close()
