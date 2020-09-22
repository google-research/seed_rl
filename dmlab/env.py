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

"""DeepMind Lab Gym wrapper."""

import hashlib
import os

from absl import flags
from absl import logging

import gym
import numpy as np
from seed_rl.common import common_flags  
from seed_rl.dmlab import games
import tensorflow as tf

import deepmind_lab

FLAGS = flags.FLAGS

flags.DEFINE_string('homepath', '', 'Labyrinth homepath.')
flags.DEFINE_string(
    'dataset_path', '', 'Path to dataset needed for psychlab_*, see '
    'https://github.com/deepmind/lab/tree/master/data/brady_konkle_oliva2008')

flags.DEFINE_string('game', 'explore_goal_locations_small', 'Game/level name.')
flags.DEFINE_integer('width', 96, 'Width of observation.')
flags.DEFINE_integer('height', 72, 'Height of observation.')
flags.DEFINE_string('level_cache_dir', None, 'Global level cache directory.')


DEFAULT_ACTION_SET = (
    (0, 0, 0, 1, 0, 0, 0),    # Forward
    (0, 0, 0, -1, 0, 0, 0),   # Backward
    (0, 0, -1, 0, 0, 0, 0),   # Strafe Left
    (0, 0, 1, 0, 0, 0, 0),    # Strafe Right
    (-20, 0, 0, 0, 0, 0, 0),  # Look Left
    (20, 0, 0, 0, 0, 0, 0),   # Look Right
    (-20, 0, 0, 1, 0, 0, 0),  # Look Left + Forward
    (20, 0, 0, 1, 0, 0, 0),   # Look Right + Forward
    (0, 0, 0, 0, 1, 0, 0),    # Fire.
)


class LevelCache(object):
  """Level cache."""

  def __init__(self, cache_dir):
    self._cache_dir = cache_dir

  def get_path(self, key):
    key = hashlib.md5(key.encode('utf-8')).hexdigest()
    dir_, filename = key[:3], key[3:]
    return os.path.join(self._cache_dir, dir_, filename)

  def fetch(self, key, pk3_path):
    path = self.get_path(key)
    try:
      tf.io.gfile.copy(path, pk3_path, overwrite=True)
      return True
    except tf.errors.OpError:
      return False

  def write(self, key, pk3_path):
    path = self.get_path(key)
    if not tf.io.gfile.exists(path):
      tf.io.gfile.makedirs(os.path.dirname(path))
      tf.io.gfile.copy(pk3_path, path)


class DmLab(gym.Env):
  """DeepMind Lab wrapper."""

  def __init__(self, game, num_action_repeats, seed, is_test, config,
               action_set=DEFAULT_ACTION_SET, level_cache_dir=None):
    if is_test:
      config['allowHoldOutLevels'] = 'true'
      # Mixer seed for evalution, see
      # https://github.com/deepmind/lab/blob/master/docs/users/python_api.md
      config['mixerSeed'] = 0x600D5EED

    if game in games.ALL_GAMES:
      game = 'contributed/dmlab30/' + game

    config['datasetPath'] = FLAGS.dataset_path

    self._num_action_repeats = num_action_repeats
    self._random_state = np.random.RandomState(seed=seed)
    if FLAGS.homepath:
      deepmind_lab.set_runfiles_path(FLAGS.homepath)
    self._env = deepmind_lab.Lab(
        level=game,
        observations=['RGB_INTERLEAVED'],
        level_cache=LevelCache(level_cache_dir) if level_cache_dir else None,
        config={k: str(v) for k, v in config.items()},
    )
    self._action_set = action_set
    self.action_space = gym.spaces.Discrete(len(self._action_set))
    self.observation_space = gym.spaces.Box(
        low=0,
        high=255,
        shape=(config['height'], config['width'], 3),
        dtype=np.uint8)

  def _observation(self):
    return self._env.observations()['RGB_INTERLEAVED']

  def reset(self):
    self._env.reset(seed=self._random_state.randint(0, 2 ** 31 - 1))
    return self._observation()

  def step(self, action):
    raw_action = np.array(self._action_set[action], np.intc)
    reward = self._env.step(raw_action, num_steps=self._num_action_repeats)
    done = not self._env.is_running()
    observation = None if done else self._observation()
    return observation, reward, done, {}

  def close(self):
    self._env.close()


def create_environment(task, config):
  logging.info('Creating environment: %s', config.game)
  return DmLab(
      config.game,
      config.num_action_repeats,
      seed=task + 1,
      is_test=False,
      level_cache_dir=config.level_cache_dir,
      config={
          'width': config.width,
          'height': config.height,
          'logLevel': 'WARN',
      })
