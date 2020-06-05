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

"""A class implementing minimal Atari 2600 preprocessing.

Adapted from Dopamine.
"""

from gym.spaces.box import Box
import numpy as np

import cv2


class UnityPreprocessing(object):
  """A class implementing image preprocessing for Atari 2600 agents.

  Specifically, this provides the following subset from the JAIR paper
  (Bellemare et al., 2013) and Nature DQN paper (Mnih et al., 2015):

    * Terminal signal when a life is lost (off by default).
    * Downsample the screen to a square image (defaults to 84x84).

  More generally, this class follows the preprocessing guidelines set down in
  Machado et al. (2018), "Revisiting the Arcade Learning Environment:
  Evaluation Protocols and Open Problems for General Agents".

  It also provides random starting no-ops, which are used in the Rainbow, Apex
  and R2D2 papers.
  """

  def __init__(self, environment, terminal_on_life_loss=False,
               screen_size=84, max_random_noops=0):
    """Constructor for Unity preprocessor.

    Args:
      environment: Gym environment whose observations are preprocessed.
      terminal_on_life_loss: bool, If True, the step() method returns
        is_terminal=True whenever a life is lost. See Mnih et al. 2015.
      screen_size: int, size of a resized Atari 2600 frame.
      max_random_noops: int, maximum number of no-ops to apply at the beginning
        of each episode to reduce determinism. These no-ops are applied at a
        low-level, before frame skipping.

    Raises:
    """
    if screen_size <= 0:
      raise ValueError('Target screen size should be strictly positive, got {}'.
                       format(screen_size))

    self.environment = environment
    self.terminal_on_life_loss = terminal_on_life_loss
    self.max_random_noops = max_random_noops

    self.obs_dims = self.environment.observation_space
    self.screen_size = environment._get_vis_obs_shape()
    print(self.obs_dims)
    # Stores temporary observations used for pooling over two successive
    # frames.
    self.screen_buffer = [
        np.empty((self.obs_dims.shape[0], self.obs_dims.shape[1], self.obs_dims.shape[2]), dtype=np.uint8),
        np.empty((self.obs_dims.shape[0], self.obs_dims.shape[1], self.obs_dims.shape[2]), dtype=np.uint8)
    ]

    self.game_over = False
    self.lives = 0  # Will need to be set by reset().

  @property
  def observation_space(self):
    # Return the observation space adjusted to match the shape of the processed
    # observations.
    return Box(low=0, high=255, shape=(self.obs_dims.shape[0], self.obs_dims.shape[1], self.obs_dims.shape[2]),
               dtype=np.uint8)

  @property
  def action_space(self):
    return self.environment.action_space

  @property
  def reward_range(self):
    return self.environment.reward_range

  @property
  def metadata(self):
    return self.environment.metadata

  def close(self):
    return self.environment.close()

  def apply_random_noops(self):
    """Steps self.environment with random no-ops."""
    if self.max_random_noops <= 0:
      return
    # Other no-ops implementations actually always do at least 1 no-op. We
    # follow them.
    no_ops = np.random.randint(1, self.max_random_noops + 1)
    for _ in range(no_ops):
      _, _, game_over, _ = self.environment.step(0)
      if game_over:
        self.environment.reset()

  def reset(self):
    """Resets the environment.

    Returns:
      observation: numpy array, the initial observation emitted by the
        environment.
    """
    self.environment.reset()
    self.apply_random_noops()

    self.lives = 1
    return self.environment.reset()

  def render(self, mode):
    """Renders the current screen, before preprocessing.

    This calls the Gym API's render() method.

    Args:
      mode: Mode argument for the environment's render() method.
        Valid values (str) are:
          'rgb_array': returns the raw ALE image.
          'human': renders to display via the Gym renderer.

    Returns:
      if mode='rgb_array': numpy array, the most recent screen.
      if mode='human': bool, whether the rendering was successful.
    """
    return self.environment.render(mode)

  def step(self, action):
    """Applies the given action in the environment.

    Remarks:

    Args:
      action: The action to be executed.

    Returns:
      observation: numpy array, the observation following the action.
      reward: float, the reward following the action.
      is_terminal: bool, whether the environment has reached a terminal state.
        This is true when a life is lost and terminal_on_life_loss, or when the
        episode is over.
      info: Gym API's info data structure.
    """
    
    return self.environment.step(action)

  def _pool_and_resize(self):
    """Transforms frames into a Nature DQN observation.

    For efficiency, the transformation is done in-place in self.screen_buffer.

    Returns:
      transformed_screen: numpy array, pooled, resized screen.
    """
    # Pool if there are enough screens to do so.
    transformed_image = cv2.resize(self.screen_buffer[0],
                                   (self.screen_size, self.screen_size),
                                   interpolation=cv2.INTER_LINEAR)
    int_image = np.asarray(transformed_image, dtype=np.uint8)
    return np.expand_dims(int_image, axis=2)