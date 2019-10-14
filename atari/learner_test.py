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

"""Tests for learner.py."""

from absl import flags
import numpy as np
from seed_rl.atari import agents
from seed_rl.atari import learner
from seed_rl.utils import utils
import tensorflow as tf


FLAGS = flags.FLAGS

OBS_SHAPE = [84, 84, 1]


class LearnerTest(tf.test.TestCase):

  def test_apply_epsilon_greedy(self):
    tf.random.set_seed(5)
    num_actors = 10000
    epsilon = 0.4
    # Actions from network are negative, random actions are non-negative. This
    # allows distinguishing where an action comes from.
    action = tf.range(-num_actors, 0)
    new_action = tf.function(learner.apply_epsilon_greedy)(
        action,
        # We always pick the first actor which has an epsilon of 0.4
        actor_ids=tf.zeros([num_actors], dtype=tf.int32),
        num_training_actors=10,
        num_eval_actors=0,
        eval_epsilon=0,
        num_actions=200)
    num_random_actions = tf.reduce_sum(
        tf.cast(tf.math.greater_equal(new_action, 0), tf.int32)).numpy()
    self.assertLess(num_random_actions, num_actors * epsilon * 1.3)
    self.assertGreater(num_random_actions, num_actors * epsilon * 0.7)
    # Check that new actions are either random actions, or equal to the input
    # actions.
    self.assertEqual(
        tf.reduce_sum(
            tf.cast(tf.logical_or(
                tf.math.greater_equal(new_action, 0),
                tf.equal(new_action, action)), tf.int32)).numpy(),
        num_actors)

  def test_get_actors_epsilon(self):
    epsilons = tf.function(learner.get_actors_epsilon)(
        tf.range(20, dtype=tf.int32),
        num_training_actors=10,
        num_eval_actors=10,
        eval_epsilon=1e-3)
    # Eval epsilons.
    self.assertAllClose(epsilons[10:], [1e-3] * 10)
    # Training epsilons.
    self.assertAllClose(epsilons[0], 0.4)
    self.assertAllClose(epsilons[9], 0.4 ** 8)



  def _create_env_output(self, batch_size, unroll_length):
    return utils.EnvOutput(
        reward=tf.random.uniform([unroll_length, batch_size]),
        done=tf.cast(tf.random.uniform([unroll_length, batch_size],
                                       maxval=2, dtype=tf.int32),
                     tf.bool),
        observation=self._random_obs(batch_size, unroll_length))

  def _random_obs(self, batch_size, unroll_length):
    return tf.cast(
        tf.random.uniform([unroll_length, batch_size] + OBS_SHAPE,
                          maxval=256, dtype=tf.int32),
        tf.uint8)

  def _create_agent_outputs(self, batch_size, unroll_length, num_actions):
    return agents.AgentOutput(
        action=tf.random.uniform([unroll_length, batch_size],
                                 maxval=num_actions, dtype=tf.int32),
        q_values=tf.random.uniform([unroll_length, batch_size, num_actions]))

  def test_compute_loss_basic(self):
    """Basic test to exercise learner.compute_loss_and_priorities()."""
    batch_size = 32
    num_actions = 3
    unroll_length = 10
    training_agent = agents.DuelingLSTMDQNNet(num_actions, OBS_SHAPE)
    prev_actions = tf.random.uniform(
        [unroll_length, batch_size], maxval=2, dtype=tf.int32)
    tf.function(learner.compute_loss_and_priorities)(
        training_agent,
        agents.DuelingLSTMDQNNet(num_actions, OBS_SHAPE),
        training_agent.initial_state(batch_size),
        prev_actions,
        self._create_env_output(batch_size, unroll_length),
        self._create_agent_outputs(batch_size, unroll_length, num_actions),
        0.99,
        burn_in=5)

  def test_value_function_rescaling(self):
    for x in np.linspace(-100., 100.):
      self.assertAllClose(
          learner.inverse_value_function_rescaling(
              learner.value_function_rescaling(x)),
          x)
    self.assertAllEqual(
        learner.value_function_rescaling(0.), 0)
    self.assertAllGreater(
        learner.value_function_rescaling(1000.), 10.)
    self.assertAllLess(
        learner.value_function_rescaling(-1000.), -10.)
    self.assertAllEqual(
        learner.inverse_value_function_rescaling(0.), 0)

    # Higher dimensional inputs:
    self.assertAllClose(
        learner.value_function_rescaling(
            tf.constant([0., 3., -3.])),
        tf.constant([0., 1 + 3e-3, -1 - 3e-3]))
    # We need a fairly high absolute precision tolerance. The re-scaling is not
    # very stable numerically.
    self.assertAllClose(
        learner.inverse_value_function_rescaling(
            tf.constant([0., 1 + 3e-3, -1 - 3e-3])),
        tf.constant([0., 3, -3]),
        atol=2e-4)

  def test_n_step_bellman_target_one_step(self):
    targets = tf.function(learner.n_step_bellman_target)(
        rewards=np.array([[1., 2., 3.]], np.float32).T,
        done=np.array([[False] * 3]).T,
        q_target=np.array([[100, 200, 300]], np.float32).T,
        gamma=0.9,
        n_steps=1)
    self.assertAllClose(
        targets,
        np.array([[1 + 0.9 * 100, 2 + 0.9 * 200, 3 + 0.9 * 300]]).T)

  def test_n_step_bellman_target_one_step_with_done(self):
    targets = tf.function(learner.n_step_bellman_target)(
        rewards=np.array([[1., 2., 3.]], np.float32).T,
        done=np.array([[False, True, False]]).T,
        q_target=np.array([[100, 200, 300]], np.float32).T,
        gamma=0.9,
        n_steps=1)
    self.assertAllClose(targets,
                        np.array([[1 + 0.9 * 100, 2, 3 + 0.9 * 300]]).T)

  def test_n_step_bellman_target_two_steps(self):
    targets = tf.function(learner.n_step_bellman_target)(
        rewards=np.array([[1., 2., 3.]], np.float32).T,
        done=np.array([[False, False, False]]).T,
        q_target=np.array([[100, 200, 300]], np.float32).T,
        gamma=0.9,
        n_steps=2)
    self.assertAllClose(
        targets,
        np.array([[
            1 + 0.9 * 2 + 0.9 ** 2 * 200,
            2 + 0.9 * 3 + 0.9 ** 2 * 300,
            # Last target is actually 1-step.
            3 + 0.9 * 300,
        ]]).T)

  def test_n_step_bellman_target_three_steps_done(self):
    targets = tf.function(learner.n_step_bellman_target)(
        rewards=np.array([[1., 2., 3., 4., 5., 6., 7.]], np.float32).T,
        done=np.array([[False, False, False, True, False, False, False]]).T,
        q_target=np.array([[100, 200, 300, 400, 500, 600, 700]], np.float32).T,
        gamma=0.9,
        n_steps=3)
    self.assertAllClose(
        targets,
        np.array([[
            1 + 0.9 * 2 + 0.9 ** 2 * 3 + 0.9 ** 3 * 300,
            2 + 0.9 * 3 + 0.9 ** 2 * 4,
            3 + 0.9 * 4,
            4,
            5 + 0.9 * 6 + 0.9 ** 2 * 7 + 0.9 ** 3 * 700,
            # Actually 2-step.
            6 + 0.9 * 7 + 0.9 ** 2 * 700,
            # Actually 1-step.
            7 + 0.9 * 700,
        ]]).T)


if __name__ == '__main__':
  tf.test.main()
