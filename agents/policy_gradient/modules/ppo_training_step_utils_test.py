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


"""Tests for ppo_training_step_utils."""

import collections

from absl.testing import parameterized
import gym
import numpy as np
from seed_rl.agents.policy_gradient.modules import advantages as ga_advantages
from seed_rl.agents.policy_gradient.modules import continuous_control_agent
from seed_rl.agents.policy_gradient.modules import generalized_onpolicy_loss
from seed_rl.agents.policy_gradient.modules import policy_losses
from seed_rl.agents.policy_gradient.modules import policy_regularizers
from seed_rl.agents.policy_gradient.modules import popart
from seed_rl.agents.policy_gradient.modules import ppo_training_step_utils
from seed_rl.agents.policy_gradient.modules import running_statistics
from seed_rl.common import parametric_distribution
from seed_rl.common import utils
import tensorflow as tf

Unroll = collections.namedtuple(
    'Unroll', 'agent_state prev_actions env_outputs agent_outputs')


class PPOUtilsTest(parameterized.TestCase):

  def test_split(self):
    # tensor dims
    timesteps = 3
    batch_size = 2
    observation_size = 2

    class DummyLoss:

      def compute_advantages(self, *_):
        return (tf.zeros(shape=(timesteps - 1, batch_size, observation_size)),
                tf.zeros(shape=(timesteps - 1, batch_size, observation_size)))

    input_args = Unroll(
        (), (), (), tf.zeros(shape=(timesteps, batch_size, observation_size)))
    output_args = ppo_training_step_utils.compute_advantages_and_split(
        DummyLoss(), input_args)

    self.assertEqual(output_args[3].shape,
                     (1, (timesteps - 1) * batch_size, observation_size))
    # Checks that the advantages have been appended.
    self.assertLen(output_args, len(input_args) + 2)

  @parameterized.parameters([
      {
          'batch_mode': 'repeat',
          'use_agent_state': False,
      },
      {
          'batch_mode': 'repeat',
          'use_agent_state': True,
      },
      {
          'batch_mode': 'shuffle',
          'use_agent_state': False,
      },
      {
          'batch_mode': 'shuffle',
          'use_agent_state': True,
      },
      {
          'batch_mode': 'split',
          'use_agent_state': False,
      },
      {
          'batch_mode': 'split_with_advantage_recomputation',
          'use_agent_state': False,
      },
  ])
  def test_ppo_training_step(self, batch_mode, use_agent_state):
    action_space = gym.spaces.Box(low=-1, high=1, shape=[128], dtype=np.float32)
    distribution = (
        parametric_distribution.get_parametric_distribution_for_action_space(
            action_space))
    training_agent = continuous_control_agent.ContinuousControlAgent(
        distribution)
    virtual_bs = 32
    unroll_length = 5
    batches_per_step = 4
    done = tf.zeros([unroll_length, virtual_bs], dtype=tf.bool)
    prev_actions = tf.reshape(
        tf.stack([action_space.sample()
                  for _ in range(unroll_length * virtual_bs)]),
        [unroll_length, virtual_bs, -1])
    env_outputs = utils.EnvOutput(
        reward=tf.random.uniform([unroll_length, virtual_bs]),
        done=done,
        observation=tf.zeros([unroll_length, virtual_bs, 128],
                             dtype=tf.float32),
        abandoned=tf.zeros_like(done),
        episode_step=tf.ones([unroll_length, virtual_bs], dtype=tf.int32))
    if use_agent_state:
      core_state = tf.zeros([virtual_bs, 64])
    else:
      core_state = training_agent.initial_state(virtual_bs)
    agent_outputs, _ = training_agent((prev_actions, env_outputs),
                                      core_state,
                                      unroll=True)
    args = Unroll(core_state, prev_actions, env_outputs, agent_outputs)

    class DummyStrategy:

      def __init__(self):
        self.num_replicas_in_sync = 1

    loss_fn = generalized_onpolicy_loss.GeneralizedOnPolicyLoss(
        training_agent,
        popart.PopArt(running_statistics.FixedMeanStd(), compensate=False),
        distribution,
        ga_advantages.GAE(lambda_=0.9),
        policy_losses.ppo(0.9),
        discount_factor=0.99,
        regularizer=policy_regularizers.KLPolicyRegularizer(entropy=0.5),
        baseline_cost=0.5,
        max_abs_reward=None,
        frame_skip=1,
        reward_scaling=10)
    loss_fn.init()
    loss, logs = ppo_training_step_utils.ppo_training_step(
        epochs_per_step=8,
        loss_fn=loss_fn,
        args=args,
        batch_mode=batch_mode,
        training_strategy=DummyStrategy(),
        virtual_batch_size=virtual_bs,
        unroll_length=unroll_length - 1,
        batches_per_step=batches_per_step,
        clip_norm=50.,
        optimizer=tf.keras.optimizers.Adam(1e-3),
        logger=utils.ProgressLogger())
    del loss
    del logs


if __name__ == '__main__':
  tf.test.main()
