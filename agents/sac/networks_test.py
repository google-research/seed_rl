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

"""Tests for network architectures."""

from absl import flags
from seed_rl.agents.sac import networks
from seed_rl.common import parametric_distribution
from seed_rl.common import utils
import tensorflow as tf

FLAGS = flags.FLAGS


class NetworkTest(tf.test.TestCase):

  def test_actor_critic_lstm(self):
    n_steps = 100
    batch_size = 10
    obs_size = 15
    action_size = 3

    action_dist = parametric_distribution.NormalTanhDistribution(action_size)
    agent = networks.ActorCriticLSTM(
        action_dist,
        n_critics=2,
        lstm_sizes=[10, 20],
        pre_mlp_sizes=[30, 40],
        post_mlp_sizes=[50],
        ff_mlp_sizes=[25, 35, 45])
    env_output = utils.EnvOutput(
        observation=tf.random.normal((n_steps, batch_size, obs_size)),
        reward=tf.random.normal((n_steps, batch_size)),
        done=tf.cast(tf.random.uniform((n_steps, batch_size), 0, 1), tf.bool),
        abandoned=tf.zeros((n_steps, batch_size), dtype=tf.bool),
        episode_step=tf.ones((n_steps, batch_size), dtype=tf.int32))
    prev_action = tf.random.normal((n_steps, batch_size, action_size))
    action = tf.random.normal((n_steps, batch_size, action_size))
    state = agent.initial_state(10)

    # Run in one call.
    v_one_call = agent.get_V(prev_action, env_output, state)
    q_one_call = agent.get_Q(prev_action, env_output, state, action)

    # Run step-by-step.
    v_many_calls = []
    q_many_calls = []
    for i in range(n_steps):
      
      env_output_i = tf.nest.map_structure(lambda t: t[i], env_output)
      expanded_env_output_i = tf.nest.map_structure(lambda t: t[i, tf.newaxis],
                                                    env_output)
      v_many_calls.append(
          agent.get_V(prev_action[i, tf.newaxis],
                      expanded_env_output_i,
                      state)[0])
      q_many_calls.append(
          agent.get_Q(prev_action[i, tf.newaxis],
                      expanded_env_output_i,
                      state,
                      action[i, tf.newaxis])[0])
      unused_action, state = agent(prev_action[i], env_output_i, state)
    v_many_calls = tf.stack(v_many_calls)
    q_many_calls = tf.stack(q_many_calls)

    # Check if results are the same.
    self.assertAllClose(v_one_call, v_many_calls, 1e-4, 1e-4)
    self.assertAllClose(q_one_call, q_many_calls, 1e-4, 1e-4)


if __name__ == '__main__':
  tf.test.main()
