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


"""Tests for continuous_control_agent."""

from absl.testing import parameterized

from seed_rl.agents.policy_gradient.modules import continuous_control_agent
from seed_rl.agents.policy_gradient.modules import input_normalization
from seed_rl.agents.policy_gradient.modules import running_statistics
from seed_rl.agents.policy_gradient.modules import test_utils
from seed_rl.common import parametric_distribution
from seed_rl.common import utils
import tensorflow as tf


def _dummy_rnn_core_state(batch_size):
  lstm_sizes = [256]
  lstm_cells = [tf.keras.layers.LSTMCell(size) for size in lstm_sizes]
  rnn = tf.keras.layers.StackedRNNCells(lstm_cells)
  return rnn.get_initial_state(batch_size=batch_size, dtype=tf.float32)


def _dummy_input(unroll):
  """Returns a dummy tuple that can be fed into an agent."""
  batch_size = 15
  base_shape = [6, batch_size] if unroll else [batch_size]
  prev_actions = tf.zeros(base_shape + [10], tf.float32)
  # Create the environment output.
  env_outputs = utils.EnvOutput(
      reward=tf.zeros(base_shape, tf.float32),
      done=tf.zeros(base_shape, tf.bool),
      observation=tf.zeros(base_shape + [17], tf.float32),
      abandoned=tf.zeros(base_shape, tf.bool),
      episode_step=tf.zeros(base_shape, tf.bool))
  core_state = _dummy_rnn_core_state(batch_size=batch_size)
  return (prev_actions, env_outputs), core_state


def _combinations():
  """Yields dictionaries with all constructor arguments to be tested."""
  yield {},
  yield {'shared': True},
  yield {'residual_connections': True},
  yield {'correct_observations': True},
  yield {'observation_normalizer': input_normalization.InputNormalization(
      running_statistics.EMAMeanStd())},
  yield {'num_layers_policy': 4},
  yield {'num_layers_value': 4},
  yield {'num_units_policy': 128},
  yield {'num_units_value': 128},
  yield {'activation': tf.sin},
  yield {'activation': tf.keras.activations.relu},
  yield {'kernel_init': tf.keras.initializers.Orthogonal()},
  yield {'last_kernel_init_value_scaling': 0.1},
  yield {'last_kernel_init_policy_scaling': 0.1},
  yield {'last_kernel_init_value': tf.keras.initializers.Orthogonal()},
  yield {'last_kernel_init_policy': tf.keras.initializers.Orthogonal()},
  yield {'layer_normalizer': tf.keras.layers.LayerNormalization},
  yield {'std_independent_of_input': True},
  yield {'num_layers_rnn': 1},


def setUpModule():
  # We want our tests to run on several devices with a mirrored strategy.
  test_utils.simulate_two_devices()


class ContinuousControlTest(test_utils.TestCase, parameterized.TestCase):
  # We manually enter a strategy scope in our tests.
  ENTER_PRIMARY_DEVICE = False

  @parameterized.parameters(list(_combinations()))
  def test_configuration(self, kwargs):
    agent, strategy = self._setup_agent(kwargs)
    self._call_from_strategy(agent, strategy)

  def _setup_agent(self, kwargs):
    """Sets up the agent and a distribution strategy to run."""
    agent = continuous_control_agent.ContinuousControlAgent(
        parametric_distribution.normal_tanh_distribution(20), **kwargs)
    strategy = test_utils.create_distribution_strategy(
        use_tpu=self.primary_device == 'TPU')
    self.assertEqual(strategy.num_replicas_in_sync, 2)
    with strategy.scope():
      wrapped_f = tf.function(agent.__call__)
      wrapped_f(*_dummy_input(False), unroll=False, is_training=True)
    return agent, strategy

  def _call_from_strategy(self, agent, strategy):
    """Updates the normalization statistics via the strategy."""
    @tf.function
    def _run_on_tpu():
      return agent(*_dummy_input(True), unroll=True, is_training=True)
    strategy.run(_run_on_tpu)


if __name__ == '__main__':
  tf.enable_v2_behavior()
  tf.test.main()
