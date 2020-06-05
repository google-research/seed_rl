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

# python3
"""Tests networks.py."""

import unittest.mock as mock
from seed_rl.atari import networks
from seed_rl.common import utils
import tensorflow as tf

OBS_DIM = 84


def stack_fn(input_t, stack_state):
  """Stacks frames.

  We use this function for testing _unroll_cell.

  Args:
    input_t: Current batched frame tensor. Expected last channel dimension is 1.
    stack_state: Input state. This is a tuple of the (stack_size - 1) last
      batched frames.

  Returns:
    A pair:
      - A tensor with the last stack_size frames concatenated along the last
          dimension.
      - New state, tuple of the last (stack_size - 1) batched frames.
  """
  new_stack_tuple = stack_state + (input_t,)
  return tf.concat(new_stack_tuple, axis=-1), new_stack_tuple[1:]


@tf.function
def stack_frames(frames, frame_stacking_state, done, stack_size):
  return networks.stack_frames(
      tf.convert_to_tensor(frames, dtype=tf.float32),
      frame_stacking_state,
      tf.convert_to_tensor(done),
      stack_size)


class AgentsTest(tf.test.TestCase):

  def _random_obs(self, batch_size, unroll_length):
    return tf.cast(
        tf.random.uniform([unroll_length, batch_size, OBS_DIM, OBS_DIM, 1],
                          maxval=256, dtype=tf.int32),
        tf.uint8)

  def _create_agent_input(self, batch_size, unroll_length):
    done = tf.cast(tf.random.uniform([unroll_length, batch_size],
                                     maxval=2, dtype=tf.int32),
                   tf.bool)
    return (
        tf.random.uniform([unroll_length, batch_size], maxval=2,
                          dtype=tf.int32),
        utils.EnvOutput(
            reward=tf.random.uniform([unroll_length, batch_size]),
            done=done,
            observation=self._random_obs(batch_size, unroll_length)))

  def test_basic(self):
    agent = networks.DuelingLSTMDQNNet(2, [OBS_DIM, OBS_DIM, 1], stack_size=4)
    batch_size = 16
    initial_agent_state = agent.initial_state(batch_size)
    _, _ = agent(self._create_agent_input(batch_size, 80),
                 initial_agent_state,
                 unroll=True)

  def test_basic_no_frame_stack(self):
    agent = networks.DuelingLSTMDQNNet(2, [OBS_DIM, OBS_DIM, 1], stack_size=1)
    batch_size = 16
    initial_agent_state = agent.initial_state(batch_size)
    with mock.patch.object(agent, '_torso', wraps=agent._torso):
      _, _ = agent(self._create_agent_input(batch_size, 80),
                   initial_agent_state,
                   unroll=True)
      self.assertEqual(agent._torso.call_args[0][1].observation.shape[-1], 1)

  def test_basic_frame_stacking(self):
    agent = networks.DuelingLSTMDQNNet(2, [OBS_DIM, OBS_DIM, 1], stack_size=4)
    batch_size = 16
    initial_agent_state = agent.initial_state(batch_size)
    with mock.patch.object(agent, '_torso', wraps=agent._torso):
      _, _ = agent(self._create_agent_input(batch_size, 80),
                   initial_agent_state,
                   unroll=True)
      self.assertEqual(agent._torso.call_args[0][1].observation.shape[-1], 4)

  def check_core_input_shape(self):
    num_actions = 37
    agent = networks.DuelingLSTMDQNNet(num_actions, [OBS_DIM, OBS_DIM, 1],
                                       stack_size=4)
    batch_size = 16
    initial_agent_state = agent.initial_state(batch_size)
    with mock.patch.object(agent, '_core', wraps=agent._core):
      _, _ = agent(self._create_agent_input(batch_size, 80),
                   initial_agent_state,
                   unroll=True)
      # conv_output_dim + num_actions + reward.
      self.assertEqual(agent._core.call_args[0][0].shape[-1],
                       512 + num_actions + 1)

  def test_unroll_cell(self):
    # Each component is [batch_size=1, channels=1]
    zero_state = (tf.constant([[0]]),) * 3
    # inputs: [time=1, batch_size=1, channels=1].
    # done: [time=1, batch_size=1].
    output, state = networks._unroll_cell(
        inputs=[[[1]]], done=[[False]], start_state=zero_state,
        zero_state=zero_state, recurrent_cell=stack_fn)

    # [time=1, batch_size=1, frame_stack=4]
    # 3 zero frames and last one coming from last inputs.
    self.assertAllEqual(output, [[[0, 0, 0, 1]]])

    output, state = networks._unroll_cell(
        inputs=[[[2]]], done=[[False]], start_state=state,
        zero_state=zero_state,
        recurrent_cell=stack_fn)
    # 2 zero frames and last 2 ones coming from the last two inputs.
    self.assertAllEqual(output, [[[0, 0, 1, 2]]])

    # A longer unroll should be used correctly.
    # inputs: [time=6, batch_size=1, channels=1].
    output, state = networks._unroll_cell(
        inputs=[[[3]], [[4]], [[5]], [[6]], [[7]], [[8]]], done=[[False]] * 6,
        start_state=state, zero_state=zero_state,
        recurrent_cell=stack_fn)

    self.assertEqual(output.shape[0], 6)
     # The first element of the output should be a stack with 1 blank frames and
     # 3 real frames.
    self.assertAllEqual(output[0], [[0, 1, 2, 3]])
    # The last element of the output should contain the last 4 frames from the
    # last inputs.
    self.assertAllEqual(output[5], [[5, 6, 7, 8]])

  def test_unroll_cell_done(self):
    # Each component is [batch_size=1, channels=1]
    zero_state = (tf.constant([[0]]),) * 3
    # inputs: [time=1, batch_size=1, channels=1].
    # done: [time=1, batch_size=1].
    output, state = networks._unroll_cell(
        inputs=[[[1]]], done=[[False]], start_state=zero_state,
        zero_state=zero_state, recurrent_cell=stack_fn)

    # [time=1, batch_size=1, frame_stack=4]
    # 3 zero frames and last one coming from last inputs.
    self.assertAllEqual(output, [[[0, 0, 0, 1]]])

    # Episode is done, stacking should be reset.
    output, state = networks._unroll_cell(
        inputs=[[[2]]], done=[[True]], start_state=state,
        zero_state=zero_state,
        recurrent_cell=stack_fn)
    self.assertAllEqual(output, [[[0, 0, 0, 2]]])

    # A longer unroll with done in the middle should be used correctly.
    # inputs: [time=6, batch_size=1, channels=1].
    output, state = networks._unroll_cell(
        inputs=[[[3]], [[4]], [[5]], [[6]], [[7]], [[8]]],
        done=[[False], [False], [False], [False], [True], [False]],
        start_state=state, zero_state=zero_state,
        recurrent_cell=stack_fn)

    self.assertEqual(output.shape[0], 6)
    self.assertAllEqual(output[0], [[0, 0, 2, 3]])
    self.assertAllEqual(output[5], [[0, 0, 7, 8]])

  def test_stack_frames(self):
    zero_state = networks.DuelingLSTMDQNNet(2, [1], stack_size=4).initial_state(
        1).frame_stacking_state
    # frames: [time=1, batch_size=1, channels=1].
    # done: [time=1, batch_size=1].
    output, state = stack_frames(
        frames=[[[1]]], done=[[False]], frame_stacking_state=zero_state,
        stack_size=4)

    # [time=1, batch_size=1, frame_stack=4]
    # 3 zero frames and last one coming from last inputs.
    self.assertAllEqual(output, [[[1, 0, 0, 0]]])

    output, state = stack_frames(
        frames=[[[2]]], done=[[False]], frame_stacking_state=state,
        stack_size=4)
    # 2 zero frames and last 2 ones coming from the last two inputs.
    self.assertAllEqual(output, [[[2, 1, 0, 0]]])

    # A longer unroll should be used correctly.
    # frames: [time=6, batch_size=1, channels=1].
    output, state = stack_frames(
        frames=[[[3]], [[4]], [[5]], [[6]], [[7]], [[8]]], done=[[False]] * 6,
        frame_stacking_state=state, stack_size=4)

    self.assertEqual(output.shape[0], 6)
     # The first element of the output should be a stack with 1 blank frames and
     # 3 real frames.
    self.assertAllEqual(output[0], [[3, 2, 1, 0]])
    # The last element of the output should contain the last 4 frames from the
    # last inputs.
    self.assertAllEqual(output[5], [[8, 7, 6, 5]])

  def test_stack_frames_done(self):
    zero_state = networks.DuelingLSTMDQNNet(2, [1], stack_size=4).initial_state(
        1).frame_stacking_state
    # frames: [time=1, batch_size=1, channels=1].
    # done: [time=1, batch_size=1].
    output, state = stack_frames(
        frames=[[[1]]], done=[[False]], frame_stacking_state=zero_state,
        stack_size=4)

    # [time=1, batch_size=1, frame_stack=4]
    # 3 zero frames and last one coming from last inputs.
    self.assertAllEqual(output, [[[1, 0, 0, 0]]])

    # Episode is done, stacking should be reset.
    output, state = stack_frames(
        frames=[[[2]]], done=[[True]], frame_stacking_state=state,
        stack_size=4)
    self.assertAllEqual(output, [[[2, 0, 0, 0]]])

    # A longer unroll with done in the middle should be used correctly.
    # frames: [time=6, batch_size=1, channels=1].
    output, state = stack_frames(
        frames=[[[3]], [[4]], [[5]], [[6]], [[7]], [[8]]],
        done=[[False], [False], [False], [False], [True], [False]],
        frame_stacking_state=state, stack_size=4)

    self.assertEqual(output.shape[0], 6)
    self.assertAllEqual(output[0], [[3, 2, 0, 0]])
    self.assertAllEqual(output[5], [[8, 7, 0, 0]])


if __name__ == '__main__':
  tf.test.main()
