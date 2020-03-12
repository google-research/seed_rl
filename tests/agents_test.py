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

"""Tests for agents."""


import os
from seed_rl.dmlab import networks
import tensorflow as tf


class AgentsTest(tf.test.TestCase):

  def test_agent_is_checkpointable(self):
    agent = networks.ImpalaDeep(9)
    output0 = _run_actor(agent)

    checkpoint_dir = '/tmp/training_checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, 'model.ckpt')
    ckpt = tf.train.Checkpoint(agent=agent)

    ckpt.save(file_prefix=checkpoint_prefix)

    for v in agent.trainable_variables:
      v.assign_add(tf.broadcast_to(1., v.shape))

    output1 = _run_actor(agent)

    ckpt_path = tf.train.latest_checkpoint(checkpoint_dir)
    ckpt.restore(ckpt_path).assert_consumed()

    output2 = _run_actor(agent)

    self.assertEqual(len(agent.trainable_variables), 39)
    self.assertAllEqual(output0[0].policy_logits, output2[0].policy_logits)
    self.assertNotAllEqual(output0[0].policy_logits, output1[0].policy_logits)


def _run_actor(agent):
  initial_agent_state = agent.initial_state(1)
  observation = tf.ones(
      shape=(1, 72, 96, 3),
      dtype=tf.uint8,
  )
  initial_env_output = (tf.constant([2.]), tf.constant([False]), observation)
  return agent(tf.zeros([1], dtype=tf.int32), initial_env_output,
               initial_agent_state)


if __name__ == '__main__':
  tf.test.main()
