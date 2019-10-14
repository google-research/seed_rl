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

r"""Actor task which uses specified Atari game as RL-environment."""

from absl import app
from absl import flags
from absl import logging

import numpy as np
from seed_rl import grpc
from seed_rl.atari import config
from seed_rl.utils import utils
import tensorflow as tf


FLAGS = flags.FLAGS


def main(_):
  while True:
    try:
      # Client to communicate with the learner.
      client = grpc.Client(FLAGS.server_address)

      env = config.create_environment(FLAGS.task)

      # Unique ID to identify a specific run of an actor.
      run_id = np.random.randint(np.iinfo(np.int64).max)
      observation = env.reset()
      reward = 0.0
      raw_reward = 0.0
      done = False

      while True:
        env_output = utils.EnvOutput(reward, done, observation)
        action = client.inference((FLAGS.task, run_id, env_output, raw_reward))
        observation, reward, done, info = env.step(action.numpy())
        raw_reward = float(info.get('score_reward', reward))
        if done:
          observation = env.reset()
    except (tf.errors.UnavailableError, tf.errors.CancelledError) as e:
      logging.exception(e)
      env.close()


if __name__ == '__main__':
  app.run(main)
