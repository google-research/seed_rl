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

r"""SEED actor."""

import os
import timeit

from absl import flags
from absl import logging
import numpy as np
from seed_rl import grpc
from seed_rl.common import common_flags  
from seed_rl.common import profiling
from seed_rl.common import utils
import tensorflow as tf


FLAGS = flags.FLAGS

flags.DEFINE_integer('task', 0, 'Task id.')
flags.DEFINE_integer('num_actors_with_summaries', 4,
                     'Number of actors that will log debug/profiling TF '
                     'summaries.')
flags.DEFINE_bool('render', False,
                  'Whether the first actor should render the environment.')


def are_summaries_enabled():
  return FLAGS.task < FLAGS.num_actors_with_summaries


def actor_loop(create_env_fn):
  """Main actor loop.

  Args:
    create_env_fn: Callable (taking the task ID as argument) that must return a
      newly created environment.
  """
  logging.info('Starting actor loop')
  is_rendering_enabled = FLAGS.render and FLAGS.task == 0
  if are_summaries_enabled():
    summary_writer = tf.summary.create_file_writer(
        os.path.join(FLAGS.logdir, 'actor_{}'.format(FLAGS.task)),
        flush_millis=20000, max_queue=1000)
    timer_cls = profiling.ExportingTimer
  else:
    summary_writer = tf.summary.create_noop_writer()
    timer_cls = utils.nullcontext

  actor_step = 0
  with summary_writer.as_default():
    while True:
      try:
        # Client to communicate with the learner.
        client = grpc.Client(FLAGS.server_address)

        env = create_env_fn(FLAGS.task)

        # Unique ID to identify a specific run of an actor.
        run_id = np.random.randint(np.iinfo(np.int64).max)
        observation = env.reset()
        reward = 0.0
        raw_reward = 0.0
        done = False
        abandoned = False

        global_step = 0
        episode_step = 0
        episode_step_sum = 0
        episode_return_sum = 0
        episode_raw_return_sum = 0
        episodes_in_report = 0

        elapsed_inference_s_timer = timer_cls('actor/elapsed_inference_s', 1000)
        last_log_time = timeit.default_timer()
        while True:
          tf.summary.experimental.set_step(actor_step)
          env_output = utils.EnvOutput(reward, done, observation,
                                       abandoned, episode_step)
          with elapsed_inference_s_timer:
            action = client.inference(
                FLAGS.task, run_id, env_output, raw_reward)
          with timer_cls('actor/elapsed_env_step_s', 1000):
            observation, reward, done, info = env.step(action.numpy())
          if is_rendering_enabled:
            env.render()
          episode_step += 1
          episode_return_sum += reward
          raw_reward = float((info or {}).get('score_reward', reward))
          episode_raw_return_sum += raw_reward
          # If the info dict contains an entry abandoned=True and the
          # episode was ended (done=True), then we need to specially handle
          # the final transition as per the explanations below.
          abandoned = (info or {}).get('abandoned', False)
          assert done if abandoned else True
          if done:
            # If the episode was abandoned, we need to report the final
            # transition including the final observation as if the episode has
            # not terminated yet. This way, learning algorithms can use the
            # transition for learning.
            if abandoned:
              # We do not signal yet that the episode was abandoned. This will
              # happen for the transition from the terminal state to the
              # resetted state.
              env_output = utils.EnvOutput(reward, False, observation,
                                           False, episode_step)
              with elapsed_inference_s_timer:
                action = client.inference(
                    FLAGS.task, run_id, env_output, raw_reward)
              reward = 0.0
              raw_reward = 0.0

            # Periodically log statistics.
            current_time = timeit.default_timer()
            episode_step_sum += episode_step
            global_step += episode_step
            episodes_in_report += 1
            if current_time - last_log_time > 1:
              logging.info(
                  'Actor steps: %i, Return: %f Raw return: %f Episode steps: %f',
                  global_step, episode_return_sum / episodes_in_report,
                  episode_raw_return_sum / episodes_in_report,
                  episode_step_sum / episodes_in_report)
              episode_return_sum = 0
              episode_raw_return_sum = 0
              episode_step_sum = 0
              episodes_in_report = 0
              last_log_time = current_time

            # Finally, we reset the episode which will report the transition
            # from the terminal state to the resetted state in the next loop
            # iteration (with zero rewards).
            with timer_cls('actor/elapsed_env_reset_s', 10):
              observation = env.reset()
              episode_step = 0
            if is_rendering_enabled:
              env.render()
          actor_step += 1
      except (tf.errors.UnavailableError, tf.errors.CancelledError) as e:
        logging.exception(e)
        env.close()
