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
"""V-trace based SEED learner."""

from asyncio import tasks
import collections
import math
import os
import time

from absl import flags
from absl import logging

from seed_rl import grpc
from seed_rl.common import common_flags  
from seed_rl.common import utils
from seed_rl.common import vtrace
from seed_rl.common.parametric_distribution import get_parametric_distribution_for_action_space

import tensorflow as tf
import numpy as np
import h5py




FLAGS = flags.FLAGS
tf.compat.v1.enable_eager_execution ()
EpisodeInfo = collections.namedtuple(
    'EpisodeInfo',
    # num_frames: length of the episode in number of frames.
    # returns: Sum of undiscounted rewards experienced in the episode.
    # raw_returns: Sum of raw rewards experienced in the episode.
    # env_ids: ID of the environment that generated this episode.
    'num_frames returns raw_returns env_ids')


Unroll = collections.namedtuple(
    'Unroll', 'agent_state prev_actions env_outputs agent_outputs')


def validate_config():
  utils.validate_learner_config(FLAGS)


def learner_loop(create_env_fn, create_agent_fn, create_optimizer_fn):
  """Main learner loop.

  Args:
    create_env_fn: Callable that must return a newly created environment. The
      callable takes the task ID as argument - an arbitrary task ID of 0 will be
      passed by the learner. The returned environment should follow GYM's API.
      It is only used for infering tensor shapes. This environment will not be
      used to generate experience.
    create_agent_fn: Function that must create a new tf.Module with the neural
      network that outputs actions and new agent state given the environment
      observations and previous agent state. See dmlab.agents.ImpalaDeep for an
      example. The factory function takes as input the environment action and
      observation spaces and a parametric distribution over actions.
    create_optimizer_fn: Function that takes the final iteration as argument
      and must return a tf.keras.optimizers.Optimizer and a
      tf.keras.optimizers.schedules.LearningRateSchedule.
  """
  total_frames = 0
  logging.info('Starting learner loop')
  validate_config()
  settings = utils.init_learner_multi_host(FLAGS.num_training_tpus)
  strategy, hosts, training_strategy, encode, decode = settings
  env = create_env_fn(0, FLAGS)
  FLAGS.num_action_repeats = env._num_action_repeats
  logging.info('Action repeats: %d', env._num_action_repeats)
  parametric_action_distribution = get_parametric_distribution_for_action_space(
      env.action_space)
  if FLAGS.extra_input:
    env_output_specs = utils.EnvOutput_extra(
        tf.TensorSpec([], tf.float32, 'reward'),
        tf.TensorSpec([], tf.bool, 'done'),
        tf.TensorSpec(env.observation_space.shape, env.observation_space.dtype, 'observation'),
        tf.TensorSpec(env.embedding_space.shape, env.embedding_space.dtype, 'embedding'),
        tf.TensorSpec([], tf.int64, 'inst_len'),
        tf.TensorSpec([], tf.bool, 'abandoned'),
        tf.TensorSpec([], tf.int32, 'episode_step'),
    )
  else:
    env_output_specs = utils.EnvOutput(
        tf.TensorSpec([], tf.float32, 'reward'),
        tf.TensorSpec([], tf.bool, 'done'),
        tf.TensorSpec(env.observation_space.shape, env.observation_space.dtype, 'observation'),
        tf.TensorSpec([], tf.bool, 'abandoned'),
        tf.TensorSpec([], tf.int32, 'episode_step'),
    )
  action_specs = tf.TensorSpec(env.action_space.shape,
                               env.action_space.dtype, 'action')
  agent_input_specs = (action_specs, env_output_specs)

  # Initialize agent and variables.
  agent = create_agent_fn(env.action_space, env.observation_space,
                          parametric_action_distribution)
  initial_agent_state = agent.initial_state(1)
  agent_state_specs = tf.nest.map_structure(
      lambda t: tf.TensorSpec(t.shape[1:], t.dtype), initial_agent_state)
  # unroll_specs = [None]  # Lazy initialization.
  input_ = tf.nest.map_structure(
      lambda s: tf.zeros([1] + list(s.shape), s.dtype), agent_input_specs)
  input_ = encode(input_)

  with strategy.scope():
    @tf.function
    def create_variables(*args):
      return agent.get_action(*decode(args))

    initial_agent_output, _ = create_variables(*input_, initial_agent_state)

    if not hasattr(agent, 'entropy_cost'):
      mul = FLAGS.entropy_cost_adjustment_speed
      agent.entropy_cost_param = tf.Variable(
          tf.math.log(FLAGS.entropy_cost) / mul,
          # Without the constraint, the param gradient may get rounded to 0
          # for very small values.
          constraint=lambda v: tf.clip_by_value(v, -20 / mul, 20 / mul),
          trainable=True,
          dtype=tf.float32)
      agent.entropy_cost = lambda: tf.exp(mul * agent.entropy_cost_param)
    # Create optimizer.
    iter_frame_ratio = (
        FLAGS.batch_size * FLAGS.unroll_length * FLAGS.num_action_repeats)
    final_iteration = int(
        math.ceil(FLAGS.total_environment_frames / iter_frame_ratio))
    optimizer, learning_rate_fn = create_optimizer_fn(final_iteration)


    iterations = optimizer.iterations
    optimizer._create_hypers()  
    optimizer._create_slots(agent.trainable_variables)  

  agent_output_specs = tf.nest.map_structure(
      lambda t: tf.TensorSpec(t.shape[1:], t.dtype), initial_agent_output)

  # Setup checkpointing and restore checkpoint.
  ckpt = tf.train.Checkpoint(agent=agent, optimizer=optimizer)
  if FLAGS.init_checkpoint is not None:
    tf.print('Loading initial checkpoint from %s...' % FLAGS.init_checkpoint)
    with strategy.scope():
      # ckpt.restore(FLAGS.init_checkpoint).assert_consumed()
      ckpt.restore(FLAGS.init_checkpoint)

  # Logging.
  summary_writer = tf.summary.create_file_writer(
      FLAGS.logdir, flush_millis=20000, max_queue=10000)
  logger = utils.ProgressLogger(summary_writer=summary_writer,
                                starting_step=total_frames)

  servers = []
  # unroll_queues = []
  info_specs = (
      tf.TensorSpec([], tf.int64, 'episode_num_frames'),
      tf.TensorSpec([], tf.float32, 'episode_returns'),
      tf.TensorSpec([], tf.float32, 'episode_raw_returns'),
  )
  episode_info_specs = EpisodeInfo(*(
      info_specs + (tf.TensorSpec([], tf.int32, 'env_ids'),)))
  info_queue = utils.StructuredFIFOQueue(-1, episode_info_specs)

  def create_host(i, host, inference_devices):
    with tf.device(host):
      server = grpc.Server([FLAGS.server_address])

      store = utils.UnrollStore(
          FLAGS.num_envs, FLAGS.unroll_length,
          (action_specs, env_output_specs, agent_output_specs))
      env_run_ids = utils.Aggregator(FLAGS.num_envs,
                                     tf.TensorSpec([], tf.int64, 'run_ids'))
      env_infos = utils.Aggregator(FLAGS.num_envs, info_specs,
                                   'env_infos')

      # First agent state in an unroll.
      first_agent_states = utils.Aggregator(
          FLAGS.num_envs, agent_state_specs, 'first_agent_states')

      # Current agent state and action.
      agent_states = utils.Aggregator(
          FLAGS.num_envs, agent_state_specs, 'agent_states')
      actions = utils.Aggregator(FLAGS.num_envs, action_specs, 'actions')

      def add_batch_size(ts):
        return tf.TensorSpec([FLAGS.inference_batch_size] + list(ts.shape),
                             ts.dtype, ts.name)

      inference_specs = (
          tf.TensorSpec([], tf.int32, 'env_id'),
          tf.TensorSpec([], tf.int64, 'run_id'),
          env_output_specs,
          tf.TensorSpec([], tf.float32, 'raw_reward'),
      )
      inference_specs = tf.nest.map_structure(add_batch_size, inference_specs)
      def create_inference_fn(inference_device):
        @tf.function(input_signature=inference_specs)
        def inference(env_ids, run_ids, env_outputs, raw_rewards):
          # Reset the environments that had their first run or crashed.
          previous_run_ids = env_run_ids.read(env_ids)
          env_run_ids.replace(env_ids, run_ids)
          reset_indices = tf.where(
              tf.not_equal(previous_run_ids, run_ids))[:, 0]
          envs_needing_reset = tf.gather(env_ids, reset_indices)
          if tf.not_equal(tf.shape(envs_needing_reset)[0], 0):
            tf.print('Environment ids needing reset:', envs_needing_reset)
          env_infos.reset(envs_needing_reset)
          store.reset(envs_needing_reset)
          initial_agent_states = agent.initial_state(
              tf.shape(envs_needing_reset)[0])
          first_agent_states.replace(envs_needing_reset, initial_agent_states)
          agent_states.replace(envs_needing_reset, initial_agent_states)
          actions.reset(envs_needing_reset)

          tf.debugging.assert_non_positive(
              tf.cast(env_outputs.abandoned, tf.int32),
              'Abandoned done states are not supported in VTRACE.')

          # Update steps and return.
          env_infos.add(env_ids, (0, env_outputs.reward, raw_rewards))
          done_ids = tf.gather(env_ids, tf.where(env_outputs.done)[:, 0])
          # dumpDones(trajBuffer, traj2Save, done_ids, num_tasks, eps_count, trans_count)
          if i == 0:
            done_episodes_info = env_infos.read(done_ids)
            info_queue.enqueue_many(EpisodeInfo(*(done_episodes_info + (done_ids,))))
          # if i == 0:
          #   info_queue.enqueue_many(env_infos.read(done_ids))
          env_infos.reset(done_ids)
          env_infos.add(env_ids, (FLAGS.num_action_repeats, 0., 0.))

          # Inference.
          prev_actions = actions.read(env_ids)
          input_ = encode((prev_actions, env_outputs))
          prev_agent_states = agent_states.read(env_ids)
          with tf.device(inference_device):
            @tf.function
            def agent_inference(*args):
              return agent(*decode(args), is_training=False)

            agent_outputs, curr_agent_states = agent_inference(
                *input_, prev_agent_states)

          # Append the latest outputs to the unroll and insert completed unrolls
          # in queue.
          completed_ids, unrolls = store.append(
              env_ids, (prev_actions, env_outputs, agent_outputs))
          # unroll_queue.enqueue_many(unrolls)
          first_agent_states.replace(completed_ids,
                                     agent_states.read(completed_ids))

          # Update current state.
          agent_states.replace(env_ids, curr_agent_states)
          actions.replace(env_ids, agent_outputs.action)
          # Return environment actions to environments.

          # appendStep(trajBuffer, env_outputs, agent_outputs.action, env_ids)
          # dataBufferQueue.enqueue((env_ids, env_outputs, agent_outputs.action))
          # dataBufferQueue.append((env_ids, env_outputs, agent_outputs.action, done_ids))
          return agent_outputs.action

        return inference

      with strategy.scope():
        server.bind([create_inference_fn(d) for d in inference_devices])
      server.start()
      # unroll_queues.append(unroll_queue)
      servers.append(server)

  for i, (host, inference_devices) in enumerate(hosts):
    create_host(i, host, inference_devices)

  def additional_logs():
    n_episodes = info_queue.size()
    n_episodes -= n_episodes % 10
    if tf.not_equal(n_episodes, 0):
      episode_stats = info_queue.dequeue_many(n_episodes)
      frames = [0 for i in range(len(FLAGS.task_names))]
      ep_returns = [0 for i in range(len(FLAGS.task_names))]
      env_counts = [0 for i in range(len(FLAGS.task_names))]
      increment = 0
      for (frame, ep_return, raw_return, env_id) in zip(*episode_stats):
        logging.info('Return: %f Raw return: %f Frames: %i Env id: %i', ep_return,
                     raw_return, frame, env_id)
        env_counts[env_id % len(FLAGS.task_names)] += 1
        frames[env_id % len(FLAGS.task_names)] += frame
        increment += frame
        ep_returns[env_id % len(FLAGS.task_names)] += ep_return
      for idx, env_count in enumerate(env_counts):
        if env_count != 0:
          tf.summary.scalar('subtasks/' + FLAGS.task_names[idx] + '/episode_num_frames', frames[idx] / env_count)
          tf.summary.scalar('subtasks/' + FLAGS.task_names[idx] + '/episode_return', ep_returns[idx] / env_count)
      logger.step_cnt.assign_add(increment)

    


  logger.start(additional_logs)
  # Execute learning.
  while True:
    continue
  logger.shutdown()
  for server in servers:
    server.shutdown()
