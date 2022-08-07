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
def reset_data():
  return {'observations': [],
          'actions': [],
          'terminals': [],
          'rewards': [],
          }

def appendStep(buffer, env_outputs, action, env_ids):
    for idx, env_id in enumerate(env_ids):
        buffer[env_id]['observations'].append(env_outputs.observation[idx])
        buffer[env_id]['actions'].append(FLAGS.action_set[action[idx]])
        buffer[env_id]['terminals'].append(env_outputs.done[idx])
        buffer[env_id]['rewards'].append(env_outputs.reward[idx])

def dumpDones(buffer, data2save, done_ids, num_tasks, eps_count, trans_count):
    print(done_ids)
    for done_id in done_ids:
        task_id = done_id % num_tasks
        eps_count[task_id] += 1
        trans_count[task_id] += (len(buffer[done_id]['observation'][:-1]))
        data2save[task_id]['observations'].extend(buffer[done_id]['observation'])
        data2save[task_id]['actions'].extend(buffer[done_id]['actions'])
        data2save[task_id]['terminals'].extend(buffer[done_id]['terminals'])
        data2save[task_id]['rewards'].extend(buffer[done_id]['rewards'])
        print(f"dumping, length: {len(buffer[done_id]['rewards'])}")
        buffer[done_id] = reset_data()


def append_data(data, obs, act, rew, done):
  data['observations'].extend(obs)
  data['actions'].extend(act)
  data['rewards'].extend(rew)
  data['terminals'].extend(done)
  data['timeouts'].extend(done)


def merge_data(data1, data2):
  data1['observations'].extend(data2['observations'])
  data1['actions'].extend(data2['actions'])
  data1['rewards'].extend(data2['rewards'])
  data1['terminals'].extend(data2['terminals'])
  data1['timeouts'].extend(data2['timeouts'])

def createDataset(k):
  res = []
  for i in range(k):
    res.append(reset_data())
  return res

def npify(data):
    for k in data:
        if k in ['terminals', 'timeouts']:
            dtype = np.bool_
            data[k] = np.array(data[k], dtype=dtype)
        else:
            data[k] = np.array(data[k])

EpisodeInfo = collections.namedtuple(
    'EpisodeInfo',
    # num_frames: length of the episode in number of frames.
    # returns: Sum of undiscounted rewards experienced in the episode.
    # raw_returns: Sum of raw rewards experienced in the episode.
    # env_ids: ID of the environment that generated this episode.
    'num_frames returns raw_returns env_ids')

def compute_loss(logger, parametric_action_distribution, agent, agent_state,
                 prev_actions, env_outputs, agent_outputs):
  learner_outputs, _ = agent(prev_actions,
                             env_outputs,
                             agent_state,
                             unroll=True,
                             is_training=True)

  # Use last baseline value (from the value function) to bootstrap.
  bootstrap_value = learner_outputs.baseline[-1]

  # At this point, we have unroll length + 1 steps. The last step is only used
  # as bootstrap value, so it's removed.
  agent_outputs = tf.nest.map_structure(lambda t: t[:-1], agent_outputs)
  rewards, done, _, _, _ = tf.nest.map_structure(lambda t: t[1:], env_outputs)
  learner_outputs = tf.nest.map_structure(lambda t: t[:-1], learner_outputs)

  if FLAGS.max_abs_reward:
    rewards = tf.clip_by_value(rewards, -FLAGS.max_abs_reward,
                               FLAGS.max_abs_reward)
  discounts = tf.cast(~done, tf.float32) * FLAGS.discounting

  target_action_log_probs = parametric_action_distribution.log_prob(
      learner_outputs.policy_logits, agent_outputs.action)
  behaviour_action_log_probs = parametric_action_distribution.log_prob(
      agent_outputs.policy_logits, agent_outputs.action)

  # Compute V-trace returns and weights.
  vtrace_returns = vtrace.from_importance_weights(
      target_action_log_probs=target_action_log_probs,
      behaviour_action_log_probs=behaviour_action_log_probs,
      discounts=discounts,
      rewards=rewards,
      values=learner_outputs.baseline,
      bootstrap_value=bootstrap_value,
      lambda_=FLAGS.lambda_)

  # Policy loss based on Policy Gradients
  policy_loss = -tf.reduce_mean(target_action_log_probs *
                                tf.stop_gradient(vtrace_returns.pg_advantages))

  # Value function loss
  v_error = vtrace_returns.vs - learner_outputs.baseline
  v_loss = FLAGS.baseline_cost * 0.5 * tf.reduce_mean(tf.square(v_error))

  # Entropy reward
  entropy = tf.reduce_mean(
      parametric_action_distribution.entropy(learner_outputs.policy_logits))
  entropy_loss = tf.stop_gradient(agent.entropy_cost()) * -entropy

  # KL(old_policy|new_policy) loss
  kl = behaviour_action_log_probs - target_action_log_probs
  kl_loss = FLAGS.kl_cost * tf.reduce_mean(kl)

  # Entropy cost adjustment (Langrange multiplier style)
  if FLAGS.target_entropy:
    entropy_adjustment_loss = agent.entropy_cost() * tf.stop_gradient(
        tf.reduce_mean(entropy) - FLAGS.target_entropy)
  else:
    entropy_adjustment_loss = 0. * agent.entropy_cost()  # to avoid None in grad

  total_loss = (policy_loss + v_loss + entropy_loss + kl_loss +
                entropy_adjustment_loss)

  # value function
  session = logger.log_session()
  logger.log(session, 'V/value function',
             tf.reduce_mean(learner_outputs.baseline))
  logger.log(session, 'V/L2 error', tf.sqrt(tf.reduce_mean(tf.square(v_error))))
  # losses
  logger.log(session, 'losses/policy', policy_loss)
  logger.log(session, 'losses/V', v_loss)
  logger.log(session, 'losses/entropy', entropy_loss)
  logger.log(session, 'losses/kl', kl_loss)
  logger.log(session, 'losses/total', total_loss)
  # policy
  dist = parametric_action_distribution.create_dist(
      learner_outputs.policy_logits)
  if hasattr(dist, 'scale'):
    logger.log(session, 'policy/std', tf.reduce_mean(dist.scale))
  logger.log(session, 'policy/max_action_abs(before_tanh)',
             tf.reduce_max(tf.abs(agent_outputs.action)))
  logger.log(session, 'policy/entropy', entropy)
  logger.log(session, 'policy/entropy_cost', agent.entropy_cost())
  logger.log(session, 'policy/kl(old|new)', tf.reduce_mean(kl))

  return total_loss, session


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
  task_set = FLAGS.task_names
  num_tasks = len(task_set)
  trajBuffer = createDataset(FLAGS.num_envs)
  traj2Save = createDataset(num_tasks)
  traj2SaveIdx = [0 for i in range(num_tasks)]
  traj2SaveEps = [0 for i in range(num_tasks)]
  traj2SaveTrans = [0 for i in range(num_tasks)]
  trans_count = [0 for i in range(num_tasks)]
  eps_count = [0 for i in range(num_tasks)]
  maxSave = 500
  dataSaveInterval = int(1e6)
  logging.info('Starting learner loop')
  validate_config()
  settings = utils.init_learner_multi_host(FLAGS.num_training_tpus)
  strategy, hosts, training_strategy, encode, decode = settings
  env = create_env_fn(0, FLAGS)
  FLAGS.num_action_repeats = env._num_action_repeats
  logging.info('Action repeats: %d', env._num_action_repeats)
  parametric_action_distribution = get_parametric_distribution_for_action_space(
      env.action_space)
  env_output_specs = utils.EnvOutput(
      tf.TensorSpec([], tf.float32, 'reward'),
      tf.TensorSpec([], tf.bool, 'done'),
      tf.TensorSpec(env.observation_space.shape, env.observation_space.dtype,
                    'observation'),
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
  unroll_specs = [None]  # Lazy initialization.
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

    # ON_READ causes the replicated variable to act as independent variables for
    # each replica.
    temp_grads = [
        tf.Variable(tf.zeros_like(v), trainable=False,
                    synchronization=tf.VariableSynchronization.ON_READ)
        for v in agent.trainable_variables
    ]

  @tf.function
  def minimize(iterator):
    data = next(iterator)

    def compute_gradients(args):
      args = tf.nest.pack_sequence_as(unroll_specs[0], decode(args, data))
      with tf.GradientTape() as tape:
        loss, logs = compute_loss(logger, parametric_action_distribution, agent,
                                  *args)
      grads = tape.gradient(loss, agent.trainable_variables)
      for t, g in zip(temp_grads, grads):
        t.assign(g)
      return loss, logs

    loss, logs = training_strategy.run(compute_gradients, (data,))
    loss = training_strategy.experimental_local_results(loss)[0]

    def apply_gradients(_):
      optimizer.apply_gradients(zip(temp_grads, agent.trainable_variables))

    strategy.run(apply_gradients, (loss,))

    getattr(agent, 'end_of_training_step_callback',
            lambda: logging.info('end_of_training_step_callback not found'))()

    logger.step_end(logs, training_strategy, iter_frame_ratio)

  agent_output_specs = tf.nest.map_structure(
      lambda t: tf.TensorSpec(t.shape[1:], t.dtype), initial_agent_output)

  # Setup checkpointing and restore checkpoint.
  ckpt = tf.train.Checkpoint(agent=agent, optimizer=optimizer)
  if FLAGS.init_checkpoint is not None:
    tf.print('Loading initial checkpoint from %s...' % FLAGS.init_checkpoint)
    with strategy.scope():
      ckpt.restore(FLAGS.init_checkpoint).assert_consumed()
  manager = tf.train.CheckpointManager(
      ckpt, FLAGS.logdir, max_to_keep=100, keep_checkpoint_every_n_hours=6)
  last_ckpt_time = 0  # Force checkpointing of the initial model.
  if manager.latest_checkpoint:
    logging.info('Restoring checkpoint: %s', manager.latest_checkpoint)
    ckpt.restore(manager.latest_checkpoint).assert_consumed()
    last_ckpt_time = time.time()

  # Logging.
  summary_writer = tf.summary.create_file_writer(
      FLAGS.logdir, flush_millis=20000, max_queue=1000)
  logger = utils.ProgressLogger(summary_writer=summary_writer,
                                starting_step=iterations * iter_frame_ratio)

  servers = []
  unroll_queues = []
  info_specs = (
      tf.TensorSpec([], tf.int64, 'episode_num_frames'),
      tf.TensorSpec([], tf.float32, 'episode_returns'),
      tf.TensorSpec([], tf.float32, 'episode_raw_returns'),
  )
  episode_info_specs = EpisodeInfo(*(
      info_specs + (tf.TensorSpec([], tf.int32, 'env_ids'),)))
  info_queue = utils.StructuredFIFOQueue(-1, episode_info_specs)
  # info_queue = utils.StructuredFIFOQueue(-1, info_specs)
  dataBufferSpecs = (
        tf.TensorSpec([], tf.int32, 'env_id'),
        env_output_specs,
        tf.TensorSpec([], tf.int64, 'action')
    )
  # dataBufferQueue = utils.StructuredFIFOQueue(-1, dataBufferSpecs)

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

      unroll_specs[0] = Unroll(agent_state_specs, *store.unroll_specs)
      unroll_queue = utils.StructuredFIFOQueue(1, unroll_specs[0])

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
          # _, _ = trajStore.append(
          #     env_ids, (prev_actions, env_outputs, agent_outputs))
          # unrolls = Unroll(first_agent_states.read(completed_ids), *unrolls)
          unroll_queue.enqueue_many(unrolls)
          first_agent_states.replace(completed_ids,
                                     agent_states.read(completed_ids))

          # Update current state.
          agent_states.replace(env_ids, curr_agent_states)
          actions.replace(env_ids, agent_outputs.action)
          # Return environment actions to environments.
          # dataBufferQueue.enqueue((env_ids, env_outputs, agent_outputs.action))
          # dataBufferQueue.append((env_ids, env_outputs, agent_outputs.action, done_ids))
          return agent_outputs.action

        return inference

      with strategy.scope():
        server.bind([create_inference_fn(d) for d in inference_devices])
      server.start()
      unroll_queues.append(unroll_queue)
      servers.append(server)

  for i, (host, inference_devices) in enumerate(hosts):
    create_host(i, host, inference_devices)

  def dequeue(ctx):
    # Create batch (time major).
    env_outputs = tf.nest.map_structure(lambda *args: tf.stack(args), *[
        unroll_queues[ctx.input_pipeline_id].dequeue()
        for i in range(ctx.get_per_replica_batch_size(FLAGS.batch_size))
    ])
    env_outputs = env_outputs._replace(
        prev_actions=utils.make_time_major(env_outputs.prev_actions),
        env_outputs=utils.make_time_major(env_outputs.env_outputs),
        agent_outputs=utils.make_time_major(env_outputs.agent_outputs))
    env_outputs = env_outputs._replace(
        env_outputs=encode(env_outputs.env_outputs))
    # tf.data.Dataset treats list leafs as tensors, so we need to flatten and
    # repack.
    return tf.nest.flatten(env_outputs)

  def dataset_fn(ctx):
    dataset = tf.data.Dataset.from_tensors(0).repeat(None)

    def _dequeue(_):
      return dequeue(ctx)

    return dataset.map(
        _dequeue, num_parallel_calls=ctx.num_replicas_in_sync // len(hosts))

  dataset = training_strategy.experimental_distribute_datasets_from_function(
      dataset_fn)
  it = iter(dataset)

  def additional_logs():
    tf.summary.scalar('learning_rate', learning_rate_fn(iterations))
    n_episodes = info_queue.size()
    n_episodes -= n_episodes % FLAGS.log_episode_frequency
    if tf.not_equal(n_episodes, 0):
      episode_stats = info_queue.dequeue_many(n_episodes)
      # episode_keys = [
      #     'episode_num_frames', 'episode_return', 'episode_raw_return'
      # ]
      # for key, values in zip(episode_keys, episode_stats):
      #   for value in tf.split(values,
      #                         values.shape[0] // FLAGS.log_episode_frequency):
      #     tf.summary.scalar(key, tf.reduce_mean(value))

      for (frames, ep_return, raw_return, env_id) in zip(*episode_stats):
        logging.info('Return: %f Raw return: %f Frames: %i Env id: %i', ep_return,
                     raw_return, frames, env_id)
        tf.summary.scalar('subtasks/' + FLAGS.task_names[env_id % len(FLAGS.task_names)] + '/episode_num_frames', frames)
        tf.summary.scalar('subtasks/' + FLAGS.task_names[env_id % len(FLAGS.task_names)] + '/episode_return', ep_return)
        tf.summary.scalar('subtasks/' + FLAGS.task_names[env_id % len(FLAGS.task_names)] + '/episode_raw_return', raw_return)
    # for idx in range(num_tasks):
    #     if len(traj2Save[idx]['observation']) > dataSaveInterval:
    #         traj2SaveEps[idx] += eps_count[idx]
    #         traj2SaveTrans[idx] += trans_count[idx]
    #         logging.info(f'saving data, save idx: {traj2SaveIdx[idx]} task: {task_set[idx]} cur transitions: {trans_count[idx]}  cur episodes: {eps_count[idx]}')
    #         logging.info(f'saved transitions: {traj2SaveTrans[idx]} saved episodes: {traj2SaveEps[idx]}')
    #         dataset2save = h5py.File(FLAGS.logdir + '/' + task_set[idx] + '_dataset/' + str(idx) + '_' + str(traj2SaveIdx[idx] % maxSave) + '.hdf5', 'w')
    #         npify(traj2Save[idx])
    #         for k in traj2Save[idx]:
    #             dataset2save.create_dataset(k, data=traj2Save[idx][k], compression='gzip')
    #         traj2Save[idx] = reset_data()
    #         traj2SaveIdx[idx] += 1
    #         eps_count[idx] = 0
    #         trans_count[idx] = 0


  logger.start(additional_logs)
  # Execute learning.
  while iterations < final_iteration:
    # Save checkpoint.
    current_time = time.time()
    if current_time - last_ckpt_time >= FLAGS.save_checkpoint_secs:
      manager.save()
      # Apart from checkpointing, we also save the full model (including
      # the graph). This way we can load it after the code/parameters changed.
      tf.saved_model.save(agent, os.path.join(FLAGS.logdir, 'saved_model'))
      last_ckpt_time = current_time
    minimize(it)
  logger.shutdown()
  manager.save()
  tf.saved_model.save(agent, os.path.join(FLAGS.logdir, 'saved_model'))
  for server in servers:
    server.shutdown()
  for unroll_queue in unroll_queues:
    unroll_queue.close()
