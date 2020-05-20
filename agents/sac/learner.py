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
"""Soft Actor-Critic (SAC) learner for SEED.

Original paper: https://arxiv.org/pdf/1801.01290.pdf
Follow-up paper: https://arxiv.org/pdf/1812.05905.pdf

Included features:
- arbitrary action distributions (for non-reparametrizable distributions
the actor gradient is computed with Policy Gradients)
- bootstrapping from V-function or directly from Q-function
- entropy coefficient adjustment (i.e. entropy constraint)
"""

import collections
import math
import os
import time

from absl import flags
from absl import logging
import numpy as np

from seed_rl import grpc
from seed_rl.common import common_flags  
from seed_rl.common import utils
from seed_rl.common.parametric_distribution import get_parametric_distribution_for_action_space

import tensorflow as tf




# Training.
flags.DEFINE_integer('save_checkpoint_secs', 1800,
                     'Checkpoint save period in seconds.')
flags.DEFINE_integer('total_environment_frames', int(1e9),
                     'Total environment frames to train for.')
flags.DEFINE_integer('batch_size', 256, 'Batch size for training.')
flags.DEFINE_integer('inference_batch_size', -1,
                     'Batch size for inference, -1 for auto-tune.')
flags.DEFINE_integer('unroll_length', 1, 'Unroll length in agent steps.')
flags.DEFINE_integer('num_training_tpus', 1, 'Number of TPUs for training.')
flags.DEFINE_string('init_checkpoint', None,
                    'Path to the checkpoint used to initialize the agent.')
flags.DEFINE_integer('unroll_queue_max_size', 100,
                     'Max size of the unroll queue')
# Replay buffer
flags.DEFINE_integer('replay_buffer_size', int(1e6),
                     'Size of the replay buffer (in number of unrolls stored).')
flags.DEFINE_float('replay_ratio', 4,
                   'Average number of times each observation is replayed.')
flags.DEFINE_integer('her_window_length', None, 'If not None, then Hindsight '
                     'Experience Replay is used and this parameter determines '
                     'the size (in environment steps) of the window from which '
                     'hindsight goals are sampled.')
flags.DEFINE_float('her_substitution_probability', 0.8, 'Probability of '
                   'substituting each goal if HER is used. Substituted goals '
                   'are sampled uniformly from subsequently achieved goals in '
                   'the same window.')
# Loss settings.
flags.DEFINE_float('entropy_cost', 0.01, 'Entropy cost/multiplier.')
flags.DEFINE_float('target_entropy', None, 'If not None, the entropy cost is '
                   'automatically adjusted to reach the desired entropy level.')
flags.DEFINE_float('entropy_cost_adjustment_speed', 1., 'Controls how fast '
                   'the entropy cost coefficient is adjusted.')
flags.DEFINE_float('discounting', .99, 'Discounting factor.')
flags.DEFINE_float('max_abs_reward', 0.,
                   'Maximum absolute reward when calculating loss.'
                   'Use 0. to disable clipping.')
flags.DEFINE_enum('bootstrap_net', 'v', ['q', 'v'],
                  'Specifies whether to bootstrap from the Q-function '
                  'or V-function. Original SAC bootstrapped from V but a later '
                  'paper (https://arxiv.org/abs/1812.05905) switched to '
                  'bootstrapping directly from Q.')
# Target network
flags.DEFINE_integer(
    'update_target_every_n_step', 1,
    'Frequency of target network updates in training minibatches.')
flags.DEFINE_float(
    'polyak', 0.9,
    'Coefficient for soft target network update. Set to 1 for hard update.'
)

# Logging
flags.DEFINE_integer('log_batch_frequency', 100, 'We average that many batches '
                     'before logging batch statistics like entropy.')
flags.DEFINE_integer('log_episode_frequency', 1, 'We average that many episodes'
                     ' before logging average episode return and length.')

FLAGS = flags.FLAGS


log_keys = []  # array of strings with names of values logged by compute_loss


def compute_loss(logger, parametric_action_distribution, agent, target_agent,
                 agent_state, prev_actions, env_outputs, agent_actions):
  # At this point, we have unroll length + 1 steps. The last step is only used
  # as bootstrap value, so it's removed.
  rewards, done, _, _, _ = tf.nest.map_structure(lambda t: t[1:], env_outputs)
  discounts = tf.cast(~done, tf.float32) * FLAGS.discounting

  if FLAGS.max_abs_reward:
    rewards = tf.clip_by_value(rewards, -FLAGS.max_abs_reward,
                               FLAGS.max_abs_reward)

  # Networks expect postprocessed prev_actions but it's done during inference.
  inputs = (prev_actions[:-1],
            tf.nest.map_structure(lambda t: t[:-1], env_outputs), agent_state)
  if FLAGS.her_window_length:
    # Shift the desired goals for the target batch so that we bootstrap from
    # the right goals.
    observation = env_outputs.observation.copy()
    observation['desired_goal'] = tf.concat(values=[
        observation['desired_goal'][:1] * np.nan,  # this value is not used
        observation['desired_goal'][:-1]  # same goals as in the main batch
    ], axis=0)
    target_inputs = (prev_actions,
                     env_outputs._replace(observation=observation), agent_state)
  else:
    target_inputs = (prev_actions, env_outputs, agent_state)

  # this is called to update observation normalization (if used)
  agent(*inputs, is_training=True, unroll=True)
  # run actor
  action_params = agent.get_action_params(*inputs)
  action = parametric_action_distribution.sample(action_params)
  entropy = parametric_action_distribution.entropy(action_params)

  v = agent.get_V(*inputs)  # state value function

  # actor loss
  with tf.GradientTape(watch_accessed_variables=False) as tape:
    tape.watch(action)
    q_action = agent.get_Q(*inputs, action=action)
    logp_action = parametric_action_distribution.log_prob(action_params, action)
    min_q = tf.reduce_min(q_action, axis=-1)  # min over 2 critics
    actor_objective = (min_q -
                       tf.stop_gradient(agent.entropy_cost()) * logp_action)

    if parametric_action_distribution.reparametrizable:  # DDPG-style gradient
      grad_action = tape.gradient(min_q, action)
      actor_loss = -tf.reduce_mean(tf.stop_gradient(grad_action) * action)
      actor_loss -= (tf.stop_gradient(agent.entropy_cost())
                     * tf.reduce_mean(entropy))
    else:  # policy gradients
      advantage = tf.stop_gradient(actor_objective - v)
      advantage -= tf.reduce_mean(advantage)
      advantage /= tf.math.reduce_std(advantage) + 0.001
      actor_loss = -tf.reduce_mean(advantage * logp_action)

  # V-function loss
  target_v = tf.stop_gradient(actor_objective)
  v_error = v - target_v
  v_loss = tf.reduce_mean(tf.square(v_error))

  # Q-function loss
  q_old_action = agent.get_Q(*inputs, action=agent_actions[:-1])

  if FLAGS.bootstrap_net == 'q':
    next_action_params = agent.get_action_params(*target_inputs)
    next_action = parametric_action_distribution.sample(
        next_action_params)
    next_q = target_agent.get_Q(*target_inputs, action=next_action)[1:]
    next_q = tf.reduce_min(next_q, axis=-1)  # minimum over 2 Q-networks
    next_entropy = parametric_action_distribution.entropy(
        next_action_params)[1:]
    next_v = next_q + tf.stop_gradient(agent.entropy_cost()) * next_entropy
  elif FLAGS.bootstrap_net == 'v':
    next_v = target_agent.get_V(*target_inputs)[1:]
  else:
    assert False, 'bootstrap_net flag must be equal "q" or "v".'

  target_q = tf.stop_gradient(rewards + discounts * next_v)
  q_error = q_old_action - tf.expand_dims(target_q, axis=-1)
  q_loss = tf.reduce_mean(tf.square(q_error))

  # Entropy cost adjustment (Langrange multiplier style)
  if FLAGS.target_entropy:
    entropy_adjustment_loss = agent.entropy_cost() * tf.stop_gradient(
        tf.reduce_mean(entropy) - FLAGS.target_entropy)
  else:
    entropy_adjustment_loss = 0. * agent.entropy_cost()  # to avoid None in grad

  total_loss = actor_loss + q_loss + v_loss + entropy_adjustment_loss

  # Q-function
  session = logger.log_session()
  logger.log(session, 'Q/value', tf.reduce_mean(q_action))
  logger.log(session, 'Q/L2 error', tf.sqrt(tf.reduce_mean(tf.square(q_error))))
  # V-function
  logger.log(session, 'V/value', tf.reduce_mean(v))
  logger.log(session, 'V/L2 error', tf.sqrt(tf.reduce_mean(tf.square(v_error))))
  # losses
  logger.log(session, 'losses/actor', actor_loss)
  logger.log(session, 'losses/Q', q_loss)
  logger.log(session, 'losses/V', v_loss)
  logger.log(session, 'losses/total', total_loss)
  # policy
  dist = parametric_action_distribution.create_dist(action_params)
  if hasattr(dist, 'scale'):
    logger.log(session, 'policy/std', tf.reduce_mean(dist.scale))
  logger.log(session, 'policy/max_action_abs(before_tanh)',
             tf.reduce_max(tf.abs(action)))
  logger.log(session, 'policy/entropy', entropy)
  logger.log(session, 'policy/entropy_cost', agent.entropy_cost())

  return total_loss, session


Unroll = collections.namedtuple(
    'Unroll', 'agent_state prev_actions env_outputs agent_actions')


def create_dataset(unroll_queue, replay_buffer, strategy, batch_size, encode):
  """Creates a dataset sampling from replay buffer.

  This dataset will consume a batch of unrolls from 'unroll_queue', add it to
  the replay buffer, and sample a batch of unrolls from it.

  Args:
    unroll_queue: Queue of 'Unroll' elements.
    replay_buffer: Replay buffer of 'Unroll' elements.
    strategy: A `distribute_lib.Strategy`.
    batch_size: Batch size used for consuming the unroll queue and sampling from
      the replay buffer.
    encode: Function to encode the data for TPU, etc.

  Returns:
    A "distributed `Dataset`", which acts like a `tf.data.Dataset` except it
    produces "PerReplica" values. Each iteration of the dataset produces
    flattened `Unroll` structures where per-timestep tensors have front
    dimensions [unroll_length, batch_size_per_replica].
  """

  @tf.function
  def dequeue(ctx):
    """Inserts into and samples from the replay buffer.

    Args:
      ctx: tf.distribute.InputContext.

    Returns:
      A flattened `Unroll` structures where per-timestep tensors have
      front dimensions [unroll_length, batch_size_per_replica].
    """
    per_replica_batch_size = ctx.get_per_replica_batch_size(batch_size)

    while tf.constant(True):
      # Each tensor in 'unrolls' has shape [insertion_batch_size, unroll_length,
      # <field-specific dimensions>].
      insert_batch_size = get_replay_insertion_batch_size(per_replica=True)
      unrolls = unroll_queue.dequeue_many(insert_batch_size)

      # The replay buffer is not threadsafe (and making it thread-safe might
      # slow it down), which is why we insert and sample in a single thread, and
      # use TF Queues for passing data between threads.
      replay_buffer.insert(unrolls, priorities=tf.ones(insert_batch_size))

      if replay_buffer.num_inserted >= FLAGS.batch_size:
        break

    _, _, sampled_unrolls = replay_buffer.sample(per_replica_batch_size,
                                                 priority_exp=0.)
    sampled_unrolls = sampled_unrolls._replace(
        prev_actions=utils.make_time_major(sampled_unrolls.prev_actions),
        env_outputs=utils.make_time_major(sampled_unrolls.env_outputs),
        agent_actions=utils.make_time_major(sampled_unrolls.agent_actions))
    sampled_unrolls = sampled_unrolls._replace(
        env_outputs=encode(sampled_unrolls.env_outputs))
    # tf.data.Dataset treats list leafs as tensors, so we need to flatten and
    # repack.
    return tf.nest.flatten(sampled_unrolls)

  def dataset_fn(ctx):
    dataset = tf.data.Dataset.from_tensors(0).repeat(None)
    return dataset.map(lambda _: dequeue(ctx))

  return strategy.experimental_distribute_datasets_from_function(dataset_fn)


def get_replay_insertion_batch_size(per_replica=False):
  insertion_per_replica = int(
      FLAGS.batch_size / FLAGS.replay_ratio / FLAGS.num_training_tpus)
  if FLAGS.her_window_length:
    # We insert chunks of length her_window_length but pull from the buffer
    # unrolls of length unroll_length.
    insertion_per_replica *= FLAGS.unroll_length
    insertion_per_replica //= FLAGS.her_window_length
  if per_replica:
    return insertion_per_replica
  else:
    return FLAGS.num_training_tpus * insertion_per_replica


def validate_config():
  assert (
      FLAGS.num_actors + FLAGS.unroll_queue_max_size >=
      get_replay_insertion_batch_size(per_replica=True)
  ), ('Insertion batch size can not be bigger than num_actors + '
      'unroll_queue_max_size.'
     )
  assert get_replay_insertion_batch_size(per_replica=True) >= 1, (
      'Replay ratio is bigger than batch size per replica.')
  if FLAGS.inference_batch_size == -1:
    FLAGS.inference_batch_size = max(1, FLAGS.num_actors // 2)
  assert FLAGS.num_actors >= FLAGS.inference_batch_size, (
      'Inference batch size is bigger than the number of actors.')
  if FLAGS.her_window_length:
    assert FLAGS.her_window_length >= FLAGS.unroll_length, (
        'The HER window can not be shorter than the unroll length.')


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
  logging.info('Starting learner loop')
  validate_config()
  settings = utils.init_learner(FLAGS.num_training_tpus)
  strategy, inference_devices, training_strategy, encode, decode = settings
  env = create_env_fn(0)
  parametric_action_distribution = get_parametric_distribution_for_action_space(
      env.action_space)
  env_output_specs = utils.EnvOutput(
      tf.TensorSpec([], tf.float32, 'reward'),
      tf.TensorSpec([], tf.bool, 'done'),
      tf.nest.map_structure(
          lambda s: tf.TensorSpec(s.shape, s.dtype, 'observation'),
          env.observation_space.__dict__.get('spaces', env.observation_space)),
      tf.TensorSpec([], tf.bool, 'abandoned'),
      tf.TensorSpec([], tf.int32, 'episode_step'),
  )
  action_specs = tf.TensorSpec(env.action_space.shape,
                               env.action_space.dtype, 'action')

  # Initialize agent and variables.
  agent = create_agent_fn(env.action_space, env.observation_space,
                          parametric_action_distribution)
  target_agent = create_agent_fn(env.action_space, env.observation_space,
                                 parametric_action_distribution)
  initial_agent_state = agent.initial_state(1)
  agent_state_specs = tf.nest.map_structure(
      lambda t: tf.TensorSpec(t.shape[1:], t.dtype), initial_agent_state)
  agent_input_specs = (action_specs, env_output_specs)

  input_ = tf.nest.map_structure(
      lambda s: tf.zeros([1, 1] + list(s.shape), s.dtype), agent_input_specs)
  input_no_time = tf.nest.map_structure(lambda t: t[0], input_)

  input_ = encode(input_ + (initial_agent_state,))
  input_no_time = encode(input_no_time + (initial_agent_state,))

  with strategy.scope():
    # Initialize variables
    def initialize_agent_variables(agent):
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
      @tf.function
      def create_variables():
        return [agent.get_action(*decode(input_no_time)),
                agent.get_V(*decode(input_)),
                agent.get_Q(*decode(input_), action=decode(input_[0]))]
      create_variables()

    initialize_agent_variables(agent)
    initialize_agent_variables(target_agent)

    # Target network update
    @tf.function
    def update_target_agent(polyak):
      """Synchronizes training and target agent variables."""
      variables = agent.variables
      target_variables = target_agent.variables
      assert len(target_variables) == len(variables), (
          'Mismatch in number of net tensors: {} != {}'.format(
              len(target_variables), len(variables)))
      for target_var, source_var in zip(target_variables, variables):
        target_var.assign(polyak * target_var +
                          (1. - polyak) * source_var)

    update_target_agent(polyak=0.)  # copy weights

    # Create optimizer.
    iter_frame_ratio = (get_replay_insertion_batch_size(per_replica=False) *
                        (FLAGS.her_window_length or FLAGS.unroll_length)
                        * FLAGS.num_action_repeats)
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
      args = tf.nest.pack_sequence_as(unroll_specs, decode(args, data))
      with tf.GradientTape() as tape:
        loss, logs = compute_loss(logger, parametric_action_distribution, agent,
                                  target_agent, *args)
      grads = tape.gradient(loss, agent.trainable_variables)
      for t, g in zip(temp_grads, grads):
        t.assign(g)
      return loss, logs

    loss, logs = training_strategy.experimental_run_v2(compute_gradients,
                                                       (data,))
    loss = training_strategy.experimental_local_results(loss)[0]

    def apply_gradients(_):
      optimizer.apply_gradients(zip(temp_grads, agent.trainable_variables))

    strategy.experimental_run_v2(apply_gradients, (loss,))

    getattr(agent, 'end_of_training_step_callback',
            lambda: logging.info('end_of_training_step_callback not found'))()

    logger.step_end(logs, training_strategy, iter_frame_ratio)

  # Logging.
  summary_writer = tf.summary.create_file_writer(
      FLAGS.logdir, flush_millis=20000, max_queue=1000)
  logger = utils.ProgressLogger(summary_writer=summary_writer)

  # Setup checkpointing and restore checkpoint.
  ckpt = tf.train.Checkpoint(agent=agent, target_agent=target_agent,
                             optimizer=optimizer)
  if FLAGS.init_checkpoint is not None:
    tf.print('Loading initial checkpoint from %s...' % FLAGS.init_checkpoint)
    ckpt.restore(FLAGS.init_checkpoint).assert_consumed()
  manager = tf.train.CheckpointManager(
      ckpt, FLAGS.logdir, max_to_keep=1, keep_checkpoint_every_n_hours=6)
  last_ckpt_time = 0  # Force checkpointing of the initial model.
  if manager.latest_checkpoint:
    logging.info('Restoring checkpoint: %s', manager.latest_checkpoint)
    ckpt.restore(manager.latest_checkpoint).assert_consumed()
    last_ckpt_time = time.time()

  server = grpc.Server([FLAGS.server_address])

  store = utils.UnrollStore(
      FLAGS.num_actors, FLAGS.her_window_length or FLAGS.unroll_length,
      (action_specs, env_output_specs, action_specs))
  actor_run_ids = utils.Aggregator(FLAGS.num_actors,
                                   tf.TensorSpec([], tf.int64, 'run_ids'))
  info_specs = (
      tf.TensorSpec([], tf.int64, 'episode_num_frames'),
      tf.TensorSpec([], tf.float32, 'episode_returns'),
      tf.TensorSpec([], tf.float32, 'episode_raw_returns'),
  )
  actor_infos = utils.Aggregator(FLAGS.num_actors, info_specs, 'actor_infos')

  # First agent state in an unroll.
  first_agent_states = utils.Aggregator(
      FLAGS.num_actors, agent_state_specs, 'first_agent_states')

  # Current agent state and action.
  agent_states = utils.Aggregator(
      FLAGS.num_actors, agent_state_specs, 'agent_states')
  actions = utils.Aggregator(FLAGS.num_actors, action_specs, 'actions')

  unroll_specs = Unroll(agent_state_specs, *store.unroll_specs)
  unroll_queue = utils.StructuredFIFOQueue(FLAGS.unroll_queue_max_size,
                                           unroll_specs)
  info_queue = utils.StructuredFIFOQueue(-1, info_specs)

  if FLAGS.her_window_length:
    replay_buffer = utils.HindsightExperienceReplay(
        FLAGS.replay_buffer_size,
        unroll_specs,
        compute_reward_fn=env.compute_reward,
        unroll_length=FLAGS.unroll_length,
        importance_sampling_exponent=0.,
        substitution_probability=FLAGS.her_substitution_probability)
  else:
    replay_buffer = utils.PrioritizedReplay(FLAGS.replay_buffer_size,
                                            unroll_specs,
                                            importance_sampling_exponent=0.)

  def add_batch_size(ts):
    return tf.TensorSpec([FLAGS.inference_batch_size] + list(ts.shape),
                         ts.dtype, ts.name)

  inference_iteration = tf.Variable(-1)
  inference_specs = (
      tf.TensorSpec([], tf.int32, 'actor_id'),
      tf.TensorSpec([], tf.int64, 'run_id'),
      env_output_specs,
      tf.TensorSpec([], tf.float32, 'raw_reward'),
  )
  inference_specs = tf.nest.map_structure(add_batch_size, inference_specs)
  @tf.function(input_signature=inference_specs)
  def inference(actor_ids, run_ids, env_outputs, raw_rewards):
    # Reset the actors that had their first run or crashed.
    previous_run_ids = actor_run_ids.read(actor_ids)
    actor_run_ids.replace(actor_ids, run_ids)
    reset_indices = tf.where(tf.not_equal(previous_run_ids, run_ids))[:, 0]
    actors_needing_reset = tf.gather(actor_ids, reset_indices)
    if tf.not_equal(tf.shape(actors_needing_reset)[0], 0):
      tf.print('Actor ids needing reset:', actors_needing_reset)
    actor_infos.reset(actors_needing_reset)
    store.reset(actors_needing_reset)
    initial_agent_states = agent.initial_state(
        tf.shape(actors_needing_reset)[0])
    first_agent_states.replace(actors_needing_reset, initial_agent_states)
    agent_states.replace(actors_needing_reset, initial_agent_states)
    actions.reset(actors_needing_reset)

    tf.debugging.assert_non_positive(
        tf.cast(env_outputs.abandoned, tf.int32),
        'Abandoned done states are not supported in SAC.')

    # Update steps and return.
    actor_infos.add(actor_ids, (0, env_outputs.reward, raw_rewards))
    done_ids = tf.gather(actor_ids, tf.where(env_outputs.done)[:, 0])
    info_queue.enqueue_many(actor_infos.read(done_ids))
    actor_infos.reset(done_ids)
    actor_infos.add(actor_ids, (FLAGS.num_action_repeats, 0., 0.))

    # Inference.
    prev_actions = parametric_action_distribution.postprocess(
        actions.read(actor_ids))
    input_ = encode((prev_actions, env_outputs))
    prev_agent_states = agent_states.read(actor_ids)
    def make_inference_fn(inference_device):
      def device_specific_inference_fn():
        with tf.device(inference_device):
          @tf.function
          def agent_inference(*args):
            return agent(*decode(args), is_training=False,
                         postprocess_action=False)

          return agent_inference(*input_, prev_agent_states)

      return device_specific_inference_fn

    # Distribute the inference calls among the inference cores.
    branch_index = inference_iteration.assign_add(1) % len(inference_devices)
    agent_actions, curr_agent_states = tf.switch_case(branch_index, {
        i: make_inference_fn(inference_device)
        for i, inference_device in enumerate(inference_devices)
    })

    # Append the latest outputs to the unroll and insert completed unrolls in
    # queue.
    completed_ids, unrolls = store.append(
        actor_ids, (prev_actions, env_outputs, agent_actions))
    unrolls = Unroll(first_agent_states.read(completed_ids), *unrolls)
    unroll_queue.enqueue_many(unrolls)
    first_agent_states.replace(completed_ids,
                               agent_states.read(completed_ids))

    # Update current state.
    agent_states.replace(actor_ids, curr_agent_states)
    actions.replace(actor_ids, agent_actions)

    # Return environment actions to actors.
    return parametric_action_distribution.postprocess(agent_actions)

  with strategy.scope():
    server.bind(inference, batched=True)
  server.start()

  dataset = create_dataset(unroll_queue, replay_buffer, training_strategy,
                           FLAGS.batch_size, encode)
  it = iter(dataset)

  def additional_logs():
    tf.summary.scalar('learning_rate', learning_rate_fn(iterations))
    tf.summary.scalar('buffer/unrolls_inserted', replay_buffer.num_inserted)
    # log data from info_queue
    n_episodes = info_queue.size()
    n_episodes -= n_episodes % FLAGS.log_episode_frequency
    if tf.not_equal(n_episodes, 0):
      episode_stats = info_queue.dequeue_many(n_episodes)
      episode_keys = [
          'episode_num_frames', 'episode_return', 'episode_raw_return'
      ]
      for key, values in zip(episode_keys, episode_stats):
        for value in tf.split(values,
                              values.shape[0] // FLAGS.log_episode_frequency):
          tf.summary.scalar(key, tf.reduce_mean(value))

      for (frames, ep_return, raw_return) in zip(*episode_stats):
        logging.info('Return: %f Raw return: %f Frames: %i', ep_return,
                     raw_return, frames)

  logger.start(additional_logs)
  # Execute learning.
  while iterations < final_iteration:
    if iterations.numpy() % FLAGS.update_target_every_n_step == 0:
      update_target_agent(FLAGS.polyak)
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
  server.shutdown()
  unroll_queue.close()
