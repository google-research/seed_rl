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
"""Deep recurrent Q-learning learner in SEED's distributed architecture.

This contains all the features from R2D2
(https://openreview.net/pdf?id=r1lyTjAqYX), namely:
- Double Q-learning.
- Dueling network architecture.
- Prioritized replay buffer with realistic initial priorities and importance
  sampling correction.
- Value function re-scaling.
- Recurrent Q-network, with recurrent state stored in replay buffer, and
  burned-in at training time.
- n-step Bellman targets.
- Target network.
"""

import collections
import concurrent.futures
import math
import time

from absl import app
from absl import flags
from absl import logging
from seed_rl import grpc
from seed_rl.atari import config
from seed_rl.utils import utils
import tensorflow as tf


FLAGS = flags.FLAGS

Unroll = collections.namedtuple(
    'Unroll', 'agent_state priority prev_actions env_outputs agent_outputs')

# Unrolls sampled from the replay buffer. Contains the corresponding indices in
# the replay buffer and importance weights.
SampledUnrolls = collections.namedtuple(
    'SampledUnrolls', 'unrolls indices importance_weights')

# Information about a finished episode.
EpisodeInfo = collections.namedtuple(
    'EpisodeInfo',
    # num_frames: length of the episode in number of frames.
    # returns: Sum of undiscounted rewards experienced in the episode.
    # raw_returns: Sum of raw rewards experienced in the episode.
    # actor_ids: ID of the actor that generated this episode.
    'num_frames returns raw_returns actor_ids')


def get_replay_insertion_batch_size(per_replica=False):
  if per_replica:
    return int(FLAGS.batch_size / FLAGS.replay_ratio / FLAGS.num_training_tpus)
  else:
    return int(FLAGS.batch_size / FLAGS.replay_ratio)


def get_num_training_actors():
  return FLAGS.num_actors - FLAGS.num_eval_actors


def is_training_actor(actor_id):
  """Training actor IDs are in range [0, num_training_actors)."""
  return actor_id < get_num_training_actors()


def get_actors_epsilon(actor_ids, num_training_actors, num_eval_actors,
                       eval_epsilon):
  """Per-actor epsilon as in Apex and R2D2.

  Args:
    actor_ids: <int32>[inference_batch_size], the actor task IDs (in range
      [0, num_training_actors+num_eval_actors)).
    num_training_actors: Number of training actors. Training actors should have
      IDs in [0, num_training_actors).
    num_eval_actors: Number of evaluation actors. Eval actors should have IDs in
      [num_training_actors, num_training_actors + num_eval_actors).
    eval_epsilon: Epsilon used for eval actors.

  Returns:
    A 1D float32 tensor with one epsilon for each input actor ID.
  """
  # <float32>[num_training_actors + num_eval_actors]
  epsilons = tf.concat(
      [tf.math.pow(0.4, tf.linspace(1., 8., num=num_training_actors)),
       tf.constant([eval_epsilon] * num_eval_actors)],
      axis=0)
  return tf.gather(epsilons, actor_ids)


def apply_epsilon_greedy(actions, actor_ids, num_training_actors,
                         num_eval_actors, eval_epsilon, num_actions):
  """Epsilon-greedy: randomly replace actions with given probability.

  Args:
    actions: <int32>[batch_size] tensor with one action per actor.
    actor_ids: <int32>[inference_batch_size], the actor task IDs (in range
      [0, num_actors)).
    num_training_actors: Number of training actors.
    num_eval_actors: Number of eval actors.
    eval_epsilon: Epsilon used for eval actors.
    num_actions: Number of environment actions.

  Returns:
    A new <int32>[batch_size] tensor with one action per actor. With probability
    epsilon, the new action is random, and with probability (1 -
    epsilon), the action is unchanged, where epsilon is chosen for each actor.
  """
  batch_size = tf.shape(actions)[0]
  epsilons = get_actors_epsilon(actor_ids, num_training_actors, num_eval_actors,
                                eval_epsilon)
  random_actions = tf.random.uniform([batch_size], maxval=num_actions,
                                     dtype=tf.int32)
  probs = tf.random.uniform(shape=[batch_size])
  return tf.where(tf.math.less(probs, epsilons), random_actions, actions)


def value_function_rescaling(x):
  """Value function rescaling per R2D2 paper, table 2."""
  eps = FLAGS.value_function_rescaling_epsilon
  return tf.math.sign(x) * (tf.math.sqrt(tf.math.abs(x) + 1.) - 1.) + eps * x


def inverse_value_function_rescaling(x):
  """See Proposition A.2 in paper "Observe and Look Further"."""
  eps = FLAGS.value_function_rescaling_epsilon
  return tf.math.sign(x) * (
      tf.math.square(((tf.math.sqrt(
          1. + 4. * eps * (tf.math.abs(x) + 1. + eps))) - 1.) / (2. * eps)) -
      1.)


def n_step_bellman_target(rewards, done, q_target, gamma, n_steps):
  r"""Computes n-step Bellman targets.

  See section 2.3 of R2D2 paper (which does not mention the logic around end of
  episode).

  Args:
    rewards: <float32>[time, batch_size] tensor. This is r_t in the equations
      below.
    done: <bool>[time, batch_size] tensor. This is done_t in the equations
      below. done_t should be true if the episode is done just after
      experimenting reward r_t.
    q_target: <float32>[time, batch_size] tensor. This is Q_target(s_{t+1}, a*)
      (where a* is an action chosen by the caller).
    gamma: Exponential RL discounting.
    n_steps: The number of steps to look ahead for computing the Bellman
      targets.

  Returns:
    y_t targets as <float32>[time, batch_size] tensor.
    When n_steps=1, this is just:

    $$r_t + gamma * (1 - done_t) * Q_{target}(s_{t+1}, a^*)$$

    In the general case, this is:

    $$(\sum_{i=0}^{n-1} \gamma ^ {i} * notdone_{t, i-1} * r_{t + i}) +
      \gamma ^ n * notdone_{t, n-1} * Q_{target}(s_{t + n}, a^*) $$

    where notdone_{t,i} is defined as:

    $$notdone_{t,i} = \prod_{k=0}^{k=i}(1 - done_{t+k})$$

    The last n_step-1 targets cannot be computed with n_step returns, since we
    run out of Q_{target}(s_{t+n}). Instead, they will use n_steps-1, .., 1 step
    returns. For those last targets, the last Q_{target}(s_{t}, a^*) is re-used
    multiple times.
  """
  # We append n_steps - 1 times the last q_target. They are divided by gamma **
  # k to correct for the fact that they are at a 'fake' indice, and will
  # therefore end up being multiplied back by gamma ** k in the loop below.
  # We prepend 0s that will be discarded at the first iteration below.
  bellman_target = tf.concat(
      [tf.zeros_like(q_target[0:1]), q_target] +
      [q_target[-1:] / gamma ** k
       for k in range(1, n_steps)],
      axis=0)
  # Pad with n_steps 0s. They will be used to compute the last n_steps-1
  # targets (having 0 values is important).
  done = tf.concat([done] + [tf.zeros_like(done[0:1])] * n_steps, axis=0)
  rewards = tf.concat([rewards] + [tf.zeros_like(rewards[0:1])] * n_steps,
                      axis=0)
  # Iteratively build the n_steps targets. After the i-th iteration (1-based),
  # bellman_target is effectively the i-step returns.
  for _ in range(n_steps):
    rewards = rewards[:-1]
    done = done[:-1]
    bellman_target = (
        rewards + gamma * (1. - tf.cast(done, tf.float32)) * bellman_target[1:])

  return bellman_target


def compute_loss_and_priorities_from_agent_outputs(
    training_agent_output,
    target_agent_output,
    env_outputs,
    agent_outputs,
    gamma, eta=0.9):
  """Computes the loss to optimize and the new priorities.

  This implements the n-step double DQN loss with the Q function computed on
  sequences of transitions.

  Args:
    training_agent_output: AgentOutput where tensors have [unroll_length,
      batch_size] front dimensions. Used for the Q values that should be
      learned, and for computing the max over next actions as part of
      double-DQN.
    target_agent_output: AgentOutput used to compute the double-DQN target.
    env_outputs: EnvOutputs where tensors have [unroll_length, batch_size]
      additional front dimensions.
    agent_outputs: AgentOutputs where tensors have [unroll_length, batch_size]
      additional front dimensions.
    gamma: RL discounting factor.
    eta: float, parameter for balancing mean and max TD errors over a sequence
      of transitions for computing its new priority.

  Returns:
    A pair:
      - The loss for each unroll. Shape <float32>[batch_size]
      - The new priorities for each unroll. Shape <float32>[batch_size].
  """
  num_actions = tf.shape(training_agent_output.q_values)[2]
  # <float32>[time, batch_size, num_actions].
  replay_action_one_hot = tf.one_hot(agent_outputs.action, num_actions, 1., 0.)
  # This is Q(s, a), where a is the action played (can come from the agent or be
  # random). This is what we learn.
  # <float32>[time, batch_size].
  replay_q = tf.reduce_sum(
      training_agent_output.q_values * replay_action_one_hot, axis=2)

  # [time, batch_size, num_actions]
  best_actions_one_hot = tf.one_hot(
      training_agent_output.action, num_actions, 1., 0.)
  # This is Q'(s) = h^(-1)(Q_target(s, \argmax_a Q(s, a)))
  # [time, batch_size]
  qtarget_max = inverse_value_function_rescaling(tf.reduce_sum(
      target_agent_output.q_values * best_actions_one_hot,
      axis=2))

  # [time, batch_size]
  bellman_target = tf.stop_gradient(n_step_bellman_target(
      env_outputs.reward,
      env_outputs.done,
      qtarget_max,
      gamma,
      FLAGS.n_steps))

  # replay_q is actually Q(s_{t+1}, a_{t+1}), so we need to shift the targets.
  bellman_target = bellman_target[1:, ...]

  replay_q = replay_q[:-1, ...]

  bellman_target = value_function_rescaling(bellman_target)
  # [time, batch_size]
  abs_td_errors = tf.abs(bellman_target - replay_q)

  # [batch_size]
  priorities = (eta * tf.reduce_max(abs_td_errors, axis=0) +
                (1 - eta) * tf.reduce_mean(abs_td_errors, axis=0))

  # Sums over time dimension.
  loss = 0.5 * tf.reduce_sum(tf.math.square(abs_td_errors), axis=0)

  return loss, priorities


def compute_loss_and_priorities(
    training_agent, target_agent,
    agent_state, prev_actions, env_outputs, agent_outputs,
    gamma, burn_in):
  """Computes the loss to optimize and the new priorities.

  This implements the n-step double DQN loss with the Q function computed on
  sequences of transitions.

  Args:
    training_agent: Keras Model representing the training agent's network.
    target_agent: Keras Model representing the target agent's network.
    agent_state: Batched agent recurrent state at the beginning of each unroll.
    prev_actions: <int32>[unroll_length, batch_size]. This is the action played
      in the environment before each corresponding "env_outputs" (i.e. after
      epsilon-greedy policy is applied).
    env_outputs: EnvOutputs where tensors have [unroll_length, batch_size]
      additional front dimensions.
    agent_outputs: AgentOutputs where tensors have [unroll_length, batch_size]
      additional front dimensions.
    gamma: RL discounting factor.
    burn_in: Number of time steps on which we update each recurrent state
      without computing the loss nor propagating gradients.

  Returns:
    A pair:
      - The loss for each unroll. Shape <float32>[batch_size]
      - The new priorities for each unroll. Shape <float32>[batch_size].
  """
  if burn_in:
    agent_input_prefix, agent_input_suffix = utils.split_structure(
        (prev_actions, env_outputs), burn_in)
    _, agent_outputs_suffix = utils.split_structure(agent_outputs, burn_in)
    _, training_state = training_agent(
        agent_input_prefix, agent_state, unroll=True)
    training_state = tf.nest.map_structure(tf.stop_gradient, training_state)
    _, target_state = target_agent(agent_input_prefix, agent_state, unroll=True)
  else:
    agent_input_suffix = (prev_actions, env_outputs)
    agent_outputs_suffix = agent_outputs
    training_state = agent_state
    target_state = agent_state

  # Agent outputs have fields with shape [time, batch_size, <field_shape>].
  training_agent_output, _ = training_agent(
      agent_input_suffix, training_state, unroll=True)
  target_agent_output, _ = target_agent(
      agent_input_suffix, target_state, unroll=True)
  _, env_outputs_suffix = agent_input_suffix
  return compute_loss_and_priorities_from_agent_outputs(
      training_agent_output, target_agent_output, env_outputs_suffix,
      agent_outputs_suffix, gamma)


def create_dataset(unroll_queue, replay_buffer, strategy, batch_size,
                   priority_exponent, encode):
  """Creates a dataset sampling from replay buffer.

  This dataset will consume a batch of unrolls from 'unroll_queue', add it to
  the replay buffer, and sample a batch of unrolls from it.

  Args:
    unroll_queue: Queue of 'Unroll' elements.
    replay_buffer: Replay buffer of 'Unroll' elements.
    strategy: A `distribute_lib.Strategy`.
    batch_size: Batch size used for consuming the unroll queue and sampling from
      the replay buffer.
    priority_exponent: Priority exponent used for sampling from the replay
      buffer.
    encode: Function to encode the data for TPU, etc.

  Returns:
    A "distributed `Dataset`", which acts like a `tf.data.Dataset` except it
    produces "PerReplica" values. Each iteration of the dataset produces
    flattened `SampledUnrolls` structures where per-timestep tensors have front
    dimensions [unroll_length, batch_size_per_replica].
  """

  @tf.function
  def dequeue(ctx):
    """Inserts into and samples from the replay buffer.

    Args:
      ctx: tf.distribute.InputContext.

    Returns:
      A flattened `SampledUnrolls` structures where per-timestep tensors have
      front dimensions [unroll_length, batch_size_per_replica].
    """
    per_replica_batch_size = ctx.get_per_replica_batch_size(batch_size)
    insertion_batch_size = get_replay_insertion_batch_size(per_replica=True)

    print_every = tf.cast(
        insertion_batch_size *
        (1 + FLAGS.replay_buffer_min_size // 50 // insertion_batch_size),
        tf.int64)
    log_summary_every = tf.cast(insertion_batch_size * 500, tf.int64)

    while tf.constant(True):
      # Each tensor in 'unrolls' has shape [insertion_batch_size, unroll_length,
      # <field-specific dimensions>].
      unrolls = unroll_queue.dequeue_many(insertion_batch_size)
      # The replay buffer is not threadsafe (and making it thread-safe might
      # slow it down), which is why we insert and sample in a single thread, and
      # use TF Queues for passing data between threads.
      replay_buffer.insert(unrolls, unrolls.priority)
      if tf.equal(replay_buffer.num_inserted % log_summary_every, 0):
        # Unfortunately, there is no tf.summary(log_every_n_sec).
        tf.summary.histogram('initial_priorities', unrolls.priority)
      if replay_buffer.num_inserted >= FLAGS.replay_buffer_min_size:
        break

      if tf.equal(replay_buffer.num_inserted % print_every, 0):
        tf.print('Waiting for the replay buffer to fill up. '
                 'It currently has', replay_buffer.num_inserted,
                 'elements, waiting for at least', FLAGS.replay_buffer_min_size,
                 'elements')

    sampled_indices, weights, sampled_unrolls = replay_buffer.sample(
        per_replica_batch_size, priority_exponent)
    sampled_unrolls = sampled_unrolls._replace(
        prev_actions=utils.make_time_major(sampled_unrolls.prev_actions),
        env_outputs=utils.make_time_major(sampled_unrolls.env_outputs),
        agent_outputs=utils.make_time_major(sampled_unrolls.agent_outputs))
    sampled_unrolls = sampled_unrolls._replace(
        env_outputs=encode(sampled_unrolls.env_outputs))
    # tf.data.Dataset treats list leafs as tensors, so we need to flatten and
    # repack.
    return tf.nest.flatten(SampledUnrolls(
        sampled_unrolls, sampled_indices, weights))

  def dataset_fn(ctx):
    dataset = tf.data.Dataset.from_tensors(0).repeat(None)
    return dataset.map(lambda _: dequeue(ctx))

  return strategy.experimental_distribute_datasets_from_function(dataset_fn)


def validate_config():
  assert FLAGS.num_actors >= FLAGS.inference_batch_size, (
      'Inference batch size is bigger than the number of actors.')
  assert FLAGS.n_steps >= 1, '--n_steps < 1 does not make sense.'
  assert FLAGS.num_actors > FLAGS.num_eval_actors, (
      'Total number of actors ({}) should be greater than number of actors '
      'reserved to eval ({})'.format(
          FLAGS.num_actors, FLAGS.num_eval_actors))


def main(_):
  validate_config()
  settings = utils.init_learner(FLAGS.num_training_tpus)
  strategy, inference_devices, training_strategy, encode, decode = settings
  # Environment specification.
  env = config.create_environment(0)
  env_output_specs = utils.EnvOutput(
      tf.TensorSpec([], tf.float32, 'reward'),
      tf.TensorSpec([], tf.bool, 'done'),
      tf.TensorSpec(env.observation_space.shape, env.observation_space.dtype,
                    'observation'),
  )
  action_specs = tf.TensorSpec([], tf.int32, 'action')
  num_actions = env.action_space.n
  agent_input_specs = (action_specs, env_output_specs)

  # Initialize agent and variables.
  agent = config.create_agent(env_output_specs, num_actions)
  target_agent = config.create_agent(env_output_specs, num_actions)
  initial_agent_state = agent.initial_state(1)
  agent_state_specs = tf.nest.map_structure(
      lambda t: tf.TensorSpec(t.shape[1:], t.dtype), initial_agent_state)
  input_ = tf.nest.map_structure(
      lambda s: tf.zeros([1] + list(s.shape), s.dtype), agent_input_specs)
  input_ = encode(input_)

  with strategy.scope():

    @tf.function
    def create_variables(*args):
      return agent(*decode(args))

    @tf.function
    def create_target_agent_variables(*args):
      return target_agent(*decode(args))

    # The first call to Keras models to create varibales for agent and target.
    initial_agent_output, _ = create_variables(input_, initial_agent_state)
    create_target_agent_variables(input_, initial_agent_state)

    @tf.function
    def update_target_agent():
      """Synchronizes training and target agent variables."""
      variables = agent.trainable_variables
      target_variables = target_agent.trainable_variables
      assert len(target_variables) == len(variables), (
          'Mismatch in number of net tensors: {} != {}'.format(
              len(target_variables), len(variables)))
      for target_var, source_var in zip(target_variables, variables):
        target_var.assign(source_var)

    # Create optimizer.
    iter_frame_ratio = (
        get_replay_insertion_batch_size() *
        FLAGS.unroll_length * FLAGS.num_action_repeats)
    final_iteration = int(
        math.ceil(FLAGS.total_environment_frames / iter_frame_ratio))
    optimizer, learning_rate_fn = config.create_optimizer(final_iteration)


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
    """Computes and applies gradients.

    Args:
      iterator: An iterator of distributed dataset that produces `PerReplica`.

    Returns:
      A tuple:
        - priorities, the new priorities. Shape <float32>[batch_size].
        - indices, the indices for updating priorities. Shape
        <int32>[batch_size].
        - gradient_norm_before_clip, a scalar.
    """
    data = next(iterator)

    def compute_gradients(args):
      """A function to pass to `Strategy` for gradient computation."""
      args = decode(args, data)
      args = tf.nest.pack_sequence_as(SampledUnrolls(unroll_specs, 0, 0), args)
      with tf.GradientTape() as tape:
        # loss: [batch_size]
        # priorities: [batch_size]
        loss, priorities = compute_loss_and_priorities(
            agent,
            target_agent,
            args.unrolls.agent_state,
            args.unrolls.prev_actions,
            args.unrolls.env_outputs,
            args.unrolls.agent_outputs,
            gamma=FLAGS.discounting,
            burn_in=FLAGS.burn_in)
        loss = tf.reduce_mean(loss * args.importance_weights)
      grads = tape.gradient(loss, agent.trainable_variables)
      gradient_norm_before_clip = tf.linalg.global_norm(grads)
      if FLAGS.clip_norm:
        grads, _ = tf.clip_by_global_norm(
            grads, FLAGS.clip_norm, use_norm=gradient_norm_before_clip)

      for t, g in zip(temp_grads, grads):
        t.assign(g)

      return loss, priorities, args.indices, gradient_norm_before_clip

    loss, priorities, indices, gradient_norm_before_clip = (
        training_strategy.experimental_run_v2(compute_gradients, (data,)))
    loss = training_strategy.experimental_local_results(loss)[0]

    def apply_gradients(loss):
      optimizer.apply_gradients(zip(temp_grads, agent.trainable_variables))
      return loss


    loss = strategy.experimental_run_v2(apply_gradients, (loss,))

    # convert PerReplica to a Tensor
    if not isinstance(priorities, tf.Tensor):

      priorities = tf.reshape(tf.stack(priorities.values), [-1])
      indices = tf.reshape(tf.stack(indices.values), [-1])
      gradient_norm_before_clip = tf.reshape(
          tf.stack(gradient_norm_before_clip.values), [-1])
      gradient_norm_before_clip = tf.reduce_max(gradient_norm_before_clip)

    return loss, priorities, indices, gradient_norm_before_clip

  agent_output_specs = tf.nest.map_structure(
      lambda t: tf.TensorSpec(t.shape[1:], t.dtype), initial_agent_output)
  # Logging.
  summary_writer = tf.summary.create_file_writer(
      FLAGS.logdir, flush_millis=20000, max_queue=1000)

  # Setup checkpointing and restore checkpoint.

  ckpt = tf.train.Checkpoint(
      agent=agent, target_agent=target_agent, optimizer=optimizer)
  manager = tf.train.CheckpointManager(
      ckpt, FLAGS.logdir, max_to_keep=1, keep_checkpoint_every_n_hours=6)
  last_ckpt_time = 0  # Force checkpointing of the initial model.
  if manager.latest_checkpoint:
    logging.info('Restoring checkpoint: %s', manager.latest_checkpoint)
    ckpt.restore(manager.latest_checkpoint).assert_consumed()
    last_ckpt_time = time.time()

  server = grpc.Server([FLAGS.server_address])

  # Buffer of incomplete unrolls. Filled during inference with new transitions.
  # This only contains data from training actors.
  store = utils.UnrollStore(
      get_num_training_actors(), FLAGS.unroll_length,
      (action_specs, env_output_specs, agent_output_specs),
      num_overlapping_steps=FLAGS.burn_in)
  actor_run_ids = utils.Aggregator(FLAGS.num_actors,
                                   tf.TensorSpec([], tf.int64, 'run_ids'))
  info_specs = (
      tf.TensorSpec([], tf.int64, 'episode_num_frames'),
      tf.TensorSpec([], tf.float32, 'episode_returns'),
      tf.TensorSpec([], tf.float32, 'episode_raw_returns'),
  )
  actor_infos = utils.Aggregator(FLAGS.num_actors, info_specs)

  # First agent state in an unroll.
  first_agent_states = utils.Aggregator(FLAGS.num_actors, agent_state_specs)

  # Current agent state and action.
  agent_states = utils.Aggregator(FLAGS.num_actors, agent_state_specs)
  actions = utils.Aggregator(FLAGS.num_actors, action_specs)

  unroll_specs = Unroll(agent_state_specs,
                        tf.TensorSpec([], tf.float32, 'priority'),
                        *store.unroll_specs)
  # Queue of complete unrolls. Filled by the inference threads, and consumed by
  # the tf.data.Dataset thread.
  unroll_queue = utils.StructuredFIFOQueue(
      FLAGS.unroll_queue_max_size, unroll_specs)
  episode_info_specs = EpisodeInfo(*(
      info_specs + (tf.TensorSpec([], tf.int32, 'actor_ids'),)))
  info_queue = utils.StructuredFIFOQueue(-1, episode_info_specs)

  replay_buffer = utils.PrioritizedReplay(FLAGS.replay_buffer_size,
                                          unroll_specs,
                                          FLAGS.importance_sampling_exponent)

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
    """Agent inference.

    This evaluates the agent policy on the provided environment data (reward,
    done, observation), and store appropriate data to feed the main training
    loop.

    Args:
      actor_ids: <int32>[inference_batch_size], the actor task IDs (in range
        [0, num_tasks)).
      run_ids: <int64>[inference_batch_size], the actor run IDs. Actor
        generate a random int64 run id at startup, so this can be used to detect
        the actors jobs that restarted.
      env_outputs: Follows env_output_specs, but with the inference_batch_size
        added as first dimension. These are the actual environment outputs
        (reward, done, observation).
      raw_rewards: <float32>[inference_batch_size], representing the raw reward
        of each step.

    Returns:
      A tensor <int32>[inference_batch_size] with one action for each actor.
    """
    # Reset the actors that had their first run or crashed.
    previous_run_ids = actor_run_ids.read(actor_ids)
    actor_run_ids.replace(actor_ids, run_ids)
    reset_indices = tf.where(tf.not_equal(previous_run_ids, run_ids))[:, 0]
    actors_needing_reset = tf.gather(actor_ids, reset_indices)
    if tf.not_equal(tf.shape(actors_needing_reset)[0], 0):
      tf.print('Actors needing reset:', actors_needing_reset)
    actor_infos.reset(actors_needing_reset)
    store.reset(tf.gather(
        actors_needing_reset,
        tf.where(is_training_actor(actors_needing_reset))[:, 0]))
    initial_agent_states = agent.initial_state(
        tf.shape(actors_needing_reset)[0])
    first_agent_states.replace(actors_needing_reset, initial_agent_states)
    agent_states.replace(actors_needing_reset, initial_agent_states)
    actions.reset(actors_needing_reset)

    # Update steps and return.
    actor_infos.add(actor_ids, (0, env_outputs.reward, raw_rewards))
    done_ids = tf.gather(actor_ids, tf.where(env_outputs.done)[:, 0])
    done_episodes_info = actor_infos.read(done_ids)
    info_queue.enqueue_many(EpisodeInfo(*(done_episodes_info + (done_ids,))))
    actor_infos.reset(done_ids)
    actor_infos.add(actor_ids, (FLAGS.num_action_repeats, 0., 0.))

    # Inference.
    prev_actions = actions.read(actor_ids)
    input_ = encode((prev_actions, env_outputs))
    prev_agent_states = agent_states.read(actor_ids)
    def make_inference_fn(inference_device):
      def device_specific_inference_fn():
        with tf.device(inference_device):
          @tf.function
          def agent_inference(*args):
            return agent(*decode(args))

          return agent_inference(input_, prev_agent_states)

      return device_specific_inference_fn

    # Distribute the inference calls among the inference cores.
    branch_index = inference_iteration.assign_add(1) % len(inference_devices)
    agent_outputs, curr_agent_states = tf.switch_case(branch_index, {
        i: make_inference_fn(inference_device)
        for i, inference_device in enumerate(inference_devices)
    })

    agent_outputs = agent_outputs._replace(
        action=apply_epsilon_greedy(
            agent_outputs.action, actor_ids,
            get_num_training_actors(),
            FLAGS.num_eval_actors, FLAGS.eval_epsilon, num_actions))

    # Append the latest outputs to the unroll, only for experience coming from
    # training actors (IDs < num_training_actors), and insert completed unrolls
    # in queue.
    # <int64>[num_training_actors]
    training_indices = tf.where(is_training_actor(actor_ids))[:, 0]
    training_actor_ids = tf.gather(actor_ids, training_indices)
    training_prev_actions, training_env_outputs, training_agent_outputs = (
        tf.nest.map_structure(lambda s: tf.gather(s, training_indices),
                              (prev_actions, env_outputs, agent_outputs)))

    append_to_store = (
        training_prev_actions, training_env_outputs, training_agent_outputs)
    completed_ids, completed_unrolls = store.append(
        training_actor_ids, append_to_store)
    _, unrolled_env_outputs, unrolled_agent_outputs = completed_unrolls
    unrolled_agent_states = first_agent_states.read(completed_ids)

    # Only use the suffix of the unrolls that is actually used for training. The
    # prefix is only used for burn-in of agent state at training time.
    _, agent_outputs_suffix = utils.split_structure(
        utils.make_time_major(unrolled_agent_outputs), FLAGS.burn_in)
    _, env_outputs_suffix = utils.split_structure(

        utils.make_time_major(unrolled_env_outputs), FLAGS.burn_in)
    _, initial_priorities = compute_loss_and_priorities_from_agent_outputs(
        # We don't use the outputs from a separated target network for computing
        # initial priorities.
        agent_outputs_suffix,
        agent_outputs_suffix,
        env_outputs_suffix,
        agent_outputs_suffix,
        gamma=FLAGS.discounting)

    unrolls = Unroll(unrolled_agent_states, initial_priorities,
                     *completed_unrolls)
    unroll_queue.enqueue_many(unrolls)
    first_agent_states.replace(completed_ids,
                               agent_states.read(completed_ids))

    # Update current state.
    agent_states.replace(actor_ids, curr_agent_states)
    actions.replace(actor_ids, agent_outputs.action)

    # Return environment actions to actors.
    return agent_outputs.action

  with strategy.scope():
    server.bind(inference, batched=True)
  server.start()

  # Execute learning and track performance.
  with summary_writer.as_default(), \
    concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
    log_future = executor.submit(lambda: None)  # No-op future.
    tf.summary.experimental.set_step(iterations * iter_frame_ratio)
    dataset = create_dataset(unroll_queue, replay_buffer, training_strategy,
                             FLAGS.batch_size, FLAGS.priority_exponent, encode)
    it = iter(dataset)

    last_num_env_frames = iterations * iter_frame_ratio
    last_log_time = time.time()
    max_gradient_norm_before_clip = 0.
    while iterations < final_iteration:
      num_env_frames = iterations * iter_frame_ratio
      tf.summary.experimental.set_step(num_env_frames)

      if iterations.numpy() % FLAGS.update_target_every_n_step == 0:
        update_target_agent()

      # Save checkpoint.
      current_time = time.time()
      if current_time - last_ckpt_time >= FLAGS.save_checkpoint_secs:
        manager.save()
        last_ckpt_time = current_time

      def log(num_env_frames):
        """Logs actor summaries."""
        summary_writer.set_as_default()
        tf.summary.experimental.set_step(num_env_frames)
        episode_info = info_queue.dequeue_many(info_queue.size())
        for n, r, _, actor_id in zip(*episode_info):
          is_training = is_training_actor(actor_id)
          logging.info(
              'Return: %f Frames: %i Actor id: %i (%s) Iteration: %i',
              r, n, actor_id,
              'training' if is_training else 'eval',
              iterations.numpy())
          if not is_training:
            tf.summary.scalar('eval/episode_return', r)
            tf.summary.scalar('eval/episode_frames', n)
      log_future.result()  # Raise exception if any occurred in logging.
      log_future = executor.submit(log, num_env_frames)

      _, priorities, indices, gradient_norm = minimize(it)

      replay_buffer.update_priorities(indices, priorities)
      # Max of gradient norms (before clipping) since last tf.summary export.
      max_gradient_norm_before_clip = max(gradient_norm.numpy(),
                                          max_gradient_norm_before_clip)
      if current_time - last_log_time >= 120:
        df = tf.cast(num_env_frames - last_num_env_frames, tf.float32)
        dt = time.time() - last_log_time
        tf.summary.scalar('num_environment_frames/sec (actors)', df / dt)
        tf.summary.scalar('num_environment_frames/sec (learner)',
                          df / dt * FLAGS.replay_ratio)

        tf.summary.scalar('learning_rate', learning_rate_fn(iterations))
        tf.summary.scalar('replay_buffer_num_inserted',
                          replay_buffer.num_inserted)
        tf.summary.scalar('unroll_queue_size', unroll_queue.size())

        last_num_env_frames, last_log_time = num_env_frames, time.time()
        tf.summary.histogram('updated_priorities', priorities)
        tf.summary.scalar('max_gradient_norm_before_clip',
                          max_gradient_norm_before_clip)
        max_gradient_norm_before_clip = 0.

  manager.save()
  server.shutdown()
  unroll_queue.close()


if __name__ == '__main__':
  app.run(main)
