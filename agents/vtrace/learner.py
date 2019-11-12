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

import collections
import concurrent.futures
import math
import time

from absl import flags
from absl import logging

from seed_rl import grpc
from seed_rl.common import common_flags  
from seed_rl.common import utils
from seed_rl.common import vtrace
from seed_rl.common.parametric_distribution import get_parametric_distribution_for_action_space

import tensorflow as tf




# Training.
flags.DEFINE_integer('save_checkpoint_secs', 1800,
                     'Checkpoint save period in seconds.')
flags.DEFINE_integer('total_environment_frames', int(1e9),
                     'Total environment frames to train for.')
flags.DEFINE_integer('batch_size', 2, 'Batch size for training.')
flags.DEFINE_integer('inference_batch_size', 2, 'Batch size for inference.')
flags.DEFINE_integer('unroll_length', 100, 'Unroll length in agent steps.')
flags.DEFINE_integer('num_training_tpus', 1, 'Number of TPUs for training.')

# Loss settings.
flags.DEFINE_float('entropy_cost', 0.00025, 'Entropy cost/multiplier.')
flags.DEFINE_float('baseline_cost', .5, 'Baseline cost/multiplier.')
flags.DEFINE_float('kl_cost', 0., 'KL(old_policy|new_policy) loss multiplier.')
flags.DEFINE_float('discounting', .99, 'Discounting factor.')
flags.DEFINE_float('lambda_', 1., 'Lambda.')
flags.DEFINE_float('max_abs_reward', 0.,
                   'Maximum absolute reward when calculating loss.'
                   'Use 0. to disable clipping.')

# Actors
flags.DEFINE_integer('num_actors', 4, 'Number of actors.')


FLAGS = flags.FLAGS


def compute_loss(parametric_action_distribution, agent, agent_state,
                 prev_actions, env_outputs, agent_outputs):
  # agent((prev_actions[t], env_outputs[t]), agent_state)
  #   -> agent_outputs[t], agent_state'
  learner_outputs, _ = agent((prev_actions, env_outputs),
                             agent_state,
                             unroll=True,
                             is_training=True)

  # Use last baseline value (from the value function) to bootstrap.
  bootstrap_value = learner_outputs.baseline[-1]

  # At this point, we have unroll length + 1 steps. The last step is only used
  # as bootstrap value, so it's removed.
  agent_outputs = tf.nest.map_structure(lambda t: t[:-1], agent_outputs)
  rewards, done, _ = tf.nest.map_structure(lambda t: t[1:], env_outputs)
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
  entropy_loss = FLAGS.entropy_cost * -entropy

  # KL(old_policy|new_policy) loss
  kl = behaviour_action_log_probs - target_action_log_probs
  kl_loss = FLAGS.kl_cost * tf.reduce_mean(kl)

  total_loss = policy_loss + v_loss + entropy_loss + kl_loss

  return total_loss


Unroll = collections.namedtuple(
    'Unroll', 'agent_state prev_actions env_outputs agent_outputs')


def validate_config():
  assert FLAGS.num_actors >= FLAGS.batch_size, (
      'Batch size is bigger than the number of actors.')
  assert FLAGS.num_actors >= FLAGS.inference_batch_size, (
      'Inference batch size is bigger than the number of actors.')


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
      tf.TensorSpec(env.observation_space.shape, env.observation_space.dtype,
                    'observation'),
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
  input_ = tf.nest.map_structure(
      lambda s: tf.zeros([1] + list(s.shape), s.dtype), agent_input_specs)
  input_ = encode(input_)

  with strategy.scope():
    @tf.function
    def create_variables(*args):
      return agent(*decode(args))

    initial_agent_output, _ = create_variables(input_, initial_agent_state)
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
      args = tf.nest.pack_sequence_as(unroll_specs, decode(args, data))
      with tf.GradientTape() as tape:
        loss = compute_loss(parametric_action_distribution, agent, *args)
      grads = tape.gradient(loss, agent.trainable_variables)
      for t, g in zip(temp_grads, grads):
        t.assign(g)
      return loss

    loss = training_strategy.experimental_run_v2(compute_gradients, (data,))
    loss = training_strategy.experimental_local_results(loss)[0]

    def apply_gradients(_):
      optimizer.apply_gradients(zip(temp_grads, agent.trainable_variables))

    strategy.experimental_run_v2(apply_gradients, (loss,))

  agent_output_specs = tf.nest.map_structure(
      lambda t: tf.TensorSpec(t.shape[1:], t.dtype), initial_agent_output)
  # Logging.
  summary_writer = tf.summary.create_file_writer(
      FLAGS.logdir, flush_millis=20000, max_queue=1000)

  # Setup checkpointing and restore checkpoint.
  ckpt = tf.train.Checkpoint(agent=agent, optimizer=optimizer)
  manager = tf.train.CheckpointManager(
      ckpt, FLAGS.logdir, max_to_keep=1, keep_checkpoint_every_n_hours=6)
  last_ckpt_time = 0  # Force checkpointing of the initial model.
  if manager.latest_checkpoint:
    logging.info('Restoring checkpoint: %s', manager.latest_checkpoint)
    ckpt.restore(manager.latest_checkpoint).assert_consumed()
    last_ckpt_time = time.time()

  server = grpc.Server([FLAGS.server_address])

  store = utils.UnrollStore(
      FLAGS.num_actors, FLAGS.unroll_length,
      (action_specs, env_output_specs, agent_output_specs))
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

  unroll_specs = Unroll(agent_state_specs, *store.unroll_specs)
  unroll_queue = utils.StructuredFIFOQueue(1, unroll_specs)
  info_queue = utils.StructuredFIFOQueue(-1, info_specs)

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

    # Update steps and return.
    actor_infos.add(actor_ids, (0, env_outputs.reward, raw_rewards))
    done_ids = tf.gather(actor_ids, tf.where(env_outputs.done)[:, 0])
    info_queue.enqueue_many(actor_infos.read(done_ids))
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
            return agent(*decode(args), is_training=True)

          return agent_inference(input_, prev_agent_states)

      return device_specific_inference_fn

    # Distribute the inference calls among the inference cores.
    branch_index = inference_iteration.assign_add(1) % len(inference_devices)
    agent_outputs, curr_agent_states = tf.switch_case(branch_index, {
        i: make_inference_fn(inference_device)
        for i, inference_device in enumerate(inference_devices)
    })

    # Append the latest outputs to the unroll and insert completed unrolls in
    # queue.
    completed_ids, unrolls = store.append(
        actor_ids, (prev_actions, env_outputs, agent_outputs))
    unrolls = Unroll(first_agent_states.read(completed_ids), *unrolls)
    unroll_queue.enqueue_many(unrolls)
    first_agent_states.replace(completed_ids,
                               agent_states.read(completed_ids))

    # Update current state.
    agent_states.replace(actor_ids, curr_agent_states)
    actions.replace(actor_ids, agent_outputs.action)

    # Return environment actions to actors.
    return parametric_action_distribution.postprocess(agent_outputs.action)

  with strategy.scope():
    server.bind(inference, batched=True)
  server.start()

  def dequeue(ctx):
    # Create batch (time major).
    actor_outputs = tf.nest.map_structure(lambda *args: tf.stack(args), *[
        unroll_queue.dequeue()
        for i in range(ctx.get_per_replica_batch_size(FLAGS.batch_size))
    ])
    actor_outputs = actor_outputs._replace(
        prev_actions=utils.make_time_major(actor_outputs.prev_actions),
        env_outputs=utils.make_time_major(actor_outputs.env_outputs),
        agent_outputs=utils.make_time_major(actor_outputs.agent_outputs))
    actor_outputs = actor_outputs._replace(
        env_outputs=encode(actor_outputs.env_outputs))
    # tf.data.Dataset treats list leafs as tensors, so we need to flatten and
    # repack.
    return tf.nest.flatten(actor_outputs)

  def dataset_fn(ctx):
    dataset = tf.data.Dataset.from_tensors(0).repeat(None)
    return dataset.map(lambda _: dequeue(ctx),
                       num_parallel_calls=ctx.num_replicas_in_sync)

  dataset = training_strategy.experimental_distribute_datasets_from_function(
      dataset_fn)
  it = iter(dataset)

  # Execute learning and track performance.
  with summary_writer.as_default(), \
    concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
    log_future = executor.submit(lambda: None)  # No-op future.
    last_num_env_frames = iterations * iter_frame_ratio
    last_log_time = time.time()
    while iterations < final_iteration:
      num_env_frames = iterations * iter_frame_ratio
      tf.summary.experimental.set_step(num_env_frames)

      # Save checkpoint.
      current_time = time.time()
      if current_time - last_ckpt_time >= FLAGS.save_checkpoint_secs:
        manager.save()
        last_ckpt_time = current_time

      def log(num_env_frames):
        """Logs actor summaries."""
        summary_writer.set_as_default()
        tf.summary.experimental.set_step(num_env_frames)
        episode_num_frames, episode_returns, episode_raw_returns = (
            info_queue.dequeue_many(info_queue.size()))
        for n, r, s in zip(episode_num_frames, episode_returns,
                           episode_raw_returns):
          logging.info('Return: %f Frames: %i', r, n)
          tf.summary.scalar('episode_return', r)
          tf.summary.scalar('episode_raw_return', s)
          tf.summary.scalar('num_episode_frames', n)
      log_future.result()  # Raise exception if any occurred in logging.
      log_future = executor.submit(log, num_env_frames)

      if current_time - last_log_time >= 120:
        df = tf.cast(num_env_frames - last_num_env_frames, tf.float32)
        dt = time.time() - last_log_time
        tf.summary.scalar('num_environment_frames/sec', df / dt)
        tf.summary.scalar('learning_rate', learning_rate_fn(iterations))

        last_num_env_frames, last_log_time = num_env_frames, time.time()

      minimize(it)

  manager.save()
  server.shutdown()
  unroll_queue.close()
