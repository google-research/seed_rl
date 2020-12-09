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
"""Configurable onpolicy learner."""

import collections
import math
import os
import time
from typing import List, Optional

from absl import flags
from absl import logging
import dataclasses
import numpy as np
from seed_rl import grpc
from seed_rl.agents.policy_gradient import eval_utils
from seed_rl.agents.policy_gradient import learner_config
from seed_rl.agents.policy_gradient.modules import advantages as gae  
from seed_rl.agents.policy_gradient.modules import generalized_onpolicy_loss
from seed_rl.agents.policy_gradient.modules import policy_losses  
from seed_rl.agents.policy_gradient.modules import policy_regularizers  
from seed_rl.agents.policy_gradient.modules import ppo_training_step_utils as ppo_utils
from seed_rl.common import common_flags  
from seed_rl.common import parametric_distribution
from seed_rl.common import utils
import tensorflow as tf
from tensorflow.python.ops import state_ops  

FLAGS = flags.FLAGS

Unroll = collections.namedtuple(
    'Unroll', 'agent_state prev_actions env_outputs agent_outputs')

# State of the learner. Can be used for quick restores.
LearnerState = collections.namedtuple('LearnerState',
                                      'training_agent loss_fn optimizer')

# Recorded snapshot.
# steps_done - at which step snapshot was taken.
# model - snapshoted model.
Snapshot = collections.namedtuple('Snapshot', 'steps_done learner_state')

# Set of objects returned from the training loop.
# snapshots - list of recorded snapshots.
TrainingResult = collections.namedtuple('TrainingResult', 'snapshots')

# Represents a training step during which some additional actions should happen.
# step - number of the step to take action.
# model - should model be saved.
# snapshot - should model be snapshoted and returned as a training result.
# checkpoint - should model be checkpointed.
SaveActionPoint = collections.namedtuple('SaveActionPoint',
                                         'step model snapshot checkpoint')


def validate_config(config, settings):
  num_hosts = len(settings.hosts)
  utils.validate_learner_config(config, num_hosts)
  assert config.num_envs > config.num_eval_envs, (
      'Total number of environments ({}) should be greater than number '
      'reserved to eval ({})'.format(config.num_envs, config.num_eval_envs))
  assert (not config.block_inference_on_training or
          settings.training_strategy.num_replicas_in_sync == num_hosts
         ), 'block_inference_on_training not supported for multi-TPU training'


@dataclasses.dataclass
class TrainingHost(object):
  """State of a training host."""
  device: str
  inference_devices: List[str]
  server: Optional[grpc.Server]
  env_infos: utils.Aggregator
  inference_iterations: tf.Variable
  completed_unrolls: tf.Variable
  store: utils.UnrollStore
  env_run_ids: utils.Aggregator
  first_agent_states: utils.Aggregator
  agent_states: utils.Aggregator
  actions: utils.Aggregator
  unroll_queue: utils.StructuredFIFOQueue
  inference_fns: Optional[List[tf.function]]
  deterministic_inference: tf.Variable
  store_unrolls_on_inference: tf.Variable


class Learner(object):
  """PPO learner."""

  def __init__(self,
               create_env_fn,
               create_agent_fn,
               create_optimizer_fn,
               settings,
               config: learner_config.TrainingConfig,
               create_loss_fn=None,
               action_distribution_config=None):
    """Creates a new PPO learner.

    This initializes the state of the learner, including the training and
    inference agents and the optimizer, that persist between training runs.

    Args:
      create_env_fn: Callable that must return a newly created environment. The
        callable takes the task ID as argument - an arbitrary task ID of 0 will
        be passed by the learner. The returned environment should follow GYM's
        API. It is only used for infering tensor shapes. This environment will
        not be used to generate experience.
      create_agent_fn: Function that must create a new tf.Module with the neural
        network that outputs actions and new agent state given the environment
        observations and previous agent state. See dmlab.agents.ImpalaDeep for
        an example. The factory function takes as input the environment action
        and observation spaces and a parametric distribution over actions.
      create_optimizer_fn: Function that takes the final iteration as argument
        and must return a tf.keras.optimizers.Optimizer and a
        tf.keras.optimizers.schedules.LearningRateSchedule.
      settings: Strategy and inference settings, e.g. as returned by
        utils.init_learner().
      config: PPO config and parameters.
      create_loss_fn: Callable that must return a newly created loss function.
        Its arguments will be the training agent and the parametric action
        distribution. If None, then GeneralizedOnPolicyLoss configured though
        Gin will be used.
      action_distribution_config: configuration for ParametricDistribution over
        actions; the actual distribution also depends on the action spec
        retrieved from the environment. If None (and actions are continuous),
        uses the setting parametric_distribution.continuous_action_config().
    """
    validate_config(config, settings)
    self.server_initialized = False
    self.training_strategy = settings.training_strategy
    self.strategy = settings.strategy
    self.encode = settings.encode
    self.decode = settings.decode
    self.config = config
    self.create_agent_fn = create_agent_fn
    self.snapshots = None

    self.virtual_batch_size = config.batch_size * config.batches_per_step
    self.frames_per_virtual_batch = (
        self.virtual_batch_size * config.unroll_length *
        config.num_action_repeats)
    self.final_iteration = int(
        math.ceil(config.total_environment_frames /
                  self.frames_per_virtual_batch))
    self.training_iterations = tf.Variable(0, trainable=False, dtype=tf.int64)
    self.summary_writer = None
    self.logger = utils.ProgressLogger()

    # Create the environment and {training, inference} agents.
    # SEED Mujoco env factory only supports FLAGS.num_action_repeats == 1.
    # We can set it to 1 here because we only use this environment to
    # get observation and action shapes.
    num_action_repeats = FLAGS.num_action_repeats
    FLAGS.num_action_repeats = 1
    self.env = create_env_fn(0, config)
    FLAGS.num_action_repeats = num_action_repeats
    if action_distribution_config is None:
      action_distribution_config = (
          parametric_distribution.continuous_action_config())
    self.parametric_action_distribution = (
        parametric_distribution.get_parametric_distribution_for_action_space(
            self.env.action_space,
            continuous_config=action_distribution_config))
    try:
      # Deterministic inference assumes that the distribution has a proper mean.
      # This is not the case for some of the distributions, e.g. categorical.
      # They throw an exception which we use to determine whether deterministic
      # inference can be supported or not.
      self.parametric_action_distribution(
          tf.zeros((self.parametric_action_distribution.param_size,))).mean()
      self.supports_deterministic_inference = True
    except:  
      logging.info('Deterministic inference is not supported.')

      self.supports_deterministic_inference = False
    self.env_output_specs = utils.EnvOutput(
        tf.TensorSpec([], tf.float32, 'reward'),
        tf.TensorSpec([], tf.bool, 'done'),
        tf.TensorSpec(self.env.observation_space.shape,
                      self.env.observation_space.dtype, 'observation'),
        tf.TensorSpec([], tf.bool, 'abandoned'),
        tf.TensorSpec([], tf.int32, 'episode_step'),
    )
    self.action_specs = tf.TensorSpec(self.env.action_space.shape,
                                      self.env.action_space.dtype, 'action')
    self.agent_input_specs = (self.action_specs, self.env_output_specs)

    # Initialize agent and variables. We create two sets of parameters, one for
    # training on the training cores and one for inference on all the cores.
    self.inference_agent = self.build_new_agent()
    self.training_agent = self.build_new_agent()

    if create_loss_fn:
      self.loss_fn = create_loss_fn(
          self.training_agent,
          parametric_action_distribution=self.parametric_action_distribution)
    else:
      # This is configured through gin.
      self.loss_fn = generalized_onpolicy_loss.GeneralizedOnPolicyLoss(
          self.training_agent,
          parametric_action_distribution=self.parametric_action_distribution,
          frame_skip=config.num_action_repeats)

    initial_agent_state = self.inference_agent.initial_state(1)
    self.agent_state_specs = tf.nest.map_structure(
        lambda t: tf.TensorSpec(t.shape[1:], t.dtype), initial_agent_state)

    input_ = tf.nest.map_structure(
        lambda s: tf.zeros([1] + list(s.shape), s.dtype),
        self.agent_input_specs)
    # input_ contains 0s for the env observation. However, this might not be a
    # valid observation (e.g. when the env returns a packed graph vector where
    # some components refer to number of nodes), so we do proper obs sampling
    # from the observation_space here.
    input_ = (input_[0], input_[1]._replace(
        observation=tf.expand_dims(self.env.observation_space.sample(), axis=0))
             )
    input_ = self.encode(input_)
    self.input_ = input_

    # Create training variables on all the replicas of training_strategy.
    with self.training_strategy.scope():

      @tf.function
      def create_training_variables(*args):
        variables = self.training_agent(*self.decode(args))
        _ = self.training_agent.get_action(*self.decode(args))
        self.loss_fn.init()  # creates Lagrange multipliers
        return variables

      initial_agent_output, _ = create_training_variables(
          input_, initial_agent_state)
      self.optimizer, self.learning_rate_fn = create_optimizer_fn(
          self.final_iteration * self.config.epochs_per_step *
          self.config.batches_per_step)


      self.optimizer_iterations = self.optimizer.iterations
      self.optimizer._create_hypers()  
      self.optimizer._create_slots(self.loss_fn.trainable_variables)  

    # Create inference variables and temporary model variables (for copying) on
    # all the replicas of strategy.
    with self.strategy.scope():

      @tf.function
      def create_inference_variables(*args):
        return self.inference_agent(*self.decode(args))

      initial_agent_output, _ = create_inference_variables(
          input_, initial_agent_state)

      # ON_READ causes the replicated variable to act as independent variables
      # for each replica. The key idea is that all these variables are zero on
      # all the replicas. We then only change them to hold the model parameters
      # on the first training_strategy replica after finishing a training step.
      # Finally, we can all_reduce with ReduceOp.SUM to update the model
      # variables on the inference_model which live on all the replicas of
      # strategy. This works around the fact that we are operating on different
      # replicas when doing training and inference and allows us to copy weights
      # in-between different strategies.
      self.temp_per_replica_variables = []
      for v in self.training_agent.variables:
        self.temp_per_replica_variables.append(
            tf.Variable(
                tf.zeros_like(v),
                trainable=False,
                synchronization=tf.VariableSynchronization.ON_READ))

    self.agent_output_specs = tf.nest.map_structure(
        lambda t: tf.TensorSpec(t.shape[1:], t.dtype), initial_agent_output)

    def add_batch_size(ts, batch_size):
      return tf.TensorSpec([batch_size] + list(ts.shape), ts.dtype, ts.name)

    inference_specs = (
        tf.TensorSpec([], tf.int32, 'env_id'),
        tf.TensorSpec([], tf.int64, 'run_id'),
        self.env_output_specs,
        tf.TensorSpec([], tf.float32, 'raw_reward'),
    )
    self.inference_specs = tf.nest.map_structure(
        lambda s: add_batch_size(s, config.inference_batch_size),
        inference_specs)
    self.batched_action_specs = add_batch_size(self.action_specs,
                                               config.inference_batch_size)

    # Only the master uses the evaluator (that's why there is no copy per host)
    self.evaluator = eval_utils.Evaluator(config.print_episode_summaries,
                                          config.log_episode_frequency)

    self.info_specs = (
        tf.TensorSpec([], tf.int64, 'episode_num_frames'),
        tf.TensorSpec([], tf.float32, 'episode_returns'),
        tf.TensorSpec([], tf.float32, 'episode_raw_returns'),
    )
    self.dummy_action = tf.constant(
        tf.zeros(
            shape=[config.inference_batch_size] +
            list(self.env.action_space.shape),
            dtype=self.env.action_space.dtype))

    # The initialization steps below depend on the number of the environments
    # and hosts.

    num_envs = config.num_envs

    self.hosts = []
    add_unroll_batch_size = (
        self.virtual_batch_size // self.training_strategy.num_replicas_in_sync)
    for (host, inference_devices) in settings.hosts:
      with tf.device(host):
        store = utils.UnrollStore(
            config.num_training_envs, config.unroll_length,
            self.agent_input_specs + (self.agent_output_specs,))
        self.unroll_specs = Unroll(self.agent_state_specs, *store.unroll_specs)
        self.add_unroll_spec = tf.nest.map_structure(
            lambda s: add_batch_size(s, add_unroll_batch_size),
            self.unroll_specs)
        self.hosts.append(
            TrainingHost(
                device=host,
                inference_devices=inference_devices,
                server=None,
                inference_iterations=tf.Variable(
                    0, trainable=False, dtype=tf.int64),
                completed_unrolls=tf.Variable(
                    0, trainable=False, dtype=tf.int64),
                store=store,
                env_infos=utils.Aggregator(num_envs, self.info_specs,
                                           'env_infos'),
                env_run_ids=utils.Aggregator(
                    num_envs, tf.TensorSpec([], tf.int64, 'run_ids')),
                # First agent state in an unroll.
                first_agent_states=utils.Aggregator(num_envs,
                                                    self.agent_state_specs,
                                                    'first_agent_states'),
                # Current agent state and action.
                agent_states=utils.Aggregator(num_envs, self.agent_state_specs,
                                              'agent_states'),
                actions=utils.Aggregator(num_envs, self.action_specs,
                                         'actions'),
                unroll_queue=utils.StructuredFIFOQueue(self.virtual_batch_size,
                                                       self.unroll_specs),
                inference_fns=None,
                deterministic_inference=tf.Variable(False, trainable=False),
                store_unrolls_on_inference=tf.Variable(
                    config.store_unrolls_on_inference, trainable=False)))

    if config.prefetch_batches:
      self.dataset = self.create_dataset()
      self.dataset_iterator = iter(self.dataset)

    # Logging.
    self.manager = None
    self.log_keys = []
    self.last_log_time = None
    self.last_num_env_frames = 0
    self.last_inference_iterations = 0

    @tf.function(input_signature=[
        tf.TensorSpec(t.shape, t.dtype) for t in self.training_agent.variables
    ])
    def copy_to_training_agent_variables(*args):
      """Copies provided variables to the training agent variables.

      Args:
        *args: Source variables to copy from.  This function will typically be
          run under self.training_strategy.
      """
      for tmp, var in zip(self.training_agent.variables, args):
        tmp.assign(var)

    self.copy_to_training_agent_variables = copy_to_training_agent_variables

    @tf.function(input_signature=[
        tf.TensorSpec(t.shape, t.dtype) for t in self.loss_fn.variables
    ])
    def copy_to_loss_fn_variables(*args):
      """Copies provided variables to the loss fn variables.

      Args:
        *args: Source variables to copy from.  This function will typically be
          run under self.training_strategy.
      """
      for tmp, var in zip(self.loss_fn.variables, args):
        tmp.assign(var)

    self.copy_to_loss_fn_variables = copy_to_loss_fn_variables

    @tf.function(input_signature=[
        tf.TensorSpec(t.shape, t.dtype) for t in self.optimizer.variables()
    ])
    def copy_to_optimizer_variables(*args):
      """Copies provided variables to the optimizer variables.

      Args:
        *args: Source variables to copy from.  This function will typically be
          run under self.training_strategy.
      """
      for tmp, var in zip(self.optimizer.variables(), args):
        tmp.assign(var)

    self.copy_to_optimizer_variables = copy_to_optimizer_variables

  def __del__(self):
    for host in self.hosts:
      host.unroll_queue.close()

  def build_new_agent(self):
    return self.create_agent_fn(self.env.action_space,
                                self.env.observation_space,
                                self.parametric_action_distribution)

  def random_agent_variables(self):
    agent = self.build_new_agent()
    initial_agent_state = self.inference_agent.initial_state(1)
    agent(*self.decode((self.input_, initial_agent_state)))
    return agent.variables

  def generate_save_action_steps(self):
    """Generates list of steps at which snapshot/checkpoint should be done."""
    steps_done = self.training_iterations * self.frames_per_virtual_batch
    snapshot_points = set(
        np.linspace(
            0,
            self.config.total_environment_frames,
            self.config.num_snapshots,
            dtype=np.int64))
    model_points = set(
        np.linspace(
            0,
            self.config.total_environment_frames,
            self.config.num_saved_models,
            dtype=np.int64))
    checkpoint_points = set(
        np.linspace(
            0,
            self.config.total_environment_frames,
            self.config.num_checkpoints,
            dtype=np.int64))
    points = []
    for s in sorted(snapshot_points.union(model_points, checkpoint_points)):
      if s > steps_done:
        points.append(
            SaveActionPoint(
                step=s,
                model=s in model_points,
                snapshot=s in snapshot_points,
                checkpoint=s in checkpoint_points))

    # Add a no-op guard at the end, so that list is never empty.
    points.append(
        SaveActionPoint(
            step=999999999999, model=False, snapshot=False, checkpoint=False))
    return points

  def run_eval(self, num_episodes: int, deterministic_inference: bool = False):
    """Performs evaluation runs of the current policy.

    Args:
      num_episodes: How many evaluation episodes to run.
      deterministic_inference: Whether to use deterministic inference or not.

    Returns:
        Evaluation score.
    """
    assert not deterministic_inference or self.supports_deterministic_inference
    # Evaluation runs without training, so do not block inference on adding
    # to the unroll queue.
    for host in self.hosts:
      host.deterministic_inference.assign(deterministic_inference)
      host.store_unrolls_on_inference.assign(False)

    # Evaluation doesn't save the model at the end.
    original_save_trained_model = self.config.save_trained_model
    self.config.save_trained_model = False

    # We want to run eval on `num_episodes`.
    original_log_episode_frequency = self.evaluator.log_episode_frequency
    self.evaluator.log_episode_frequency = num_episodes

    logging.info('Starting eval loop')
    self.start_actor_loops()
    while True:
      stats = self.evaluator.process(write_tf_summaries=False)
      if stats:
        break
      time.sleep(0.1)
    self.finalize()
    self.evaluator.log_episode_frequency = original_log_episode_frequency

    for host in self.hosts:
      host.deterministic_inference.assign(False)
      host.store_unrolls_on_inference.assign(
          self.config.store_unrolls_on_inference)

    self.config.save_trained_model = original_save_trained_model
    return stats

  def run_training(self):
    """Performs an E2E training run.

    For now can be called only once per learner instance (work in progress).

    Returns:
        Result of a training in a form of TrainingResult tuple.
    """
    logging.info('Starting learner loop')
    self.start_actor_loops()

    # calculate when to evaluate or when to save the model for evaluation later.

    snapshots = []
    action_points = self.generate_save_action_steps()
    snapshots_todo = sum([s.snapshot for s in action_points])
    if snapshots_todo:
      if self.snapshots is None:
        self.snapshots = [self.get_state_fn() for _ in range(snapshots_todo)]
      assert len(self.snapshots) == snapshots_todo, (
          'Not allowed to change the number of snapshots between runs')
    reporting_step = 0
    # Start measuring speed etc after first minimize finishes (so that we wait
    # for all actors to schedule). Unless we are profiling inference (in that
    # case, the minimize loop will be blocked). We use the value in the config
    # rather than per-host copies.
    if (self.config.profile_inference_return !=
        learner_config.InferenceReturn.END):
      self.logger.start(self.additional_logs)
      self.minimize_loop(tf.constant(1))
    else:
      self.minimize_loop(tf.constant(1))
      self.logger.start(self.additional_logs)
    current_action_point = 0
    while True:
      action_point = action_points[current_action_point]
      steps_done = self.training_iterations * self.frames_per_virtual_batch
      if action_point.step <= steps_done:
        start_time = time.time()
        if action_point.snapshot:
          snapshots.append(
              Snapshot(steps_done.numpy(),
                       self.snapshots[current_action_point]()))
        if action_point.model:
          self.save_model(eval_id=steps_done.numpy())
        if action_point.checkpoint:
          self.save_checkpoint()
        logging.info('Checkpoint/snapshot at %d steps took %f sec.', steps_done,
                     time.time() - start_time)
        current_action_point += 1
      if steps_done >= self.config.total_environment_frames:
        break
      # Make sure to run minimize at least once.
      reporting_step += max(FLAGS.log_frequency, self.frames_per_virtual_batch)
      reporting_step = min(reporting_step, action_point.step)
      self.minimize_loop(
          tf.constant(
              math.ceil(reporting_step / self.frames_per_virtual_batch)))
    self.logger.shutdown()
    self.finalize()
    return TrainingResult(snapshots)

  def get_state_fn(self):
    """Returns a function used for obtaining current learner's state.

       Each call to this function creates a new copy of tf.Variables.
       When returned function is called multiple times it returns the same
       set of variables.
    """
    with self.training_strategy.scope():
      state = LearnerState(
          training_agent=[
              tf.Variable(v) for v in self.training_agent.variables
          ],
          loss_fn=[tf.Variable(v) for v in self.loss_fn.variables],
          optimizer=[tf.Variable(v) for v in self.optimizer.variables()])

    @tf.function
    def copy_internal():
      for src, dest in zip(self.training_agent.variables, state.training_agent):
        dest.assign(src)
      for src, dest in zip(self.loss_fn.variables, state.loss_fn):
        dest.assign(src)
      for src, dest in zip(self.optimizer.variables(), state.optimizer):
        dest.assign(src)

    def copy():
      self.training_strategy.run(copy_internal, ())
      return state

    return copy

  def prepare_for_run(self,
                      logdir: str = None,
                      init_checkpoint: str = None,
                      restore_latest_checkpoint: bool = True,
                      reuse_summary_writer: bool = False,
                      init_server: bool = True) -> None:
    """Prepares the learner for a new run.

    This initializes the checkpoint manager for logging under `logdir`
    and restores a previous checkpoint if there is one and
    restore_latest_checkpoint is True. This functinality is useful for handling
    task preemptions.

    Args:
      logdir: Base directory for the logs and checkpoints.
      init_checkpoint: Path of the initial checkpoint to restore, or model
        returned by run_training method. Used only when previous checkpoint is
        not loaded.
      restore_latest_checkpoint: Should latest checkpoint created by
        CheckpointManager be restored.
      reuse_summary_writer: Whether to reuse the summary writer from the
        previous prepare_for_run() call if any.
      init_server: Initializes but does not start the gRPC server.
    """
    if init_checkpoint is None and 'init_checkpoint' in FLAGS:
      init_checkpoint = FLAGS.init_checkpoint
    if logdir is None:
      logdir = FLAGS.logdir
    self.logdir = logdir
    self.training_iterations.assign(0)

    # Setup checkpointing and restore checkpoint.
    self.ckpt = tf.train.Checkpoint(
        loss_fn=self.loss_fn,
        optimizer=self.optimizer,
        training_iterations=self.training_iterations)
    self.manager = tf.train.CheckpointManager(
        self.ckpt,
        self.logdir,
        max_to_keep=1,
        keep_checkpoint_every_n_hours=6,
    )

    # If there is a recent checkpoint in the current logdir, we use this one
    # (this indicates that this job was probably preempted).
    if restore_latest_checkpoint and self.manager.latest_checkpoint:
      latest_checkpoint = self.manager.latest_checkpoint
      logging.info('Restoring checkpoint: %s', latest_checkpoint)
      self.ckpt.restore(latest_checkpoint).assert_existing_objects_matched()
    elif init_checkpoint is not None:
      if isinstance(init_checkpoint, str):
        logging.info('Loading initial checkpoint from %s...', init_checkpoint)
        self.ckpt.restore(init_checkpoint).assert_existing_objects_matched()
        self.training_iterations.assign(0)
      else:
        with self.training_strategy.scope():
          if init_checkpoint.training_agent:
            self.copy_to_training_agent_variables(
                *init_checkpoint.training_agent)
          if init_checkpoint.loss_fn:
            self.copy_to_loss_fn_variables(*init_checkpoint.loss_fn)
          if init_checkpoint.optimizer:
            self.copy_to_optimizer_variables(*init_checkpoint.optimizer)

    # Logging.
    if self.summary_writer is None or not reuse_summary_writer:
      self.summary_writer = tf.summary.create_file_writer(
          self.logdir,
          flush_millis=self.config.summary_flush_seconds * 1000,
          max_queue=self.config.summary_max_queue_size)

    self.logger.reset(
        summary_writer=self.summary_writer,
        starting_step=self.frames_per_virtual_batch *
        self.training_iterations.read_value())

    # We need to explicitly copy the training agent variables to the inference
    # agent variables. This copying normally happens at the end of each training
    # step, which only happens after we've performed enough inference and
    # accumulated enough unrolls to train on.
    # This is needed both with the training agent is restored from checkpoint
    # (we need to propagate the weights to the inference agent) and when the
    # training agent is randomly initialized (we also want to propagate those
    # random weights to the inference agent, so that actors use on-policy data
    # with respect to the training agent).
    self.training_strategy.run(self.copy_to_temp_per_replica_variables,
                               (self.training_agent.variables,))
    self.strategy.run(self.copy_to_inference_variables, ())

    # Resets evaluator's inner state.
    self.evaluator.reset()

    if init_server:
      self.init_server()

    for host in self.hosts:
      host.completed_unrolls.assign(0)
      unroll_queue = host.unroll_queue
      while unroll_queue.size():
        unroll_queue.dequeue_up_to(unroll_queue.size())

  def finalize(self) -> None:
    """Finalizes the learner after training ends."""
    self.stop_actor_loops()

    # Save the model.
    if self.config.save_trained_model:
      self.manager.save()
      tf.saved_model.save(self.training_agent,
                          os.path.join(self.logdir, 'saved_model'))

  @property
  def done(self) -> bool:
    """Returns true if the training is done."""
    return self.training_iterations >= self.final_iteration

  @tf.function
  def copy_to_temp_per_replica_variables(self, src):
    """Copies provided variables to the temp ON_READ per replica variables.

    Args:
      src: Source variables to copy from.  Source variables will be copied to
        the first replica of temp_per_replica variables. Other replicas of
        temp_per_replica variables will be set to 0. This allows updating the
        inference agent variables later using a ReduceOp.Sum.  This function
        will typically be run under self.training_strategy.
    """
    for tmp, var in zip(self.temp_per_replica_variables, src):
      # Only copy the variables for the first replica of training_strategy.
      num = tf.distribute.get_replica_context().replica_id_in_sync_group
      value = tf.where(num == 0, var, tf.zeros_like(var))
      tmp.assign(value)

  @tf.function
  def copy_to_inference_variables(self):
    """Copies temp_per_replica variables to the inference agent variables.

    This assumes that only a single replica of the temp_per_replica_variables is
    set (as is done in copy_to_temp_per_replica_variables). This allows using
    ReduceOp.SUM to retrieve the value set in that single replica.

    This function will typically be run under self.strategy.
    """

    def distributed_copy(strategy, src_variables, dest_variables):
      reduced_src_variables = strategy.extended.batch_reduce_to(
          tf.distribute.ReduceOp.SUM, zip(src_variables, dest_variables))
      for src, dest in zip(reduced_src_variables, dest_variables):
        strategy.extended.update(dest, state_ops.assign, args=(src,))

    tf.distribute.get_replica_context().merge_call(
        distributed_copy,
        args=(self.temp_per_replica_variables, self.inference_agent.variables))

  def create_dataset(self):
    """Creates distributed tf.Dataset which pulls unrolls from unroll queues."""

    def dequeue(ctx):
      # Create batch (time major).
      # NOTE: dequeue_many seems to be performing worse than simple dequeue in
      # a loop.
      env_outputs = tf.nest.map_structure(
          lambda *args: tf.stack(args), *[
              self.hosts[ctx.input_pipeline_id].unroll_queue.dequeue()
              for i in range(
                  ctx.get_per_replica_batch_size(self.virtual_batch_size))
          ])
      env_outputs = env_outputs._replace(
          prev_actions=utils.make_time_major(env_outputs.prev_actions),
          env_outputs=utils.make_time_major(env_outputs.env_outputs),
          agent_outputs=utils.make_time_major(env_outputs.agent_outputs))
      env_outputs = env_outputs._replace(
          env_outputs=self.encode(env_outputs.env_outputs))
      # tf.data.Dataset treats list leafs as tensors, so we need to flatten and
      # repack.
      return tf.nest.flatten(env_outputs)

    def dataset_fn(ctx):
      dataset = tf.data.Dataset.from_tensors(0).repeat(None)

      def dequeue_fn(_):
        return dequeue(ctx)

      return dataset.map(
          dequeue_fn,
          num_parallel_calls=ctx.num_replicas_in_sync // len(self.hosts))

    # Dataset and iterator objects should not change between successive minimize
    # calls. Otherwise, it will be re-traced and compiled.
    return (self.training_strategy
            .experimental_distribute_datasets_from_function(dataset_fn))

  @tf.function
  def pull_batch_from_unroll_queue(self):
    """Returns the next batch distributed across replicas."""

    def dequeue(ctx):
      # Create batch (time major).
      # NOTE: dequeue_many seems to be performing worse than simple dequeue in
      # a loop.
      assert ctx.num_replicas_in_sync % len(self.hosts) == 0
      num_replicas_per_host = ctx.num_replicas_in_sync // len(self.hosts)
      
      env_outputs = tf.nest.map_structure(
          lambda *args: tf.stack(args), *[
              self.hosts[ctx.replica_id_in_sync_group //
                         num_replicas_per_host].unroll_queue.dequeue()
              for i in range(self.virtual_batch_size //
                             ctx.num_replicas_in_sync)
          ])
      env_outputs = env_outputs._replace(
          prev_actions=utils.make_time_major(env_outputs.prev_actions),
          env_outputs=utils.make_time_major(env_outputs.env_outputs),
          agent_outputs=utils.make_time_major(env_outputs.agent_outputs))
      env_outputs = env_outputs._replace(
          env_outputs=self.encode(env_outputs.env_outputs))
      return tf.nest.flatten(env_outputs)

    return self.training_strategy.experimental_distribute_values_from_function(
        dequeue)

  @tf.function
  def minimize(self):
    """Orchestrates a training step."""
    logging.info('Tracing minimize')
    if self.config.prefetch_batches:
      data = next(self.dataset_iterator)
    else:
      data = self.pull_batch_from_unroll_queue()

    def training_step(args):
      """Executes a single training step of PPO.

      Args:
        args: List of variables whose leading dimensions are time and batch_size
          * batches_per_step and that corresponds to a flattened Unroll struct.

      Returns:
        Loss and logs of the last virtual mini-batch.
      """
      args = self.decode(args, data)
      args = tf.nest.pack_sequence_as(self.unroll_specs, args)

      # Update input normalization stats once at the beginning of the training
      # step. Not all agents have this function defined.
      if hasattr(self.training_agent,
                 'update_observation_normalization_statistics'):
        self.training_agent.update_observation_normalization_statistics(
            args.env_outputs.observation[:-1])

      config = self.config
      loss, logs = ppo_utils.ppo_training_step(
          config.epochs_per_step, self.loss_fn, args, config.batch_mode,
          training_strategy, self.virtual_batch_size, config.unroll_length,
          config.batches_per_step, config.clip_norm, self.optimizer,
          self.logger)

      self.copy_to_temp_per_replica_variables(self.training_agent.variables)
      return loss, logs

    training_strategy = self.training_strategy
    loss, logs = training_strategy.run(training_step, (data,))
    loss = training_strategy.experimental_local_results(loss)[0]
    self.logger.step_end(
        logs,
        strategy=training_strategy,
        step_increment=self.frames_per_virtual_batch)

    with tf.control_dependencies([loss]):
      self.strategy.run(self.copy_to_inference_variables, ())
      self.training_iterations.assign_add(1)

    logging.info('End of tracing minimize')

  @tf.function
  def minimize_loop(self, iterations):
    iterations_int64 = tf.cast(iterations, tf.int64)
    while iterations_int64 > self.training_iterations.read_value():
      self.minimize()

  def init_server(self) -> None:
    """Initializes the TF gRPC server, but doesn't start it yet."""
    assert not self.server_initialized
    config = self.config
    for host_index, host in enumerate(self.hosts):
      with tf.device(host.device):
        inference_devices = host.inference_devices
        logging.info('Creating grpc %s %s %s', config.server_address,
                     host.device, inference_devices)
        server = grpc.Server([config.server_address])

        @tf.function(input_signature=[])
        def pending_minimize():
          training_batch = tf.math.floordiv(
              host.completed_unrolls,  
              self.virtual_batch_size)
          return self.training_iterations.read_value() < training_batch

        def get_inference_fn(host, inference_device):

          def log_inference():
            # make sure we log inference stats when we are profiling
            # inference
            inference_iterations = tf.reduce_sum(
                [host.inference_iterations for host in self.hosts])
            session = self.logger.log_session()
            # unless we log a scalar, the logger will fail

            self.logger.step_end(
                session,
                strategy=self.training_strategy,
                step_increment=inference_iterations)

          @tf.function
          def add_episode_stats(done_ids, env_infos):
            """Adds episode info of done environments to the evaluator."""
            training_ids = tf.boolean_mask(done_ids,
                                           config.is_training_env(done_ids))
            eval_ids = tf.boolean_mask(done_ids,
                                       not config.is_training_env(done_ids))
            training_infos = (tf.repeat(
                tf.constant(['training/'], dtype='string'),
                tf.shape(training_ids)[0]), *env_infos.read(training_ids))
            eval_infos = (tf.repeat(
                tf.constant(['eval_actors/'], dtype='string'),
                tf.shape(eval_ids)[0]), *env_infos.read(eval_ids))
            self.evaluator.add_many(training_infos)
            self.evaluator.add_many(eval_infos)

          # Moving this function to class or global scope introduces
          # performance hit.
          @tf.function(input_signature=self.inference_specs)
          def inference(env_ids, run_ids, env_outputs, raw_reward):
            host.inference_iterations.assign_add(1)
            # If we only want to profile the network performance, we immediately
            # return an arbitrary action.
            if (config.profile_inference_return ==
                learner_config.InferenceReturn.INSTANTLY):
              log_inference()
              return self.dummy_action

            # Reset the environments that had their first run or crashed.
            previous_run_ids = host.env_run_ids.read(env_ids)
            reset_mask = tf.not_equal(previous_run_ids, run_ids)
            envs_needing_reset = tf.boolean_mask(env_ids, reset_mask)
            nondying_envs_mask, nondying_env_ids = (
                utils.get_non_dying_envs(envs_needing_reset, reset_mask,
                                         env_ids))
            host.env_run_ids.replace(
                nondying_env_ids, tf.boolean_mask(run_ids, nondying_envs_mask))

            if tf.not_equal(tf.shape(envs_needing_reset)[0], 0):
              tf.print('Environment ids needing reset:', envs_needing_reset)
            initial_agent_states = self.inference_agent.initial_state(
                tf.shape(envs_needing_reset)[0])
            if host.store_unrolls_on_inference:
              host.store.reset(
                  tf.boolean_mask(envs_needing_reset,
                                  config.is_training_env(envs_needing_reset)))
              host.first_agent_states.replace(
                  envs_needing_reset,
                  initial_agent_states,
                  'reset_agents',
                  debug_tensors=[envs_needing_reset, previous_run_ids])
            host.agent_states.replace(envs_needing_reset, initial_agent_states)
            host.actions.reset(envs_needing_reset)

            if not config.add_eval_stats_from_actors:
              host.env_infos.reset(envs_needing_reset)
              host.env_infos.add(env_ids, (0, env_outputs.reward, raw_reward))
              done_ids = tf.gather(env_ids, tf.where(env_outputs.done)[:, 0])
              if host_index == 0:  
                add_episode_stats(done_ids, host.env_infos)
              host.env_infos.reset(done_ids)
              host.env_infos.add(env_ids, (config.num_action_repeats, 0., 0.))

            # Get previous actions.
            prev_actions = host.actions.read(env_ids)

            # We want to profile right before inference.
            if (config.profile_inference_return ==
                learner_config.InferenceReturn.BEFORE_INFERENCE):
              log_inference()
              return self.dummy_action

            # Prepare code for inference.
            input_ = self.encode((prev_actions, env_outputs))
            prev_agent_states = host.agent_states.read(env_ids)

            # Note: For performance reasons it is crucial to wrap the
            # self.inference_agent() call in a tf.function. If not, the method
            # does not fully run on TPU, instead it gets split between CPU and
            # TPU which results in significant performance hit.
            @tf.function
            def agent_inference(inputs, states):
              return self.inference_agent(
                  *self.decode((inputs, states)), is_training=True)

            with tf.device(inference_device):
              agent_outputs, curr_agent_states = agent_inference(
                  input_, prev_agent_states)

            # After inference, we directly return instead of unrolling.
            if (config.profile_inference_return ==
                learner_config.InferenceReturn.AFTER_INFERENCE):
              log_inference()
              return self.dummy_action

            if (host.store_unrolls_on_inference and
                config.profile_inference_return ==
                learner_config.InferenceReturn.END):
              # <bool>[inference_batch_size], True for transitions that should
              # be added to the store.
              # We only add experience coming from training environments, and
              # remove transitions from dying environments to make sure there is
              # no duplicate. See b/162235884.
              should_append_to_store = tf.logical_and(
                  config.is_training_env(env_ids), nondying_envs_mask)

              # <int64>[num_transitions_to_append_to_store] with the indices of
              # the transitions to append to the store.
              append_to_store_indices = tf.where(should_append_to_store)[:, 0]
              append_to_store_env_ids = tf.gather(env_ids,
                                                  append_to_store_indices)

              append_to_store = tf.nest.map_structure(
                  lambda s: tf.gather(s, append_to_store_indices),
                  (prev_actions, env_outputs, agent_outputs))

              # Add training experience to the unroll store and get finished
              # unrolls.
              completed_ids, unrolls = host.store.append(
                  append_to_store_env_ids, append_to_store)

              # Add finished unrolls to queue.
              unrolls = Unroll(
                  host.first_agent_states.read(completed_ids), *unrolls)
              host.completed_unrolls.assign_add(
                  tf.cast(len(unrolls[1]), tf.int64))
              host.unroll_queue.enqueue_many(unrolls)
              host.first_agent_states.replace(
                  completed_ids, host.agent_states.read(completed_ids),
                  'new_unrolls')

            # Update current state.
            host.agent_states.replace(
                nondying_env_ids,
                tf.nest.map_structure(
                    lambda t: tf.boolean_mask(t, nondying_envs_mask),
                    curr_agent_states))
            host.actions.replace(
                nondying_env_ids,
                tf.boolean_mask(agent_outputs.action, nondying_envs_mask))

            if (config.profile_inference_return ==
                learner_config.InferenceReturn.AFTER_UNROLL):
              log_inference()

            # Return environment actions to environments.
            if (self.supports_deterministic_inference and
                host.deterministic_inference):
              return self.parametric_action_distribution(
                  agent_outputs.policy_logits).mean()
            else:
              return agent_outputs.action

          return inference

        def get_config_fn(config):
          """Called by actors when re-connecting to get configuration."""

          @tf.function(input_signature=[])
          def get_config():
            return config

          return get_config

        @tf.function(input_signature=[self.evaluator.info_specs])
        def add_stats(data):
          """Called by actors to add {train,eval} episode stats."""
          if host_index == 0:  
            self.evaluator.add(data)

        @tf.function(input_signature=[self.add_unroll_spec])
        def add_unroll(data):
          # Data will be batched. enqueue_many will unstack the input tensors to
          # generate the list of unrolls.
          host.unroll_queue.enqueue_many(Unroll(*data))  

        with self.strategy.scope():
          # Avoid retracing inference function. This saves on time and
          # reduces memory usage.
          if host.inference_fns is None:
            host.inference_fns = [
                get_inference_fn(host, d) for d in inference_devices
            ]
          host.pending_minimize = pending_minimize
          server.bind(host.inference_fns)
          # Config is serialized outside of tf.function.
          server.bind(get_config_fn(utils.serialize_config(config)))
          server.bind(add_stats)
          server.bind(pending_minimize)
          # Binding uses the config and not the tf.Variable.
          if not config.store_unrolls_on_inference:
            server.bind(add_unroll)

        host.server = server

    self.server_initialized = True

  def start_actor_loops(self) -> None:
    """Starts accepting connections from actors."""
    assert self.server_initialized
    for host in self.hosts:
      with tf.device(host.device):
        host.server.start()

  def stop_actor_loops(self) -> None:
    """Stops accepting connections from actors."""
    assert self.server_initialized
    for host in self.hosts:
      if host.server:
        host.server.shutdown()
      host.server = None
    self.server_initialized = False

  def save_checkpoint(self) -> str:
    """Saves a checkpoint."""
    self.manager.save()
    logging.info('Saved checkpoint %s', self.manager.latest_checkpoint)
    return self.manager.latest_checkpoint

  def save_model(self,
                 directory: Optional[str] = None,
                 eval_id: Optional[int] = -1) -> None:
    """Saves the current model.

    Args:
      directory: Directory to save the model. If not set, then model will be
        saved under 'saved_model' directory in logdir.
      eval_id: Id to attach to the model directory name. If <0, it's not used.
        Usually, this is the number of steps.
    """
    if not directory:
      directory = os.path.join(self.logdir, 'saved_model')
    if eval_id >= 0:
      directory = os.path.join(directory, str(eval_id))
    tf.saved_model.save(self.training_agent, directory)

  def additional_logs(self):
    """Adds extra data to the logs and returns the episode summary stats.

    Returns:
      A dictionary of average values of the metrics over the logged episodes
      keyed by their names.
    """
    tf.summary.scalar('learning_rate',
                      self.learning_rate_fn(self.optimizer.iterations))
    tf.summary.scalar('unroll_queue_size', self.hosts[0].unroll_queue.size())

    log_time = time.time()
    inference_iterations = tf.reduce_sum(
        [host.inference_iterations for host in self.hosts])
    # log the number of inference_batches/sec
    if self.last_log_time is None:
      self.last_log_time = log_time
    dt = log_time - self.last_log_time
    if dt > 0.1:
      df = tf.cast(inference_iterations - self.last_inference_iterations,
                   tf.float32)
      tf.summary.scalar('speed/num_inference_batches/sec', df / dt)
      self.last_inference_iterations = inference_iterations
      self.last_log_time = log_time

    return self.evaluator.process()


def learner_loop(create_env_fn,
                 create_agent_fn,
                 create_optimizer_fn,
                 config: learner_config.TrainingConfig,
                 settings: utils.MultiHostSettings,
                 action_distribution_config=None):
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
    create_optimizer_fn: Function that takes the final iteration as argument and
      must return a tf.keras.optimizers.Optimizer and a
      tf.keras.optimizers.schedules.LearningRateSchedule.
    config: Training config.
    settings: Settings for training and inference strategies. You can set this
      to avoid re-initialization of the TPU system. If not set, we use the
      settings as returned by utils.init_learner.
    action_distribution_config: configuration for ParametricDistribution over
      actions; the actual distribution also depends on the action spec retrieved
      from the environment. If None (and actions are continuous), uses the
      setting parametric_distribution.continuous_action_config().
  """
  learner = Learner(
      create_env_fn,
      create_agent_fn,
      create_optimizer_fn,
      settings=settings,
      config=config,
      action_distribution_config=action_distribution_config)
  learner.prepare_for_run()
  learner.run_training()
