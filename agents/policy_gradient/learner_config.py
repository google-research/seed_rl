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

"""Configs for the learner."""

import enum
from typing import Optional

import dataclasses


# Possible options for early returns in inference.
class InferenceReturn(enum.IntEnum):
  END = 0
  INSTANTLY = 1
  BEFORE_INFERENCE = 2
  AFTER_INFERENCE = 3
  AFTER_UNROLL = 4


@dataclasses.dataclass
class TrainingConfig(object):
  """PPO training configuration and parameters."""
  # If true, then the evaluator stats will be added by the actors using the
  # add_stats() call. Otherwise, they will be added in inference by the learner.
  # Currently, this is only used by the Kernel actor.
  add_eval_stats_from_actors: bool = False
  # Defines how to handle virtual minibatches. Possible values are 'repeat',
  # 'shuffle', 'split', 'split_with_advantage_recomputation'.
  batch_mode: str = 'split'
  # Batch size for training.
  batch_size: int = 2
  # How many batches to pass over in a training step.
  batches_per_step: int = 1
  # Should inference block until current training batch is processed.
  block_inference_on_training: bool = True
  # We clip gradient global norm to this value.
  clip_norm: Optional[float] = None
  # How many environments should be batched together per single inference call.
  env_batch_size: int = 1
  # Name of the environment to use.
  env_name: str = ''
  # How many times to pass over all the batches during a training step.
  epochs_per_step: int = 1
  # Inference batch size for the learner side.
  inference_batch_size: int = 2
  # We average at least that many episodes before logging average episode return
  # and length.
  log_episode_frequency: int = 1
  # Number of action repeats.
  num_action_repeats: int = 1
  # Total number of environments to run in all actors (training and eval envs).
  num_envs: int = 4
  # Number of environments that will be used for eval.
  # Must be less than num_envs.
  num_eval_envs: int = 0
  # If true, episode summaries will be logged.
  print_episode_summaries: bool = True
  # Allows early returns in the inference function to profile performance.
  profile_inference_return: InferenceReturn = InferenceReturn.END
  # Period for flushing the summary.
  summary_flush_seconds: int = 20
  # Maximum queue size for the summary.
  summary_max_queue_size: int = 1000
  # Total environment frames to train for.
  total_environment_frames: int = int(1e9)
  # Unroll length in agent steps.
  unroll_length: int = 100
  # How many checkpoints to generate besides the one created when the training
  # is done.
  num_checkpoints: int = 0
  # How many times to save a model to disk periodically during training.
  # Can be used for offline evaluation. Saved models are available even if
  # training doesn't complete.
  num_saved_models: int = 0
  # How many model snapshots to return from the training loop. Models are not
  # persisted on disk, so snapshoting is faster than full model save, but
  # in case of crash all snapshots are lost. Can be used for offline evaluation
  # process.
  num_snapshots: int = 0
  # Address of the Learner's GRPC server.
  server_address: str = 'localhost:8686'
  # Should model be saved at the end of the training.
  save_trained_model: bool = True
  # Whether to prefetch batches with tf.Dataset.
  prefetch_batches: bool = True
  # Whether to store unrolls on inference.
  store_unrolls_on_inference: bool = True
  # Checkpoint save period in seconds. These checkpoints are in addition to the
  # regular checkpoints based on the training iterations; see num_checkpoints
  # above. Set to 0 to disable.
  save_checkpoint_secs: int = 0

  @property
  def num_training_envs(self) -> int:
    return self.num_envs - self.num_eval_envs

  def is_training_env(self, env_id) -> bool:
    """Training env IDs are in range [0, num_training_envs)."""
    return env_id < self.num_training_envs
