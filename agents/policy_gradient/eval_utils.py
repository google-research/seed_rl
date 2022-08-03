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

r"""Evaluataion utils."""
import collections
from absl import logging
from seed_rl.common import utils
import tensorflow as tf

# Information about a finished episode.
EpisodeInfo = collections.namedtuple(
    'EpisodeInfo',
    # episode_num_frames: length of the episode in number of frames.
    # episode_returns: Sum of undiscounted rewards experienced in the episode.
    # episode_raw_returns: Sum of raw rewards experienced in the episode.
    'eval_name episode_num_frames episode_returns episode_raw_returns')


class Evaluator(object):
  """Utility module that processes environment evaluation results."""

  def __init__(self, print_episode_summaries, log_episode_frequency=1):
    self.log_episode_frequency = log_episode_frequency
    self.info_specs = (
        tf.TensorSpec([], tf.string, 'eval_name'),
        tf.TensorSpec([], tf.int64, 'episode_num_frames'),
        tf.TensorSpec([], tf.float32, 'episode_returns'),
        tf.TensorSpec([], tf.float32, 'episode_raw_returns'),
    )
    self.episode_info_queue = utils.StructuredFIFOQueue(
        -1, EpisodeInfo(*self.info_specs))
    # A map from env eval name to 4 lists that contain EpisodeInfo stats.
    self.eval_data = collections.defaultdict(
        lambda: tuple([[] for _ in range(len(EpisodeInfo._fields))]))
    self.print_episode_summaries = print_episode_summaries

  def add(self, data):
    """Adds data (which should have self.info_specs signature) to the queue."""
    self.episode_info_queue.enqueue(EpisodeInfo(*data))

  def add_many(self, data):
    """Adds data (several items) to the queue.

    Args:
      data: tuple with shape self.info_specs with an additional batch front
        dimension.
    """
    self.episode_info_queue.enqueue_many(EpisodeInfo(*data))

  def reset(self):
    self.eval_data.clear()
    while self.episode_info_queue.size():
      self.episode_info_queue.dequeue_many(self.episode_info_queue.size())

  def process(self, write_tf_summaries=True):
    """Processes the data from the queue.

    The following steps are executed:
     - stats are logged if print_episode_summaries is enabled
     - stats are groupped into buckets of size >= self.log_episode_frequency
       and then dumped as tf.summary.scalar and also returned in a dictionary.

    Args:
      write_tf_summaries: should TF.summaries be written on top of returning
        stats.

    Returns:
        Evaluation stats.
    """
    episode_stats = self.episode_info_queue.dequeue_many(
        self.episode_info_queue.size())
    episode_summary_stats = {}

    for episode_info in zip(*episode_stats):
      epinfo = EpisodeInfo(*episode_info)
      eval_name = epinfo.eval_name.numpy()

      if self.print_episode_summaries:
        episode_returns = epinfo.episode_returns.numpy()
        episode_raw_returns = epinfo.episode_raw_returns.numpy()
        episode_num_frames = epinfo.episode_num_frames.numpy()
        logging.info('Return: %f Raw return: %f (key_prefix="%s") Frames: %i',
                     episode_returns, episode_raw_returns, eval_name,
                     episode_num_frames)

      for i in range(1, len(EpisodeInfo._fields)):
        self.eval_data[eval_name][i].append(episode_info[i].numpy())

    for key, value in self.eval_data.items():
      for i in range(1, len(EpisodeInfo._fields)):
        if len(value[i]) >= self.log_episode_frequency:
          v_mean = tf.reduce_mean(value[i])
          v_std = tf.math.reduce_std(tf.constant(value[i], dtype=tf.float32))
          name = key.decode('utf-8') + EpisodeInfo._fields[i]
          name_std = name + '_std'
          value[i].clear()
          if write_tf_summaries:
            tf.summary.scalar(name, v_mean)
            tf.summary.scalar(name_std, v_std)
          episode_summary_stats[name] = v_mean
          episode_summary_stats[name_std] = v_std
    return episode_summary_stats
