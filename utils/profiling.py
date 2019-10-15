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


"""Profiling utils.
"""

import collections
import time
import tensorflow as tf


class Aggregator(object):
  """Allows accumulating values and computing their mean."""

  def __init__(self):
    self.reset()

  def reset(self):
    self.sum = 0.
    self.count = 0

  def average(self):
    return self.sum / self.count if self.count else 0.

  def add(self, v):
    self.sum += v
    self.count += 1


class ExportingTimer(object):
  """A context-manager timer with automatic tf.summary export.

  ExportingTimer is thread-hostile because of the state shared across instances.
  One could protect this shared state under a mutex if using multiple instances
  of ExportingTimer in multiple threads becomes needed.

  Example usage:

  with ExportingTimer('actor/env_step_s', aggregation_window_size=100):
    env.step()

  which will record it takes to execute 'env.step()' in seconds, and export the
  average as a tf.summary under 'actor/env_steps_s' every 100 invocations.
  """

  # Maps tf.summary names to the sum and counts of elapsed times (seconds).
  # This is global for all instances of ExportingTimer.
  aggregators = collections.defaultdict(Aggregator)

  def __init__(self, summary_name, aggregation_window_size):
    self.summary_name = summary_name
    self.aggregation_window_size = aggregation_window_size

  def __enter__(self):
    self.start_time_s = time.time()
    return self

  def __exit__(self, exc_type, exc_value, traceback):
    self.elapsed_s = time.time() - self.start_time_s
    aggregator = self.aggregators[self.summary_name]
    aggregator.add(self.elapsed_s)
    if aggregator.count >= self.aggregation_window_size:
      tf.summary.scalar(self.summary_name, aggregator.average())
      aggregator.reset()
