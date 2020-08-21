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


"""Flags common across learners and actors.
"""

from absl import flags

flags.DEFINE_string('logdir', '/tmp/agent', 'TensorFlow log directory.')
flags.DEFINE_alias('job-dir', 'logdir')
flags.DEFINE_string('server_address', 'localhost:8686', 'Server address.',
                    allow_hide_cpp=True)


flags.DEFINE_enum(
    'run_mode', None, ['learner', 'actor'],
    'Whether we run the learner or the actor. Each actor runs the environment '
    'and sends to the learner each env observation and receives the action to '
    'play. A learner performs policy inference for batches of observations '
    'coming from multiple actors, and use the generated trajectories to learn.')

flags.DEFINE_integer('num_actors', 4,
                     'Total number of actors (training and eval actors).')
flags.DEFINE_integer('num_eval_actors', 0,
                     'Number of actors that will be used for eval (for agents '
                     'that support eval actors). Must be less than num_actors.')
flags.DEFINE_integer('envs_per_actor', 1,
                     'Number of environments to run in a batch by one actor.')

flags.DEFINE_integer('num_action_repeats', 1, 'Number of action repeats.')
