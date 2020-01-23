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

"""Starts actor or learner job depending on the GCP node type."""

import json
import os
import subprocess
import sys

from absl import app
from absl import flags
from absl import logging
import concurrent.futures

flags.DEFINE_string('environment', 'football', 'Environment to run.')
flags.DEFINE_string('agent', 'vtrace', 'Agent to run.')
flags.DEFINE_integer('workers', 1, 'Number of workers.')
flags.DEFINE_integer('actors_per_worker', 1,
                     'Number of actors to run on a single worker.')
FLAGS = flags.FLAGS


def get_py_main():
  return os.path.join('/seed_rl', FLAGS.environment,
                      FLAGS.agent + '_main.py')


def run_learner(executor, config):
  """Runs learner job using executor."""
  _, master_port = config.get('cluster').get('master')[0].split(':', 1)
  args = [
      'python', get_py_main(),
      '--run_mode=learner',
      '--server_address=[::]:{}'.format(master_port),
      '--num_actors={}'.format(FLAGS.workers * FLAGS.actors_per_worker)
  ]
  if '--' in sys.argv:
    args.extend(sys.argv[sys.argv.index('--') + 1:])
  return executor.submit(subprocess.check_call, args)


def run_actor(executor, config, actor_id):
  """Runs actor job using executor."""
  master_addr = config.get('cluster').get('master')[0]
  args = [
      'python', get_py_main(),
      '--run_mode=actor',
      '--server_address={}'.format(master_addr),
      '--num_actors={}'.format(FLAGS.workers * FLAGS.actors_per_worker)
  ]
  worker_index = config.get('task').get('index')
  args.append('--task={}'.format(worker_index * FLAGS.actors_per_worker +
                                 actor_id))
  if '--' in sys.argv:
    args.extend(sys.argv[sys.argv.index('--') + 1:])
  return executor.submit(subprocess.check_call, args)


def main(_):
  tf_config = os.environ.get('TF_CONFIG', None)
  logging.info(tf_config)
  config = json.loads(tf_config)
  job_type = config.get('task', {}).get('type')
  executor = concurrent.futures.ThreadPoolExecutor(
      max_workers=FLAGS.actors_per_worker)
  futures = []
  if job_type == 'master':
    futures.append(run_learner(executor, config))
  else:
    assert job_type == 'worker', 'Unexpected task type: {}'.format(job_type)
    for actor_id in range(FLAGS.actors_per_worker):
      futures.append(run_actor(executor, config, actor_id))
  for f in futures:
    f.result()


if __name__ == '__main__':
  app.run(main)
