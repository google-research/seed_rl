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

flags.DEFINE_string('config', 'football', 'Config to run.')
flags.DEFINE_integer('workers', 1, 'Number of workers.')
flags.DEFINE_integer('actors_per_worker', 1,
                     'Number of actors to run on a single worker.')
FLAGS = flags.FLAGS


def run_learner(executor, directory, config):
  """Runs learner job using executor."""
  _, master_port = config.get('cluster').get('master')[0].split(':', 1)
  args = [
      'python', directory + 'learner.py',
      '--server_address=[::]:{}'.format(master_port),
      '--num_actors={}'.format(FLAGS.workers * FLAGS.actors_per_worker)
  ]
  if '--' in sys.argv:
    args.extend(sys.argv[sys.argv.index('--') + 1:])
  return executor.submit(subprocess.check_call, args)


def run_actor(executor, directory, config, actor_id):
  """Runs actor job using executor."""
  master_addr = config.get('cluster').get('master')[0]
  args = [
      'python', directory + 'actor.py',
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
  new_env = {}
  new_env[
      'PYTHONPATH'] = '/tensorflow_src/bazel-bin/tensorflow/cc/seed_rl/grpc/ops_test.runfiles/org_tensorflow/:/'
  os.environ.update(new_env)
  executor = concurrent.futures.ThreadPoolExecutor(
      max_workers=FLAGS.actors_per_worker)
  futures = []
  directory = '/seed_rl/{}/'.format(FLAGS.config)
  if job_type == 'master':
    futures.append(run_learner(executor, directory, config))
  else:
    assert job_type == 'worker', 'Unexpected task type: {}'.format(job_type)
    for actor_id in range(FLAGS.actors_per_worker):
      futures.append(run_actor(executor, directory, config, actor_id))
  for f in futures:
    f.result()


if __name__ == '__main__':
  app.run(main)
