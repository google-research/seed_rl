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

"""A script for loading a policy and running it locally."""
from absl import app
from absl import flags
from seed_rl.common import google_policy_viewer
from seed_rl.mujoco.env import create_environment

flags.DEFINE_string('env_name', None, 'Environment name from OpenAI Gym.')

FLAGS = flags.FLAGS


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
  create_env_fn = lambda: create_environment(FLAGS.env_name)
  google_policy_viewer.policy_viewer(create_env_fn)


if __name__ == '__main__':
  app.run(main)
