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

r"""SEED actor."""

import os
import timeit

from absl import flags
import numpy as np
from seed_rl.common import common_flags  
from seed_rl.common import env_wrappers
from seed_rl.common import profiling
from seed_rl.common import utils
import tensorflow as tf
import h5py
from seed_rl.common import common_flags  
from seed_rl.dmlab import env
from seed_rl.dmlab import networks
import tensorflow as tf
from seed_rl.dmlab import games

def create_agent(action_space):
  return networks.ImpalaDeep(action_space.n)

def reset_data():
    return {'observations': [],
            'actions': [],
            'terminals': [],
            'timeouts': [],
            'rewards': [],
            'infos': [],
            }

def append_data(data, obs, act, rew, infos, done):
    data['observations'].extend(obs)
    data['actions'].extend(act)
    data['rewards'].extend(rew)
    data['terminals'].extend(done)
    data['timeouts'].extend(done)
    data['infos'].extend(infos)


def npify(data):
    for k in data:
        if k in ['terminals', 'timeouts']:
            dtype = np.bool_
            data[k] = np.array(data[k], dtype=dtype)
        else:
            data[k] = np.array(data[k])



FLAGS = flags.FLAGS
flags.DEFINE_integer('task', 0, 'Task id.')
flags.DEFINE_integer('save_interval', int(1e6), 'save interval')
flags.DEFINE_integer('env', int(1e6), 'save interval')
flags.DEFINE_string('experts_path', None, 'experts path')
flags.DEFINE_string('sub_task', 'rooms_watermaze', 'sub tasks, i.e. dmlab30, dmlab26, all, others')
flags.DEFINE_float('reward_threshold', 0., 'reward threshold for sampling')

def actor_loop(create_env_fn, log_period=10):
  """Main actor loop.

  Args:
    create_env_fn: Callable (taking the task ID as argument) that must return a
      newly created environment.
    config: Configuration of the training.
    log_period: How often to log in seconds.
  """
  config = FLAGS
  save_idx = 0
  total_transitions = 0
  total_eps = 0
  cur_trans_num = 0
  cur_ep_num = 0
  avg_ep_reward = 0
  actor_idx = FLAGS.task
  env_batch_size = FLAGS.env_batch_size
  dummy_infos = np.array(0, dtype=np.uint8)
  actor_step = 0
  while True:
    print('creating environment')
    batched_env = env_wrappers.BatchedEnvironment(
        create_env_fn, env_batch_size, 0, config)
    agent = create_agent(9)
    observation = batched_env.reset()
    reward = np.zeros(env_batch_size, np.float32)
    raw_reward = np.zeros(env_batch_size, np.float32)
    done = np.zeros(env_batch_size, np.bool_)
    core_states = agent.initial_state(env_batch_size)
    prev_action = np.zeros(env_batch_size)

    global_step = 0
    episode_step = np.zeros(env_batch_size, np.int32)
    episode_return = np.zeros(env_batch_size, np.float32)
    episode_raw_return = np.zeros(env_batch_size, np.float32)
    episode_step_sum = 0
    episode_return_sum = 0
    episode_raw_return_sum = 0
    episodes_in_report = 0
    obsBuffer = [[] for i in range(env_batch_size)]
    actionsBuffer = [[] for i in range(env_batch_size)]
    rewardBuffer = [[] for i in range(env_batch_size)]
    terminalBuffer = [[] for i in range(env_batch_size)]
    infosBuffer = [[] for i in range(env_batch_size)]

    data2save = reset_data()
    last_log_time = timeit.default_timer()
    last_global_step = 0
    while True:
      for i in range(env_batch_size):
        obsBuffer[i].append(observation[i])
      env_outputs = [raw_reward, done, observation, [], []]
      action = agent(prev_action, env_outputs, core_states)
      observation, reward, done, info = batched_env.step(action.numpy())
      for i in range(env_batch_size):
        actionsBuffer[i].append(action.numpy()[i])
        rewardBuffer[i].append(reward[i])
        terminalBuffer[i].append(done[i])
        infosBuffer[i].append(dummy_infos)
        episode_step[i] += 1
        episode_return[i] += reward[i]
        raw_reward[i] = float((info[i] or {}).get('score_reward',
                                                  reward[i]))
        episode_raw_return[i] += raw_reward[i]
        if done[i]:
          current_time = timeit.default_timer()
          episode_step_sum += episode_step[i]
          episode_return_sum += episode_return[i]
          episode_raw_return_sum += episode_raw_return[i]
          global_step += episode_step[i]
          episodes_in_report += 1

          if episode_raw_return[i] >= FLAGS.reward_threshold:
            cur_trans_num += episode_step[i]
            cur_ep_num += 1
            avg_ep_reward += episode_raw_return[i]
            append_data(data2save, obsBuffer[i], actionsBuffer[i], rewardBuffer[i], infosBuffer[i],terminalBuffer[i])
            total_eps += 1
          if cur_trans_num >= FLAGS.save_interval:
            total_transitions += cur_trans_num
            pid = os.getpid()
            print(f'pid: {pid} saving data, save idx: {save_idx} env idx: {actor_idx} cur transitions: {cur_trans_num} tt transitions: {total_transitions} episodes: {cur_ep_num}')
            dataset2save = h5py.File(FLAGS.logdir + '/' + FLAGS.task_names[actor_idx % len(FLAGS.task_names)] + '_dataset/' + str(actor_idx) + '_' + str(save_idx) + '.hdf5', 'w')
            save_idx += 1
            cur_trans_num = 0
            cur_ep_num = 0
            npify(data2save)
            for k in data2save:
                dataset2save.create_dataset(k, data=data2save[k], compression='gzip')
            data2save = reset_data()

          if current_time - last_log_time >= log_period:
            pid = os.getpid()
            print(
                f'PID: {pid} Actor steps: {global_step}, Return: {episode_return_sum / episodes_in_report} '
                f'Episode steps: {episode_step_sum / episodes_in_report}, '
                f'Speed: {(global_step - last_global_step) / (current_time - last_log_time)} steps/s'
                )
            last_global_step = global_step
            episode_return_sum = 0
            episode_raw_return_sum = 0
            episode_step_sum = 0
            episodes_in_report = 0
            last_log_time = current_time
          episode_step[i] = 0
          episode_return[i] = 0
          episode_raw_return[i] = 0
          obsBuffer[i] = []
          actionsBuffer[i] = []
          rewardBuffer[i] = []
          terminalBuffer[i] = []
          infosBuffer[i] = []

      observation = batched_env.reset_if_done(done)
      actor_step += 1

if __name__ == '__main__':
  actor_loop(env.create_environment)