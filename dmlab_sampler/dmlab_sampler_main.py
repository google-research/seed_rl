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
from absl import app
from absl import flags
import numpy as np
import env_wrappers
import h5py
import env
import networks
import tensorflow as tf
import games
import utils

def create_agent(action_space):
  return networks.ImpalaDeep(action_space)

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

def create_optimizer(final_iteration):
  learning_rate_fn = tf.keras.optimizers.schedules.PolynomialDecay(
      1e-2, final_iteration, 0)
  optimizer = tf.keras.optimizers.Adam(learning_rate_fn, beta_1=0)
  # optimizer = tf.keras.optimizers.RMSprop(learning_rate_fn, FLAGS.rms_decay, FLAGS.rms_momentum,
  #                                      FLAGS.rms_epsilon)
  return optimizer, learning_rate_fn

FLAGS = flags.FLAGS
flags.DEFINE_integer('task', 0, 'Task id.')
flags.DEFINE_integer('save_interval', int(1e6), 'save interval')
flags.DEFINE_integer('env', int(1e6), 'save interval')
flags.DEFINE_string('experts_path', None, 'experts path')
flags.DEFINE_string('sub_task', 'rooms_watermaze', 'sub tasks, i.e. dmlab30, dmlab26, all, others')
flags.DEFINE_float('reward_threshold', 0., 'reward threshold for sampling')
flags.DEFINE_integer('env_batch_size', int(4), 'env batch size')
flags.DEFINE_integer('cuda_device', 0, 'cuda id, -1 for cpu')
flags.DEFINE_integer('num_training_tpus', 1, 'Number of TPUs for training.')

def actor_loop(create_env_fn, create_optimizer_fn, log_period=10):
  """Main actor loop.

  Args:
    create_env_fn: Callable (taking the task ID as argument) that must return a
      newly created environment.
    config: Configuration of the training.
    log_period: How often to log in seconds.
  """
  os.environ['CUDA_VISIBLE_DEVICES'] = str(FLAGS.cuda_device)
  settings = utils.init_learner_multi_host(FLAGS.num_training_tpus)
  strategy, hosts, training_strategy, encode, decode = settings
  host, inference_devices = hosts[0]
  env = create_env_fn(0, FLAGS)
  env_output_specs = utils.EnvOutput(
      tf.TensorSpec([], tf.float32, 'reward'),
      tf.TensorSpec([], tf.bool, 'done'),
      tf.TensorSpec(env.observation_space.shape, env.observation_space.dtype,
                    'observation'),
      tf.TensorSpec([], tf.bool, 'abandoned'),
      tf.TensorSpec([], tf.int32, 'episode_step'),
  )
  action_specs = tf.TensorSpec(env.action_space.shape,
                               env.action_space.dtype, 'action')
  agent_input_specs = (action_specs, env_output_specs)

  # Initialize agent and variables.
  agent = create_agent(9)
  initial_agent_state = agent.initial_state(1)
  agent_state_specs = tf.nest.map_structure(
      lambda t: tf.TensorSpec(t.shape[1:], t.dtype), initial_agent_state)
  # unroll_specs = [None]  # Lazy initialization.
  input_ = tf.nest.map_structure(
      lambda s: tf.zeros([1] + list(s.shape), s.dtype), agent_input_specs)
  input_ = encode(input_)

  with strategy.scope():
    if not hasattr(agent, 'entropy_cost'):
      mul = 10.
      agent.entropy_cost_param = tf.Variable(
          tf.math.log(1e-2) / mul,
          constraint=lambda v: tf.clip_by_value(v, -20 / mul, 20 / mul),
          trainable=True,
          dtype=tf.float32)
      agent.entropy_cost = lambda: tf.exp(mul * agent.entropy_cost_param)
    # Create optimizer.
    # iter_frame_ratio = (
    #     FLAGS.batch_size * FLAGS.unroll_length * FLAGS.num_action_repeats)
    final_iteration = 100000
    optimizer, learning_rate_fn = create_optimizer_fn(final_iteration)
    # iterations = optimizer.iterations
    optimizer._create_hypers()  
    optimizer._create_slots(agent.trainable_variables)  
  ckpt = tf.train.Checkpoint(agent=agent, optimizer=optimizer)
  with strategy.scope():
    status = ckpt.restore('./dmlab_experts/rooms_watermaze/ckpt-42')

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
  encode = lambda x: x
  decode = lambda x, s=None: x if s is None else tf.nest.pack_sequence_as(s, x)
  pid = os.getpid()
  while True:
    print('creating environment')
    batched_env = env_wrappers.BatchedEnvironment(
        create_env_fn, env_batch_size, 0, config)
    core_states = agent.initial_state(env_batch_size)
    
    observation = batched_env.reset()
    reward = np.zeros(env_batch_size, np.float32)
    raw_reward = np.zeros(env_batch_size, np.float32)
    done = np.zeros(env_batch_size, np.bool_)
    
    action = np.zeros(env_batch_size, dtype=np.int32)

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
      env_output = utils.EnvOutput(reward, done, observation,
                              [], episode_step)
      input_ = encode((action, env_output))
      with tf.device(inference_devices[0]):
          action, core_states = agent(*input_, core_states)
      action = action[0]
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
            print(f'pid: {pid} dumping traj, env idx: {actor_idx} cur transitions: {cur_trans_num} episodes: {cur_ep_num}')
          if cur_trans_num >= FLAGS.save_interval:
            total_transitions += cur_trans_num
            print(f'pid: {pid} saving data, save idx: {save_idx} env idx: {actor_idx} cur transitions: {cur_trans_num} tt transitions: {total_transitions} episodes: {cur_ep_num}')
            dataset2save = h5py.File(FLAGS.logdir + '/' + FLAGS.task_names[actor_idx % len(FLAGS.task_names)] + '_dataset/' + str(actor_idx) + '_' + str(save_idx) + '.hdf5', 'w')
            save_idx += 1
            cur_trans_num = 0
            cur_ep_num = 0
            npify(data2save)
            for k in data2save:
                dataset2save.create_dataset(k, data=data2save[k], compression='gzip')
            del data2save
            data2save = reset_data()

          if current_time - last_log_time >= log_period:
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
          obsBuffer[i].clear()
          actionsBuffer[i].clear()
          rewardBuffer[i].clear()
          terminalBuffer[i].clear()
          infosBuffer[i].clear()

      observation = batched_env.reset_if_done(done)
      actor_step += 1

def main(_):
  actor_loop(env.create_environment, create_optimizer)

if __name__ == '__main__':
  app.run(main)
  