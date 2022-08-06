from procgen import ProcgenEnv
import gym
import numpy as np
import timeit
import h5py
import os


def reset_data():
    return {'observations': [],
            'actions': [],
            'terminals': [],
            'timeouts': [],
            'rewards': [],
            'infos/prev_level_seed': [],
            'infos/prev_level_complete': [],
            'infos/level_seed': [],
            }


def append_data(data, obs, act, rew, infos1, infos2, infos3, done):
    data['observations'].extend(obs)
    data['actions'].extend(act)
    data['rewards'].extend(rew)
    data['terminals'].extend(done)
    data['timeouts'].extend(done)
    data['infos/prev_level_seed'].extend(infos1)
    data['infos/prev_level_complete'].extend(infos2)
    data['infos/level_seed'].extend(infos3)


def npify(data):
    for k in data:
        if k in ['terminals', 'timeouts']:
            dtype = np.bool_
            data[k] = np.array(data[k], dtype=dtype)
        else:
            data[k] = np.array(data[k])


# env = gym.make("procgen:procgen-bigfish-v0", distribution_mode='hard',
#                         start_level=0, num_levels=100000)
env_batch_size = 32
k = 1000000000
save_idx = 0
total_transitions = 0
total_eps = 0
cur_trans_num = 0
cur_ep_num = 0
avg_ep_reward = 0
actor_idx = 0
reward_threshold = 10
save_interval = 100000
log_period = 10

while True:
    procgen_env = ProcgenEnv(num_envs=env_batch_size, env_name='bigfish', num_levels=100000, start_level=0,
                             distribution_mode='hard')
    observation = procgen_env.reset()
    reward = np.zeros(env_batch_size, np.float32)
    raw_reward = np.zeros(env_batch_size, np.float32)
    done = np.zeros(env_batch_size, np.bool_)
    abandoned = np.zeros(env_batch_size, np.bool_)
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
    infos1Buffer = [[] for i in range(env_batch_size)]
    infos2Buffer = [[] for i in range(env_batch_size)]
    infos3Buffer = [[] for i in range(env_batch_size)]
    data2save = reset_data()
    last_global_step = 0
    last_log_time = timeit.default_timer()
    while True:
        for i in range(env_batch_size):
            obsBuffer[i].append(observation['rgb'][i])
        action = np.random.randint(0, 16, env_batch_size)
        observation, reward, done, info = procgen_env.step(action)
        for i in range(env_batch_size):
            actionsBuffer[i].append(action[i])
            rewardBuffer[i].append(reward[i])
            terminalBuffer[i].append(done[i])
            infos1Buffer[i].append(info[i]['prev_level_seed'])
            infos3Buffer[i].append(info[i]['prev_level_complete'])
            infos2Buffer[i].append(info[i]['level_seed'])
            episode_step[i] += 1
            episode_return[i] += reward[i]
            raw_reward[i] = float((info[i] or {}).get('score_reward',
                                                      reward[i]))
            episode_raw_return[i] += raw_reward[i]
            assert done[i] if abandoned[i] else True
            if done[i]:
                current_time = timeit.default_timer()
                episode_step_sum += episode_step[i]
                episode_return_sum += episode_return[i]
                episode_raw_return_sum += episode_raw_return[i]
                global_step += episode_step[i]
                episodes_in_report += 1

                if episode_raw_return[i] >= reward_threshold:
                    cur_trans_num += episode_step[i]
                    cur_ep_num += 1
                    avg_ep_reward += episode_raw_return[i]
                    append_data(data2save, obsBuffer[i], actionsBuffer[i], rewardBuffer[i], infos1Buffer[i],
                                infos2Buffer[i],
                                infos3Buffer[i], terminalBuffer[i])
                    total_eps += 1
                obsBuffer[i] = []
                actionsBuffer[i] = []
                rewardBuffer[i] = []
                terminalBuffer[i] = []
                infos1Buffer[i] = []
                infos3Buffer[i] = []
                infos2Buffer[i] = []
                if cur_trans_num >= save_interval:
                    total_transitions += cur_trans_num
                    pid = os.getpid()
                    print(
                        f'saving data, pid: {pid} save idx: {save_idx} env idx: {actor_idx} cur transitions: {cur_trans_num} tt transitions: {total_transitions} episodes: {cur_ep_num}')
                    dataset2save = h5py.File(str(actor_idx) + '_' + str(save_idx) + '.hdf5', 'w')
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
