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

"""Atari env factory."""

import tempfile

from absl import flags
from absl import logging
from procgen import ProcgenEnv
from seed_rl.common import common_flags


games = {
        "bigfish": ["procgen:procgen-bigfish-v0", "hard", 20.],
        "bossfight": ["procgen:procgen-bossfight-v0", "hard", 10.],
        "caveflyer": ["procgen:procgen-caveflyer-v0", "hard", 7.5],
        "chaser": ["procgen:procgen-chaser-v0", "hard", 7.],
        "climber": ["procgen:procgen-climber-v0", "hard", 8.],
        "coinrun": ["procgen:procgen-coinrun-v0", "hard", 8.],
        "dodgeball": ["procgen:procgen-dodgeball-v0", "hard", 8.],
        "fruitbot": ["procgen:procgen-fruitbot-v0", "hard", 18.],
        "heist": ["procgen:procgen-heist-v0", "easy", 7.5],
        "jumper": ["procgen:procgen-jumper-v0", "hard", 5.],
        "leaper": ["procgen:procgen-leaper-v0", "hard", 7.5],
        "maze": ["procgen:procgen-maze-v0", "easy", 7.5],
        "miner": ["procgen:procgen-miner-v0", "hard", 15.],
        "ninja": ["procgen:procgen-ninja-v0", "hard", 8.],
        "plunder": ["procgen:procgen-plunder-v0", "hard", 15.],
        "starpilot": ["procgen:procgen-starpilot-v0", "hard", 15.],
}

FLAGS = flags.FLAGS

# Environment settings.
flags.DEFINE_integer('game', 0, 'Game idx.')


def create_environment(task, config):
    if config.sub_task != "all":
        full_game_name = config.sub_task
        game_difficulty = games[config.sub_task][1]
    else:
        game_list = games.values()
        full_game_name = game_list[task % 16]
        game_difficulty = games[game_list[task % 16]][1]
    # env = gym.make(full_game_name, distribution_mode=game_difficulty,
    #                 start_level=0, num_levels=100000)
    env = ProcgenEnv(num_envs=config.env_batch_size, env_name=full_game_name, num_levels=100000, start_level=0, distribution_mode=game_difficulty)
    env.observation_space = env.observation_space['rgb']
    logging.info('Creating environment: %s', full_game_name)
    logging.info('Distribution mode: %s', game_difficulty)
    logging.info(env.observation_space)
    logging.info(env.action_space)

    return env
