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
import gym
from seed_rl.common import common_flags


games = {
        "bigfish": ["procgen:procgen-bigfish-v0", "hard"],
        "bossfight": ["procgen:procgen-bossfight-v0", "hard"],
        "caveflyer": ["procgen:procgen-caveflyer-v0", "hard"],
        "chaser": ["procgen:procgen-chaser-v0", "hard"],
        "climber": ["procgen:procgen-climber-v0", "hard"],
        "coinrun": ["procgen:procgen-coinrun-v0", "hard"],
        "dodgeball": ["procgen:procgen-dodgeball-v0", "hard"],
        "fruitbot": ["procgen:procgen-fruitbot-v0", "hard"],
        "heist": ["procgen:procgen-heist-v0", "easy"],
        "jumper": ["procgen:procgen-jumper-v0", "hard"],
        "leaper": ["procgen:procgen-leaper-v0", "hard"],
        "maze": ["procgen:procgen-maze-v0", "easy"],
        "miner": ["procgen:procgen-miner-v0", "hard"],
        "ninja": ["procgen:procgen-ninja-v0", "hard"],
        "plunder": ["procgen:procgen-plunder-v0", "hard"],
        "starpilot": ["procgen:procgen-starpilot-v0", "hard"],
}

FLAGS = flags.FLAGS

# Environment settings.
flags.DEFINE_integer('game', 0, 'Game idx.')


def create_environment(task, config):
    if config.sub_task != "all":
        full_game_name = games[config.sub_task][0]
        game_difficulty = games[config.sub_task][1]
        env = gym.make(full_game_name, distribution_mode=game_difficulty,
                        start_level=0, num_levels=100000)
    else:
        game_list = games.values()
        full_game_name = games[game_list[task % 16]][0]
        game_difficulty = games[game_list[task % 16]][1]
        env = gym.make(full_game_name, distribution_mode=game_difficulty,
                        start_level=0, num_levels=100000)
    logging.info('Creating environment: %s', full_game_name)
    logging.info('Distribution mode: %s', game_difficulty)

    # Change TimeLimit wrapper to 108,000 steps (30 min) as default in the
    # litterature instead of OpenAI Gym's default of 100,000 steps.
    env = gym.wrappers.TimeLimit(env.env, max_episode_steps=108000)
    return env
