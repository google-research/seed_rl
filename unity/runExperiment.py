import os

import collections
import concurrent.futures
import math
import time

from absl import app
from absl import flags
from absl import logging
import numpy as np
from seed_rl import grpc
from seed_rl.common import common_flags	
from seed_rl.common import profiling
from seed_rl.common import utils
from seed_rl.unity import env
from seed_rl.unity import networks

import tensorflow as tf

def validate_config():
	assert FLAGS.n_steps >= 1, '--n_steps < 1 does not make sense.'
	
def is_rendering_enabled():
	return FLAGS.render

def runExperiment(create_env_fn, create_agent_fn):
	"""Main learner loop.
	Args:
		create_env_fn: Callable that must return a newly created environment. The
			callable takes the task ID as argument - an arbitrary task ID of 0 will be
			passed by the learner. The returned environment should follow GYM's API.
			It is only used for infering tensor shapes. This environment will not be
			used to generate experience.
		create_agent_fn: Function that must create a new tf.Module with the neural
			network that outputs actions, Q values and new agent state given the
			environment observations and previous agent state. See
			unity.agents.DuelingLSTMDQNNet for an example. The factory function takes
			as input the environment output specs and the number of possible actions
			in the env.
	"""
	
	run_id = 0
	
	logging.info('Starting experiment')
	validate_config()
	settings = utils.init_learner(1)
	strategy, inference_devices, training_strategy, encode, decode = settings
	
	env = create_env_fn(run_id)
	env_output_specs = utils.EnvOutput(
			tf.TensorSpec([], tf.float32, 'reward'),
			tf.TensorSpec([], tf.bool, 'done'),
			tf.TensorSpec(env.observation_space.shape, env.observation_space.dtype,
										'observation'),
	)
	num_actions = env.action_space.n

	# Initialize agent and variables.
	action_specs = tf.TensorSpec([], tf.int32, 'action')
	agent_input_specs = (action_specs, env_output_specs)
	agent = create_agent_fn(env_output_specs, num_actions)
	initial_agent_state = agent.initial_state(1)
	agent_state_specs = tf.nest.map_structure(
			lambda t: tf.TensorSpec(t.shape[1:], t.dtype), initial_agent_state)
	input_ = tf.nest.map_structure(
			lambda s: tf.zeros([1] + list(s.shape), s.dtype), agent_input_specs)
	input_ = encode(input_)
	
	with strategy.scope():
		def create_variables(*args):
			return agent(*decode(args))

		initial_agent_output, _ = create_variables(input_, initial_agent_state)


	# Logging.
	summary_writer = tf.summary.create_file_writer(
			FLAGS.logdir, flush_millis=20000, max_queue=1000)

	# Setup checkpointing and restore checkpoint.
	ckpt = tf.train.Checkpoint(agent=agent)
	manager = tf.train.CheckpointManager(
			ckpt, FLAGS.logdir, max_to_keep=1, keep_checkpoint_every_n_hours=6)
	if manager.latest_checkpoint:
		logging.info('Restoring checkpoint: %s', manager.latest_checkpoint)
		ckpt.restore(manager.latest_checkpoint).expect_partial()
	
	with strategy.scope():
		@tf.function
		def inference(*args):
			return agent(*decode(args))

	
	agent_outputs = utils.Aggregator(1, action_specs, 'actions')
	env_outputs = utils.Aggregator(1, env_output_specs, 'env_outputs')
	agent_states = utils.Aggregator(1, agent_state_specs, 'agent_states')
	
	zeroIndex = tf.constant([0], dtype=tf.int32)
		
	## Loop for multiple runs
	for i in range(0, 10)
		observation = env.reset()
		reward = 0.0
		raw_reward = 0.0
		done = False
		
		agent_outputs.reset(zeroIndex)
		agent_states.replace(zeroIndex, agent.initial_state(1))
		
		while not done:
			env_output = utils.EnvOutput(reward, done, observation)
		
			env_outputs.replace(zeroIndex, env_output)
			
			input_ = encode((agent_outputs.read(zeroIndex), env_outputs.read(zeroIndex)))
			agent_output, agent_state = inference(input_, agent_states.read(zeroIndex))
				
			agent_outputs.replace(zeroIndex, agent_output.action)	
			agent_states.replace(zeroIndex, agent_state)	
			
			observation, reward, done, info = env.step(agent_output.action.numpy()[0])
			
			# save game state using info
			
			#if is_rendering_enabled():
			#	env.render()
			
			raw_reward += reward

		
def create_agent(env_output_specs, num_actions):
	return networks.DuelingLSTMDQNNet(
			num_actions, env_output_specs.observation.shape)

		
def main(argv):
	if len(argv) > 1:
		raise app.UsageError('Too many command-line arguments.')
	
	runExperiment(env.create_environment, create_agent)


if __name__ == '__main__':
	app.run(main)
	
	