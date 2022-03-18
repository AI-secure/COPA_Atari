# coding=utf-8
# Copyright 2018 The Dopamine Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Module defining classes and helper methods for general agents."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import os.path as osp
import pickle
import sys
import time
from tqdm import tqdm

from absl import logging

from dopamine.discrete_domains import atari_lib
from dopamine.discrete_domains import checkpointer
from dopamine.discrete_domains import iteration_statistics
from dopamine.discrete_domains import logger

import numpy as np
import tensorflow as tf
import tensorflow.compat.v1 as tf1

import gin.tf

import logging


def setup_logger(logger_name, log_file, level=logging.INFO):
    l = logging.getLogger(logger_name)
    formatter = logging.Formatter('%(asctime)s : %(message)s')
    fileHandler = logging.FileHandler(log_file, mode='w')
    fileHandler.setFormatter(formatter)
    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(formatter)

    l.setLevel(level)
    l.addHandler(fileHandler)
    l.addHandler(streamHandler)


@gin.configurable
class TestRunner(object):
  """Object that handles running Dopamine experiments.

  Here we use the term 'experiment' to mean simulating interactions between the
  agent and the environment and reporting some statistics pertaining to these
  interactions.

  A simple scenario to train a DQN agent is as follows:

  ```python
  import dopamine.discrete_domains.atari_lib
  base_dir = '/tmp/simple_example'
  def create_agent(sess, environment):
    return dqn_agent.DQNAgent(sess, num_actions=environment.action_space.n)
  runner = Runner(base_dir, create_agent, atari_lib.create_atari_environment)
  runner.run()
  ```
  """

  def __init__(self,
               base_dir,
               model_dir,
               n_runs,
               total_num,
               create_agent_fn,
               create_environment_fn=atari_lib.create_atari_environment,
               checkpoint_file_prefix='ckpt',
               logging_file_prefix='log',
               log_every_n=1,
               num_iterations=200,
               training_steps=250000,
              #  evaluation_steps=125000,
               evaluation_steps=1000,
               window_size=2,
               max_steps_per_episode=27000,
               clip_rewards=True):
    """Initialize the Runner object in charge of running a full experiment.

    Args:
      base_dir: str, the base directory to host all required sub-directories.
      create_agent_fn: A function that takes as args a Tensorflow session and an
        environment, and returns an agent.
      create_environment_fn: A function which receives a problem name and
        creates a Gym environment for that problem (e.g. an Atari 2600 game).
      checkpoint_file_prefix: str, the prefix to use for checkpoint files.
      logging_file_prefix: str, prefix to use for the log files.
      log_every_n: int, the frequency for writing logs.
      num_iterations: int, the iteration number threshold (must be greater than
        start_iteration).
      training_steps: int, the number of training steps to perform.
      evaluation_steps: int, the number of evaluation steps to perform.
      max_steps_per_episode: int, maximum number of steps after which an episode
        terminates.
      clip_rewards: bool, whether to clip rewards in [-1, 1].

    This constructor will take the following actions:
    - Initialize an environment.
    - Initialize a `tf.compat.v1.Session`.
    - Initialize a logger.
    - Initialize an agent.
    - Reload from the latest checkpoint, if available, and initialize the
      Checkpointer object.
    """
    tf.compat.v1.disable_v2_behavior()

    self._logging_file_prefix = logging_file_prefix
    self._log_every_n = log_every_n
    self._num_iterations = num_iterations
    self._training_steps = training_steps
    self._evaluation_steps = evaluation_steps
    self._max_steps_per_episode = max_steps_per_episode
    self._window_size = window_size
    self._clip_rewards = clip_rewards
    self._n_runs = n_runs
    self._total_num = total_num

    self._base_dir = base_dir
    self._model_dir = model_dir
    self._create_directories()

    self._environment = create_environment_fn()

    self.config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
    # Allocate only subset of the GPU memory as needed which allows for running
    # multiple agents/workers on the same GPU.
    self.config.gpu_options.allow_growth = True
    # Set up a session and initialize variables.

    self.create_agent_fn = create_agent_fn
    self.checkpoint_file_prefix = checkpoint_file_prefix

  def _create_directories(self):
    """Create necessary sub-directories."""
    if not os.path.exists(self._base_dir):
      os.makedirs(self._base_dir)
    setup_logger('window action cert', os.path.join(self._base_dir, 'test.log'))
    self._logger = logging.getLogger('window action cert')

  def _create_agent(self, config, create_agent_fn):
    sess = tf.compat.v1.Session('', config=config)
    sess.run(tf.compat.v1.global_variables_initializer())

    agent = create_agent_fn(sess, self._environment,
                                  summary_writer=None)
    return agent

  def _init_agent_from_ckpt(self, agent, checkpoint_dir, checkpoint_file_prefix):
    self._checkpointer = checkpointer.Checkpointer(checkpoint_dir,
                                                   checkpoint_file_prefix)
    latest_checkpoint_version = checkpointer.get_latest_checkpoint_number(
        checkpoint_dir)
    if latest_checkpoint_version >= 0:
      experiment_data = self._checkpointer.load_checkpoint(
          latest_checkpoint_version)
      agent.unbundle(
          checkpoint_dir, latest_checkpoint_version, experiment_data)
    return agent


  def _initialize_episode(self):
    """Initialization for a new episode.

    Returns:
      action: int, the initial action chosen by the agent.
    """
    initial_observation = self._environment.reset()
    return self._agent.begin_episode(initial_observation)

  def _run_one_step(self, action):
    """Executes a single step in the environment.

    Args:
      action: int, the action to perform in the environment.

    Returns:
      The observation, reward, and is_terminal values returned from the
        environment.
    """
    observation, reward, is_terminal, _ = self._environment.step(action)
    return observation, reward, is_terminal

  def _end_episode(self, reward, terminal=True):
    """Finalizes an episode run.

    Args:
      reward: float, the last reward from the environment.
      terminal: bool, whether the last state-action led to a terminal state.
    """
    self._agent.end_episode(reward)

  def _compute_poison_size(self, action, a, n, c, W):
    # self._logger.info(f'c mat = {c}')
    h = c[action,:] + W - c[a,:]
    # self._logger.info(f'[compute_poison_size] h arr (before sort) = {h}')
    h = np.sort(h)[::-1]
    # self._logger.info(f'[compute_poison_size] h arr (after sort) = {h}')

    delta = n[action] - (n[a] + (a < action))
    # self._logger.info(f'[compute_poison_size] delta = {delta}')

    # self._logger.info(f'delta = {delta}, h arr = {h}')

    prefix_sum_h = np.cumsum(h)
    return np.where(prefix_sum_h > delta)[0][0] + 1

  def _compute_action_and_certificate(self, actions):
    actions_arr = np.asarray(actions)
    actions_arr_f = actions_arr.flatten()

    # self._logger.info(f'actions_arr = {actions_arr}')

    n = np.zeros((self.num_actions), dtype=np.int)
    for a in range(self.num_actions):
      n[a] = len(np.where(actions_arr_f == a)[0])
    action = np.argmax(n)

    c = np.zeros((self.num_actions, self._total_num), dtype=np.int)
    for a in range(self.num_actions):
      pos = np.where(actions_arr == a)[1]
      for i in range(self._total_num):
        c[a][i] = len(np.where(pos == i)[0])

    min_val = 1e10
    for a in range(self.num_actions):
      if a == action: continue
      min_val = min(min_val, self._compute_poison_size(action, a, n, c, W=len(actions_arr)))

    return action, min_val-1

  def _run_one_episode_multi_agent(self, save_fig=False):
    """Executes a full trajectory of the agent interacting with the environment.

    Returns:
      The number of steps taken and the total reward.
    """
    all_obs = []
    step_number = 0
    total_reward = 0.
    all_cert = []
    all_reward = []
    all_action = []

    action_seqs = []

    initial_observation = self._environment.reset()
    if save_fig:
      all_obs.append(initial_observation)

    actions = []
    for agent in self._agents:
      actions.append(agent.begin_episode(initial_observation))
    action_seqs.append(actions)

    action, cert = self._compute_action_and_certificate(action_seqs)
    all_cert.append(cert)
    all_action.append(action)

    is_terminal = False


    # Keep interacting until we reach a terminal state.
    while True:
      if len(action_seqs) == self._window_size:
        action_seqs.pop(0)

      observation, reward, is_terminal = self._run_one_step(action)
      if save_fig:
        all_obs.append(observation)
      all_reward.append(reward)

      total_reward += reward
      step_number += 1

      if self._clip_rewards:
        # Perform reward clipping.
        reward = np.clip(reward, -1, 1)

      if step_number % 100 == 0:
        self._logger.info(f'step_number = {step_number}, total reward = {total_reward}')

      if (self._environment.game_over or
          step_number == self._max_steps_per_episode):
        # Stop the run loop once we reach the true end of episode.
        break
      elif is_terminal:
        # If we lose a life but the episode is not over, signal an artificial
        # end of episode to the agent.
        actions = []
        for agent in self._agents:
          actions.append(agent.begin_episode(observation))
        action_seqs.append(actions)

        action, cert = self._compute_action_and_certificate(action_seqs)
        all_cert.append(cert)
        all_action.append(action)
      else:
        actions = []
        for agent in self._agents:
          actions.append(agent.step(reward, observation))
        action_seqs.append(actions)

        # with open('result_2.pkl', 'wb') as f:
        #   pickle.dump(agent.state, f)
        #   exit(-1)
        t = time.time()
        action, cert = self._compute_action_and_certificate(action_seqs)
        # self._logger.info(f'voting takes {time.time() - t} seconds!')
        all_cert.append(cert)
        all_action.append(action)

    return step_number, total_reward, all_cert, all_obs, all_reward, all_action

  def run_experiment(self):
    """Runs a full experiment, spread over multiple iterations."""
    # evaluate for oniteration
    id_list = list(range(1, self._total_num+1))

    sess = tf.compat.v1.Session('', config=self.config)
    sess.run(tf.compat.v1.global_variables_initializer())

    t_0 = time.time()
    self._agents = []
    for cur_id in id_list:
      with tf.name_scope(f"net{cur_id}"):
        agent = self.create_agent_fn(sess, self._environment,
                                      summary_writer=None)
      net1_varlist = {v.op.name.lstrip(f"net{cur_id}/"): v
                    for v in tf1.get_collection(tf1.GraphKeys.VARIABLES, scope=f"net{cur_id}/")}
      print(net1_varlist)
      net1_saver = tf1.train.Saver(var_list=net1_varlist)

      t = time.time()
      net1_saver.restore(sess, f'{self._model_dir}test{cur_id}/checkpoints/tf_ckpt-49')
      self._logger.info(f'loading ckpts {cur_id} taking {time.time() - t} seconds!')

      self._logger.info(f'loading {cur_id} done!')

      self._agents.append(agent)

    self._logger.info(f'loading all models using {time.time() - t_0} seconds!')

    self.num_actions = self._agents[0].num_actions


    for idx in tqdm(range(self._n_runs)):

      t = time.time()
      step_number, total_reward, all_cert, all_obs, all_reward, all_action = self._run_one_episode_multi_agent()
      self._logger.info(f'running one episode takes {time.time() - t} seconds!')

      self._logger.info(f'step_number = {step_number}')
      self._logger.info(f'total_reward = {total_reward}')
      self._logger.info(f'all_cert = {all_cert}')
      self._logger.info(f'all_reward = {all_reward}')
      self._logger.info(f'all_action = {all_action}')

      result = {
        'step_number': step_number,
        'total_reward': total_reward,
        'all_cert': all_cert,
        'all_obs': all_obs,
        'all_reward': all_reward,
        'all_action': all_action
      }
      save_filename = os.path.join(self._base_dir, f'result-{idx}.pkl')
      with open(save_filename, 'wb') as f:
        pickle.dump(result, f)

      self._logger.info(f'result saved to {save_filename}')
