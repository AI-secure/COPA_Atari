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
import copy
from dataclasses import dataclass, field
import json
import pickle
from queue import PriorityQueue
import sys
import time

from absl import logging

from dopamine.discrete_domains import atari_lib
from dopamine.discrete_domains import checkpointer
from dopamine.discrete_domains import iteration_statistics
from dopamine.discrete_domains import logger

import numpy as np
import tensorflow as tf
import tensorflow.compat.v1 as tf1

import gin.tf

global_id = 0

from collections import OrderedDict

class LimitedSizeDict(OrderedDict):
    def __init__(self, *args, **kwds):
        # self.size_limit = kwds.pop("size_limit", None)
        self.size_limit = 50000
        OrderedDict.__init__(self, *args, **kwds)
        self._check_size_limit()

    def __setitem__(self, key, value):
        OrderedDict.__setitem__(self, key, value)
        self._check_size_limit()

    def _check_size_limit(self):
        if self.size_limit is not None:
            while len(self) > self.size_limit:
                self.popitem(last=False)

def to_str(d):
  return ",".join("{}: {}".format(*i) for i in d.items())

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


class Elem(object):
    def __init__(self, snapshot, state, a, rho, re, no):
        global global_id
        self.gid = global_id
        global_id += 1

        self.snapshot = snapshot
        self.state = state
        self.a = a
        self.rho = rho
        self.re = re
        self.no = no

    def __lt__(self, other):
        return self.rho < other.rho

@dataclass(order=True)
class PrioritizedItem:
    priority: int
    item: object = field()


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
               max_steps_per_episode=100,
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
    assert base_dir is not None
    tf.compat.v1.disable_v2_behavior()

    self._logging_file_prefix = logging_file_prefix
    self._log_every_n = log_every_n
    self._num_iterations = num_iterations
    self._training_steps = training_steps
    self._evaluation_steps = evaluation_steps
    self._max_steps_per_episode = max_steps_per_episode
    self._base_dir = base_dir
    self._model_dir = model_dir
    self._total_num = total_num
    self._clip_rewards = clip_rewards
    self._create_directories()
    self._create_environment_fn = create_environment_fn

    self._environment = create_environment_fn(sticky_actions=False)
    self._environment.environment.seed(42)
    self._logger.info(f'environment seed set to 42')
    self.config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
    # Allocate only subset of the GPU memory as needed which allows for running
    # multiple agents/workers on the same GPU.
    self.config.gpu_options.allow_growth = True
    # Set up a session and initialize variables.

    self.create_agent_fn = create_agent_fn
    self.checkpoint_file_prefix = checkpoint_file_prefix

    # initialize
    self.re_min = 1e100
    self.certify_map = {}
    self.state_set = [set() for _ in range(self._max_steps_per_episode+1)]
    self.state_dict = LimitedSizeDict()
    # self.p_que = PriorityQueue()
    self.queues = [[] for _ in range(self._total_num//2+2)]
    self.fout = [open(osp.join(self._base_dir, f'queue-{i}.pkl'), 'wb')  for i in range(self._total_num//2+2)]
    self.queue_maxlen = 5000

  def _create_directories(self):
    """Create necessary sub-directories."""
    if not os.path.exists(self._base_dir):
      os.makedirs(self._base_dir)
    log_file = os.path.join(self._base_dir, 'test')

    suffix = 0
    while 1:
      if osp.exists(log_file + ('' if not suffix else '_'+str(suffix)) + '.log'):
        suffix += 1
      else:
        break
    log_file += ('' if not suffix else '_'+str(suffix)) + '.log'
    setup_logger('reward cert', log_file)
    self._logger = logging.getLogger('reward cert')

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

  def update_queue_into_file(self, rho):
    t = time.time()
    # write queue content to file
    if self.queues[rho]:
      pickle.dump(self.queues[rho], self.fout[rho])
    self._logger.info(f'dump queue of length={len(self.queues[rho])} into file for rho={rho} using {time.time()-t} seconds done!')
    # clear queue content
    self.queues[rho] = []

  def read_queue_from_file(self, rho):
    queue = []
    fin = open(osp.join(self._base_dir, f'queue-{rho}.pkl'), 'rb')
    while 1:
        try:
            queue += pickle.load(fin)
            self._logger.info(f'reading...[len queue = {len(queue)}]')
        except EOFError:
            break
    self._logger.info('reading done!')
    return queue

  # whether satisfy val + \sum (val - (thre-1)) + \sum (val - thre) >= thre
  def _satisfy(self, thre, actions_cnt, a_tar):
    expected_cnt = np.concatenate((np.ones(a_tar, dtype=np.int)*(thre-1), np.asarray([0]), np.ones(self.num_actions-a_tar-1, dtype=np.int)*thre))
    lhs = np.sum((actions_cnt >= expected_cnt) * (actions_cnt - expected_cnt))
    rhs = thre
    self._logger.info(f'a_tar = {a_tar}, thre = {thre}, expected_cnt = {expected_cnt}, lhs = {lhs}, rhs = {rhs}')
    return lhs >= rhs, lhs - actions_cnt[a_tar]

  def _compute_poison_size(self, actions_cnt, a_star, a):
    lo = actions_cnt[a]
    hi = actions_cnt[a_star] + 1
    while hi - lo > 1:
      mid = (lo + hi) >> 1
      flag, poison_size = self._satisfy(mid, actions_cnt, a)
      if flag:
        lo = mid
      else:
        hi = mid - 1

    # check whether hi satisfies
    flag_hi, poison_size_hi = self._satisfy(hi, actions_cnt, a)
    if flag_hi: return poison_size_hi

    # hi must satisfies
    flag_lo, poison_size_lo = self._satisfy(lo, actions_cnt, a)
    assert flag_lo, f'actions_cnt = {actions_cnt}, a = {a}, lo = {lo}, hi = {hi}, failed!'
    return poison_size_lo

  def _compute_action_and_certifications(self, actions):
    self._logger.info(f'[compute_action_and_certifications] actions = {actions}')
    actions_cnt = np.zeros(self.num_actions, dtype=np.int)
    for i in range(self.num_actions):
      actions_cnt[i] = np.count_nonzero(actions == i)
    a_star = np.argmax(actions_cnt)

    poison_sizes = np.zeros(self.num_actions, dtype=np.int)
    for a in range(self.num_actions):
      if a == a_star:
        poison_sizes[a] = 0
      else:
        poison_sizes[a] = self._compute_poison_size(actions_cnt, a_star, a)

    return a_star, poison_sizes

  def take_action(self, state, rho_lim, re_cur, no):

      # check whether state has been visited and determine whether needed to put in queue
      state_bytes = state.tobytes()
      if state_bytes in self.state_set[no]:
        self._logger.info(f'########################################### examining duplicated states, state_set size at no={no} is {len(self.state_set[no])}')
        vis = True
      else:
        self.state_set[no].add(state_bytes)
        vis = False

      actions = []
      for agent in self._agents:
        agent.state = state
        actions.append(agent._select_action())

      a_star, poison_sizes = self._compute_action_and_certifications(np.asarray(actions))

      self._logger.info(f'poison_sizes = {poison_sizes} at no = {no}')
      a_list = []

      snapshot = self._environment.environment.ale.cloneState()
      for a in range(self.num_actions):
          self._environment.environment.ale.restoreState(snapshot)

          _, reward, done = self._run_one_step(a)

          if done:
              continue

          # reward shaping for Pong
          reward = max(reward, 0)

          rho = poison_sizes[a]
          self._logger.info(f'action = {a}, rho = {rho}')

          if rho <= rho_lim:
            a_list.append(a)
          elif not vis:
            elem = Elem(snapshot, state, a, rho, re_cur, no)
            self.queues[elem.rho].append(elem)
            self._logger.info(f'appending to queue[{elem.rho}], cur length={len(self.queues[elem.rho])}')
            if len(self.queues[elem.rho]) >= self.queue_maxlen:
              self.update_queue_into_file(elem.rho)

      return a_list

  def _save_image(self, img, img_file):
    with open(osp.join(self._base_dir, img_file), 'wb') as f:
        pickle.dump(img, f)

  def update_dict_1_level(self, dic, k, v):
    dic[k] = v if k not in dic else min(dic[k], v)

  def update_dict_2_level(self, state_bytes, new_dict, notice=False):
    if state_bytes not in self.state_dict:
      self.state_dict[state_bytes] = new_dict
    else:
      for k, v in new_dict.items():
        if notice:
          assert k not in self.state_dict[state_bytes], f'k = {k} in self.state_dict[state_bytes], previous = {to_str(self.state_dict[state_bytes])}'
        self.update_dict_1_level(self.state_dict[state_bytes], k, v)

  def expand(self, state, rho_lim=0, re_cur=0, no=0):
    state_bytes = state.tobytes()

    if re_cur >= self.re_min:
      self.update_dict_2_level(state_bytes, {self._max_steps_per_episode-no: 0})
      self._logger.info(f'************************************************ pruning at no={no}')
      return self.state_dict[state_bytes]

    if no >= self._max_steps_per_episode:
      self.update_dict_2_level(state_bytes, {self._max_steps_per_episode-no: 0})
      self.re_min = min(self.re_min, re_cur)
      self._logger.info(f'================================================ run to the end with re={re_cur}, updated re_min={self.re_min}')
      return self.state_dict[state_bytes]

    snapshot = self._environment.environment.ale.cloneState()

    if state_bytes in self.state_dict:
      if self._max_steps_per_episode - no in self.state_dict[state_bytes]:
        re_all = re_cur+self.state_dict[state_bytes][self._max_steps_per_episode-no]
        self.re_min = min(self.re_min, re_all)
        self._logger.info(f'%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% expanding duplicated states with same length={self._max_steps_per_episode-no} at no={no}, achieve re_all={re_all}, updated re_min={self.re_min}')
        return self.state_dict[state_bytes]
      else:
        self._logger.info(f'%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% expanding duplicated states without length={self._max_steps_per_episode-no} at no={no}')
    self._logger.info(f'++++++++++++++++++++++++++++++++++++++++++++++++ state dict size is {len(self.state_dict)}')

    a_list = self.take_action(state, rho_lim, re_cur, no)

    if not len(a_list):
      self.update_dict_2_level(state_bytes, {self._max_steps_per_episode-no: 0})
      self.re_min = min(self.re_min, re_cur)
      self._logger.info(f'################################################ no action at no={no}, achieve re={re_cur}, updated re_min={self.re_min}!')
      return self.state_dict[state_bytes]

    if state_bytes in self.state_dict:
      self._logger.info(f'at no={no}, before taking action, self.state_dict[state_bytes] = {to_str(self.state_dict[state_bytes])}')

    cur_dict = dict()

    for a in a_list:
      self._environment.environment.ale.restoreState(snapshot)

      observation, reward, done = self._run_one_step(a)

      assert not done

      # reward shaping for Pong
      reward = max(reward, 0)

      self._logger.info(f'no = {no+1}, re_cur = {re_cur+reward}, rho_lim = {rho_lim}, map = {len(self.certify_map)}, a = {a}, a_list = {a_list}')

      next_state = self._update_observation(state, observation)

      next_dict = self.expand(next_state, rho_lim, re_cur+reward, no+1)
      self._logger.info(f'action = {a}, next_dict = {to_str(next_dict)}')

      for k, v in next_dict.items():
        self.update_dict_1_level(cur_dict, k+1, reward+v)

    self._logger.info(f'at no={no}, after exploring all actions, cur_dict = {to_str(cur_dict)}')

    # self.update_dict_2_level(state_bytes, cur_dict, notice=True)
    self.update_dict_2_level(state_bytes, cur_dict, notice=False)
    self._logger.info(f'at no={no}, after merging dict, self.state_dict[state_bytes] = {to_str(self.state_dict[state_bytes])}')

    return self.state_dict[state_bytes]

  def _update_observation(self, state, observation):
    """Records an observation and update state.

    Extracts a frame from the observation vector and overwrites the oldest
    frame in the state buffer.

    Args:
      observation: numpy array, an observation from the environment.
    """
    # Set current observation. We do the reshaping to handle environments
    # without frame stacking.
    _observation = np.reshape(observation, self.observation_shape)
    # Swap out the oldest frame with the current frame.
    next_state = np.roll(state, -1, axis=-1)
    next_state[0, ..., -1] = _observation
    return next_state

  def get_first(self, queue):
      elem = queue.pop(0)
      return elem.snapshot, elem.state, elem.a, elem.rho, elem.re, elem.no, elem.gid

  def update_map(self, rho):
      self.certify_map[rho] = self.re_min
      self._logger.info(f'------------------------ putting elem into certify_map: {rho} : {self.re_min}')

  def run_experiment(self):
    self._logger.info('exp begin!')

    """Runs a full experiment, spread over multiple iterations."""
    id_list = list(range(1, self._total_num+1))
    # id_list = list(range(1, 31))

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

    t = time.time()

    self.observation_shape = self._agents[0].observation_shape

    state = np.zeros_like(self._agents[0].state)  # maintain the state

    initial_observation = self._environment.reset()
    state = self._update_observation(state, initial_observation)

    self.expand(state, rho_lim=0, re_cur=0, no=0)
    for i in range(self._total_num//2+2):
      self.update_queue_into_file(rho=i)

    self.update_map(rho=0)

    cur_queue = []
    global_rho = 0
    self._logger.info(f'global_rho set to {global_rho}')

    while 1:
      while not cur_queue:
        os.remove(osp.join(self._base_dir, f'queue-{global_rho}.pkl'))
        self._logger.info(f'global_rho = {global_rho} processing done! file removed.')

        global_rho += 1

        if global_rho == self._total_num // 2 + 2:
          # growing search tree done
          self._logger.info(f'grow search tree using {time.time() - t} seconds!')

          save_filename = os.path.join(self._base_dir, 'certify_map.pkl')
          with open(save_filename, 'wb') as f:
            pickle.dump(self.certify_map, f)

          self._logger.info(f'certify map saved to {save_filename}')

          exit(-1)

        self.fout[global_rho].close()
        t_read = time.time()
        cur_queue = self.read_queue_from_file(global_rho)
        self._logger.info(f'file queue-{global_rho}.pkl close, read queue of length={len(cur_queue)} from the file using {time.time() - t_read} seconds!')

        self.state_dict = LimitedSizeDict()
        self._logger.info(f'************************************************ clearing all state dict due to increase to rho={global_rho}')

      snapshot, state, a, rho, re, no, gid = self.get_first(cur_queue)
      self._logger.info(f'get elem from cur_queue, now size = {len(cur_queue)}')
      assert rho == global_rho, f'rho = {rho} != global_rho = {global_rho}'

      self._logger.info(f'start from {no} with rho={rho} and re={re}')

      self._environment.environment.ale.restoreState(snapshot)
      observation, reward, done = self._run_one_step(a)
      next_state = self._update_observation(state, observation)

      # reward shaping for Pong
      reward = max(reward, 0)

      assert not done

      self.expand(next_state, rho, re+reward, no+1)
      for i in range(global_rho+1, self._total_num//2+2):
        self.update_queue_into_file(rho=i)

      self.update_map(rho)

