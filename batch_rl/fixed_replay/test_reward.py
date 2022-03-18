# coding=utf-8
# Copyright 2021 The Google Research Authors.
#
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

r"""The entry point for running experiments with fixed replay datasets.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import json
import os


from absl import app
from absl import flags

from batch_rl.fixed_replay import run_experiment_test_reward,\
                                  run_experiment_test_reward_tight, \
                                  run_experiment_test_reward_window, \
                                  run_experiment_test_reward_window_dynamic
from batch_rl.fixed_replay.agents import dqn_agent
from batch_rl.fixed_replay.agents import multi_head_dqn_agent
from batch_rl.fixed_replay.agents import quantile_agent
from batch_rl.fixed_replay.agents import rainbow_agent

from dopamine.discrete_domains import run_experiment as base_run_experiment
import tensorflow.compat.v1 as tf

# from dopamine.google import xm_utils

flags.DEFINE_integer('re_min', 2147438647, 'min reward')
flags.DEFINE_integer('start_rho', 0, 'start rho')
flags.DEFINE_integer('max_steps_per_episode', 100, 'max_steps_per_episode')
flags.DEFINE_integer('max_window_size', 5, 'max window size')
flags.DEFINE_integer('window_size', 2, 'window size')
flags.DEFINE_integer('total_num', 50, 'total number of models')
flags.DEFINE_string('cert_alg', 'vanilla', 'Name of the cert alg.')
flags.DEFINE_string('agent_name', 'dqn', 'Name of the agent.')
flags.DEFINE_string('base_dir', None,
                    'Base directory to host all required sub-directories.')
flags.DEFINE_multi_string(
    'gin_files', [], 'List of paths to gin configuration files (e.g.'
    '"third_party/py/dopamine/agents/dqn/dqn.gin").')                    
flags.DEFINE_multi_string(
    'gin_bindings', [],
    'Gin bindings to override the values set in the config files '
    '(e.g. "DQNAgent.epsilon_train=0.1",'
    '      "create_environment.game_name="Pong"").')
flags.DEFINE_string('replay_dir', None, 'Directory from which to load the '
                    'replay data')
flags.DEFINE_string('init_checkpoint_dir', None, 'Directory from which to load '
                    'the initial checkpoint before training starts.')
flags.DEFINE_string('model_dir', None, 'Directory from which to load the '
                    'model')

FLAGS = flags.FLAGS


def create_agent(sess, environment, replay_data_dir, summary_writer=None):
  """Creates a DQN agent.

  Args:
    sess: A `tf.Session`object  for running associated ops.
    environment: An Atari 2600 environment.
    replay_data_dir: Directory to which log the replay buffers periodically.
    summary_writer: A Tensorflow summary writer to pass to the agent
      for in-agent training statistics in Tensorboard.

  Returns:
    A DQN agent with metrics.
  """

  if FLAGS.agent_name == 'dqn':
    agent = dqn_agent.FixedReplayDQNAgent
  elif FLAGS.agent_name == 'c51':
    agent = rainbow_agent.FixedReplayRainbowAgent
  elif FLAGS.agent_name == 'quantile':
    agent = quantile_agent.FixedReplayQuantileAgent
  elif FLAGS.agent_name == 'multi_head_dqn':
    agent = multi_head_dqn_agent.FixedReplayMultiHeadDQNAgent
  else:
    raise ValueError('{} is not a valid agent name'.format(FLAGS.agent_name))

  return agent(sess, num_actions=environment.action_space.n,
               replay_data_dir=replay_data_dir, summary_writer=summary_writer,
               init_checkpoint_dir=FLAGS.init_checkpoint_dir,
               eval_mode=True, epsilon_eval=0)




def main(unused_argv):
  tf.logging.set_verbosity(tf.logging.INFO)
  base_run_experiment.load_gin_configs(FLAGS.gin_files, FLAGS.gin_bindings)
  # replay_data_dir = os.path.join(FLAGS.replay_dir, 'replay_logs')

  create_agent_fn = functools.partial(
      create_agent, replay_data_dir=None)

  if FLAGS.cert_alg == 'vanilla':
    runner = run_experiment_test_reward.TestRunner(FLAGS.base_dir, FLAGS.model_dir, FLAGS.total_num,
                                                  create_agent_fn,
                                                  max_steps_per_episode=FLAGS.max_steps_per_episode)
  elif FLAGS.cert_alg == 'tight':
    runner = run_experiment_test_reward_tight.TestRunner(FLAGS.base_dir, FLAGS.model_dir, FLAGS.total_num,
                                                  create_agent_fn,
                                                  max_steps_per_episode=FLAGS.max_steps_per_episode)

  elif FLAGS.cert_alg == 'window':
    runner = run_experiment_test_reward_window.TestRunner(FLAGS.base_dir, FLAGS.model_dir, FLAGS.total_num,
                                                  create_agent_fn,
                                                  max_steps_per_episode=FLAGS.max_steps_per_episode,
                                                  window_size=FLAGS.window_size)
  elif FLAGS.cert_alg == 'dynamic':
    runner = run_experiment_test_reward_window_dynamic.TestRunner(FLAGS.base_dir, FLAGS.model_dir, FLAGS.total_num,
                                                  create_agent_fn,
                                                  max_steps_per_episode=FLAGS.max_steps_per_episode,
                                                  max_window_size=FLAGS.max_window_size)
  else:
    raise NotImplementedError(f'certification algorithm = {FLAGS.cert_alg} not implemented!')

  runner.run_experiment()


if __name__ == '__main__':
  app.run(main)
