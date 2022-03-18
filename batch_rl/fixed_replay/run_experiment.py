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

"""Runner for experiments with a fixed replay buffer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

from dopamine.discrete_domains import checkpointer
from dopamine.discrete_domains import iteration_statistics
from dopamine.discrete_domains import run_experiment

import gin
import tensorflow.compat.v1 as tf

import os
import os.path as osp
import shutil

@gin.configurable
class FixedReplayRunner(run_experiment.Runner):
  """Object that handles running Dopamine experiments with fixed replay buffer."""

  def _initialize_checkpointer_and_maybe_resume(self, checkpoint_file_prefix):
    super(FixedReplayRunner, self)._initialize_checkpointer_and_maybe_resume(
        checkpoint_file_prefix)

    self._num_buffers = min(len(os.listdir(self._agent._replay_data_dir)) // 6, 5)
    print(f'num_buffers = {self._num_buffers}')

    # Code for the loading a checkpoint at initialization
    init_checkpoint_dir = self._agent._init_checkpoint_dir  # pylint: disable=protected-access
    if (self._start_iteration == 0) and (init_checkpoint_dir is not None):
      if checkpointer.get_latest_checkpoint_number(self._checkpoint_dir) < 0:
        # No checkpoint loaded yet, read init_checkpoint_dir
        init_checkpointer = checkpointer.Checkpointer(
            init_checkpoint_dir, checkpoint_file_prefix)
        latest_init_checkpoint = checkpointer.get_latest_checkpoint_number(
            init_checkpoint_dir)
        if latest_init_checkpoint >= 0:
          experiment_data = init_checkpointer.load_checkpoint(
              latest_init_checkpoint)
          if self._agent.unbundle(
              init_checkpoint_dir, latest_init_checkpoint, experiment_data):
            if experiment_data is not None:
              assert 'logs' in experiment_data
              assert 'current_iteration' in experiment_data
              self._logger.data = experiment_data['logs']
              self._start_iteration = experiment_data['current_iteration'] + 1
            tf.logging.info(
                'Reloaded checkpoint from %s and will start from iteration %d',
                init_checkpoint_dir, self._start_iteration)

  def _run_train_phase(self):
    """Run training phase."""
    self._agent.eval_mode = False
    start_time = time.time()
    from tqdm import tqdm
    for _ in tqdm(range(self._training_steps)):
      self._agent._train_step()  # pylint: disable=protected-access
    time_delta = time.time() - start_time
    tf.logging.info('Average training steps per second: %.2f',
                    self._training_steps / time_delta)

  def _run_one_iteration(self, iteration):
    """Runs one iteration of agent/environment interaction."""
    statistics = iteration_statistics.IterationStatistics()
    tf.logging.info('Starting iteration %d', iteration)
    # pylint: disable=protected-access
    if not self._agent._replay_suffix:
      # Reload the replay buffer
      self._agent._replay.memory.reload_buffer(num_buffers=self._num_buffers)
    # pylint: enable=protected-access
    self._run_train_phase()

    num_episodes_eval, average_reward_eval = self._run_eval_phase(statistics)

    self._save_tensorboard_summaries(
        iteration, num_episodes_eval, average_reward_eval)
    return statistics.data_lists

  def _save_tensorboard_summaries(self, iteration,
                                  num_episodes_eval,
                                  average_reward_eval):
    """Save statistics as tensorboard summaries.

    Args:
      iteration: int, The current iteration number.
      num_episodes_eval: int, number of evaluation episodes run.
      average_reward_eval: float, The average evaluation reward.
    """
    summary = tf.Summary(value=[
        tf.Summary.Value(tag='Eval/NumEpisodes',
                         simple_value=num_episodes_eval),
        tf.Summary.Value(tag='Eval/AverageReturns',
                         simple_value=average_reward_eval)
    ])
    self._summary_writer.add_summary(summary, iteration)

  def _checkpoint_experiment(self, iteration):
    """Checkpoint experiment data.

    Args:
      iteration: int, iteration number for checkpointing.
    """
    experiment_data = self._agent.bundle_and_checkpoint(self._checkpoint_dir,
                                                        iteration)
    if experiment_data:
      experiment_data['current_iteration'] = iteration
      experiment_data['logs'] = self._logger.data
      self._checkpointer.save_checkpoint(iteration, experiment_data)

    if (iteration + 1) % 10 == 0:
      shutil.copyfile(osp.join(self._checkpoint_dir, f'ckpt.{iteration}'), 
                      osp.join(self._checkpoint_dir, f'stored_ckpt.{iteration}'))
      shutil.copyfile(osp.join(self._checkpoint_dir, f'sentinel_checkpoint_complete.{iteration}'), 
                      osp.join(self._checkpoint_dir, f'stored_sentinel_checkpoint_complete.{iteration}'))
      shutil.copyfile(osp.join(self._checkpoint_dir, f'tf_ckpt-{iteration}.data-00000-of-00001'), 
                      osp.join(self._checkpoint_dir, f'stored_tf_ckpt-{iteration}.data-00000-of-00001'))
      shutil.copyfile(osp.join(self._checkpoint_dir, f'tf_ckpt-{iteration}.index'), 
                      osp.join(self._checkpoint_dir, f'stored_tf_ckpt-{iteration}.index'))
      shutil.copyfile(osp.join(self._checkpoint_dir, f'tf_ckpt-{iteration}.meta'), 
                      osp.join(self._checkpoint_dir, f'stored_tf_ckpt-{iteration}.meta'))
