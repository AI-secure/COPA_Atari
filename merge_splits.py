import concurrent.futures
import logging
import torch

from tqdm import tqdm

import numpy as np

import matplotlib.pyplot as plt

import gzip

import IPython.display
import PIL.Image
import time

import os
import os.path as osp

import shutil

from helper import read_gz_file, write_gz_file, setup_logger

import argparse

SPLIT_SIZE = 1000000
count = SPLIT_SIZE - 1

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-folder', type=str, default='')
    parser.add_argument('--hash-num', type=int, default=0)
    return parser.parse_args()

args = parse_args()


output_folder = osp.join(osp.join(args.input_folder, f'hash_{args.hash_num}'), 'replay_logs')
if not osp.exists(output_folder):
    os.makedirs(output_folder)

task_name = 'merge partitioned data'
setup_logger(task_name, osp.join(osp.join(args.input_folder, f'hash_{args.hash_num}'), 'gen.log'))
logger = logging.getLogger(task_name)

logger.info(f'args = {args}')


prev_ob = np.empty((0, 84, 84), dtype=np.uint8)
prev_action = np.empty(0, dtype=np.int32)
prev_reward = np.empty(0, dtype=np.float32)
prev_terminal = np.empty(0, dtype=np.uint8)

gid = 0

for i in range(1, 6):
    logger.info(f'--------------------- dataset {i} ----------------------------------')

    folder = f'{args.input_folder}_{i}/dataset/hash_{args.hash_num}/replay_logs'

    ckpt_id = 0
    while 1:
        ob_path = osp.join(folder, f'$store$_observation_ckpt.{ckpt_id}.gz')
        if not osp.exists(ob_path): break
        action_path = osp.join(folder, f'$store$_action_ckpt.{ckpt_id}.gz')
        reward_path = osp.join(folder, f'$store$_reward_ckpt.{ckpt_id}.gz')
        terminal_path = osp.join(folder, f'$store$_terminal_ckpt.{ckpt_id}.gz')

        logger.info(f'++++++++++++++++++++ ckpt {ckpt_id} ++++++++++++++++++++++++++++++++')

        start = time.time()
        ob = read_gz_file(ob_path)
        action = read_gz_file(action_path)
        reward = read_gz_file(reward_path)
        terminal = read_gz_file(terminal_path)
        logger.info(f'loading dataset {i} ckpt {ckpt_id} of size {len(ob)} done using {time.time() - start} seconds!')

        start = time.time()
        ob = np.concatenate([prev_ob, np.roll(ob, 1, axis=0)])
        action = np.concatenate([prev_action, np.roll(action, 1, axis=0)])
        reward = np.concatenate([prev_reward, np.roll(reward, 1, axis=0)])
        terminal = np.concatenate([prev_terminal, np.roll(terminal, 1, axis=0)])
        logger.info(f'concatenating done using {time.time() - start} seconds! current size = {len(ob)}')

        if len(ob) >= SPLIT_SIZE:
            logger.info(f'start writing gid {gid}!')
            start = time.time()
            write_gz_file(np.roll(ob[:SPLIT_SIZE], -1, axis=0), osp.join(output_folder, f'$store$_observation_ckpt.{gid}.gz'))
            write_gz_file(np.roll(action[:SPLIT_SIZE], -1, axis=0), osp.join(output_folder, f'$store$_action_ckpt.{gid}.gz'))
            write_gz_file(np.roll(reward[:SPLIT_SIZE], -1, axis=0), osp.join(output_folder, f'$store$_reward_ckpt.{gid}.gz'))
            write_gz_file(np.roll(terminal[:SPLIT_SIZE], -1, axis=0), osp.join(output_folder, f'$store$_terminal_ckpt.{gid}.gz'))
            write_gz_file(np.asarray([999998, 999999, 0, 1, 2], dtype=np.int64), osp.join(output_folder, f'invalid_range_ckpt.{gid}.gz'))
            write_gz_file(np.asarray(count, dtype=np.int64), osp.join(output_folder, f'add_count_ckpt.{gid}.gz'))
            count += SPLIT_SIZE
            logger.info(f'saving gid {gid} done using {time.time() - start} seconds!')

            prev_ob = ob[SPLIT_SIZE:]
            prev_action = action[SPLIT_SIZE:]
            prev_reward = reward[SPLIT_SIZE:]
            prev_terminal = terminal[SPLIT_SIZE:]

            gid += 1

        else:
            prev_ob = ob
            prev_action = action
            prev_reward = reward
            prev_terminal = terminal

        ckpt_id += 1

if len(ob):
    start = time.time()
    write_gz_file(np.roll(ob, -1, axis=0), osp.join(output_folder, f'$store$_observation_ckpt.{gid}.gz'))
    write_gz_file(np.roll(action, -1, axis=0), osp.join(output_folder, f'$store$_action_ckpt.{gid}.gz'))
    write_gz_file(np.roll(reward, -1, axis=0), osp.join(output_folder, f'$store$_reward_ckpt.{gid}.gz'))
    write_gz_file(np.roll(terminal, -1, axis=0), osp.join(output_folder, f'$store$_terminal_ckpt.{gid}.gz'))
    count = len(ob)
    write_gz_file(np.asarray([count-1, count, count+1, count+2, count+3], dtype=np.int64), osp.join(output_folder, f'invalid_range_ckpt.{gid}.gz'))
    write_gz_file(np.asarray(count, dtype=np.int64), osp.join(output_folder, f'add_count_ckpt.{gid}.gz'))
    logger.info(f'saving gid {gid} done using {time.time() - start} seconds!')
