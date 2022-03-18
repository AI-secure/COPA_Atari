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

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-data-folder', type=str, default='')

    parser.add_argument('--output-folder', type=str, default='')

    return parser.parse_args()

args = parse_args()

# output_folder = osp.join(args.output_folder, 'replay_logs')
output_folder = args.output_folder
if not osp.exists(output_folder):
    os.makedirs(output_folder)

task_name = 'partition data via hash'
setup_logger(task_name, osp.join(args.output_folder, 'gen.log'))
logger = logging.getLogger(task_name)

logger.info(f'args = {args}')


prev_ob = np.empty((0, 84, 84), dtype=np.uint8)
prev_action = np.empty(0, dtype=np.int32)
prev_reward = np.empty(0, dtype=np.float32)
prev_terminal = np.empty(0, dtype=np.uint8)


def compute_hash(cur_ind, start, end):
    return cur_ind, int(np.sum(ob[start:end]) % 50)


cum = 0

partition = [[] for _ in range(50)]

for cur_epi in range(50):
    start = time.time()
    ob = read_gz_file(osp.join(args.train_data_folder, f'$store$_observation_ckpt.{cur_epi}.gz'))
    logger.info(f'-----------------------------------------------------------------')
    logger.info(f'loading {cur_epi} done using {time.time() - start} seconds!')
    
    action = read_gz_file(osp.join(args.train_data_folder, f'$store$_action_ckpt.{cur_epi}.gz'))
    reward = read_gz_file(osp.join(args.train_data_folder, f'$store$_reward_ckpt.{cur_epi}.gz'))
    terminal = read_gz_file(osp.join(args.train_data_folder, f'$store$_terminal_ckpt.{cur_epi}.gz'))
    ind_cur = np.where(terminal)[0]

    ob = np.concatenate([prev_ob, np.roll(ob, 1, axis=0)])
    action = np.concatenate([prev_action, np.roll(action, 1, axis=0)])
    reward = np.concatenate([prev_reward, np.roll(reward, 1, axis=0)])
    terminal = np.concatenate([prev_terminal, np.roll(terminal, 1, axis=0)])

    ind = np.where(terminal)[0]

    ind = np.insert(ind, 0, -1)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        pool = [executor.submit(compute_hash, i, ind[i]+1, ind[i+1]+1) for i in range(len(ind)-1)]
        for i in concurrent.futures.as_completed(pool):
            cur_ind, hash_val = i.result()
            partition[hash_val].append(cum + cur_ind)

    prev_ob = ob[ind[-1]+1:]
    prev_action = action[ind[-1]+1:]
    prev_reward = reward[ind[-1]+1:]
    prev_terminal = terminal[ind[-1]+1:]
    cum += len(ind) - 1


for i in range(50):
    torch.save(partition[i], osp.join(args.output_folder, f'partition_{i}.pt'))

