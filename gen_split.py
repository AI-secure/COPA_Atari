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
    parser.add_argument('--epi-index-path', type=str, default='')
    parser.add_argument('--train-data-folder', type=str, default='')

    parser.add_argument('--output-folder', type=str, default='')

    parser.add_argument('--start-id', type=int, default=40)
    parser.add_argument('--end-id', type=int, default=50)
    return parser.parse_args()

args = parse_args()


output_folder = osp.join(args.output_folder, 'replay_logs')
if not osp.exists(output_folder):
    os.makedirs(output_folder)

task_name = 'generate data based on episode index'
setup_logger(task_name, osp.join(args.output_folder, 'gen.log'))
logger = logging.getLogger(task_name)

logger.info(f'args = {args}')

epi_sel = torch.load(args.epi_index_path)

logger.info(f'episode indices loading done! length = {len(epi_sel)}')

# epi_sel = list(range(100000))

new_ob = np.empty((0, 84, 84), dtype=np.uint8)
new_action = np.empty(0, dtype=np.int32)
new_reward = np.empty(0, dtype=np.float32)
new_terminal = np.empty(0, dtype=np.uint8)


i = 0
cum = 0
wid = 0


for cur_epi in range(args.start_id, args.end_id):
    if cur_epi > args.start_id:
        prev_ob = ob
        prev_reward = reward
        prev_action = action
        prev_terminal = terminal
        prev_ind_cur = ind_cur
        cum += len(ind_cur)
    
    start = time.time()
    ob = read_gz_file(osp.join(args.train_data_folder, f'$store$_observation_ckpt.{cur_epi}.gz'))
    logger.info(f'-----------------------------------------------------------------')
    logger.info(f'loading {cur_epi} done using {time.time() - start} seconds! [cur len = {len(new_action)}]')
    
    reward = read_gz_file(osp.join(args.train_data_folder, f'$store$_reward_ckpt.{cur_epi}.gz'))
    action = read_gz_file(osp.join(args.train_data_folder, f'$store$_action_ckpt.{cur_epi}.gz'))
    terminal = read_gz_file(osp.join(args.train_data_folder, f'$store$_terminal_ckpt.{cur_epi}.gz'))
    ind_cur = np.where(terminal)[0]
    
    idx = epi_sel[i]

    if idx == cum:
        logger.info('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        end = ind_cur[0] + 1

        if cur_epi == args.start_id:
            new_ob = np.concatenate([ob[-1:], ob[:end]])
            new_reward = np.concatenate([reward[-1:], reward[:end]])
            new_action = np.concatenate([action[-1:], action[:end]])
            new_terminal = np.concatenate([terminal[-1:], terminal[:end]])
        else:
            new_ob = np.concatenate([new_ob, prev_ob[prev_ind_cur[-1]+1:-1], ob[-1:], ob[:end]])
            new_reward = np.concatenate([new_reward, prev_reward[prev_ind_cur[-1]+1:-1], reward[-1:], reward[:end]])
            new_action = np.concatenate([new_action, prev_action[prev_ind_cur[-1]+1:-1], action[-1:], action[:end]])
            new_terminal = np.concatenate([new_terminal, prev_terminal[prev_ind_cur[-1]+1:-1], terminal[-1:], terminal[:end]])

        logger.info(f'including the first segment and len = {len(new_action)}')

        i += 1
        idx = epi_sel[i]


    new_ind = []

    while 1:            
        if idx-cum >= len(ind_cur):
            break

        beg = ind_cur[idx-cum-1]+1
        end = ind_cur[idx-cum]+1
        
        # logger.info(beg, '-', end)
        new_ind += list(range(beg, end))

        i += 1
        if i >= len(epi_sel):
            break

        idx = epi_sel[i]

    if len(new_ind):

        logger.info('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
        logger.info(f'start concat!')
        logger.info(f'len new_ob = {len(new_ob)}, len new_ind = {len(new_ind)}')
        start = time.time()
        new_ind = np.asarray(new_ind)
        new_ob = np.concatenate([new_ob, ob[new_ind]])
        new_reward = np.concatenate([new_reward, reward[new_ind]])
        new_action = np.concatenate([new_action, action[new_ind]])
        new_terminal = np.concatenate([new_terminal, terminal[new_ind]])
        logger.info(f'concat finish using {time.time() - start} seconds!')

        while len(new_action) >= 1000000:
            logger.info('************************************************************')
            logger.info(f'start saving {wid} with len = {len(new_action)}!')

            start = time.time()
            write_gz_file(np.roll(new_ob[:1000000], -1, axis=0), osp.join(output_folder, f'$store$_observation_ckpt.{wid}.gz'))
            logger.info(f'write observation {wid} using {time.time() - start} seconds!')


            write_gz_file(np.roll(new_reward[:1000000], -1), osp.join(output_folder, f'$store$_reward_ckpt.{wid}.gz'))
            write_gz_file(np.roll(new_action[:1000000], -1), osp.join(output_folder, f'$store$_action_ckpt.{wid}.gz'))
            write_gz_file(np.roll(new_terminal[:1000000], -1), osp.join(output_folder, f'$store$_terminal_ckpt.{wid}.gz'))

            new_ob = new_ob[1000000:]
            new_reward = new_reward[1000000:]
            new_action = new_action[1000000:]
            new_terminal = new_terminal[1000000:]

            logger.info(f'len = {len(new_action)} after saving!')

            wid += 1


if len(new_action):

    logger.info('************************************************************')
    logger.info(f'start saving {wid} with len = {len(new_action)}!')

    start = time.time()
    write_gz_file(np.roll(new_ob, -1, axis=0), osp.join(output_folder, f'$store$_observation_ckpt.{wid}.gz'))
    logger.info(f'write observation {wid} using {time.time() - start} seconds!')


    write_gz_file(np.roll(new_reward, -1), osp.join(output_folder, f'$store$_reward_ckpt.{wid}.gz'))
    write_gz_file(np.roll(new_action, -1), osp.join(output_folder, f'$store$_action_ckpt.{wid}.gz'))
    write_gz_file(np.roll(new_terminal, -1), osp.join(output_folder, f'$store$_terminal_ckpt.{wid}.gz'))

    logger.info(f'len = {len(new_action)} after saving!')

    wid += 1
