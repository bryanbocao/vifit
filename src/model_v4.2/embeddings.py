'''
!/usr/bin/env python
-*- coding: utf-8 -*-
@Time    : 13/1/2021
@Author  : Huatao
@Email   : 735820057@qq.com
@File    : embedding.py
@Description : check the output and embedding of pretrained model

Modified by
@Author : Bryan Bo Cao
@Email  : boccao@cs.stonybrook.edu, bryan.bo.cao@outlook.com
@Time   : October 2022
'''

import os

import numpy as np
import json
import torch
from torch import nn
from torch.utils.data import DataLoader

import train
from config import Config
from models import LIMUBertModel4Pretrain
from utils import get_device, LIBERTDataset4Pretrain, \
    Preprocess4Normalization, Preprocess4Mask, \
    prepare_sync_training_data, prepare_XT_data, \
    preprocess_config

def generate_embeddings(C, trainer, data_loader, data_type='IMU', save=False):
    def func_forward(model, batch):
        # DEBUG:
        # print('\n\n embedding.py - generate_embeddings_or_output() - (batch)): ', (batch))
        # print('\n\n embedding.py - generate_embeddings_or_output() - np.shape(np.array(batch)): ', np.shape(np.array(batch)))
        # full_win_seq, masked_pos, masked_frames_ORI = batch # New # Note that only full_win_seq is used to generate embeddings
        full_win_seq = batch
        embed = model(full_win_seq) # batch)
        return embed

    pretrain_model_path = C.checkpoint_path_dict['pretrain'] + '/epoch_best.pt'
    print('\n\n embeddings.py - generate_embeddings() - pretrain_model_path: ', pretrain_model_path)
    embeds = trainer.run(func_forward, None, data_loader, pretrain_model_path, output_embed=True)
    # args.pretrain_model: path
    # DEBUG:
    print('\n\n embeddings.py - generate_embeddings() -  np.shape(embeds): ', np.shape(embeds)) # e.g. (31648, 10, 72)
    print('\n\n embeddings.py - generate_embeddings() -  embeds[-1]: ', embeds[-1])

    # trainer.save()
    if save:
        embeds_file_path = C.checkpoint_path_dict['embed'] + '/embeds.npy'
        print('embeddings.py - __main__() - generate_embeddings() - embeds_file_path: ', embeds_file_path)
        print('embeddings.py - __main__() - generate_embeddings() - C.args_dict.keys(): ', C.args_dict.keys())
        np.save(embeds_file_path, embeds)
        print(embeds_file_path, 'saved!')
    return embeds

def load_embeddings(C):
    embeds_file_path = C.checkpoint_path_dict['embed'] + '/embeds.npy'
    pretrain_checkpoint_path = C.checkpoint_path_dict['pretrain']
    print('\n embeddings.py - load_embeddings() - embeds_file_path: ', embeds_file_path)
    print('\n embeddings.py - load_embeddings() - pretrain_checkpoint_path: ', pretrain_checkpoint_path)
    print('\n C.len_win: ', C.len_win)
    embeds = np.load(embeds_file_path).astype(np.float32)
    return embeds

if __name__ == "__main__":
    C = Config()
    print('\n\n C.__dict__: ', C.__dict__)
    print('\n\n C: ', C)

    trainer, train_data_loader, val_data_loader, \
        train_seq_in_correspondence, val_seq_in_correspondence, \
        criterion, optimizer, C = preprocess_config(C, 'embed')
    embeds = generate_embeddings(C, trainer, train_data_loader, 'IMU', save=True)
    # DEBUG:
    print('\n\n embeddings.py - main() - np.shape(embeds): ', np.shape(embeds))
