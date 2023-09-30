import os
import argparse
from scipy.special import factorial
from scipy.optimize import linear_sum_assignment
import cv2
from PIL import Image, ImageDraw
from scipy.signal import savgol_filter
from numpy import (array, dot, arccos, clip)
from numpy.linalg import norm
import math

from torchsummary import summary
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.ops as tops
from torch_utils import distance_box_iou_loss, generalized_box_iou_loss, _diou_iou_loss
# import torchvision.ops.distance_box_iou_loss, torchvision.ops.generalized_box_iou_loss
# https://pytorch.org/vision/main/generated/torchvision.ops.distance_box_iou_loss.html
# https://pytorch.org/vision/main/generated/torchvision.ops.generalized_box_iou_loss.html

# Baseline of ViTag Xtranslator >>>
from numpy import array
# import tensorflow as tf
from keras import backend as K
from keras.models import Model
from keras.layers import Input
from keras.layers import GRU, LSTM, Bidirectional
from keras.layers import concatenate, Concatenate, Add, Layer
from keras.layers import Dense, MaxPooling1D, Conv1D
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.layers import Masking
from keras.layers import Lambda
from keras.layers import BatchNormalization
# from keras.utils import plot_model
from keras.callbacks import ModelCheckpoint
import keras

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
import tensorflow as tf
from tensorflow.python.keras import backend as K
config = tf.compat.v1.ConfigProto(device_count = {'GPU': 1})
sess = tf.compat.v1.Session(config=config)
K.set_session(sess)
# Baseline of ViTag Xtranslator <<<

import time
import random
import numpy as np
import numpy.ma as ma
import torch
import sys
from sys import getsizeof
import json
import pickle
import copy
from collections import defaultdict
from sklearn.metrics import mean_squared_error

from config import Config
import train
from dtw import *
from models import LIMUBertModel4Pretrain, Xformer_IemcC2C, Xformer_IFcC2C, \
    Xlstm_IFcC2C, Xtranslator_IFcC2C, Width_regressor, Height_regressor, Depth_regressor, \
    MyKalmanFilter, MyKalmanFilter_v2

# ================
#  From LIMU-BERT
def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def get_device(gpu):
    if gpu is None:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cuda:' + gpu if torch.cuda.is_available() else 'cpu')
    n_gpu = torch.cuda.device_count()
    print('%s (%d GPUs)' % (device, n_gpu))
    return device

def bert_mask(seq_len, goal_num_predict):
    return random.sample(range(seq_len), goal_num_predict)

def span_mask(seq_len, max_gram=3, p=0.2, goal_num_predict=15):
    ngrams = np.arange(1, max_gram + 1, dtype=np.int64)
    pvals = p * np.power(1 - p, np.arange(max_gram))
    # alpha = 6
    # pvals = np.power(alpha, ngrams) * np.exp(-alpha) / factorial(ngrams)
    pvals /= pvals.sum(keepdims=True)
    mask_pos = set()
    while len(mask_pos) < goal_num_predict:
        n = np.random.choice(ngrams, p=pvals)
        n = min(n, goal_num_predict - len(mask_pos))
        anchor = np.random.randint(seq_len)
        if anchor in mask_pos:
            continue
        for i in range(anchor, min(anchor + n, seq_len - 1)):
            mask_pos.add(i)
    # DEBUG:
    # print('\n\n')
    # print('utils.py - span_mask() - len(list(mask_pos)): ', len(list(mask_pos))) # e.g. 18
    # print('utils.py - span_mask() - list(mask_pos): ', list(mask_pos))
    '''
    e.g.
    list(mask_pos):  [6, 7, 8, 9, 10, 11, 12, 13, 14, 78, 79, 49, 50, 51, 80, 81, 82, 94]
    list(mask_pos):  [89, 101, 5, 102, 103, 8, 9, 10, 91, 19, 20, 51, 52, 53, 86, 87, 90, 88]
    list(mask_pos):  [32, 75, 76, 77, 29, 61, 63, 50, 117, 62, 24, 25, 26, 27, 28, 93, 30, 31]
    list(mask_pos):  [65, 66, 67, 5, 13, 110, 14, 49, 50, 115, 21, 22, 23, 24, 25, 26, 27, 28]
    '''
    # print('utils.py - span_mask() - seq_len: ', seq_len) # e.g. 120
    return list(mask_pos)

def regularization_loss(model, lambda1, lambda2):
    l1_regularization, l2_regularization = 0.0, 0.0
    for param in model.parameters():
        l1_regularization += torch.norm(param, 1)
        l2_regularization += torch.norm(param, 2)
    return lambda1 * l1_regularization, lambda2 * l2_regularization

class Pipeline():
    ''' Pre-process Pipeline Class: callable '''
    def __init__(self):
        super().__init__()

    def __call__(self, instance):
        raise NotImplementedError

class Preprocess4Normalization(Pipeline):
    ''' Pre-processing steps for pretraining transformer '''
    def __init__(self, feature_len, norm_acc=True, norm_mag=True, gamma=1.0):
        super().__init__()
        self.feature_len = feature_len
        self.norm_acc = norm_acc
        self.norm_mag = norm_mag
        self.eps = 1e-5
        self.acc_norm = 9.8
        self.gamma = gamma

    def __call__(self, instance):
        # print('\n utils.py - Preprocess4Normalization - start __call__ - np.shape(instance): ', np.shape(instance))
        # e.g. (10, 1, 9)
        if isinstance(instance, np.ndarray):
            # instance_new = instance.copy()[:, :self.feature_len] # CPU memory # prev dim
            instance_new = instance.copy()[:, :, :self.feature_len] # CPU memory
        elif torch.is_tensor(instance):
            # instance_new = instance.clone()[:, :self.feature_len] # PyTorch tensor # prev dim
            instance_new = instance.clone()[:, :, :self.feature_len] # PyTorch tensor
        if instance_new.shape[2] >= 6 and self.norm_acc:
        # if instance_new.shape[1] >= 6 and self.norm_acc: # prev dim
            # instance_new[:, :3] = instance_new[:, :3] / self.acc_norm
            instance_new[:, :, :3] = instance_new[:, :, :3] / self.acc_norm
        # if instance_new.shape[1] == 9 and self.norm_mag:
        if instance_new.shape[2] == 9 and self.norm_mag:
            # mag_norms = np.linalg.norm(instance_new[:, 6:9], axis=1) + self.eps # prev dim
            # mag_norms = np.linalg.norm(instance_new[:, 3:6], axis=1) + self.eps # prev dim
            mag_norms = np.linalg.norm(instance_new[:, :, 6:9], axis=2) + self.eps
            # print('\n\n After np.linalg.norm np.shape(mag_norms): ', np.shape(mag_norms)) # e.g. (10,) # prev dim
            # print('\n\n After np.linalg.norm mag_norms.size: ', mag_norms.size) # e.g. 10 # prev dim
            # mag_norms = np.repeat(mag_norms.reshape(mag_norms.size, 1), 3, axis=1) # prev dim
            mag_norms = np.repeat(mag_norms.reshape(np.shape(mag_norms)[0], np.shape(mag_norms)[1], 1), 3, axis=2)
            # print('\n\n After np.repeat np.shape(mag_norms): ', np.shape(mag_norms)) # e.g. (10, 3) # prev dim
            # instance_new[:, 6:9] = instance_new[:, 6:9] / mag_norms * self.gamma # prev dim
            # instance_new[:, 3:6] = instance_new[:, 3:6] / mag_norms * self.gamma # prev dim
            instance_new[:, :, 6:9] = instance_new[:, :, 6:9] / mag_norms * self.gamma
        # print('\n utils.py - Preprocess4Normalization - after __call__ - np.shape(instance_new): ', np.shape(instance_new))
        return instance_new

class Preprocess4Mask:
    ''' Pre-processing steps for pretraining transformer '''
    def __init__(self, mask_cfg):
        self.mask_ratio = mask_cfg['mask_ratio']  # masking probability
        self.mask_alpha = mask_cfg['mask_alpha']
        self.max_gram = mask_cfg['max_gram']
        self.mask_prob = mask_cfg['mask_prob']
        self.replace_prob = mask_cfg['replace_prob']
        self.imu200 = mask_cfg['imu200']

    def gather(self, data, position1, position2):
        result = []
        for i in range(position1.shape[0]):
            result.append(data[position1[i], position2[i]])
        return np.array(result)

    def mask(self, data, position1, position2):
        for i in range(position1.shape[0]):
            data[position1[i], position2[i]] = np.zeros(position2[i].size)
        return data

    def replace(self, data, position1, position2):
        for i in range(position1.shape[0]):
            data[position1[i], position2[i]] = np.random.random(position2[i].size)
        return data

    def __call__(self, instance):
        shape = instance.shape
        # DEBUG:
        # print('\n\n')
        # print('utils - Preprocess4Mask __call__() - instance.shape: ', shape) # e.g. (10, 1, 9), prev (10, 9)
        # print('utils - Preprocess4Mask __call__() - instance: ', instance)
        '''
        e.g.
         [-6.38073310e-02  1.15500355e+00  3.49583983e-01 -1.50652006e-01
          -1.17542839e+00  6.02743328e-02]
         [ 4.05685008e-01  1.08684552e+00 -6.52769953e-02 -1.02216303e+00
           3.19218010e-01 -6.20173991e-01]
         [-1.36139005e-01  6.97820008e-01  1.22639999e-01 -1.30809236e+00
          -7.26900026e-02 -1.73101795e+00]
         [ 1.13280997e-01  7.77283013e-01  2.35054001e-01  1.65377295e+00
           6.90211475e-01  1.51632595e+00]
         [ 2.86773652e-01  9.28425968e-01 -9.08659995e-02 -2.66830325e-01
           8.94294024e-01  3.65170985e-01]]
        '''

        # the number of prediction is sometimes less than max_pred when sequence is short
        n_pred = max(1, int(round(shape[0] * self.mask_ratio)))
        # DEBUG:
        # print('\n\n')
        # print('utils - Preprocess4Mask __call__() - n_pred: ', n_pred) # e.g. 18

        # For masked Language Models
        # mask_pos = bert_mask(shape[0], n_pred)
        mask_pos = span_mask(shape[0], self.max_gram,  goal_num_predict=n_pred)

        if isinstance(instance, np.ndarray):
            instance_mask = instance.copy() # CPU memory
        elif torch.is_tensor(instance):
            instance_mask = instance.clone() # Pytorch tensor

        if isinstance(mask_pos, tuple):
            mask_pos_index = mask_pos[0]
            if np.random.rand() < self.mask_prob:
                self.mask(instance_mask, mask_pos[0], mask_pos[2])
                # print('line 349')
            elif np.random.rand() < self.replace_prob:
                self.replace(instance_mask, mask_pos[0], mask_pos[2])
                # print('line 352')
        else:
            mask_pos_index = mask_pos
            # DEBUG:
            # print('\n\n')
            # print('- Preprocess4Mask __call__() - mask_pos_index: ', mask_pos_index)
            '''
            e.g.
            [96, 5, 6, 7, 8, 9, 10, 79, 80, 84, 85, 86, 87, 59, 92, 93, 94, 95]
            [96, 9, 10, 11, 12, 13, 14, 15, 16, 85, 86, 87, 88, 89, 90, 93, 94, 95]
            [100, 101, 102, 103, 104, 105, 112, 113, 114, 115, 116, 117, 118, 51, 52, 53, 54, 55]
            '''
            if np.random.rand() < self.mask_prob:
                # print('\n\n np.shape(instance_mask): ', np.shape(instance_mask))
                # e.g. (10, 1, 9), prev (10, 9)
                if self.imu200: instance_mask[mask_pos, :, :] = torch.from_numpy(np.zeros((len(mask_pos), 200, shape[2])))
                else: instance_mask[mask_pos, :, :] = torch.from_numpy(np.zeros((len(mask_pos), 1, shape[2])))
            elif np.random.rand() < self.replace_prob:
                # print('\n\n np.shape(instance_mask): ', np.shape(instance_mask))
                # e.g. (10, 1, 9), prev (10, 9)
                if self.imu200: instance_mask[mask_pos, :, :] = np.random.random((len(mask_pos), 200, shape[2]))
                else: instance_mask[mask_pos, :, :] = np.random.random((len(mask_pos), 1, shape[2]))
        # DEBUG:
        # print('\n\n')
        # print('utils - Preprocess4Mask __call__() - mask_pos_index: ', mask_pos_index)
        '''
        e.g.
        [96, 5, 6, 7, 8, 9, 10, 79, 80, 84, 85, 86, 87, 59, 92, 93, 94, 95]
        [96, 9, 10, 11, 12, 13, 14, 15, 16, 85, 86, 87, 88, 89, 90, 93, 94, 95]
        [100, 101, 102, 103, 104, 105, 112, 113, 114, 115, 116, 117, 118, 51, 52, 53, 54, 55]
        '''
        sub_seqs_ORI = instance[mask_pos_index, :, :]
        # DEBUG:
        # print('\n\n')
        # print('utils - Preprocess4Mask __call__() - len(sub_seqs_ORI): ', len(sub_seqs_ORI)) # e.g. 18
        return instance_mask, np.array(mask_pos_index), np.array(sub_seqs_ORI)

class LIBERTDataset4Pretrain(Dataset):
    """ Load sentence pair (sequential or random order) from corpus """
    def __init__(self, data, pipeline=[], correspondence=False):
        super().__init__()
        self.pipeline = pipeline
        self.correspondence = correspondence
        # self.correspondence: True, then self.data = seq_in_correspondence
        #     seq_in_correspondence = [seq_in_BBX5, seq_in_FTM2, seq_in_IMU19, seq_in_TX] (5, 2, 19, 9)
        #     will be concatenated into one tensor.
        # self.correspondence: False, then self.data = seq_in_IMU_TX
        if self.correspondence:
            # DEBUG:
            # print('\n\n utils.py - LIBERTDataset4Pretrain() - torch.is_tensor(data[0])): ', torch.is_tensor(data[0]))
            # print('\n\n utils.py - LIBERTDataset4Pretrain() - np.shape(np.array(data[0])): ', np.shape(np.array(data[0])))
            # print('\n\n utils.py - LIBERTDataset4Pretrain() - np.shape(np.array(data[1])): ', np.shape(np.array(data[1])))
            # print('\n\n utils.py - LIBERTDataset4Pretrain() - np.shape(np.array(data[2])): ', np.shape(np.array(data[2])))
            # print('\n\n utils.py - LIBERTDataset4Pretrain() - np.shape(np.array(data[3])): ', np.shape(np.array(data[3])))
            '''
            e.g.
            utils.py - LIBERTDataset4Pretrain() - torch.is_tensor(data[0])):  False
            utils.py - LIBERTDataset4Pretrain() - np.shape(np.array(data[0])):  (33025, 10, 1, 5)
            utils.py - LIBERTDataset4Pretrain() - np.shape(np.array(data[1])):  (33025, 10, 1, 2)
            utils.py - LIBERTDataset4Pretrain() - np.shape(np.array(data[2])):  (33025, 10, 1, 19)
            utils.py - LIBERTDataset4Pretrain() - np.shape(np.array(data[3])):  (33025, 10, 1, 9)
            '''
            # self.data = torch.cat((data[0], data[1], data[3]), 2) # shape in indices: (5, 2, 19, 9)
            self.data = np.concatenate((data[0], data[1], data[3]), 3) # 2) # shape in indices: (5, 2, 19, 9)
            # data[0]: seq_in_BBX5
            # data[1]: seq_in_FTM2
            # data[2]: seq_in_IMU19
            self.data_shapes = [np.shape(data[0]), np.shape(data[1]), np.shape(data[3])]
            # print('\n utils.py - LIBERTDataset4Pretrain - if self.correspondence - self.data_shapes: ', self.data_shapes)
            # e.g. self.data_shapes:  [(33025, 10, 1, 5), (33025, 10, 1, 2), (33025, 10, 1, 9)]
        else:
            self.data = data
        # DEBUG:
        # print('\n\n utils.py - LIMUBERTDataset4Pretrain() - __init__() len(data): ', len(self.data)) # e.g. 31648 # correct
        # print('\n\n utils.py - LIMUBERTDataset4Pretrain() - __init__() np.shape(data): ', np.shape(self.data)) # e.g. torch.Size([31648, 10, 9]) or torch.Size([31088, 20, 16]) # correct

    def __getitem__(self, index):
        instance = self.data[index]
        # print('\n utils.py - LIMUBERTDataset4Pretrain() - __getitem__() np.shape(instance): ', np.shape(instance)) # e.g. (10, 1, 16)
        if self.correspondence:
        # np.shape(self.data) = torch.Size([, len_win, feature_num])
            # print('\n\n utils.py - LIMUBERTDataset4Pretrain() - __getitem__() - if self.correspondence - np.shape(instance): ', np.shape(instance)) # e.g. torch.Size([20, 16])
            # seq_in_BBX5, seq_in_FTM2, seq_in_IMU_TX = instance[:, :5], instance[:, 5:7], instance[:, 7:16]
            # seq_in_BBX5, seq_in_FTM2, seq_in_IMU_TX = instance[:, :self.data_shapes[0][-1]], \
            #     instance[:, self.data_shapes[0][-1] : self.data_shapes[0][-1] + self.data_shapes[1][-1]], \
            #     instance[:, self.data_shapes[0][-1] + self.data_shapes[1][-1] : self.data_shapes[0][-1] + self.data_shapes[1][-1] + self.data_shapes[2][-1]] # prev
            seq_in_BBX5, seq_in_FTM2, seq_in_IMU_TX = instance[:, :, :self.data_shapes[0][-1]], \
                instance[:, :, self.data_shapes[0][-1] : self.data_shapes[0][-1] + self.data_shapes[1][-1]], \
                instance[:, :, self.data_shapes[0][-1] + self.data_shapes[1][-1] : self.data_shapes[0][-1] + self.data_shapes[1][-1] + self.data_shapes[2][-1]]
            # DEBUG:
            # print('\n utils.py - LIMUBERTDataset4Pretrain() - __getitem__() - self.correspondenc - np.shape(instance): ', np.shape(instance))
            # print('\n utils.py - LIMUBERTDataset4Pretrain() - __getitem__() - self.correspondenc - np.shape(seq_in_BBX5): ', np.shape(seq_in_BBX5))
            # print('\n utils.py - LIMUBERTDataset4Pretrain() - __getitem__() - self.correspondenc - np.shape(seq_in_FTM2): ', np.shape(seq_in_FTM2))
            # print('\n utils.py - LIMUBERTDataset4Pretrain() - __getitem__() - self.correspondenc - np.shape(seq_in_IMU_TX): ', np.shape(seq_in_IMU_TX))
            '''
            e.g.
            utils.py - LIMUBERTDataset4Pretrain() - __getitem__() - self.correspondenc - np.shape(instance):  (10, 1, 16)
            utils.py - LIMUBERTDataset4Pretrain() - __getitem__() - self.correspondenc - np.shape(seq_in_BBX5):  (10, 1, 5)
            utils.py - LIMUBERTDataset4Pretrain() - __getitem__() - self.correspondenc - np.shape(seq_in_FTM2):  (10, 1, 2)
            utils.py - LIMUBERTDataset4Pretrain() - __getitem__() - self.correspondenc - np.shape(seq_in_IMU_TX):  (10, 1, 9)
            '''
            if self.pipeline:
                for process in self.pipeline:
                    seq_in_IMU_TX = process(seq_in_IMU_TX)
                    # print('\n\n utils.py - LIMUBERTDataset4Pretrain() - __getitem__() - if self.correspondence - np.shape(seq_in_IMU_TX): ', np.shape(seq_in_IMU_TX)) # e.g.
            return seq_in_BBX5, seq_in_FTM2, seq_in_IMU_TX
        else:
        # self.data = seq_in_IMU_TX
            if self.pipeline:
                for process in self.pipeline:
                    instance = process(instance)
            # DEBUG:
            # print('\n\n utils.py - LIMUBERTDataset4Pretrain() - __getitem__() - if not self.correspondence - np.shape(instance): ', np.shape(instance)) # e.g. (3,)
            partial_masked_full_win_seq, masked_pos, masked_frames_ORI = instance
            # DEBUG:
            # print('\n\n utils.py - LIMUBERTDataset4Pretrain() - __getitem__() - np.shape(partial_masked_full_win_seq): ', np.shape(partial_masked_full_win_seq))
            # print('\n\n utils.py - LIMUBERTDataset4Pretrain() - __getitem__() - np.shape(masked_pos): ', np.shape(masked_pos))
            # print('\n\n utils.py - LIMUBERTDataset4Pretrain() - __getitem__() - np.shape(masked_frames_ORI): ', np.shape(masked_frames_ORI))
            '''
            np.shape(partial_masked_full_win_seq):  (120, 6) # (len_win, dim_feature)
            np.shape(masked_pos):  (18,) # n_masked_frames
            np.shape(masked_frames_ORI):  (18, 6) # (n_masked_frames, dim_feature)
            '''
            # return torch.from_numpy(partial_masked_full_win_seq), torch.from_numpy(masked_pos).long(), torch.from_numpy(masked_frames_ORI)
            return partial_masked_full_win_seq, masked_pos, masked_frames_ORI

    def __len__(self):
        return len(self.data)

    def proc_dataset(self):
        # res_dataset = []
        # for instance_i in range(len(self.data)):
        #     instance = self.__getitem__(instance_i)
        #     res_dataset.append(instance)
        return [self.__getitem__(instance_i) for instance_i in range(len(self.data))]
'''
def arg_helper(model_type, config_train):
    parser = argparse.ArgumentParser(description='PyTorch LIMU-BERT Model')
    parser.add_argument('-g', '--gpu', type=str, default=None, help='Set specific GPU')
    parser.add_argument('-t', '--train_cfg', type=str, default='./config/' + config_train, \
                        help='Training config json file path')
    parser.add_argument('-a', '--mask_cfg', type=str, default='./config/mask.json',
                        help='Mask strategy json file path')
    parser.add_argument('-s', '--save_model', type=str, default='model',
                        help='The saved model name')
    # From model_dfv4 >>>
    parser.add_argument('-m', '--machine', type=str, default='snow', help='snow | aw | AI') # edit # self.machine
    parser.add_argument('-tsid_idx', '--test_seq_id_idx', type=int, default='0', help='0-14') # edit
    parser.add_argument('-lw', '--len_win', type=int, default=10, help='Window length') # edit
    parser.add_argument('-rt', '--resume_training', type=bool, default=False)
    # From model_dfv4 <<<
    try:
        args = parser.parse_args()
    except:
        parser.print_help()
        sys.exit(0)
    model_cfg = load_model_config(model_type, args.model_version) # HERE
    if model_cfg is None:
        print("Unable to find corresponding model config!")
        sys.exit()
    args.model_cfg = model_cfg
    dataset_cfg = load_dataset_stats(args.dataset, args.dataset_version)
    if dataset_cfg is None:
        print("Unable to find corresponding dataset config!")
        sys.exit()
    args.dataset_cfg = dataset_cfg
    args = create_io_config(args, args.dataset, args.dataset_version, pretrain_model=args.model_file, model_type=model_type)
    return args

def load_pretrain_data_config(args):
    model_cfg = args.model_cfg
    train_cfg = TrainConfig.from_json(args.train_cfg)
    mask_cfg = MaskConfig.from_json(args.mask_cfg)
    dataset_cfg = args.dataset_cfg
    if model_cfg.feature_num > dataset_cfg.dimension:
        print('model_cfg.feature_num > dataset_cfg.dimension')
        sys.exit()
    set_seeds(train_cfg.seed)
    data = np.load(args.data_path).astype(np.float32)
    return data, train_cfg, model_cfg, mask_cfg, dataset_cfg
'''

def count_model_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

#  From LIMU-BERT
# ================

# ===========
#  ViFiT
def BBX5_fill_nan(BBX5_sync_dfv4, C: Config):
    if C.nan_fill == '0':
        BBX5_sync_dfv4 = np.nan_to_num(BBX5_sync_dfv4, nan=0)
    elif C.nan_fill == 'linear_interp':
        # Converting to -1 is for the convenience of checking nan values
        BBX5_sync_dfv4 = np.nan_to_num(BBX5_sync_dfv4, nan=-1)

        # print(); print()
        # print('BBX5_sync_dfv4: ', BBX5_sync_dfv4)
        for subj_i in range(np.shape(BBX5_sync_dfv4)[1]):
            last_bbox_i, first_bbox_after_nan_i, first_nan_i, last_nan_i = -2, -2, -2, -2 # -2: not assigned yet
            last_bbox_i = 1
            for f_i in range(np.shape(BBX5_sync_dfv4)[0]):
                bbox = BBX5_sync_dfv4[f_i, subj_i, 0, :]
                # print(); print()
                # print('bbox: ', bbox)
                # e.g. [1225.          786.            2.18386579  198.          354.        ]
                # print('np.shape(bbox): ', np.shape(bbox))

                if -1 in bbox:
                    # nan
                    if first_nan_i == -2:
                        first_nan_i = f_i
                    elif f_i == np.shape(BBX5_sync_dfv4)[0]:
                        # Last frame is a nan bbox
                        # Fill nan values with the last bbox
                        # print('\n Fill nan values with the last bbox')
                        for j in range(last_bbox_i + 1, f_i):
                            BBX5_sync_dfv4[j, subj_i, 0, :] = last_bbox
                    last_nan_i = f_i
                else:
                    # bbox
                    # if last_nan_i == -2:
                        # nan leading video stream
                        # Fill nan with the last bbox
                        # for j in range(0, f_i):
                        #     BBX5_sync_dfv4[j, subj_i, 0, :] = bbox
                        # print('\n Fill nan with the last bbox')
                    # elif last_nan_i == f_i - 1:
                    if last_nan_i == f_i - 1:
                        # first bbox of a new stream
                        # Fill nan
                        last_bbox = BBX5_sync_dfv4[last_bbox_i, subj_i, 0, :]
                        first_bbox_after_nan = BBX5_sync_dfv4[f_i, subj_i, 0, :]
                        nan_len = f_i - last_bbox_i
                        diff = first_bbox_after_nan - last_bbox
                        diff /= nan_len
                        # Fill nan
                        last_new_nan = copy.deepcopy(last_bbox)
                        for j in range(last_bbox_i + 1, f_i):
                            last_new_nan += diff
                            BBX5_sync_dfv4[j, subj_i, 0, :] = last_new_nan
                            # print('\n Fill nan values')
                    last_bbox_i = f_i

    return BBX5_sync_dfv4

def psp_tf(img, C: Config):
    # BEV = np.zeros((np.shape(img)[0] * C.BEV_scale, np.shape(img)[1] * C.BEV_scale, np.shape(img)[2]), dtype=np.uint8)
    # BEV[:np.shape(img)[0], :np.shape(img)[1]] = np.copy(img)

    # BEV >>>
    BEV = np.copy(img)
    BEV_src_pnts = np.array([[C.BEV_src_pnts[0][0], C.BEV_src_pnts[0][1]], [C.BEV_src_pnts[1][0], C.BEV_src_pnts[1][1]], \
                             [C.BEV_src_pnts[2][0], C.BEV_src_pnts[2][1]], [C.BEV_src_pnts[3][0], C.BEV_src_pnts[3][1]]], np.float32)
    BEV_dst_pnts = np.array([[C.BEV_dst_pnts[0][0], C.BEV_dst_pnts[0][1]], [C.BEV_dst_pnts[1][0], C.BEV_dst_pnts[1][1]], \
                             [C.BEV_dst_pnts[2][0], C.BEV_dst_pnts[2][1]], [C.BEV_dst_pnts[3][0], C.BEV_dst_pnts[3][1]]], np.float32)

	# compute the perspective transform matrix and then apply it
    BEV_M  = cv2.getPerspectiveTransform(BEV_src_pnts, BEV_dst_pnts)
    warped = cv2.warpPerspective(BEV, BEV_M, (np.shape(BEV)[1], np.shape(BEV)[0]))

    # Final BEV image
    BEV = copy.deepcopy(warped) # cropped_warped_img)
    # BEV <<<

    # Inverse Perspective Transformation >>>
    img_f_BEV = np.copy(BEV)
    BEV_M_iv = cv2.getPerspectiveTransform(BEV_dst_pnts, BEV_src_pnts) # Note the reverse of DST and SRC
    warped_iv = cv2.warpPerspective(img_f_BEV, BEV_M_iv, (np.shape(img_f_BEV)[1], np.shape(img_f_BEV)[0]), cv2.WARP_INVERSE_MAP)
    # Final BEV image
    img_f_BEV = copy.deepcopy(warped_iv)
    # Inverse Perspective Transformation <<<
    return BEV, BEV_M, img_f_BEV, BEV_M_iv

# def psp_tf_pnt(pnt, C: Config):
#     pnt_ = copy.deepcopy(pnt)
#     pnt_ = (pnt_[0], pnt_[1], 1)
#     pnt_BEV = tuple(C.BEV_M.dot(pnt_))
#     pnt_BEV /= pnt_BEV[2]
#     pnt_BEV = tuple([int(pnt_BEV[0]), int(pnt_BEV[1])]) # remove Cartesian coords
#     return pnt_BEV
def psp_tf_pnt(pnt, C: Config, flag):
    pnt_ = copy.deepcopy(pnt)
    pnt_ = (pnt_[0], pnt_[1], 1)
    if flag == 'Cam2BEV': pnt_DST = tuple(C.BEV_M.dot(pnt_))
    elif flag == 'BEV2Cam': pnt_DST = tuple(C.BEV_M_iv.dot(pnt_))
    pnt_DST /= pnt_DST[2]
    pnt_DST = tuple([int(pnt_DST[0]), int(pnt_DST[1])]) # remove Cartesian coords
    return pnt_DST

def vis_tracklet(img_type, img, seq_in_, subj_i, seq_in_2, C: Config):
    print('\n\n subj_i: ', subj_i)
    if subj_i < C.n_subjects: subj_color = C.color_dict[C.color_ls[subj_i]]
    else: subj_color = C.color_dict[C.color_ls[-1]]

    if img_type == 'Cam':
        # BBX5 shape: (1, 10, 1, 5)
        # IMU_NED_pos2 shape: (1, 10, 1, 2)
        seq_in_BBX5_ = torch.squeeze(torch.squeeze(seq_in_, dim=0), dim=1)
        print(); print() # debug
        print('np.shape(seq_in_BBX5_): ', np.shape(seq_in_BBX5_)) # e.g. (10, 5)
        print('seq_in_BBX5_[:, 0]: ', seq_in_BBX5_[:, 0]) # col
        print('seq_in_BBX5_[:, 1]: ', seq_in_BBX5_[:, 1]) # row
        print('seq_in_BBX5_[:, 2]: ', seq_in_BBX5_[:, 2]) # depth
        print('seq_in_BBX5_[:, 3]: ', seq_in_BBX5_[:, 3]) # width
        print('seq_in_BBX5_[:, 4]: ', seq_in_BBX5_[:, 4]) # height
        '''
        e.g.
        Sid , np.shape(seq_in_BBX5_):  (10, 5)
        seq_in_BBX5_[:, 0]:  [866. 833. 809.   0.   0.   0.   0. 676. 653. 638.]
        seq_in_BBX5_[:, 1]:  [427. 427. 432.   0.   0.   0.   0. 446. 451. 485.]
        seq_in_BBX5_[:, 2]:  [9.25371265 9.26818466 8.887537   0.         0.         0.
         0.         7.5010891  8.03569031 8.17784595]
        seq_in_BBX5_[:, 3]:  [40. 35. 32.  0.  0.  0.  0. 34. 64. 67.]
        seq_in_BBX5_[:, 4]:  [ 46.  46.  62.   0.   0.   0.   0.  63.  77. 144.]
        '''

        for k_i in range(C.len_win):
            if k_i < np.shape(seq_in_BBX5_)[0]:
                if C.vis_color_change == 'gradient':
                    brighter = int((C.len_win - k_i) * 255 / C.len_win)
                else: brighter = 0
                BBX_subj_color = (subj_color[0] + brighter, subj_color[1] + brighter, subj_color[2] + brighter)
                # top_left = (int(seq_in_BBX5_[k_i, 0]) - int(seq_in_BBX5_[k_i, 3] / 2), \
                #             int(seq_in_BBX5_[k_i, 1]) - int(seq_in_BBX5_[k_i, 4] / 2))
                # bottom_right = (int(seq_in_BBX5_[k_i, 0]) + int(seq_in_BBX5_[k_i, 3] / 2), \
                #             int(seq_in_BBX5_[k_i, 1]) + int(seq_in_BBX5_[k_i, 4] / 2))
                # img = cv2.circle(img, (int(seq_in_BBX5_[k_i, 0]), int(seq_in_BBX5_[k_i, 1])), 4, BBX_subj_color, 4) # Note (col, row) or (x, y) here
                # img = cv2.rectangle(img, top_left, bottom_right, BBX_subj_color, 2)
                img = cv2.circle(img, (int(seq_in_BBX5_[k_i, 0]), int(seq_in_BBX5_[k_i, 1]) + int(seq_in_BBX5_[k_i, 4] / 2)), 4, BBX_subj_color, 4) # Note (col, row) or (x, y) here

        subj_color_d = (subj_color[0] - 80, subj_color[1] - 80, subj_color[2] - 80) # Darker
        img = cv2.putText(img, '{:.2f}'.format(seq_in_BBX5_[-1, 2]), (int(seq_in_BBX5_[-1, 0]), int(seq_in_BBX5_[-1, 1])), \
                            cv2.FONT_HERSHEY_SIMPLEX, 1, subj_color_d, 2, cv2.LINE_AA)
        # img = cv2.putText(img, 'HS:red, NM: green, BC: blue', (10, 40), \
        #                     cv2.FONT_HERSHEY_SIMPLEX, 1, C.color_dict[C.color_ls[-2]], 2, cv2.LINE_AA)
        return img

    elif img_type == 'BEV':
        # BBX5 shape: (1, 10, 1, 5)
        # IMU_NED_pos2 shape: (1, 10, 1, 2)
        seq_in_BBX5_ = torch.squeeze(torch.squeeze(seq_in_, dim=0), dim=1)
        # print('\n\n np.shape(seq_in_2): ', np.shape(seq_in_2)) # (10, 2)
        # seq_in_BEV_Feet_Center_ = torch.squeeze(torch.squeeze(seq_in_2, dim=0), dim=1) # OBS
        seq_in_BEV_Feet_Center_ = seq_in_2
        BEV = img
        for k_i in range(C.len_win):
            if k_i < np.shape(seq_in_BEV_Feet_Center_)[0]:
                if C.vis_color_change == 'gradient':
                    brighter = int((C.len_win - k_i) * 255 / C.len_win)
                else: brighter = 0
                BEV_subj_color = (subj_color[0] + brighter, subj_color[1] + brighter, subj_color[2] + brighter)
                # feet_center = (int(seq_in_BBX5_[k_i, 0]), int(seq_in_BBX5_[k_i, 1] + int(seq_in_BBX5_[k_i, 4] / 2)))
                # BEV_feet_center = psp_tf_pnt(feet_center)
                BEV = cv2.circle(BEV, seq_in_BEV_Feet_Center_[k_i], 2, BEV_subj_color, 2) # Note (col, row) or (x, y) here
        subj_color_d = (subj_color[0] - 80, subj_color[1] - 80, subj_color[2] - 80) # Darker
        if seq_in_BBX5_[-1, 2] != 0:
            BEV = cv2.putText(BEV, '{:.2f}'.format(seq_in_BBX5_[-1, 2]), \
                            (int(seq_in_BEV_Feet_Center_[-1][0]), int(seq_in_BEV_Feet_Center_[-1][1])), \
                            cv2.FONT_HERSHEY_SIMPLEX, 1, subj_color_d, 2, cv2.LINE_AA)
        return BEV

    elif img_type == 'IMU_NED':
        seq_in_ = torch.squeeze(torch.squeeze(seq_in_, dim=0), dim=1).cpu().detach().numpy()
        seq_in_2 = torch.squeeze(torch.squeeze(seq_in_2, dim=0), dim=1).cpu().detach().numpy()
        IMU_NED = img
        seq_in_FTM_ = seq_in_2

        print(); print()
        print('np.shape(seq_in_): ', np.shape(seq_in_))
        print('np.shape(seq_in_2): ', np.shape(seq_in_2))
        '''
        np.shape(seq_in_):  torch.Size([10, 2])
        np.shape(seq_in_2):  torch.Size([10, 2])
        '''

        # >>> rescale >>>
        last_IMU_NED_pos = (int(seq_in_[C.len_win - 1, 0] * C.IMU_NED_pos_scales[subj_i] + C.IMU_NED_origins[subj_i][0]), \
                            int(seq_in_[C.len_win - 1, 1] * C.IMU_NED_pos_scales[subj_i] + C.IMU_NED_origins[subj_i][1]))

        # Scale img
        if last_IMU_NED_pos[0] <= 0 or last_IMU_NED_pos[0] >= np.shape(img)[1] or \
            last_IMU_NED_pos[1] <= 0 or last_IMU_NED_pos[1] >= np.shape(img)[0]:
            IMU_NED = np.zeros(np.shape(img), dtype=np.uint8)
            # print(); print() # debug
            # print(np.shape(img)) # (720, 1280, 3)
            # resize image
            resized_h, resized_w = int(np.shape(img)[1] * 0.8), int(np.shape(img)[0] * 0.8)
            resized = cv2.resize(img, (resized_h, resized_w), interpolation = cv2.INTER_AREA)
            # print(); print() # debug
            # print(np.shape(resized)) # (576, 1024, 3)
            padding_rows = (np.shape(IMU_NED)[0] - np.shape(resized)[0]) // 2
            padding_cols = (np.shape(IMU_NED)[1] - np.shape(resized)[1]) // 2
            IMU_NED[padding_rows : np.shape(IMU_NED)[0] - padding_rows, padding_cols : np.shape(IMU_NED)[1] - padding_cols] = resized
            C.IMU_NED_origins[subj_i] = (np.shape(IMU_NED)[1] // 2, np.shape(IMU_NED)[0] // 2)
            C.IMU_NED_pos_scales[subj_i] *= 0.8
        # <<< rescale <<<

        # IMU_NED = cv2.circle(IMU_NED, C.IMU_NED_origins[subj_i], 5, (255, 255, 255), 5) # Note (col, row) or (x, y) here
        IMU_NED = cv2.circle(IMU_NED, C.IMU_NED_origins[subj_i], 3, (255, 255, 255), 3) # Note (col, row) or (x, y) here
        for k_i in range(C.len_win):
            if k_i < np.shape(seq_in_)[0]:
                if C.vis_color_change == 'gradient':
                    brighter = int((C.len_win - k_i) * 255 / C.len_win)
                else: brighter = 0
                IMU_NED_subj_color = (subj_color[0] + brighter, \
                                        subj_color[1] + brighter, \
                                        subj_color[2] + brighter)

                if C.vis_win_NED:
                    IMU_NED_pos = (int((seq_in_[k_i, 0] - seq_in_[0, 0]) * C.IMU_NED_pos_scales[subj_i] + C.IMU_NED_origins[subj_i][0]), \
                                   int((seq_in_[k_i, 1] - seq_in_[0, 1]) * C.IMU_NED_pos_scales[subj_i] + C.IMU_NED_origins[subj_i][1]))
                else:
                    IMU_NED_pos = (int(seq_in_[k_i, 0] * C.IMU_NED_pos_scales[subj_i] + C.IMU_NED_origins[subj_i][0]), \
                                   int(seq_in_[k_i, 1] * C.IMU_NED_pos_scales[subj_i] + C.IMU_NED_origins[subj_i][1]))
                IMU_NED = cv2.circle(IMU_NED, IMU_NED_pos, int(C.IMU_NED_pos_scales[subj_i] / 8), \
                                    IMU_NED_subj_color, 2) # Note (col, row) or (x, y) here

        IMU_NED_ori = copy.deepcopy(IMU_NED)
        IMU_NED = cv2.putText(IMU_NED, 'FTM DistanceM: {:.2f}, StdDevM: {:.2f}'.format(seq_in_FTM_[-1][0], seq_in_FTM_[-1][1]), \
                            (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, subj_color, 1, cv2.LINE_AA)

        return IMU_NED, IMU_NED_ori, C
    elif img_type == 'PDR_BEV':
        # Usage: C.BEV = vis_tracklet('PDR_BEV', C.BEV, seq_in_IMU_NED_pos2_c, subj_i_r, seq_in_BEV_Feet_Center_, C)
        seq_in_ = torch.squeeze(torch.squeeze(seq_in_, dim=0), dim=1).cpu().detach().numpy()
        seq_in_BEV_Feet_Center_ = seq_in_2
        BEV = img

        print(); print()
        print('np.shape(seq_in_): ', np.shape(seq_in_))
        print('np.shape(seq_in_BEV_Feet_Center_): ', np.shape(seq_in_BEV_Feet_Center_))
        '''
        e.g.
        np.shape(seq_in_):  torch.Size([10, 2])
        np.shape(seq_in_BEV_Feet_Center_):  torch.Size([10, 2])
        '''
        print('seq_in_[0]: ', seq_in_[0])
        print('seq_in_BEV_Feet_Center_[0]: ', seq_in_BEV_Feet_Center_[0])
        '''
        e.g.
        seq_in_[0]:  [-0.42530174  0.42322385]
        seq_in_BEV_Feet_Center_[0]:  (1043, 362)
        '''
        print('seq_in_: ', seq_in_)

        PDR_pos_img = copy.deepcopy(seq_in_)
        PDR_pos_img[0] = seq_in_BEV_Feet_Center_[0]
        for t in range(1, np.shape(seq_in_)[0]):
            PDR_pos_img[t,:] = (seq_in_[t] - seq_in_[t - 1]) * C.BEV_px2meters_ratio + PDR_pos_img[t - 1,:]
        # print('PDR_pos_img[0]: ', PDR_pos_img[0])
        # print('PDR_pos_img: ', PDR_pos_img)

        for k_i in range(C.len_win):
            if k_i < np.shape(PDR_pos_img)[0]:
                if C.vis_color_change == 'gradient':
                    brighter = int((C.len_win - k_i) * 255 / C.len_win)
                else: brighter = 0
                brighter -= 200
                BEV_subj_color = (subj_color[0] + brighter, subj_color[1] + brighter, subj_color[2] + brighter)
                # BEV = cv2.circle(BEV, (int(PDR_pos_img[k_i][0]), int(PDR_pos_img[k_i][1])), 2, BEV_subj_color, 2) # Note (col, row) or (x, y) here
                BEV = cv2.circle(BEV, (int(PDR_pos_img[k_i][0]), int(PDR_pos_img[k_i][1])), 5, BEV_subj_color, 5) # Note (col, row) or (x, y) here
        subj_color_d = (subj_color[0] - 80, subj_color[1] - 80, subj_color[2] - 80) # Darker
        return BEV
    elif img_type == 'img_f_BEV':
        # BBX5 shape: (1, 10, 1, 5)
        # IMU_NED_pos2 shape: (1, 10, 1, 2)
        seq_in_BBX5_ = torch.squeeze(torch.squeeze(seq_in_, dim=0), dim=1)
        print(); print() # debug
        print('np.shape(seq_in_BBX5_): ', np.shape(seq_in_BBX5_)) # e.g. (10, 5)
        print('seq_in_BBX5_[:, 0]: ', seq_in_BBX5_[:, 0]) # col
        print('seq_in_BBX5_[:, 1]: ', seq_in_BBX5_[:, 1]) # row
        print('seq_in_BBX5_[:, 2]: ', seq_in_BBX5_[:, 2]) # depth
        print('seq_in_BBX5_[:, 3]: ', seq_in_BBX5_[:, 3]) # width
        print('seq_in_BBX5_[:, 4]: ', seq_in_BBX5_[:, 4]) # height
        '''
        e.g.
        Sid , np.shape(seq_in_BBX5_):  (10, 5)
        seq_in_BBX5_[:, 0]:  [866. 833. 809.   0.   0.   0.   0. 676. 653. 638.]
        seq_in_BBX5_[:, 1]:  [427. 427. 432.   0.   0.   0.   0. 446. 451. 485.]
        seq_in_BBX5_[:, 2]:  [9.25371265 9.26818466 8.887537   0.         0.         0.
         0.         7.5010891  8.03569031 8.17784595]
        seq_in_BBX5_[:, 3]:  [40. 35. 32.  0.  0.  0.  0. 34. 64. 67.]
        seq_in_BBX5_[:, 4]:  [ 46.  46.  62.   0.   0.   0.   0.  63.  77. 144.]
        '''
        img_f_BEV_pos2_ = seq_in_2 # torch.squeeze(torch.squeeze(seq_in_2, dim=0), dim=1)
        print(); print()
        print('np.shape(img_f_BEV_pos2_): ', np.shape(img_f_BEV_pos2_))
        # (10, 2)
        print('\n img_f_BEV_pos2_: ', img_f_BEV_pos2_)

        for k_i in range(C.len_win):
            if k_i < np.shape(img_f_BEV_pos2_)[0]:
                if C.vis_color_change == 'gradient':
                    brighter = int((C.len_win - k_i) * 255 / C.len_win)
                else: brighter = 0
                BBX_subj_color = (subj_color[0] + brighter, subj_color[1] + brighter, subj_color[2] + brighter)
                img = cv2.circle(img, img_f_BEV_pos2_[k_i], 4, BBX_subj_color, 4) # Note (col, row) or (x, y) here
                if k_i > 0: img = cv2.arrowedLine(img, img_f_BEV_pos2_[k_i - 1], img_f_BEV_pos2_[k_i], BBX_subj_color, 3, tipLength = 0.3)

        # subj_color_d = (subj_color[0] - 80, subj_color[1] - 80, subj_color[2] - 80) # Darker
        return img

def prepare_sync_training_data(C: Config):
    print('prepare_sync_training_data() C.macro_scene: ', C.macro_scene)

    seq_in_BBX5_dfv4_ls, seq_in_FTM2_dfv4_ls = [], []
    if C.imu200:
        # seq_in_IMU19_200_dfv4_ls = []
        seq_in_IMUaccgym9_200_dfv4_ls = []
    elif C.imu_ned_pos2:
        seq_in_IMU_NED_pos2_dfv4bl_ls_dfv3_ls = []
    seq_in_IMU19_dfv4_ls = []

    if C.macro_scene == 'indoor': C.img_type = 'RGBh_ts16_dfv4'
    elif C.macro_scene == 'outdoor': C.img_type = 'RGB_ts16_dfv4'
    C.img_type += '_anonymized'

    # DEBUG:
    print('\n\n C.seq_id_ls: ', C.seq_id_ls)
    print('\n\n len(C.seq_id_ls): ', len(C.seq_id_ls))
    print('\n\n C.test_seq_id: ', C.test_seq_id)
    C.train_seq_id_ls = copy.deepcopy(C.seq_id_ls)
    if C.test_seq_id in C.train_seq_id_ls:
        C.train_seq_id_ls.remove(C.test_seq_id)

    # ---------------------------------------
    #  Iterate Over All Train Seq_id - Start
    # ---------------------------------------
    # for C.seq_id_idx, C.seq_id in enumerate(C.seq_id_ls):
    for C.seq_id_idx, C.seq_id in enumerate(C.train_seq_id_ls):
        # DEBUG:
        print('\n\n C.seq_id: ', C.seq_id)
        print('\n\n C.seq_id != C.test_seq_id: ', C.seq_id != C.test_seq_id)
        if C.seq_id != C.test_seq_id:
            C.seq_path = C.seq_id_path_ls[C.seq_id_idx]
            C.img_path = C.seq_path + '/' + C.img_type
            C.img_type += '_anonymized'
            C.img_path = C.img_path.replace('RAN4model_dfv4.2', 'RAN')

            C.seq_date = C.seq_id[:8]
            C.seq_path_for_model = C.seq4model_root_path + '/' + C.seq_id
            # if C.macro_scene == 'indoor':
            #     C.RGBg_ts16_dfv4_ls_path = C.seq_path_for_model + '/RGBh_ts16_dfv4_ls.json'
            # elif C.macro_scene == 'outdoor':
            #     C.RGBg_ts16_dfv4_ls_path = C.seq_path_for_model + '/RGB_ts16_dfv4_ls.json'
            C.RGBg_ts16_dfv4_ls_path = C.seq_path_for_model + '/RGBg_ts16_dfv4_ls.json'
            with open(C.RGBg_ts16_dfv4_ls_path, 'r') as f:
                RGBg_ts16_dfv4_ls = json.load(f)
                print(C.RGBg_ts16_dfv4_ls_path, 'loaded!')
                print('RGBg_ts16_dfv4_ls[:5]: ', RGBg_ts16_dfv4_ls[:5]) # verification

            # start_ts16_i = list(map(lambda i: i > ots26_to_ts16_dfv4(C.start_ots), C.RGB_ts16_dfv4_ls)).index(True)
            # end_ts16_i = list(map(lambda i: i > ots26_to_ts16_dfv4(C.end_ots), C.RGB_ts16_dfv4_ls)).index(True)
            # print(); print() # debug
            # print('ots26_to_ts16_dfv4(C.start_ots): ', ots26_to_ts16_dfv4(C.start_ots))
            # print('C.RGB_ts16_dfv4_ls[start_ts16_i]: ', C.RGB_ts16_dfv4_ls[start_ts16_i])
            # print('ots26_to_ts16_dfv4(C.end_ots): ', ots26_to_ts16_dfv4(C.end_ots))
            # print('C.RGB_ts16_dfv4_ls[end_ts16_i]: ', C.RGB_ts16_dfv4_ls[end_ts16_i])
            # '''
            # e.g.
            # ots26_to_ts16_dfv4(C.start_ots):  1633372099.829258
            # C.RGB_ts16_dfv4_ls[start_ts16_i]:  1633372099.929300
            # ots26_to_ts16_dfv4(C.end_ots):  1633372277.437527
            # C.RGB_ts16_dfv4_ls[end_ts16_i]:  1633372277.537532
            # '''
            # C.RGB_ts16_dfv4_valid_ls = C.RGB_ts16_dfv4_ls[start_ts16_i : end_ts16_i + 1]

            # print(); print() # debug
            # print('C.seq_id: ', C.seq_id, ', C.seq_id: ', C.seq_id)
            # print('C.seq_path: ', C.seq_path)
            # print('C.seq_path_for_model: ', C.seq_path_for_model)
            # print('len(C.RGB_ts16_dfv4_valid_ls): ', len(C.RGB_ts16_dfv4_valid_ls)) # e.g. 1700
            if C.vis:
                C.img_path = C.seq_path + '/' + C.img_type
                C.img_type += '_anonymized'
                C.img_path = C.img_path.replace('RAN4model_dfv4.2', 'RAN')

            # ------------------------------------------
            #  Synchronized data: BBX5,IMU19_sync_dfv4
            # ------------------------------------------
            C.sync_dfv4_path = C.seq_path_for_model + '/sync_ts16_dfv4'

            # -----------------
            #  Load BBX5 Data
            # -----------------
            C.BBX5_dim = 5
            if C.macro_scene == 'indoor':
                C.BBX5_sync_dfv4_path = C.sync_dfv4_path + '/BBX5H_sync_dfv4.pkl'
            elif C.macro_scene == 'outdoor':
                C.BBX5_sync_dfv4_path = C.sync_dfv4_path + '/BBX5_sync_dfv4.pkl'
            BBX5_sync_dfv4 = pickle.load(open(C.BBX5_sync_dfv4_path, 'rb'))
            print(); print() # debug
            print('np.shape(BBX5_sync_dfv4): ', np.shape(BBX5_sync_dfv4))
            # e.g. (559, 5, 1, 5), (1775, 2, 1, 5)

            # BBX5_sync_dfv4 = np.nan_to_num(BBX5_sync_dfv4, nan=0)
            BBX5_sync_dfv4 = BBX5_fill_nan(BBX5_sync_dfv4, C)

            # ---------------
            #  Load IMU Data
            # ---------------
            C.IMU19_data_types = ['ACCEL', 'GYRO', 'MAG', 'GRAV', 'LINEAR', 'Quaternion'] # ['ACCEL', 'GRAV', 'LINEAR', 'Quaternion', 'MAG', 'GYRO']
            C.IMU19_data_type_dict = {'ACCEL' : 3, 'GYRO' : 3, 'MAG' : 3, \
                                            'GRAV' : 3, 'LINEAR' : 3, 'Quaternion' : 4}
            C.IMU19_dim = np.sum(list(C.IMU19_data_type_dict.values())) # 19

            if C.imu200:
                C.IMU19_200_sync_dfv4_path = C.sync_dfv4_path + '/IMU19_200_sync_dfv4.pkl'
                IMU19_200_sync_dfv4 = pickle.load(open(C.IMU19_200_sync_dfv4_path, 'rb'))
                print(); print() # debug
                print('np.shape(IMU19_200_sync_dfv4): ', np.shape(IMU19_200_sync_dfv4))
                # e.g. (559, 5, 200, 19)

                # savgol_filter >>>
                if C.sm_filter == 'savgol_filter':
                    for subj_i in range(C.n_subjects):
                        for feat_i in range(19):
                            # Filter with a window length of 11 and a degree 2 polynomial
                            shape_ = np.shape(IMU19_200_sync_dfv4[:, subj_i, :, feat_i])
                            IMU19_200_sync_dfv4[:, subj_i, :, feat_i] = \
                                np.reshape(savgol_filter(\
                                    np.squeeze(IMU19_200_sync_dfv4[:, subj_i, :, feat_i]), \
                                    11, 2), shape_)
                # savgol_filter <<<

                C.IMUaccgym9_200_sync_dfv4_path = C.sync_dfv4_path + '/IMUaccgym9_200_sync_dfv4.pkl' # To be updated
                IMUaccgym9_200_sync_dfv4 = pickle.load(open(C.IMUaccgym9_200_sync_dfv4_path, 'rb'))
                print(); print() # debug
                print('np.shape(IMUaccgym9_200_sync_dfv4): ', np.shape(IMUaccgym9_200_sync_dfv4))
                # e.g. (559, 5, 200, 9)

                # savgol_filter >>>
                if C.sm_filter == 'savgol_filter':
                    for subj_i in range(C.n_subjects):
                        for feat_i in range(9):
                            # Filter with a window length of 11 and a degree 2 polynomial
                            shape_ = np.shape(IMUaccgym9_200_sync_dfv4[:, subj_i, :, feat_i])
                            IMUaccgym9_200_sync_dfv4[:, subj_i, :, feat_i] = \
                                np.reshape(savgol_filter(\
                                    np.squeeze(IMUaccgym9_200_sync_dfv4[:, subj_i, :, feat_i]), \
                                    11, 2), shape_)
                # savgol_filter <<<

            else:
                C.IMU19_sync_dfv4_path = C.sync_dfv4_path + '/IMU19_sync_dfv4.pkl'
                IMU19_sync_dfv4 = pickle.load(open(C.IMU19_sync_dfv4_path, 'rb'))
                print(); print() # debug
                print('np.shape(IMU19_sync_dfv4): ', np.shape(IMU19_sync_dfv4))
                # e.g. (559, 5, 1, 19)

                # savgol_filter >>>
                if C.sm_filter == 'savgol_filter':
                    for subj_i in range(C.n_subjects):
                        for feat_i in range(19):
                            # Filter with a window length of 11 and a degree 2 polynomial
                            shape_ = np.shape(IMU19_sync_dfv4[:, subj_i, :, feat_i])
                            IMU19_sync_dfv4[:, subj_i, :, feat_i] = \
                                np.reshape(savgol_filter(\
                                    np.squeeze(IMU19_sync_dfv4[:, subj_i, :, feat_i]), \
                                    11, 2), shape_)
                # savgol_filter <<<

            # ----------------
            #  Load FTM2 Data
            # ----------------
            C.FTM2_dim = 2
            C.FTM2_sync_dfv4_path = C.sync_dfv4_path + '/FTM_sync_dfv4.pkl'
            FTM2_sync_dfv4 = pickle.load(open(C.FTM2_sync_dfv4_path, 'rb'))
            print(); print() # debug
            print('np.shape(FTM2_sync_dfv4): ', np.shape(FTM2_sync_dfv4))
            # e.g. (535, 5, 2)

            # --------------
            #  Video Window
            # --------------
            C.crr_ts16_dfv4_ls_all_i = 0
            C.video_len = min(1700, len(RGBg_ts16_dfv4_ls)) # len(C.ts12_BBX5_all)
            print(); print() # debug
            print('C.video_len: ', C.video_len) # e.g. 1700
            C.n_wins = min(1700, C.video_len - C.len_win + 1)
            print('C.n_wins: ', C.n_wins) # e.g. 1791

            # --------------
            #  Prepare BBX5
            # --------------
            curr_in_view_i_ls = []
            for win_i in range(C.n_wins):
                # print('C.seq_id_to_start_end_ts16_dfv4: ', C.seq_id_to_start_end_ts16_dfv4)
                ts16_dfv4 = RGBg_ts16_dfv4_ls[win_i]
                if ts16_dfv4 < C.seq_id_to_start_end_ts16_dfv4[C.seq_id]['start']: continue
                elif ts16_dfv4 >= C.seq_id_to_start_end_ts16_dfv4[C.seq_id]['end']: break
                for subj_i in range(C.n_subjects):
                    # seq_in_BBX5_ = BBX5_sync_dfv4[subj_i, win_i : win_i + C.len_win, :] # old
                    # seq_in_BBX5_ = BBX5_sync_dfv4[win_i : win_i + C.len_win, subj_i, :] # prev
                    # seq_in_BBX5_ = BBX5_sync_dfv4[win_i : win_i + C.len_win, subj_i, 0, :]
                    seq_in_BBX5_ = BBX5_sync_dfv4[win_i : win_i + C.len_win, subj_i, :, :]
                    # print(); print() # debug
                    # print('np.shape(seq_in_BBX5_): ', np.shape(seq_in_BBX5_)) # e.g. (10, 1, 5)
                    # print('seq_in_BBX5_[:, :, 0]: ', seq_in_BBX5_[:, :, 0]) # col
                    # print('seq_in_BBX5_[:, :, 1]: ', seq_in_BBX5_[:, :, 1]) # row
                    # print('seq_in_BBX5_[:, :, 2]: ', seq_in_BBX5_[:, :, 2]) # depth
                    # print('seq_in_BBX5_[:, :, 3]: ', seq_in_BBX5_[:, :, 3]) # width
                    # print('seq_in_BBX5_[:, :, 4]: ', seq_in_BBX5_[:, :, 4]) # height
                    '''
                    e.g.
                    seq_in_BBX5_[:, :, 0]:  [641. 631. 618. 604. 592. 583. 577. 570. 565. 562.]
                    seq_in_BBX5_[:, :, 1]:  [635. 630. 627. 623. 619. 615. 611. 607. 604. 602.]
                    seq_in_BBX5_[:, :, 2]:  [1.73513258 1.75361669 1.78351653 1.84246898 1.86370301 1.86906254
                     1.90441883 1.93990803 1.98963535 2.04343772]
                    seq_in_BBX5_[:, :, 3]:  [157. 152. 147. 146. 147. 148. 149. 145. 142. 140.]
                    seq_in_BBX5_[:, :, 4]:  [163. 173. 180. 188. 195. 203. 211. 218. 225. 228.]
                    '''
                    # -----------------------------------------------------------------------------
                    #  Note that RGB_ts16_dfv4_valid_ls only works for New Dataset in this version
                    # -----------------------------------------------------------------------------
                    if seq_in_BBX5_.ndim == 3 and np.shape(seq_in_BBX5_)[0] == C.len_win:
                        curr_in_view_i_ls.append(win_i * C.n_subjects + subj_i)
                        # DEBUG:
                        # print('\n\n')
                        # print('win_i: ', win_i, ', subj_i: ', subj_i)
                        # print('np.shape(seq_in_BBX5_): ', np.shape(seq_in_BBX5_))
                        # print('seq_in_BBX5_: ', seq_in_BBX5_)
                        # print(); print() # debug
                        # print('np.shape(seq_in_BBX5_): ', np.shape(seq_in_BBX5_)) # e.g. (10, 1, 5)
                        # print('seq_in_BBX5_[:, :, 0]: ', seq_in_BBX5_[:, :, 0]) # col
                        # print('seq_in_BBX5_[:, :, 1]: ', seq_in_BBX5_[:, :, 1]) # row
                        # print('seq_in_BBX5_[:, :, 2]: ', seq_in_BBX5_[:, :, 2]) # depth
                        # print('seq_in_BBX5_[:, :, 3]: ', seq_in_BBX5_[:, :, 3]) # width
                        # print('seq_in_BBX5_[:, :, 4]: ', seq_in_BBX5_[:, :, 4]) # height

                        # Update max_depth
                        C.max_depth = max(C.max_depth, max(seq_in_BBX5_[:, :, 2]))

                        seq_in_BBX5_dfv4_ls.append(seq_in_BBX5_)

            # -----------------
            #  BBX5 Conversion
            # -----------------
            # DEBUG:
            print('\n np.shape(np.array(seq_in_BBX5_dfv4_ls)): ', np.shape(np.array(seq_in_BBX5_dfv4_ls)))
            # e.g. (2540, 30, 1, 5)
            seq_in_BBX5_dfv4_ls_new = []
            for BBX5_ in seq_in_BBX5_dfv4_ls:
                if C.delta_BBX5_version > 0:
                    BBX5_ = BBX5_to_dBBX5(C, BBX5_, C.delta_BBX5_version)
                seq_in_BBX5_dfv4_ls_new.append(BBX5_)
            # <<<

            # ----------------------
            #  Prepare IMU19 & FTM2
            # ----------------------
            for win_i in range(C.n_wins):
                ts16_dfv4 = RGBg_ts16_dfv4_ls[win_i]
                if ts16_dfv4 < C.seq_id_to_start_end_ts16_dfv4[C.seq_id]['start']: continue
                elif ts16_dfv4 >= C.seq_id_to_start_end_ts16_dfv4[C.seq_id]['end']: break
                for subj_i in range(C.n_subjects):
                    curr_in_view_i = win_i * C.n_subjects + subj_i
                    if curr_in_view_i in curr_in_view_i_ls:
                        # seq_in_BBX5_ = BBX5_sync_dfv4[subj_i, win_i : win_i + C.len_win, :] # old
                        # seq_in_BBX5_ = BBX5_sync_dfv4[win_i : win_i + C.len_win, subj_i, :] # prev
                        # seq_in_BBX5_ = BBX5_sync_dfv4[win_i : win_i + C.len_win, subj_i, 0, :]
                        seq_in_BBX5_ = BBX5_sync_dfv4[win_i : win_i + C.len_win, subj_i, :, :]
                        # print('np.shape(seq_in_BBX5_): ', np.shape(seq_in_BBX5_))
                        k_start_i = np.shape(seq_in_BBX5_)[0] # C.len_win
                        for k_i in range(np.shape(seq_in_BBX5_)[0]): # range(C.len_win):
                            # print(type(seq_in_BBX5_[k_i])) # numpy.ndarray
                            if 0 not in seq_in_BBX5_[k_i]:
                                k_start_i = k_i
                                break

                        if C.imu200:
                            seq_in_IMUaccgym9_200_ = IMUaccgym9_200_sync_dfv4[win_i : win_i + C.len_win, subj_i, :]
                            for k_i in range(C.len_win):
                                if k_i < k_start_i:
                                    IMUaccgym9_200_not_in_view = np.full((1, 200, C.IMU_XT_dim), 0)
                                    # print('IMUIMUaccgym9_200_not_in_view: ', IMUIMUaccgym9_200_not_in_view)
                                    seq_in_IMUaccgym9_200_[k_i] = IMUaccgym9_200_not_in_view
                            seq_in_IMUaccgym9_200_dfv4_ls.append(seq_in_IMUaccgym9_200_)
                        else:
                            # seq_in_IMU19_ = IMU19_sync_dfv4[subj_i, win_i : win_i + C.len_win, :] # old
                            seq_in_IMU19_ = IMU19_sync_dfv4[win_i : win_i + C.len_win, subj_i, :]
                            # seq_in_IMU19_ = IMU19_sync_dfv4[win_i : win_i + C.len_win, subj_i, 0, :]
                            for k_i in range(C.len_win):
                                if k_i < k_start_i:
                                    IMU19_not_in_view = np.full((1, C.IMU19_dim), 0)
                                    # print('IMU19_not_in_view: ', IMU19_not_in_view)
                                    seq_in_IMU19_[k_i] = IMU19_not_in_view

                            # print('np.shape(seq_in_IMU19_): ', np.shape(seq_in_IMU19_)) # e.g. (10, 1, 19)
                            seq_in_IMU19_dfv4_ls.append(seq_in_IMU19_)

                        # >>> FTM2 >>>
                        seq_in_FTM2_ = FTM2_sync_dfv4[win_i : win_i + C.len_win, subj_i, :]
                        for k_i in range(C.len_win):
                            if k_i < k_start_i:
                                FTM2_not_in_view = np.full((1, C.FTM2_dim), 0)
                                seq_in_FTM2_[k_i] = FTM2_not_in_view
                        seq_in_FTM2_dfv4_ls.append(seq_in_FTM2_)
                        # <<< FTM2 <<<

            print(); print() # debug
            if C.imu200: print('len(seq_in_IMUaccgym9_200_dfv4_ls): ', len(seq_in_IMUaccgym9_200_dfv4_ls))
            else: print('len(seq_in_IMU19_dfv4_ls): ', len(seq_in_IMU19_dfv4_ls))
            print('len(seq_in_FTM2_dfv4_ls): ', len(seq_in_FTM2_dfv4_ls))

    # -------------------------------------
    #  Iterate Over All Train Seq_id - End
    # -------------------------------------
    seq_in_BBX5 = np.array(seq_in_BBX5_dfv4_ls_new) # CPU memory, use instance.copy() in utils.py
    seq_out_BBX5 = copy.deepcopy(seq_in_BBX5) # CPU memory, use instance.copy() in utils.py
    print('np.shape(seq_in_BBX5): ', np.shape(seq_in_BBX5)) # e.g. (27376, 60, 1, 5) # (n_win, len_win, samples, feature_dim)
    print('np.shape(seq_out_BBX5): ', np.shape(seq_out_BBX5))

    seq_in_FTM2 = np.array(seq_in_FTM2_dfv4_ls) # CPU memory, use instance.copy() in utils.py
    seq_out_FTM2 = copy.deepcopy(seq_in_FTM2) # CPU memory, use instance.copy() in utils.py
    print('np.shape(seq_in_FTM2): ', np.shape(seq_in_FTM2)) # e.g. (27376, 60, 1, 2) # (n_win, len_win, samples, feature_dim)
    print('np.shape(seq_out_FTM2): ', np.shape(seq_out_FTM2))

    if C.imu200:
        seq_in_IMUaccgym9_200 = np.array(seq_in_IMUaccgym9_200_dfv4_ls) # CPU memory, use instance.copy() in utils.py
        seq_out_IMUaccgym9_200 = copy.deepcopy(seq_in_IMUaccgym9_200) # CPU memory, use instance.copy() in utils.py
        print('np.shape(seq_in_IMUaccgym9_200): ', np.shape(seq_in_IMUaccgym9_200)) # e.g. (27376, 60, 200, 9) # (n_win, len_win, samples, feature_dim)
        print('np.shape(seq_out_IMUaccgym9_200): ', np.shape(seq_out_IMUaccgym9_200))
    else:
        seq_in_IMU19 = np.array(seq_in_IMU19_dfv4_ls) # CPU memory, use instance.copy() in utils.py
        seq_out_IMU19 = copy.deepcopy(seq_in_IMU19) # CPU memory, use instance.copy() in utils.py
        print('np.shape(seq_in_IMU19): ', np.shape(seq_in_IMU19)) # e.g. (27376, 60, 1, 19) # (n_win, len_win, samples, feature_dim)
        print('np.shape(seq_out_IMU19): ', np.shape(seq_out_IMU19))

    assert np.shape(seq_in_BBX5)[1] == np.shape(seq_in_FTM2)[1]
    if C.imu200: assert np.shape(seq_in_BBX5)[1] == np.shape(seq_in_IMUaccgym9_200)[1]
    else: assert np.shape(seq_in_BBX5)[1] == np.shape(seq_in_IMU19)[1]
    assert np.shape(seq_in_BBX5)[3] == C.BBX5_dim
    assert np.shape(seq_in_FTM2)[3] == C.FTM2_dim
    if not C.imu200: assert np.shape(seq_in_IMU19)[3] == C.IMU19_dim

    # Mask features >>>
    # Only for IMU19 in this version
    if 'a' in C.mask_features:
        seq_in_IMU19[:,:,:,:3] = np.zeros(np.shape(seq_in_IMU19[:,:,:,:3]))
        seq_out_IMU19[:,:,:,:3] = np.zeros(np.shape(seq_out_IMU19[:,:,:,:3]))
        print('Mark a')
    if 'g' in C.mask_features:
        seq_in_IMU19[:,:,:,3:6] = np.zeros(np.shape(seq_in_IMU19[:,:,:,3:6]))
        seq_out_IMU19[:,:,:,3:6] = np.zeros(np.shape(seq_out_IMU19[:,:,:,3:6]))
        print('Mark g')
    if 'm' in C.mask_features:
        seq_in_IMU19[:,:,:,9:12] = np.zeros(np.shape(seq_in_IMU19[:,:,:,9:12]))
        seq_out_IMU19[:,:,:,9:12] = np.zeros(np.shape(seq_out_IMU19[:,:,:,9:12]))
        print('Mark m')
    if 'f' in C.mask_features:
        seq_in_FTM2 = np.zeros(np.shape(seq_in_FTM2))
        seq_out_FTM2 = np.zeros(np.shape(seq_out_FTM2))
        print('Mark f')
    # Mask features <<<

    # DEBUG:
    print('\n\n')
    print('\n\n seq_id_path_ls: ', C.seq_id_path_ls)
    print('\n\n seq_id_ls: ', C.seq_id_ls)
    print('\n\n C.seq_id_ls after removing test_seq_id: ', C.seq_id_ls)
    print('\n\n C.test_seq_id: ', C.test_seq_id)
    print('\n\n len(C.seq_id_ls): ', len(C.seq_id_ls))
    print('\n\n len(C.train_seq_id_ls): ', len(C.train_seq_id_ls))
    print('\n\n C.BBX5_sync_dfv4_path: ', C.BBX5_sync_dfv4_path)
    if C.imu200:
        return seq_in_BBX5, seq_out_BBX5, \
                seq_in_FTM2, seq_out_FTM2, \
                seq_in_IMUaccgym9_200, seq_out_IMUaccgym9_200
    else:
        return seq_in_BBX5, seq_out_BBX5, \
                seq_in_FTM2, seq_out_FTM2, \
                seq_in_IMU19, seq_out_IMU19

def prepare_sync_testing_data(C: Config):
    set_seeds(C.seed)
    XT_filter_indices = []

    # seq_in_BBX5_dfv4_ls, seq_in_BBX5_Others_dfv4_ls = [], []
    seq_in_BBX5_dfv4_ls = []
    seq_in_IMU19_dfv4_ls = []
    if C.imu200:
        # seq_in_IMU19_200_dfv4_ls = []
        seq_in_IMUaccgym9_200_dfv4_ls = []
    elif C.imu_ned_pos2:
        seq_in_IMU_NED_pos2_dfv4bl_ls = []
        seq_in_BEV_Feet_Center_dfv3_ls = []
        seq_in_BEV_Feet_Center_Others_dfv3_ls = []
    seq_in_FTM2_dfv4_ls, seq_in_FTM2_Others_dfv4_ls = [], []
    seq_in_FTM2_dfv4_ls = []
    if C.macro_scene == 'indoor': C.img_type = 'RGBh_ts16_dfv4'
    elif C.macro_scene == 'outdoor': C.img_type = 'RGB_ts16_dfv4'
    C.img_type += '_anonymized'

    # prev
    # C.seq_in_BBX5_r_shape = (1, C.len_win, C.BBX5_dim)
    # C.seq_in_FTM2_c_shape = (1, C.len_win, C.FTM2_dim)
    # C.seq_in_IMU19_c_shape = (1, C.len_win, C.IMU19_dim)
    # C.seq_in_BBX5_Others_r_shape = (1, C.len_win, C.BBX5_dim)

    C.seq_in_BBX5_r_shape = (1, C.len_win, 1, C.BBX5_dim)
    C.seq_in_FTM2_c_shape = (1, C.len_win, 1, C.FTM2_dim)
    C.seq_in_IMU19_c_shape = (1, C.len_win, 1, C.IMU19_dim)
    if C.imu_ned_pos2:
        C.seq_in_IMU_NED_pos2_c_shape = (1, C.len_win, 1, C.IMU_NED_pos2_dim)
    C.seq_in_BBX5_Others_r_shape = (1, C.len_win, 1, C.BBX5_dim)

    print(); print() # debug
    print('C.seq_id_ls: ', C.seq_id_ls)
    print('len(C.seq_id_ls): ', len(C.seq_id_ls))

    # --------------------------------------
    #  Iterate Over One Test Seq_id
    # --------------------------------------
    # for C.seq_id in C.test_seq_id_ls:
    # C.seq_id_idx = C.args.test_seq_id_idx # OBS
    C.seq_id = C.seq_id_ls[C.test_seq_id_idx] # edit
    print('\n\n utils.py - prepare_sync_testing_data() - C.seq_id_idx: ', C.seq_id_idx)
    print('\n\n utils.py - prepare_sync_testing_data() - C.test_seq_id_idx: ', C.test_seq_id_idx)
    print('\n\n utils.py - prepare_sync_testing_data() - C.seq_id: ', C.test_seq_id_idx)

    C.seq_path = C.seq_id_path_ls[C.test_seq_id_idx]
    print('\n\n utils.py - prepare_sync_testing_data() - C.seq_path: ', C.seq_path)
    # print('\n\n utils.py - prepare_sync_testing_data() - C.seq_id_path_ls: ', C.seq_id_path_ls)
    print('\n\n utils.py - prepare_sync_testing_data() - C.test_seq_id_idx: ', C.test_seq_id_idx)

    C.img_path = C.seq_path + '/' + C.img_type
    C.img_type += '_anonymized'
    C.img_path = C.img_path.replace('RAN4model_dfv4.2', 'RAN')
    C.seq_date = C.seq_id[:8]
    C.seq_path_for_model = C.seq4model_root_path + '/' + C.seq_id
    # C.img_path = C.seq_path + '/' + C.img_type
    # if C.scene_id == 0: C.RGBg_ts16_dfv4_ls_path = C.seq_path_for_model + '/RGBh_ts16_dfv4_ls.json'
    # else: C.RGBg_ts16_dfv4_ls_path = C.seq_path_for_model + '/RGB_ts16_dfv4_ls.json'
    C.RGBg_ts16_dfv4_ls_path = C.seq_path_for_model + '/RGBg_ts16_dfv4_ls.json'
    print('C.RGBg_ts16_dfv4_ls_path: ', C.RGBg_ts16_dfv4_ls_path)
    with open(C.RGBg_ts16_dfv4_ls_path, 'r') as f:
        RGBg_ts16_dfv4_ls = json.load(f)
        print(C.RGBg_ts16_dfv4_ls_path, 'loaded!')
        print('C.RGBg_ts16_dfv4_ls[:5]: ', C.RGBg_ts16_dfv4_ls[:5]) # verification

    print('\n\n utils.py - prepare_testing_data()')
    print('\n\n C.seq_id: ', C.seq_id, ', C.seq_id: ', C.seq_id)

    print('\n\n C.seq_path: ', C.seq_path)
    print('\n\n len(C.seq_id_path_ls): ', len(C.seq_id_path_ls))
    print('\n\n C.seq_path_for_model: ', C.seq_path_for_model)

    C.seq_path_for_modelbl = C.seq4modelbl_root_path + '/' + C.macro_scene + '/scene' + str(C.scene_id) + '/' + C.seq_id # bl: baseline
    C.sync_dfv4bl_path = C.seq_path_for_modelbl + '/sync_ts16_dfv4bl'
    print('\n\n utilp.py - prepare_sync_testing_data - C.sync_dfv4bl_path: ', C.sync_dfv4bl_path)

    # start_ts16_i = list(map(lambda i: i > ots26_to_ts16_dfv4(C.start_ots), C.RGB_ts16_dfv4_ls)).index(True)
    # end_ts16_i = list(map(lambda i: i > ots26_to_ts16_dfv4(C.end_ots), C.RGB_ts16_dfv4_ls)).index(True)
    # print(); print() # debug
    # print('ots26_to_ts16_dfv4(C.start_ots): ', ots26_to_ts16_dfv4(C.start_ots))
    # print('C.RGB_ts16_dfv4_ls[start_ts16_i]: ', C.RGB_ts16_dfv4_ls[start_ts16_i])
    # print('ots26_to_ts16_dfv4(C.end_ots): ', ots26_to_ts16_dfv4(C.end_ots))
    # print('C.RGB_ts16_dfv4_ls[end_ts16_i]: ', C.RGB_ts16_dfv4_ls[end_ts16_i])
    # '''
    # e.g.
    # ots26_to_ts16_dfv4(C.start_ots):  1633372099.829258
    # C.RGB_ts16_dfv4_ls[start_ts16_i]:  1633372099.929300
    # ots26_to_ts16_dfv4(C.end_ots):  1633372277.437527
    # C.RGB_ts16_dfv4_ls[end_ts16_i]:  1633372277.537532
    # '''
    # C.RGB_ts16_dfv4_valid_ls = C.RGB_ts16_dfv4_ls[start_ts16_i : end_ts16_i + 1]

    # if C.vis:
    #     C.img_path = C.seq_path + '/' + C.img_type
    #     C.img_type += '_anonymized'
    #     C.img_path = C.img_path.replace('RAN4model_dfv4.2', 'RAN')
    print('utils.py - prepare_testing_data()')
    print('C.seq_id: ', C.seq_id, ', C.seq_id: ', C.seq_id)
    print('C.seq_path: ', C.seq_path)
    print('C.seq_path_for_model: ', C.seq_path_for_model)
    # print('len(C.RGBg_ts16_dfv4_valid_ls): ', len(C.RGBg_ts16_dfv4_valid_ls)) # e.g. 1700

    # ------------------------------------------
    #  Synchronized data: BBX5,IMU19_sync_dfv4
    # ------------------------------------------
    C.sync_dfv4_path = C.seq_path_for_model + '/sync_ts16_dfv4'
    # ----------------
    #  Load BBX Data
    # ----------------
    C.BBX5_dim = 5
    if C.macro_scene == 'indoor':
        C.BBX5_sync_dfv4_path = C.sync_dfv4_path + '/BBX5H_sync_dfv4.pkl'
    elif C.macro_scene == 'outdoor':
        C.BBX5_sync_dfv4_path = C.sync_dfv4_path + '/BBX5_sync_dfv4.pkl'
    BBX5_sync_dfv4 = pickle.load(open(C.BBX5_sync_dfv4_path, 'rb'))
    # BBX5_sync_dfv4 = np.nan_to_num(BBX5_sync_dfv4, nan=0)
    BBX5_sync_dfv4 = BBX5_fill_nan(BBX5_sync_dfv4, C)
    print(); print() # debug
    print('C.macro_scene: ', C.macro_scene)
    print('np.shape(BBX5_sync_dfv4): ', np.shape(BBX5_sync_dfv4))
    # e.g. (573, 5, 1, 5)

    # -----------------
    #  Load IMU19 Data
    # -----------------
    C.IMU19_data_types = ['ACCEL', 'GYRO', 'MAG', 'GRAV', 'LINEAR', 'Quaternion'] # ['ACCEL', 'GRAV', 'LINEAR', 'Quaternion', 'MAG', 'GYRO']
    C.IMU19_data_type_dict = {'ACCEL' : 3, 'GYRO' : 3, 'MAG' : 3, \
                                    'GRAV' : 3, 'LINEAR' : 3, 'Quaternion' : 4}
    C.IMU19_dim = np.sum(list(C.IMU19_data_type_dict.values())) # 19

    C.IMU19_sync_dfv4_path = C.sync_dfv4_path + '/IMU19_sync_dfv4.pkl'
    IMU19_sync_dfv4 = pickle.load(open(C.IMU19_sync_dfv4_path, 'rb'))
    print(); print() # debug
    print('np.shape(IMU19_sync_dfv4): ', np.shape(IMU19_sync_dfv4))
    # e.g. (573, 5, 1, 19)

    # savgol_filter >>>
    if C.sm_filter == 'savgol_filter':
        for subj_i in range(C.n_subjects):
            for feat_i in range(19):
                # Filter with a window length of 11 and a degree 2 polynomial
                shape_ = np.shape(IMU19_sync_dfv4[:, subj_i, :, feat_i])
                IMU19_sync_dfv4[:, subj_i, :, feat_i] = \
                    np.reshape(savgol_filter(\
                        np.squeeze(IMU19_sync_dfv4[:, subj_i, :, feat_i]), \
                        11, 2), shape_)
    # savgol_filter <<<

    if C.imu200:
        C.IMU19_200_sync_dfv4_path = C.sync_dfv4_path + '/IMU19_200_sync_dfv4.pkl'
        IMU19_200_sync_dfv4 = pickle.load(open(C.IMU19_200_sync_dfv4_path, 'rb'))
        print(); print() # debug
        print('np.shape(IMU19_200_sync_dfv4): ', np.shape(IMU19_200_sync_dfv4))
        # e.g. (573, 5, 200, 19)

        # savgol_filter >>>
        if C.sm_filter == 'savgol_filter':
            for subj_i in range(C.n_subjects):
                for feat_i in range(19):
                    # Filter with a window length of 11 and a degree 2 polynomial
                    shape_ = np.shape(IMU19_200_sync_dfv4[:, subj_i, :, feat_i])
                    IMU19_200_sync_dfv4[:, subj_i, :, feat_i] = \
                        np.reshape(savgol_filter(\
                            np.squeeze(IMU19_200_sync_dfv4[:, subj_i, :, feat_i]), \
                            11, 2), shape_)
        # savgol_filter <<<

        C.IMUaccgym9_200_sync_dfv4_path = C.sync_dfv4_path + '/IMUaccgym9_200_sync_dfv4.pkl' # To be updated
        IMUaccgym9_200_sync_dfv4 = pickle.load(open(C.IMUaccgym9_200_sync_dfv4_path, 'rb'))
        print(); print() # debug
        print('np.shape(IMUaccgym9_200_sync_dfv4): ', np.shape(IMUaccgym9_200_sync_dfv4))
        # e.g. (573, 5, 200, 9)

        # savgol_filter >>>
        if C.sm_filter == 'savgol_filter':
            for subj_i in range(C.n_subjects):
                for feat_i in range(9):
                    # Filter with a window length of 11 and a degree 2 polynomial
                    shape_ = np.shape(IMUaccgym9_200_sync_dfv4[:, subj_i, :, feat_i])
                    IMUaccgym9_200_sync_dfv4[:, subj_i, :, feat_i] = \
                        np.reshape(savgol_filter(\
                            np.squeeze(IMUaccgym9_200_sync_dfv4[:, subj_i, :, feat_i]), \
                            11, 2), shape_)
        # savgol_filter <<<
    elif C.imu_ned_pos2:
        C.IMU_NED_pos2_dim = 2
        C.IMU_NED_pos2_sync_dfv4bl_path = C.sync_dfv4bl_path + '/IMU_NED_pos2_sync_dfv4bl.pkl'
        IMU_NED_pos2_sync_dfv4bl = pickle.load(open(C.IMU_NED_pos2_sync_dfv4bl_path, 'rb'))
        print(); print() # debug
        print('np.shape(IMU_NED_pos2_sync_dfv4bl): ', np.shape(IMU_NED_pos2_sync_dfv4bl))
        # e.g. np.shape(IMU_NED_pos2_sync_dfv4bl):  (1816, 2, 1, 2)


    # ----------------
    #  Load FTM2 Data
    # ----------------
    C.FTM2_dim = 2
    C.FTM2_sync_dfv4_path = C.sync_dfv4_path + '/FTM_sync_dfv4.pkl'
    FTM2_sync_dfv4 = pickle.load(open(C.FTM2_sync_dfv4_path, 'rb'))
    print(); print() # debug
    print('np.shape(FTM2_sync_dfv4): ', np.shape(FTM2_sync_dfv4))
    # e.g. (573, 5, 1, 2)

    # -------------------------------
    #  Load BBX Data for Passers-by
    # -------------------------------
    # C.Others_id_ls_path = C.sync_dfv4_path + '/Others_id_ls.pkl'
    # C.Others_id_ls = pickle.load(open(C.Others_id_ls_path, 'rb'))
    # C.BBX5_Others_sync_dfv4_path = C.sync_dfv4_path + '/BBX5_Others_sync_dfv4.pkl'
    # C.BBX5_Others_sync_dfv4 = pickle.load(open(C.BBX5_Others_sync_dfv4_path, 'rb'))
    # C.BBX5_Others_sync_dfv4 = np.nan_to_num(C.BBX5_Others_sync_dfv4, nan=0)
    # print(); print() # debug
    # print('len(C.Others_id_ls): ',len(C.Others_id_ls))
    # print('np.shape(C.BBX5_Others_sync_dfv4): ', np.shape(C.BBX5_Others_sync_dfv4))
    #
    # C.Others_id_ls_path = C.sync_dfv4_path + '/Others_id_ls.pkl'
    # C.Others_id_ls = pickle.load(open(C.Others_id_ls_path, 'rb'))
    # C.BBX5_Others_sync_dfv4_path = C.sync_dfv4_path + '/BBX5_Others_sync_dfv4.pkl'
    # C.BBX5_Others_sync_dfv4 = pickle.load(open(C.BBX5_Others_sync_dfv4_path, 'rb'))
    # C.BBX5_Others_sync_dfv4 = np.nan_to_num(C.BBX5_Others_sync_dfv4, nan=0)
    # print(); print() # debug
    # print('len(C.Others_id_ls): ',len(C.Others_id_ls))
    # print('np.shape(C.BBX5_Others_sync_dfv4): ', np.shape(C.BBX5_Others_sync_dfv4))

    # --------------
    #  Video Window
    # --------------
    C.crr_ts16_dfv4_ls_all_i = 0
    C.video_len = min(1700, len(RGBg_ts16_dfv4_ls)) # len(C.ts12_BBX5_all)
    print(); print() # debug
    print('C.video_len: ', C.video_len) # e.g. 1700
    C.n_wins = min(1700, C.video_len - C.len_win + 1)
    print('C.n_wins: ', C.n_wins) # e.g. 1791

    # -------------------
    #  Prepare Tracklets
    # -------------------
    # -------------
    #  Prepare BBX
    # -------------
    for win_i in range(C.n_wins):
        ts16_dfv4 = RGBg_ts16_dfv4_ls[win_i]
        if ts16_dfv4 < C.seq_id_to_start_end_ts16_dfv4[C.seq_id]['start']: continue
        elif ts16_dfv4 >= C.seq_id_to_start_end_ts16_dfv4[C.seq_id]['end']: break

        # if C.vis:
        #     print('\n win_i: ', win_i, ', ts16_dfv4: ', ts16_dfv4)
        #     img_path = C.img_path + '/' + ts16_dfv4 + '_anonymized.jpg'
        #     print('\n img_path: ', img_path)
        #     img = cv2.imread(img_path)

        seq_subj_i_in_view_ls_ = []

        for subj_i in range(C.n_subjects):
            seq_in_BBX5_ = BBX5_sync_dfv4[win_i : win_i + C.len_win, subj_i, :, :]
            if seq_in_BBX5_[:, :, 3].any() != 0 and seq_in_BBX5_[:, :, 4].any() != 0:
                # BBX5 in the view
                # print('\n\n np.shape(seq_in_BBX5_): ', np.shape(seq_in_BBX5_)) # e.g. (10, 1, 5)
                seq_in_BBX5_dfv4_ls.append(seq_in_BBX5_)
                seq_subj_i_in_view_ls_.append(subj_i)
                C.test_seq_in_BBX5_dict[(win_i, subj_i)] = np.expand_dims(seq_in_BBX5_, axis=0)

            else:
                # BBX5 not in the view
                if C.imu200: seq_in_BBX5_ = np.zeros((C.len_win, 200, 5))
                else: seq_in_BBX5_ = np.zeros((C.len_win, 1, 5))
                seq_in_BBX5_dfv4_ls.append(seq_in_BBX5_)
                seq_subj_i_in_view_ls_.append(subj_i)
                C.test_seq_in_BBX5_dict[(win_i, subj_i)] = np.expand_dims(seq_in_BBX5_, axis=0)

            #  >>> Vis >>>
            # if C.vis: vis_tracklet(img, seq_in_BBX5_, C.subjects[subj_i]) # C.subjects to be updated
            #  <<< Vis <<<

        #  >>> Vis >>>
        # if C.vis:
        #     img = cv2.putText(img, 'HS:red, MH: green, BC: blue', (30, 30), \
        #                         cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 200, 200), 2, cv2.LINE_AA)
        #     cv2.imshow('img', img); cv2.waitKey(0)
        #  <<< Vis <<<
        # C.seq_subj_i_in_view_dict[C.RGB_ts16_dfv4_valid_ls[win_i]] = seq_subj_i_in_view_ls_
        C.test_seq_subj_i_in_view_dict[RGBg_ts16_dfv4_ls[win_i]] = seq_subj_i_in_view_ls_

    # -----------------
    #  BBX5 Conversion
    # -----------------
    # DEBUG:
    print('\n np.shape(np.array(seq_in_BBX5_dfv4_ls)): ', np.shape(np.array(seq_in_BBX5_dfv4_ls)))
    # e.g. (2540, 30, 1, 5)
    seq_in_BBX5_dfv4_ls_new = []
    for BBX5_ in seq_in_BBX5_dfv4_ls:
        if C.delta_BBX5_version > 0:
            BBX5_ = BBX5_to_dBBX5(C, BBX5_, C.delta_BBX5_version)
        seq_in_BBX5_dfv4_ls_new.append(BBX5_)
    # <<<

    # ----------------------
    #  Prepare IMU19 & FTM2
    # ----------------------
    for win_i in range(C.n_wins):
        ts16_dfv4 = RGBg_ts16_dfv4_ls[win_i]
        if ts16_dfv4 < C.seq_id_to_start_end_ts16_dfv4[C.seq_id]['start']: continue
        elif ts16_dfv4 >= C.seq_id_to_start_end_ts16_dfv4[C.seq_id]['end']: break

        for subj_i in range(C.n_subjects):
            print('np.shape(IMU19_sync_dfv4): ', np.shape(IMU19_sync_dfv4))
            if np.shape(IMU19_sync_dfv4)[0] == 0:
                print('np.shape(IMU19_sync_dfv4): ', np.shape(IMU19_sync_dfv4))
            seq_in_IMU19_ = IMU19_sync_dfv4[win_i : win_i + C.len_win, subj_i, :, :]
            if C.imu200:
                seq_in_IMUaccgym9_200_ = IMUaccgym9_200_sync_dfv4[win_i : win_i + C.len_win, subj_i, :]

                if len(seq_in_IMU19_) == C.len_win:
                    # >>> Add noise >>>
                    if C.noise_level > 0.0:
                        noise19 = []
                        for k in range(C.len_win):
                            noise19_in_k = [np.random.normal(0, C.IMU19_range_dct['range_arr'][d]) * C.noise_level for d in range(C.IMU19_dim)]
                            noise19.append(noise19_in_k)
                        noise19 = np.array(noise19)
                        # print(); print() # debug
                        # print('np.shape(noise19): ', np.shape(noise19)) # (10, 19)
                        seq_in_IMU19_ += noise19
                    # <<< Add noise <<<

                seq_in_IMUaccgym9_200_dfv4_ls.append(seq_in_IMUaccgym9_200_)
                C.test_seq_in_IMUaccgym9_200_dict[(win_i, subj_i)] = np.expand_dims(seq_in_IMUaccgym9_200_, axis=0)
            elif C.imu_ned_pos2:
                seq_in_IMU_NED_pos2_ = IMU_NED_pos2_sync_dfv4bl[win_i : win_i + C.len_win, subj_i, :]
                seq_in_IMU_NED_pos2_dfv4bl_ls.append(seq_in_IMU_NED_pos2_)
                C.test_seq_in_IMU_NED_pos2_dict[(win_i, subj_i)] = np.expand_dims(seq_in_IMU_NED_pos2_, axis=0)

            # IMU19 >>>
            # seq_in_IMU19_ = IMU19_sync_dfv4[subj_i, win_i : win_i + C.len_win, :] # old
            seq_in_IMU19_ = IMU19_sync_dfv4[win_i : win_i + C.len_win, subj_i, :]
            # seq_in_IMU19_ = IMU19_sync_dfv4[win_i : win_i + C.len_win, subj_i, 0, :]
            print('win_i: ', win_i, ', subj_i: ', subj_i)
            print('np.shape(seq_in_IMU19_): ', np.shape(seq_in_IMU19_)) # e.g. (10, 1, 19)
            # print('seq_in_IMU19_: ', seq_in_IMU19_)

            if len(seq_in_IMU19_) == C.len_win:
                # >>> Add noise >>>
                if C.noise_level > 0.0:
                    noise19 = []
                    for k in range(C.len_win):
                        noise19_in_k = [np.random.normal(0, C.IMU19_range_dct['range_arr'][d]) * C.noise_level for d in range(C.IMU19_dim)]
                        noise19.append(noise19_in_k)
                    noise19 = np.array(noise19)
                    # print(); print() # debug
                    # print('np.shape(noise19): ', np.shape(noise19)) # (10, 19)
                    seq_in_IMU19_ += noise19
                # <<< Add noise <<<

            seq_in_IMU19_dfv4_ls.append(seq_in_IMU19_)
            C.test_seq_in_IMU19_dict[(win_i, subj_i)] = np.expand_dims(seq_in_IMU19_, axis=0)
            # IMU19 <<<

            # >>> FTM2 >>>
            seq_in_FTM2_ = FTM2_sync_dfv4[win_i : win_i + C.len_win, subj_i, :, :]
            if len(seq_in_FTM2_) == C.len_win:
                seq_in_FTM2_dfv4_ls.append(seq_in_FTM2_)
                C.test_seq_in_FTM2_dict[(win_i, subj_i)] = np.expand_dims(seq_in_FTM2_, axis=0)
            # <<< FTM2 <<<

    # ---------------------
    #  Prepare BBX5 Others
    # ---------------------
    # To be deleted
    # for win_i in range(C.n_wins):
    #     ts16_dfv4 = RGBg_ts16_dfv4_ls[win_i]
    #     if ts16_dfv4 < C.seq_id_to_start_end_ts16_dfv4[C.seq_id]['start']: continue
    #     elif ts16_dfv4 >= C.seq_id_to_start_end_ts16_dfv4[C.seq_id]['end']: break
    #
    #     seq_subj_i_in_view_ls_ = []
    #     #  >>> Vis >>>
    #     if C.vis_Others:
    #         subj_i_RGB_ts16_dfv4_img_path = C.img_path + '/' + ts16_dfv4_to_ots26(C.RGB_ts16_dfv4_valid_ls[win_i + C.len_win - 1]) + '.png'
    #         print(); print() # debug
    #         print('subj_i_RGB_ts16_dfv4_img_path: ', subj_i_RGB_ts16_dfv4_img_path)
    #         img = cv2.imread(subj_i_RGB_ts16_dfv4_img_path)
    #     #  <<< Vis <<<
    #     for subj_i_, subj_ in enumerate(C.Others_id_ls):
    #         # print(); print() # debug
    #         # print('subj_i_: ', subj_i_, ', subj_', subj_)
    #         subj_i = subj_i_ + C.n_subjects - 1
    #         seq_in_BBX5_Others_ = C.BBX5_Others_sync_dfv4[win_i : win_i + C.len_win, subj_i_, :]
    #         # print('np.shape(seq_in_BBX5_Others_): ', np.shape(seq_in_BBX5_Others_))
    #
    #         # BBX5 in the view
    #         if seq_in_BBX5_Others_[:, 3].any() != 0 and seq_in_BBX5_Others_[:, 4].any() != 0 \
    #             and len(seq_in_BBX5_Others_) == C.len_win:
    #             # print(); print() # debug
    #             # print('seq_in_BBX5_Others_: ', seq_in_BBX5_Others_)
    #             seq_in_BBX5_Others_dfv4_ls.append(seq_in_BBX5_Others_)
    #             seq_subj_i_in_view_ls_.append(subj_i)
    #             C.seq_in_BBX5_Others_dict[(win_i, subj_i)] = np.expand_dims(seq_in_BBX5_Others_, axis=0)
    #
    #         #  >>> Vis >>>
    #         if C.vis_Others: vis_tracklet(img, seq_in_BBX5_Others_, subj_)
    #         #  <<< Vis <<<
    #     #  >>> Vis >>>
    #     if C.vis_Others:
    #         img = cv2.putText(img, 'HS:red, MH: green, BC: blue', (30, 30), \
    #                             cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 200, 200), 2, cv2.LINE_AA)
    #         cv2.imshow('img', img); cv2.waitKey(0)
    #     #  <<< Vis <<<
    #     if C.RGB_ts16_dfv4_valid_ls[win_i] not in C.seq_subj_i_in_view_dict:
    #         C.seq_subj_i_in_view_dict[C.RGB_ts16_dfv4_valid_ls[win_i]] = seq_subj_i_in_view_ls_
    #     else:
    #         C.seq_subj_i_in_view_dict[C.RGB_ts16_dfv4_valid_ls[win_i]].extend(seq_subj_i_in_view_ls_)
    #     # print(); print() # debug
    #     # print('C.seq_subj_i_in_view_dict[C.RGB_ts16_dfv4_valid_ls[win_i]]: ', C.seq_subj_i_in_view_dict[C.RGB_ts16_dfv4_valid_ls[win_i]])
    #     # e.g. [0, 1, 2, 10, 11, 13, 14]

    # C.seq_in_BBX5 = np.array(seq_in_BBX5_dfv4_ls)
    # print(); print() # debug
    # print('np.shape(C.seq_in_BBX5): ', np.shape(C.seq_in_BBX5)) # e.g. (2335, 10, 1, 5)
    # C.seq_out_BBX5 = copy.deepcopy(C.seq_in_BBX5)
    #
    # C.seq_in_IMU19 = np.array(seq_in_IMU19_dfv4_ls)
    # print(); print() # debug
    # print('np.shape(C.seq_in_IMU19): ', np.shape(C.seq_in_IMU19)) # e.g. (2820, 10, 1, 19)
    # C.seq_out_IMU19 = copy.deepcopy(C.seq_in_IMU19)
    #
    # # C.seq_in_BBX5_Others = np.array(seq_in_BBX5_Others_dfv4_ls)
    # # print(); print() # debug
    # # print('np.shape(C.seq_in_BBX5_Others): ', np.shape(C.seq_in_BBX5_Others)) # e.g.
    # # C.seq_out_BBX5_Others = copy.deepcopy(C.seq_in_BBX5_Others)
    #
    # C.seq_in_FTM2 = np.array(seq_in_FTM2_dfv4_ls)
    # C.seq_out_FTM2 = copy.deepcopy(C.seq_in_FTM2)
    # print('np.shape(C.seq_in_FTM2): ', np.shape(C.seq_in_FTM2)) # e.g. (2820, 10, 1, 2)
    # print('np.shape(C.seq_out_FTM2): ', np.shape(C.seq_out_FTM2)) # e.g. (2820, 10, 1, 2)
    #
    # assert np.shape(C.seq_in_BBX5)[1] == np.shape(C.seq_in_IMU19)[1] # == np.shape(C.seq_out_BBX5_Others)[1]
    # assert np.shape(C.seq_in_BBX5)[3] == C.BBX5_dim # np.shape(C.seq_out_BBX5_Others)[2] == C.BBX5_dim
    # assert np.shape(C.seq_in_IMU19)[3] == C.IMU19_dim
    #
    # print(); print() # debug
    # print('seq_id_path_ls: ', C.seq_id_path_ls)
    # print('seq_id_ls: ', C.seq_id_ls)

    seq_in_BBX5 = np.array(seq_in_BBX5_dfv4_ls) # CPU memory, use instance.copy() in utils.py
    seq_out_BBX5 = copy.deepcopy(seq_in_BBX5) # CPU memory, use instance.copy() in utils.py
    print('np.shape(seq_in_BBX5): ', np.shape(seq_in_BBX5)) # e.g. (27376, 60, 1, 5) # (n_win, len_win, samples, feature_dim)
    print('np.shape(seq_out_BBX5): ', np.shape(seq_out_BBX5))

    seq_in_FTM2 = np.array(seq_in_FTM2_dfv4_ls) # CPU memory, use instance.copy() in utils.py
    seq_out_FTM2 = copy.deepcopy(seq_in_FTM2) # CPU memory, use instance.copy() in utils.py
    print('np.shape(seq_in_FTM2): ', np.shape(seq_in_FTM2)) # e.g. (27376, 60, 1, 3) # (n_win, len_win, samples, feature_dim)
    print('np.shape(seq_out_FTM2): ', np.shape(seq_out_FTM2))

    if C.imu200:
        seq_in_IMUaccgym9_200 = np.array(seq_in_IMUaccgym9_200_dfv4_ls) # CPU memory, use instance.copy() in utils.py
        seq_out_IMUaccgym9_200 = copy.deepcopy(seq_in_IMUaccgym9_200) # CPU memory, use instance.copy() in utils.py
        print('np.shape(seq_in_IMUaccgym9_200): ', np.shape(seq_in_IMUaccgym9_200)) # e.g. (27376, 60, 200, 9) # (n_win, len_win, samples, feature_dim)
        print('np.shape(seq_out_IMUaccgym9_200): ', np.shape(seq_out_IMUaccgym9_200))
    elif C.imu_ned_pos2:
        seq_in_IMU_NED_pos2 = np.array(seq_in_IMU_NED_pos2_dfv4bl_ls) # CPU memory, use instance.copy() in utils.py
        seq_out_IMU_NED_pos2 = copy.deepcopy(seq_in_IMU_NED_pos2) # CPU memory, use instance.copy() in utils.py
        print('np.shape(seq_in_IMU_NED_pos2): ', np.shape(seq_in_IMU_NED_pos2)) # e.g. (27376, 60, 200, 9) # (n_win, len_win, samples, feature_dim)
        print('np.shape(seq_out_IMU_NED_pos2): ', np.shape(seq_out_IMU_NED_pos2))
    seq_in_IMU19 = np.array(seq_in_IMU19_dfv4_ls) # CPU memory, use instance.copy() in utils.py
    seq_out_IMU19 = copy.deepcopy(seq_in_IMU19) # CPU memory, use instance.copy() in utils.py
    print('np.shape(seq_in_IMU19): ', np.shape(seq_in_IMU19)) # e.g. (27376, 60, 1, 19) # (n_win, len_win, samples, feature_dim)
    print('np.shape(seq_out_IMU19): ', np.shape(seq_out_IMU19))

    assert np.shape(seq_in_BBX5)[1] == np.shape(seq_in_FTM2)[1]
    if C.imu200: assert np.shape(seq_in_BBX5)[1] == np.shape(seq_in_IMUaccgym9_200)[1]
    elif C.imu_ned_pos2: assert np.shape(seq_in_BBX5)[1] == np.shape(seq_in_IMU_NED_pos2)[1]
    else: assert np.shape(seq_in_BBX5)[1] == np.shape(seq_in_IMU19)[1]
    assert np.shape(seq_in_BBX5)[3] == C.BBX5_dim
    assert np.shape(seq_in_FTM2)[3] == C.FTM2_dim
    if not C.imu200: assert np.shape(seq_in_IMU19)[3] == C.IMU19_dim

    # -------------------------------------------------------------------
    #  Verify if nan_cnt are consistent between BBX5 and IMU19 after the
    #    previous steps.
    # -------------------------------------------------------------------
    BBX5_nan_cnt = 0
    for win_i in range(np.shape(seq_in_BBX5)[0]):
        # print(seq_in_BBX5[win_i])
        if 0 in seq_in_BBX5[win_i]:
            BBX5_nan_cnt += 1
    print('BBX5_nan_cnt: ', BBX5_nan_cnt)

    IMU19_nan_cnt = 0
    for win_i in range(np.shape(seq_in_BBX5)[0]):
        # print(seq_in_BBX5[win_i])
        if 0 in seq_in_BBX5[win_i]:
            IMU19_nan_cnt += 1
    print('IMU19_nan_cnt: ', IMU19_nan_cnt)

    # BBX5_Others_nan_cnt = 0
    # for win_i in range(np.shape(C.seq_in_BBX5_Others)[0]):
    #     # print(C.seq_in_BBX5[win_i])
    #     if 0 in C.seq_in_BBX5_Others[win_i]:
    #         BBX5_Others_nan_cnt += 1
    # print('BBX5_Others_nan_cnt: ', BBX5_Others_nan_cnt)

    assert BBX5_nan_cnt == IMU19_nan_cnt

    print(); print() # debug
    print('len(C.seq_subj_i_in_view_dict.keys()): ', len(C.seq_subj_i_in_view_dict.keys())) # e.g. 55
    print('C.RGBg_ts16_dfv4_ls[:5]: ', C.RGBg_ts16_dfv4_ls[:5])
    print('C.running_mode: ', C.running_mode)

    # ----------
    #  Test Log
    # ----------
    C = update_test_log_config(C)

    if C.imu200:
        return C, seq_in_BBX5, seq_out_BBX5, \
                  seq_in_FTM2, seq_out_FTM2, \
                  seq_in_IMUaccgym9_200, seq_out_IMUaccgym9_200
    elif C.imu_ned_pos2:
        return C, seq_in_BBX5, seq_out_BBX5, \
                  seq_in_FTM2, seq_out_FTM2, \
                  seq_in_IMU_NED_pos2, seq_out_IMU_NED_pos2
    else:
        return C, seq_in_BBX5, seq_out_BBX5, \
                  seq_in_FTM2, seq_out_FTM2, \
                  seq_in_IMU19, seq_out_IMU19

def prepare_XT_data(C: Config, seq_in_correspondence, data_type):
    set_seeds(C.seed)
    if C.imu200:
        seq_in_BBX5, seq_in_FTM2, seq_in_IMUaccgym9_200 = seq_in_correspondence[0], seq_in_correspondence[1], seq_in_correspondence[2]
        seq_in_TX = copy.deepcopy(seq_in_IMUaccgym9_200)
        seq_in_correspondence = [seq_in_BBX5, seq_in_FTM2, seq_in_IMUaccgym9_200, seq_in_TX]
    else:
        seq_in_BBX5, seq_in_FTM2, seq_in_IMU19 = seq_in_correspondence[0], seq_in_correspondence[1], seq_in_correspondence[2]
        # IMU >>>
        XT_filter_indices = []
        i = 0
        for IMU_data_type in C.IMU19_data_types:
            dim = C.IMU19_data_type_dict[IMU_data_type]
            # print('dim: ', dim) # debug
            if IMU_data_type in C.IMU_XT_data_type_dict:
                XT_filter_indices.extend([i + d for d in range(dim)])
            i += dim
        print('XT_filter_indices: ', XT_filter_indices)
        # XT_filter_indices:  [0, 1, 2, 13, 14, 15, 16, 17, 18]
        # C.IMU_XT_data_types = ['ACCEL', 'GYRO',  'MAG']
        # C.IMU19_data_types = ['ACCEL', 'GYRO', 'MAG', 'GRAV', 'LINEAR', 'Quaternion'] # ['ACCEL', 'GRAV', 'LINEAR', 'Quaternion', 'MAG', 'GYRO']

        seq_in_TX = seq_in_IMU_TX = np.take(seq_in_IMU19, XT_filter_indices, axis=3)
        print('np.shape(seq_in_IMU_TX): ', np.shape(seq_in_IMU_TX)) # torch.Size([28848, 60, 9])
        # IMU <<<
        seq_in_correspondence = [seq_in_BBX5, seq_in_FTM2, seq_in_IMU19, seq_in_TX]

    # DEBUG:
    # print('utils.py - prepare_XT_data() - ', data_type, ' - len(seq_in_correspondence): ', len(seq_in_correspondence))
    # print('utils.py - prepare_XT_data() - ', data_type, ' - np.shape(seq_in_BBX5): ', np.shape(seq_in_BBX5))
    # print('utils.py - prepare_XT_data() - ', data_type, ' - np.shape(seq_in_FTM2): ', np.shape(seq_in_FTM2))
    # if C.imu200:
    #     print('utils.py - prepare_XT_data() - np.shape(seq_in_IMUaccgym9_200): ', np.shape(seq_in_IMUaccgym9_200))
    # else:
    #     print('utils.py - prepare_XT_data() - np.shape(seq_in_IMU19): ', np.shape(seq_in_IMU19))
    # print('utils.py - prepare_XT_data() - np.shape(seq_in_TX): ', np.shape(seq_in_TX))
    return seq_in_correspondence
    # return [seq_in_BBX5, seq_in_FTM2, seq_in_IMU19, seq_in_TX]

def train_val_split(seq_in_correspondence, train_val_rate=0.9):
    train_seq_in_correspondence, val_seq_in_correspondence = [], []
    # [seq_in_BBX5, seq_in_FTM2, seq_in_IMU19]
    n_seqs = np.shape(seq_in_correspondence[0])[0]
    # DEBUG:
    print('\n\n utils.py - train_val_split() - n_seqs: ', n_seqs) # e.g. 31088
    n_train = int(n_seqs * train_val_rate)
    n_val = n_seqs - n_train
    val_i = int(random.random() * train_val_rate * n_seqs)
    print('\n\n utils.py - train_val_split() - val_i: ', val_i) # e.g. 31088
    while val_i + n_val > n_seqs:
        val_i = int(random.random() * train_val_rate * n_seqs)

    for seq_in_ in seq_in_correspondence:
        val_seq_in_ = np.array(seq_in_[val_i : val_i + n_val, :, :])
        # DEBUG:
        print('\n\n utils.py - train_val_split() - np.shape(val_seq_in_): ', np.shape(val_seq_in_))
        # print('\n\n utils.py - train_val_split() - np.shape(seq_in_[ : val_i, :, :]): ', np.shape(seq_in_[ : val_i, :, :]))
        # print('\n\n utils.py - train_val_split() - np.shape(seq_in_[val_i + n_val :, :, :]): ', np.shape(seq_in_[val_i + n_val :, :, :]))
        train_seq_in_ = np.concatenate((seq_in_[ : val_i, :, :], \
                                        seq_in_[val_i + n_val :, :, :]), axis=0)
        print('\n\n utils.py - train_val_split() - np.shape(train_seq_in_): ', np.shape(train_seq_in_))
        assert np.shape(train_seq_in_)[0] + np.shape(val_seq_in_)[0] == n_seqs
        train_seq_in_correspondence.append(train_seq_in_)
        val_seq_in_correspondence.append(val_seq_in_)
    return train_seq_in_correspondence, val_seq_in_correspondence

def preprocess_config(C: Config, running_mode: str, update_running_mode_in_Config: bool, \
        load_pretrained_embeds: bool): #, flag: bool):
    if update_running_mode_in_Config: C.running_mode = running_mode

    train_data_loader, val_data_loader, test_data_loader = None, None, None
    train_seq_in_correspondence, val_seq_in_correspondence, test_seq_id_idx = None, None, None

    print('\n\n utils.py - preprocess_config() - running_mode: ', running_mode)
    if load_pretrained_embeds:
        # Update C.len_win from pretrain config
        pretrain_C_path = C.checkpoint_path_dict['pretrain'] + '/all_cfg.json'
        with open(pretrain_C_path, 'r') as f:
            pretrain_C = json.load(f)
            print(pretrain_C_path, 'loaded!')
            C.len_win = pretrain_C['len_win']
            print('Updated C[\'len_win\'] to', C.len_win)

    if C.imu200:
        seq_in_BBX5, seq_out_BBX5, \
        seq_in_FTM2, seq_out_FTM2, \
        seq_in_IMUaccgym9_200, seq_out_IMUaccgym9_200 = prepare_sync_training_data(C)
        seq_in_BBX5 = torch.from_numpy(seq_in_BBX5); seq_out_BBX5 = torch.from_numpy(seq_out_BBX5)
        seq_in_FTM2 = torch.from_numpy(seq_in_FTM2); seq_out_FTM2 = torch.from_numpy(seq_out_FTM2)
        seq_in_IMUaccgym9_200 = torch.from_numpy(seq_in_IMUaccgym9_200)
        seq_out_IMUaccgym9_200 = torch.from_numpy(seq_out_IMUaccgym9_200)
    elif C.imu_ned_pos2:
        seq_in_BBX5, seq_out_BBX5, \
        seq_in_FTM2, seq_out_FTM2, \
        seq_in_IMU_NED_pos2, seq_out_IMU_NED_pos2 = prepare_sync_training_data(C)
    else:
        seq_in_BBX5, seq_out_BBX5, \
        seq_in_FTM2, seq_out_FTM2, \
        seq_in_IMU19, seq_out_IMU19 = prepare_sync_training_data(C)
        seq_in_BBX5 = torch.from_numpy(seq_in_BBX5); seq_out_BBX5 = torch.from_numpy(seq_out_BBX5)
        seq_in_FTM2 = torch.from_numpy(seq_in_FTM2); seq_out_FTM2 = torch.from_numpy(seq_out_FTM2)
        seq_in_IMU19 = torch.from_numpy(seq_in_IMU19); seq_out_IMU19 = torch.from_numpy(seq_out_IMU19)
    # DEBUG:
    print('\n\n embeddings.py - __main__() - np.shape(seq_in_BBX5): ', np.shape(seq_in_BBX5))
    print('\n\n embeddings.py - __main__() - np.shape(seq_in_FTM2): ', np.shape(seq_in_FTM2))
    # print('\n\n embeddings.py - __main__() - np.shape(seq_in_IMU19): ', np.shape(seq_in_IMU19))
    # print(HERE)

    # generate the correspondence of [seq_in_BBX5, seq_in_FTM2, seq_in_IMU19]
    print('\n\n running_mode: ', running_mode)
    # if running_mode in ['pretrain', 'train', 'embed']: # OBS
    if C.imu200: seq_in_correspondence = [seq_in_BBX5, seq_in_FTM2, seq_in_IMUaccgym9_200]
    elif C.imu_ned_pos2: seq_in_correspondence = [seq_in_BBX5, seq_in_FTM2, seq_in_IMU_NED_pos2]
    else: seq_in_correspondence = [seq_in_BBX5, seq_in_FTM2, seq_in_IMU19]
    pretrain_seq_in_correspondence = prepare_XT_data(C, seq_in_correspondence, 'pretrain')

    test_seq_in_correspondence = None
    if running_mode in ['embed', 'test']:
        if C.imu200:
            C, seq_in_BBX5, seq_out_BBX5, \
               seq_in_FTM2, seq_out_FTM2, \
               seq_in_IMUaccgym9_200, seq_out_IMUaccgym9_200 = prepare_sync_testing_data(C)
            seq_in_BBX5 = torch.from_numpy(seq_in_BBX5); seq_out_BBX5 = torch.from_numpy(seq_out_BBX5)
            seq_in_FTM2 = torch.from_numpy(seq_in_FTM2); seq_out_FTM2 = torch.from_numpy(seq_out_FTM2)
            seq_in_IMUaccgym9_200 = torch.from_numpy(seq_in_IMUaccgym9_200)
            seq_out_IMUaccgym9_200 = torch.from_numpy(seq_out_IMUaccgym9_200)
            seq_in_correspondence = [seq_in_BBX5, seq_in_FTM2, seq_in_IMUaccgym9_200]
            test_seq_in_correspondence = prepare_XT_data(C, seq_in_correspondence, 'test')
        elif C.imu_ned_pos2:
            C, seq_in_BBX5, seq_out_BBX5, \
               seq_in_FTM2, seq_out_FTM2, \
               seq_in_IMU_NED_pos2, seq_out_IMU_NED_pos2 = prepare_sync_testing_data(C)
            seq_in_correspondence = [seq_in_BBX5, seq_in_FTM2, seq_in_IMU_NED_pos2]
            test_seq_in_correspondence = copy.deepcopy(seq_in_correspondence)
        else:
            C, seq_in_BBX5, seq_out_BBX5, \
               seq_in_FTM2, seq_out_FTM2, \
               seq_in_IMU19, seq_out_IMU19 = prepare_sync_testing_data(C)
            seq_in_BBX5 = torch.from_numpy(seq_in_BBX5); seq_out_BBX5 = torch.from_numpy(seq_out_BBX5)
            seq_in_FTM2 = torch.from_numpy(seq_in_FTM2); seq_out_FTM2 = torch.from_numpy(seq_out_FTM2)
            seq_in_IMU19 = torch.from_numpy(seq_in_IMU19); seq_out_IMU19 = torch.from_numpy(seq_out_IMU19)
            seq_in_correspondence = [seq_in_BBX5, seq_in_FTM2, seq_in_IMU19]
            test_seq_in_correspondence = prepare_XT_data(C, seq_in_correspondence, 'test')
    # print('\n\n utils.py - C.test_seq_subj_i_in_view_dict: ', C.test_seq_subj_i_in_view_dict)
    # print('\n\n utils.py - np.shape(pretrain_seq_in_correspondence): ', np.shape(pretrain_seq_in_correspondence))

    if C.imu_ned_pos2:
        train_seq_in_correspondence, val_seq_in_correspondence = None, None # edit later
    else:
        train_seq_in_correspondence, val_seq_in_correspondence = train_val_split(pretrain_seq_in_correspondence, C.train_val_rate)
        # [seq_in_BBX5, seq_in_FTM2, seq_in_IMU19]

        # Now pretrain_seq_in_correspondence = [seq_in_BBX5, seq_in_FTM2, seq_in_IMU19, seq_in_TX]
        train_seq_in_IMU_TX, val_seq_in_IMU_TX = train_seq_in_correspondence[-1], val_seq_in_correspondence[-1]
        print('np.shape(train_seq_in_IMU_TX): ', np.shape(train_seq_in_IMU_TX))
        print('np.shape(val_seq_in_IMU_TX): ', np.shape(val_seq_in_IMU_TX))
        # test_seq_in_IMU_TX = test_seq_in_IMU_TX[-1] if test_seq_in_IMU_TX is not None else None
        test_seq_in_IMU_TX = test_seq_in_correspondence[-1] if test_seq_in_correspondence is not None else None
        print('np.shape(test_seq_in_IMU_TX): ', np.shape(test_seq_in_IMU_TX))
        '''
        e.g.
        np.shape(train_seq_in_IMU_TX):  (35469, 10, 1, 9)
        np.shape(val_seq_in_IMU_TX):  (3941, 10, 1, 9)
        np.shape(test_seq_in_IMU_TX):  torch.Size([2820, 10, 1, 9])
        '''
        print('np.shape(train_seq_in_correspondence[0]): ', np.shape(train_seq_in_correspondence[0]))
        print('np.shape(train_seq_in_correspondence[1]): ', np.shape(train_seq_in_correspondence[1]))
        print('np.shape(train_seq_in_correspondence[2]): ', np.shape(train_seq_in_correspondence[2]))
        print('np.shape(train_seq_in_correspondence[3]): ', np.shape(train_seq_in_correspondence[3]))
        '''
        e.g.
        np.shape(train_seq_in_correspondence[0]):  (33025, 10, 1, 5)
        np.shape(train_seq_in_correspondence[1]):  (33025, 10, 1, 2)
        np.shape(train_seq_in_correspondence[2]):  (33025, 10, 1, 19)
        np.shape(train_seq_in_correspondence[3]):  (33025, 10, 1, 9)
        '''
        print('\n\n running_mode: ', running_mode)

        # DEBUG:
        # print('embeddings.py - __main__() - np.shape(np.array(test_seq_in_IMU_TX)): ', np.shape(np.array(test_seq_in_IMU_TX)))
        # print(HERE)

        print('C.normalization:  ', C.normalization)

    # ======================
    # Model and Trainer >>>
    if running_mode == 'pretrain':
        if C.normalization:
            pipeline = [Preprocess4Normalization(C.IMU_XT_dim), Preprocess4Mask(C.mask_cfg)]
        else: pipeline = None
        train_data_set = LIBERTDataset4Pretrain(train_seq_in_IMU_TX, pipeline=pipeline)
        train_data_loader = DataLoader(train_data_set, shuffle=True, batch_size=C.train_cfg['batch_size'])
        val_data_set = LIBERTDataset4Pretrain(val_seq_in_IMU_TX, pipeline=pipeline)
        val_data_loader = DataLoader(val_data_set, shuffle=False, batch_size=C.train_cfg['batch_size'])
        model = LIMUBertModel4Pretrain(C.model_cfg)
    elif running_mode == 'embed':
        if C.normalization:
            pipeline = [Preprocess4Normalization(C.IMU_XT_dim)] #, Preprocess4Mask(C.mask_cfg)]
        else: pipeline = None
        train_data_set = LIBERTDataset4Pretrain(train_seq_in_correspondence, pipeline=pipeline, correspondence=True)
        train_data_loader = DataLoader(train_data_set, shuffle=False, batch_size=C.train_cfg['batch_size']) # np.shape(seq_in_IMU_TX)[0]) # C.train_cfg['batch_size'])
        # train_data_loader: item: [seq_in_BBX5, seq_in_FTM2, seq_in_IMU_TX]
        val_data_set = LIBERTDataset4Pretrain(val_seq_in_correspondence, pipeline=pipeline, correspondence=True)
        val_data_loader = DataLoader(val_data_set, shuffle=False, batch_size=C.train_cfg['batch_size']) # np.shape(seq_in_IMU_TX)[0]) # C.train_cfg['batch_size'])
        # val_data_loader: item: [seq_in_BBX5, seq_in_FTM2, seq_in_IMU_TX]
        test_data_set = LIBERTDataset4Pretrain(test_seq_in_correspondence, pipeline=pipeline, correspondence=True)
        test_data_loader = DataLoader(test_data_set, shuffle=False, batch_size=C.train_cfg['batch_size']) # np.shape(seq_in_IMU_TX)[0]) # C.train_cfg['batch_size'])
        # test_data_loader: item: [seq_in_BBX5, seq_in_FTM2, seq_in_IMU_TX]
        model = LIMUBertModel4Pretrain(C.model_cfg, output_embed=True)
    elif running_mode == 'train' or running_mode == 'investigate':
        if C.normalization:
            pipeline = [Preprocess4Normalization(C.IMU_XT_dim)] #, Preprocess4Mask(C.mask_cfg)]
        else: pipeline = None

        def remove_outliers(y_feet_center, s):
            print('\n\n Before remove_outliers')
            print('\n\n np.shape(y_feet_center): ', np.shape(y_feet_center))
            print('\n\n np.shape(y_feet_center): ', np.shape(y_feet_center))
            # arr_1D = np.delete(arr_1D, np.where(arr_1D == 8))

            ol_indices = np.where(y_feet_center > 700)
            # print('\n\n ol_indices: ', ol_indices)
            # print('\n\n np.shape(ol_indices): ', np.shape(ol_indices))
            # e.g. np.shape(ol_indices):  (3, 299)
            y_feet_center = np.delete(y_feet_center, ol_indices)
            s = np.delete(s, ol_indices)

            ol_indices = np.where(y_feet_center < 5)
            y_feet_center = np.delete(y_feet_center, ol_indices)
            s = np.delete(s, ol_indices)

            # print('\n\n After remove_outliers')
            # print('\n\n np.shape(y_feet_center): ', np.shape(y_feet_center))
            # print('\n\n np.shape(s): ', np.shape(s))
            y_feet_center = np.expand_dims(np.expand_dims(y_feet_center, axis=1), axis=2)
            s = np.expand_dims(np.expand_dims(s, axis=1), axis=2)
            print('\n\n np.shape(y_feet_center): ', np.shape(y_feet_center))
            return y_feet_center, s

        if 'Width_regressor' in C.model_cfg['train_model_id']:
            # ===================
            # trainining data >>>
            y = train_seq_in_correspondence[0][:, :, :, 1][0:-1:C.slicing_step]
            w = train_seq_in_correspondence[0][:, :, :, 3][0:-1:C.slicing_step]
            h = train_seq_in_correspondence[0][:, :, :, 4][0:-1:C.slicing_step]
            print('\n pdr_baseline.py - Width_regressor - np.shape(y): ', np.shape(y))
            print('\n pdr_baseline.py - Width_regressor - np.shape(w): ', np.shape(w))
            print('\n pdr_baseline.py - Width_regressor - np.shape(h): ', np.shape(h))
            # x_feet_center = x
            y_feet_center = y + h / 2
            print('\n utils.py - preprocess_config - np.shape(y_feet_center): ', np.shape(y_feet_center))
            y_feet_center, w = remove_outliers(y_feet_center, w)
            train_yfc_w = np.concatenate((y_feet_center, w), axis = 2)
            print('\n utils.py - preprocess_config - np.shape(train_yfc_w): ', np.shape(train_yfc_w))
            '''
            e.g.
             pdr_baseline.py - Width_regressor - np.shape(y):  (87210, 1, 1)
             pdr_baseline.py - Width_regressor - np.shape(w):  (87210, 1, 1)
             pdr_baseline.py - Width_regressor - np.shape(h):  (87210, 1, 1)
             utils.py - preprocess_config - np.shape(y_feet_center):  (87210, 1, 1)
             utils.py - preprocess_config - np.shape(train_yfc_w):  (87210, 1, 2)
            '''
            # C.batch_size = min(C.batch_size, np.shape(y)[0]) - 1000
            # C.train_cfg['batch_size'] = min(C.train_cfg['batch_size'], C.batch_size) - 1000
            # C.model_cfg['batch_size'] = min(C.model_cfg['batch_size'], C.batch_size) - 1000
            train_data_loader = DataLoader(train_yfc_w, shuffle=True, batch_size=C.train_cfg['batch_size'])
            print('\n\n\n')
            print('\n utils.py - preprocess_config - len(train_data_loader): ', len(train_data_loader))
            # trainining data <<<
            # ===================
            # ===================
            # val data >>>
            y = val_seq_in_correspondence[0][:, :, :, 1]#[0:-1:C.slicing_step]
            w = val_seq_in_correspondence[0][:, :, :, 3]#[0:-1:C.slicing_step]
            h = val_seq_in_correspondence[0][:, :, :, 4]#[0:-1:C.slicing_step]
            print('\n pdr_baseline.py - Width_regressor - np.shape(y): ', np.shape(y))
            print('\n pdr_baseline.py - Width_regressor - np.shape(w): ', np.shape(w))
            print('\n pdr_baseline.py - Width_regressor - np.shape(h): ', np.shape(h))
            # HERE
            # x_feet_center = x
            y_feet_center = y + h / 2
            print('\n utils.py - preprocess_config - np.shape(y_feet_center): ', np.shape(y_feet_center))
            # e.g. (8608, 1, 1)
            y_feet_center, w = remove_outliers(y_feet_center, w)
            val_yfc_w = np.concatenate((y_feet_center, w), axis = 2)
            print('\n utils.py - preprocess_config - np.shape(val_yfc_w): ', np.shape(val_yfc_w))
            '''
            e.g.
             pdr_baseline.py - Width_regressor - np.shape(y):  (87210, 1, 1)
             pdr_baseline.py - Width_regressor - np.shape(w):  (87210, 1, 1)
             pdr_baseline.py - Width_regressor - np.shape(h):  (87210, 1, 1)
             utils.py - preprocess_config - np.shape(y_feet_center):  (87210, 1, 1)
             utils.py - preprocess_config - np.shape(val_yfc_w):  (87210, 1, 2)
            '''
            val_data_loader = DataLoader(val_yfc_w, shuffle=False, batch_size=C.train_cfg['batch_size'])
            print('\n\n\n')
            print('\n utils.py - preprocess_config - len(val_data_loader): ', len(val_data_loader))
            # val data <<<
            # ===================
        elif 'Height_regressor' in C.model_cfg['train_model_id']:
            y = train_seq_in_correspondence[0][:, :, :, 1][0:-1:C.slicing_step]
            w = train_seq_in_correspondence[0][:, :, :, 3][0:-1:C.slicing_step]
            h = train_seq_in_correspondence[0][:, :, :, 4][0:-1:C.slicing_step]
            y_feet_center = y + h / 2
            y_feet_center, h = remove_outliers(y_feet_center, h)
            train_yfc_h = np.concatenate((y_feet_center, h), axis = 2)
            train_data_loader = DataLoader(train_yfc_h, shuffle=True, batch_size=C.train_cfg['batch_size'])
            y = val_seq_in_correspondence[0][:, :, :, 1]#[0:-1:C.slicing_step]
            w = val_seq_in_correspondence[0][:, :, :, 3]#[0:-1:C.slicing_step]
            h = val_seq_in_correspondence[0][:, :, :, 4]#[0:-1:C.slicing_step]
            print('\n pdr_baseline.py - Height_regressor - np.shape(h): ', np.shape(h))
            y_feet_center = y + h / 2
            y_feet_center, h = remove_outliers(y_feet_center, h)
            val_yfc_h = np.concatenate((y_feet_center, h), axis = 2)
            val_data_loader = DataLoader(val_yfc_h, shuffle=False, batch_size=C.train_cfg['batch_size'])
        elif 'Depth_regressor' in C.model_cfg['train_model_id']:
            y = train_seq_in_correspondence[0][:, :, :, 1][0:-1:C.slicing_step]
            d = train_seq_in_correspondence[0][:, :, :, 2][0:-1:C.slicing_step]
            h = train_seq_in_correspondence[0][:, :, :, 4][0:-1:C.slicing_step]
            y_feet_center = y + h / 2
            y_feet_center, d = remove_outliers(y_feet_center, d)
            train_yfc_d = np.concatenate((y_feet_center, d), axis = 2)
            train_data_loader = DataLoader(train_yfc_d, shuffle=True, batch_size=C.train_cfg['batch_size'])
            y = val_seq_in_correspondence[0][:, :, :, 1]#[0:-1:C.slicing_step]
            d = val_seq_in_correspondence[0][:, :, :, 2]#[0:-1:C.slicing_step]
            h = val_seq_in_correspondence[0][:, :, :, 4]#[0:-1:C.slicing_step]
            print('\n pdr_baseline.py - Depth_regressor - np.shape(h): ', np.shape(h))
            y_feet_center = y + h / 2
            y_feet_center, d = remove_outliers(y_feet_center, d)
            val_yfc_d = np.concatenate((y_feet_center, d), axis = 2)
            val_data_loader = DataLoader(val_yfc_d, shuffle=False, batch_size=C.train_cfg['batch_size'])
        else:
            train_data_set = LIBERTDataset4Pretrain(train_seq_in_correspondence, pipeline=pipeline, correspondence=True)
            train_data_loader = DataLoader(train_data_set, shuffle=True, batch_size=C.train_cfg['batch_size'])
            # train_data_loader: item: [seq_in_BBX5, seq_in_FTM2, seq_in_IMU_TX]
            val_data_set = LIBERTDataset4Pretrain(val_seq_in_correspondence, pipeline=pipeline, correspondence=True)
            val_data_loader = DataLoader(val_data_set, shuffle=False, batch_size=C.train_cfg['batch_size'])
            # val_data_loader: item: [seq_in_BBX5, seq_in_FTM2, seq_in_IMU_TX]

        if C.model_cfg['train_model_id'] == 'Xformer_IemcC2C':
            model = Xformer_IemcC2C(C.model_cfg) # I2C: IMU to Camera
        elif C.model_cfg['train_model_id'] == 'Xformer_IFcC2C':
            model = Xformer_IFcC2C(C.model_cfg) # I2C: IMU to Camera
        elif C.model_cfg['train_model_id'] == 'Xlstm_IFcC2C':
            model = Xlstm_IFcC2C(C.model_cfg) # I2C: IMU to Camera
        elif C.model_cfg['train_model_id'] == 'Xtranslator_IFcC2C':
            model = Xtranslator_IFcC2C(C.model_cfg).net # I2C: IMU to Camera # Baseline model from ViTag
        elif 'Width_regressor' in C.model_cfg['train_model_id']: # pdr: Pedestrian Dead Reckoning
            model = Width_regressor(C.model_cfg)
        elif 'Height_regressor' in C.model_cfg['train_model_id']: # pdr: Pedestrian Dead Reckoning
            model = Height_regressor(C.model_cfg)
        elif 'Depth_regressor' in C.model_cfg['train_model_id']: # pdr: Pedestrian Dead Reckoning
            model = Depth_regressor(C.model_cfg)
        C.model_cfg['batch_size'] = C.train_cfg['batch_size']
    elif running_mode == 'test':
        if C.normalization:
            pipeline = [Preprocess4Normalization(C.IMU_XT_dim)] #, Preprocess4Mask(C.mask_cfg)]
        else: pipeline = None
        if C.imu_ned_pos2:
            test_data_set = None
            test_data_loader = None
        else:
            test_data_set = LIBERTDataset4Pretrain(test_seq_in_correspondence, pipeline=pipeline, correspondence=True)
            test_data_loader = DataLoader(test_data_set, shuffle=False, batch_size=C.test_cfg['batch_size']) # np.shape(seq_in_IMU_TX)[0]) # C.train_cfg['batch_size'])

        # test_data_loader: item: [seq_in_BBX5, seq_in_FTM2, seq_in_IMU_TX]
        if C.model_cfg['train_model_id'] == 'Xformer_IemcC2C':
            model = Xformer_IemcC2C(C.model_cfg) # I2C: IMU to Camera
        elif C.model_cfg['train_model_id'] == 'Xformer_IFcC2C':
            model = Xformer_IFcC2C(C.model_cfg) # I2C: IMU to Camera
        elif C.model_cfg['train_model_id'] == 'Xlstm_IFcC2C':
            model = Xlstm_IFcC2C(C.model_cfg).model # I2C: IMU to Camera
        elif C.model_cfg['train_model_id'] == 'Xtranslator_IFcC2C':
            model = Xtranslator_IFcC2C(C.model_cfg).net # I2C: IMU to Camera # Baseline model from ViTag
        elif 'Width_regressor' in C.model_cfg['train_model_id']: # pdr: Pedestrian Dead Reckoning
            model = Width_regressor(C.model_cfg)
        elif 'Height_regressor' in C.model_cfg['train_model_id']: # pdr: Pedestrian Dead Reckoning
            model = Height_regressor(C.model_cfg)
        elif 'Depth_regressor' in C.model_cfg['train_model_id']: # pdr: Pedestrian Dead Reckoning
            model = Depth_regressor(C.model_cfg)
        elif 'pdr' in C.model_cfg['train_model_id']:
            model_w = Width_regressor(C.model_cfg)
            model_h = Height_regressor(C.model_cfg)
            model_d = Depth_regressor(C.model_cfg)
            model = (model_w, model_h, model_d)
        elif C.model_cfg['train_model_id'] == 'Tc_interp':
            model = None
        elif C.model_cfg['train_model_id'] == 'kalman_filter':
            model = MyKalmanFilter(C.model_cfg)
        elif C.model_cfg['train_model_id'] == 'kalman_filter_v2':
            model = MyKalmanFilter_v2(C.model_cfg)
        C.model_cfg['batch_size'] = C.test_cfg['batch_size']
        # DEBUG:
        # print('\n utils.py - preprocess_config - C.__dict__: ', C.__dict__)
        # print('\n utils.py - preprocess_config - dir(test_data_loader): ', dir(test_data_loader))
        # print('\n utils.py - preprocess_config - test_data_loader.__len__(): ', test_data_loader.__len__())
        # e.g. 2685
        print('\n utils.py - preprocess_config - model: ', model)
        

    # print('\n utils.py - preprocess_config - summary(model): ', summary(model))

    print(dir(tops))
    # if 'MSE' in C.loss_func: criterion = nn.MSELoss(reduction='none')
    # elif 'IOU' in C.loss_func:
    #     if 'DIOU' in C.loss_func:
    #         if 'depth' in C.loss_func: criterion = combined_NMSE_IOU_loss
    #         else: criterion = proc_distance_box_iou_loss # IOU_loss
    #     if 'GIOU' in C.loss_func:
    #         if 'depth' in C.loss_func: criterion = combined_NMSE_IOU_loss
    #         else: criterion = proc_generalized_box_iou_loss # IOU_loss
    # https://pytorch.org/vision/main/generated/torchvision.ops.distance_box_iou_loss.html

    # Baseline of ViTag >>>
    if 'Xtranslator' in C.model_cfg['train_model_id']:
        if C.loss_func == 'MSE_xydwh':
            criterion = 'mse'
        else:
            criterion = None
        if C.model_cfg['optimizer'] == 'Adam':
            optimizer = 'adam'
        # TODO: device
        device = None
    # Baseline of ViTag <<<
    if C.loss_func in ['MSE_xydwh', 'MSE_xyd', 'MSE_xy']:
        criterion = nn.MSELoss(reduction='none')
    elif C.loss_func in ['DIOU_depth', 'GIOU_depth', 'NMSE_xydwh_DIOU', 'NMSE_xydwh_GIOU']:
        criterion = combined_NMSE_IOU_loss
    elif C.loss_func == 'DIOU':
        criterion = proc_distance_box_iou_loss # IOU_loss
    elif C.loss_func == 'GIOU':
        criterion = proc_generalized_box_iou_loss # IOU_loss
    else:
        # Default
        criterion = nn.MSELoss(reduction='none')

    # Default
    if C.model_cfg['optimizer'] == 'Adam':
        if 'pdr' in C.model_cfg['train_model_id']:
            optimizer_w = torch.optim.Adam(params=model_w.parameters(), lr=C.train_cfg['lr'])
            optimizer_h = torch.optim.Adam(params=model_h.parameters(), lr=C.train_cfg['lr'])
            optimizer_d = torch.optim.Adam(params=model_d.parameters(), lr=C.train_cfg['lr'])
            optimizer = (optimizer_w, optimizer_h, optimizer_d)
        elif C.model_cfg['train_model_id'] == 'Tc_interp' or C.model_cfg['train_model_id'] == 'kalman_filter' \
            or C.model_cfg['train_model_id'] == 'kalman_filter_v2': optimizer = None
        elif 'Xtranslator' in C.model_cfg['train_model_id']: optimizer = 'adam'
        else: optimizer = torch.optim.Adam(params=model.parameters(), lr=C.train_cfg['lr'])
    elif C.model_cfg['optimizer'] == 'SGD':
        if 'pdr' in C.model_cfg['train_model_id']:
            optimizer_w = torch.optim.SGD(optimizer_w.parameters(), lr=0.01, momentum=0.9)
            optimizer_h = torch.optim.SGD(optimizer_h.parameters(), lr=0.01, momentum=0.9)
            optimizer_d = torch.optim.SGD(optimizer_d.parameters(), lr=0.01, momentum=0.9)
            optimizer = (optimizer_w, optimizer_h, optimizer_d)
        elif C.model_cfg['train_model_id'] == 'Tc_interp' or C.model_cfg['train_model_id'] == 'kalman_filter' \
            or C.model_cfg['train_model_id'] == 'kalman_filter_v2': optimizer = None
        elif 'Xtranslator' in C.model_cfg['train_model_id']: optimizer = 'SGD'
        else: optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    device = get_device(C.gpu)

    # trainer = train.Trainer(train_cfg, model, optimizer, C.checkpoint_root_path, device) # ORI
    trainer = train.Trainer(C, model, optimizer, C.checkpoint_path_dict[C.running_mode], device)

    # DEBUG:prepare_sync_testing_data(C)
    # print('\n\n C.__dict__: ', C.__dict__)
    # print('\n\n C.checkpoint_path_dict[C.running_mode]: ', C.checkpoint_path_dict[C.running_mode])
    # print('\n\n C.checkpoint_path_dict[train]: ', C.checkpoint_path_dict['train'])
    # HERE
    # Model and Trainer <<<
    # ======================

    return trainer, train_data_loader, val_data_loader, test_data_loader, \
            train_seq_in_correspondence, val_seq_in_correspondence, test_seq_in_correspondence, \
            criterion, optimizer, C

class EmbedDataset(Dataset):
    def __init__(self, data, labels, correspondence=False):
        super().__init__()
        self.data = data
        self.labels = labels

    def __getitem__(self, index):
        return torch.from_numpy(self.data[index]).float(), torch.from_numpy(self.labels[index]).float()

    def __len__(self):
        return len(self.data)

def update_test_log_config(C: Config):
    C.test_log_path = C.test_log_root_path = C.checkpoint_path_dict[C.running_mode] + '/log'
    os.makedirs((C.test_log_path), exist_ok=True)

    C.test_log_file_path = C.test_log_path + '/results.log'
    C.test_log_file = open((C.test_log_file_path), 'a')
    C.test_log_pkl_file_path = C.test_log_path + '/results.pkl'
    C.test_log_plk_file = open((C.test_log_pkl_file_path), 'w')

    C.test_log_per_frame_file_path = C.test_log_path + '/results_per_frame.log'
    C.test_log_per_frame_file = open((C.test_log_per_frame_file_path), 'a')

    # DEBUG:
    print('utils.py - update_test_log_config() - C.test_log_path: ', C.test_log_path)

    '''
    # C.test_log_id = 'trained_scene1_' + C.test_type + '_' + C.seq_id + '_' + C.log_time
    C.test_log_id = C.seq_id + '_f_d_' + C.FTM_dist + '_nl_' + str(C.noise_level) + '_' + C.log_time
    C.test_log_id += '_w_ls'
    for w in C.w_ls: C.test_log_id += '_' + str(w)

    C.test_log_file_path = C.test_log_path + '/' + C.test_log_id + '_dfv4_test.log'
    C.test_log_file = open((C.test_log_file_path), 'a')
    C.test_log_file.write(str(C.__dict__) + '\n\n')
    C.test_log_file.flush()

    C.ts16_dfv4_subj_i_to_BBX5_prime = defaultdict()
    C.ts16_dfv4_subj_i_to_BBX5_prime_path_to_save = C.checkpoint_path_dict[C.running_mode] + '/' + C.test_log_id + '/ts16_dfv4_subj_i_to_BBX5_prime.pkl'
    C.ts16_dfv4_to_pred_BBX5_labels = defaultdict()
    C.ts16_dfv4_to_pred_BBX5_labels_path_to_save = C.checkpoint_path_dict[C.running_mode] + '/' + C.test_log_id + '/ts16_dfv4_to_pred_BBX5_labels.pkl'

    C.ts16_dfv4_to_test_stats = defaultdict()
    C.ts16_dfv4_to_test_stats_path_to_save = C.checkpoint_path_dict[C.running_mode] + '/' + C.test_log_id + '/ts16_dfv4_to_test_stats.pkl'

    C.prev_gd_pred_phone_i_BBX_ls, C.prev_gd_pred_phone_i_IMU_ls = [], []
    C.prev_hg_pred_phone_i_BBX_ls, C.prev_hg_pred_phone_i_IMU_ls = [], []
    C.scene_test_stats = {'gd': {'Cam': {'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0, 'correct_num': 0, 'total_num': 0, \
                                        'IDSWITCH': np.nan, 'ts16_dfv4_Cam_IDP': 0.0, 'cumu_Cam_IDP': 0.0}, \
                                    'Phone': {'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0, 'correct_num': 0, 'total_num': 0, \
                                        'IDSWITCH': np.nan, 'ts16_dfv4_Phone_IDP': 0.0, 'cumu_Phone_IDP': 0.0}}, \
                             'hg': {'Cam': {'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0, 'correct_num': 0, 'total_num': 0, \
                                        'IDSWITCH': np.nan, 'ts16_dfv4_Cam_IDP': 0.0, 'cumu_Cam_IDP': 0.0}, \
                                    'Phone': {'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0, 'correct_num': 0, 'total_num': 0, \
                                        'IDSWITCH': np.nan, 'ts16_dfv4_Phone_IDP': 0.0, 'cumu_Phone_IDP': 0.0}}} # hg: Hungarian, gd: greedy-matching
    # FN: misses of Phone Holders, TN: Others -> None
    C.scene_test_stats_path_to_save = C.checkpoint_path_dict[C.running_mode] + '/scene_test_stats.pkl'
    '''

    print('End of utils.py - prepare_sync_testing_data - C.img_path: ', C.img_path)
    return C


# ---------------------
#  Eval Reconstruction
# ---------------------
def eval_recon_core(C: Config, func_forward, model, test_seq_in_correspondence, device):
    print('\n\n utils.py - eval_recon_core() - C.normalization: ', C.normalization)
    # print('\n\n utils.py - eval_recon_core() - len(test_seq_in_correspondence): ', len(test_seq_in_correspondence))
    # print('\n\n utils.py - eval_recon_core() - np.shape(test_seq_in_correspondence[0]): ', np.shape(test_seq_in_correspondence[0]))
    # print('\n\n utils.py - eval_recon_core() - np.shape(test_seq_in_correspondence[1]): ', np.shape(test_seq_in_correspondence[1]))
    # print('\n\n utils.py - eval_recon_core() - np.shape(test_seq_in_correspondence[2]): ', np.shape(test_seq_in_correspondence[2]))
    # print('\n\n utils.py - eval_recon_core() - np.shape(test_seq_in_correspondence[3]): ', np.shape(test_seq_in_correspondence[3]))
    '''
    utils.py - eval_recon_core() - C.normalization:  True
    utils.py - eval_recon_core() - len(test_data_set):  3382
    utils.py - eval_recon_core() - len(test_data_set[0]):  3
    utils.py - eval_recon_core() - len(test_data_set[1]):  3
    utils.py - eval_recon_core() - len(test_data_set[2]):  3
    utils.py - eval_recon_core() - np.shape(test_data_set[0][0]):  (10, 1, 5)
    utils.py - eval_recon_core() - np.shape(test_data_set[0][1]):  (10, 1, 2)
    utils.py - eval_recon_core() - np.shape(test_data_set[0][2]):  (10, 1, 9)
    utils.py - eval_recon_core() - len(test_seq_in_correspondence):  4
    utils.py - eval_recon_core() - np.shape(test_seq_in_correspondence[0]):  torch.Size([2970, 10, 1, 5]) # [len_video, len_win, samples_per_win, features]
    utils.py - eval_recon_core() - np.shape(test_seq_in_correspondence[1]):  torch.Size([2970, 10, 1, 2]) # [len_video, len_win, samples_per_win, features]
    utils.py - eval_recon_core() - np.shape(test_seq_in_correspondence[2]):  torch.Size([2970, 10, 1, 19]) # [len_video, len_win, samples_per_win, features]
    utils.py - eval_recon_core() - np.shape(test_seq_in_correspondence[3]):  torch.Size([2970, 10, 1, 9]) # [len_video, len_win, samples_per_win, features]
    '''
    # seq_in_BBX5_video = test_seq_in_correspondence[0]
    # seq_in_FTM2_video = test_seq_in_correspondence[1]
    # seq_in_IMU19_video = test_seq_in_correspondence[2]
    # seq_in_IMU_XT_video = test_seq_in_correspondence[3]

    # print('\n\n utils.py - eval_recon_core - C.max_depth: ', C.max_depth)
    reconstruction_path = C.reconstruction_path

    # if C.scene_id == 0: C.RGBg_ts16_dfv4_ls_path = C.seq_path_for_model + '/RGBh_ts16_dfv4_ls.json'
    # else: C.RGBg_ts16_dfv4_ls_path = C.seq_path_for_model + '/RGB_ts16_dfv.shape(BBX5_sync_dfv4)4_ls.json'
    # print('C.RGBg_ts16_dfv4_ls_path: ', C.RGBg_ts16_dfv4_ls_path)
    C.RGBg_ts16_dfv4_ls_path = C.seq_path_for_model + '/RGBg_ts16_dfv4_ls.json'
    print('C.RGBg_ts16_dfv4_ls_path: ', C.RGBg_ts16_dfv4_ls_path)

    with open(C.RGBg_ts16_dfv4_ls_path, 'r') as f:
        C.RGBg_ts16_dfv4_ls = json.load(f)
        print(C.RGBg_ts16_dfv4_ls_path, 'loaded!')
        print('C.RGBg_ts16_dfv4_ls[:5]: ', C.RGBg_ts16_dfv4_ls[:5]) # verification
    print('len(C.RGBg_ts16_dfv4_ls): ', len(C.RGBg_ts16_dfv4_ls))
    # e.g. 1816
    # phone_win_offset = np.shape(seq_in_IMU_XT_video)[0] - len(C.RGBg_ts16_dfv4_ls)

    if C.imu_ned_pos2:
        # ---------------------
        #  Adjusted BEV points
        # ---------------------
        C.BEV_src_pnts = C.BEV_src_pnts_ls[C.scene_id]
        C.BEV_dst_pnts = C.BEV_dst_pnts_ls[C.scene_id]
        if C.scene_id == 0:
            C.BEV_px2meters_ratio = float(1280 - 1089) / 5
        elif C.scene_id == 1:
            dx = C.BEV_dst_pnts[0][0] - C.BEV_dst_pnts[1][0]
            dy = C.BEV_dst_pnts[0][1] - C.BEV_dst_pnts[1][1]
            BEV_dst_pnts_0 = (C.BEV_dst_pnts[0][0] + 24, C.BEV_dst_pnts[0][1] - 5)
            BEV_dst_pnts_1 = (C.BEV_dst_pnts[1][0], C.BEV_dst_pnts[1][1])
            BEV_dst_pnts_2_0 = C.BEV_dst_pnts[1][0] + dy
            BEV_dst_pnts_2_1 = C.BEV_dst_pnts[1][1] - dx
            BEV_dst_pnts_3_0 = C.BEV_dst_pnts[0][0] + dy
            BEV_dst_pnts_3_1 = C.BEV_dst_pnts[0][1] - dx
            C.BEV_dst_pnts = (C.BEV_dst_pnts[0], BEV_dst_pnts_1, (BEV_dst_pnts_2_0, BEV_dst_pnts_2_1), (BEV_dst_pnts_3_0, BEV_dst_pnts_3_1))
            C.BEV_px2meters_ratio = float(1280 - 1089) / 5
        elif C.scene_id == 2:
            BEV_dst_pnts_0_0 = C.BEV_dst_pnts[0][0] - 2
            BEV_dst_pnts_0_1 = C.BEV_dst_pnts[0][1] - 48
            BEV_dst_pnts_1_0 = C.BEV_dst_pnts[1][0] - 37 - 11 - 2 - 2 - 1 - 1 - 2
            BEV_dst_pnts_1_1 = C.BEV_dst_pnts[1][1] + 30 - 2 - 3 - 2 - 1
            dx = C.BEV_dst_pnts[0][0] - C.BEV_dst_pnts[1][0]
            dy = C.BEV_dst_pnts[0][1] - C.BEV_dst_pnts[1][1]

            BEV_dst_pnts_2_0 = C.BEV_dst_pnts[1][0] + dy - 20 - 10 + 10 - 2 + 1
            BEV_dst_pnts_2_1 = C.BEV_dst_pnts[1][1] - dx - 8 + 30 - 4
            BEV_dst_pnts_3_0 = C.BEV_dst_pnts[0][0] + dy + 17 + 10 + 2 + 3 + 1
            BEV_dst_pnts_3_1 = C.BEV_dst_pnts[0][1] - dx -58 + 1
            C.BEV_dst_pnts = ((BEV_dst_pnts_0_0, BEV_dst_pnts_0_1), \
                            (BEV_dst_pnts_1_0, BEV_dst_pnts_1_1), \
                            (BEV_dst_pnts_2_0, BEV_dst_pnts_2_1), \
                            (BEV_dst_pnts_3_0, BEV_dst_pnts_3_1))
            C.BEV_px2meters_ratio = float(1280 - 1091) / 5
        elif C.scene_id == 3:
            BEV_dst_pnts_0_0 = C.BEV_dst_pnts[0][0]
            BEV_dst_pnts_0_1 = C.BEV_dst_pnts[0][1]
            BEV_dst_pnts_1_0 = C.BEV_dst_pnts[1][0] + 19
            BEV_dst_pnts_1_1 = C.BEV_dst_pnts[1][1] + 50
            dx = C.BEV_dst_pnts[0][0] - C.BEV_dst_pnts[1][0]
            dy = C.BEV_dst_pnts[0][1] - C.BEV_dst_pnts[1][1]

            BEV_dst_pnts_2_0 = C.BEV_dst_pnts[1][0] + dy + 170
            BEV_dst_pnts_2_1 = C.BEV_dst_pnts[1][1] + dx + 30
            BEV_dst_pnts_3_0 = C.BEV_dst_pnts[0][0] + dy + 150
            BEV_dst_pnts_3_1 = C.BEV_dst_pnts[0][1] + dx - 20
            C.BEV_dst_pnts = ((BEV_dst_pnts_0_0, BEV_dst_pnts_0_1), \
                            (BEV_dst_pnts_1_0, BEV_dst_pnts_1_1), \
                            (BEV_dst_pnts_2_0, BEV_dst_pnts_2_1), \
                            (BEV_dst_pnts_3_0, BEV_dst_pnts_3_1))
            C.BEV_px2meters_ratio = float(1280 - 1116) / 5
        elif C.scene_id == 4:
            BEV_dst_pnts_0_0 = C.BEV_dst_pnts[0][0]
            BEV_dst_pnts_0_1 = C.BEV_dst_pnts[0][1]
            BEV_dst_pnts_1_0 = BEV_dst_pnts_0_0 - 130 # C.BEV_dst_pnts[1][0]
            BEV_dst_pnts_1_1 = BEV_dst_pnts_0_1 - 90
            dx = BEV_dst_pnts_0_0 - BEV_dst_pnts_1_0
            dy = BEV_dst_pnts_0_1 - BEV_dst_pnts_1_1

            BEV_dst_pnts_2_0 = BEV_dst_pnts_1_0 - dy - 16
            BEV_dst_pnts_2_1 = BEV_dst_pnts_1_1 + dx + 11
            BEV_dst_pnts_3_0 = BEV_dst_pnts_0_0 - dy - 16
            BEV_dst_pnts_3_1 = BEV_dst_pnts_0_1 + dx + 15
            C.BEV_dst_pnts = ((BEV_dst_pnts_0_0, BEV_dst_pnts_0_1), \
                            (BEV_dst_pnts_1_0, BEV_dst_pnts_1_1), \
                            (BEV_dst_pnts_2_0, BEV_dst_pnts_2_1), \
                            (BEV_dst_pnts_3_0, BEV_dst_pnts_3_1))
            C.BEV_px2meters_ratio = float(1280 - 1112) / 5
        C.BEV_M = None # Bird's Eye View Perspective Transformation Matrix

    # --------------
    #  Video Window
    # --------------
    C.crr_ts16_dfv4_ls_all_i = 0
    C.video_len = min(1700, len(C.RGBg_ts16_dfv4_ls)) # len(C.ts12_BBX5_all)
    print(); print() # debug
    print('C.video_len: ', C.video_len) # e.g. 1700
    C.n_wins = min(1700, C.video_len - C.len_win + 1)
    print('C.n_wins: ', C.n_wins) # e.g. 1791

    print('\n\n utils.py - eval_recon_core() - C.seq_id_to_start_end_ts16_dfv4[C.seq_id][start]: ', C.seq_id_to_start_end_ts16_dfv4[C.seq_id]['start'])
    print('\n\n utils.py - eval_recon_core() - C.seq_id_to_start_end_ts16_dfv4[C.seq_id][end]: ', C.seq_id_to_start_end_ts16_dfv4[C.seq_id]['end'])
    # print('\n\n utils.py - eval_recon_core() - C.test_seq_subj_i_in_view_dict: ', C.test_seq_subj_i_in_view_dict)
    print('\n\n utils.py - eval_recon_core() - C.seq_id]: ', C.seq_id)

    C.test_log_per_frame_file.write('ts16_dfv4,iou,mrf\n')

    def print_bbox_win(C: Config, img, seq_in_BBX5_r, pred_BBX5_c, vis_len_win):
        
        subj_color = C.color_dict[C.color_ls[r_i]]
        subj_color_PRED = (subj_color[0] + 100, subj_color[1] + 100, subj_color[2] + 100)
        print('\n\n r_i: ', r_i)
        print('\n\n subj_color: ', subj_color)

        start_frame = vis_len_win - 1

        # GND BBX >>>
        if 'GND' in C.vis_bbox_type or 'GT' in C.vis_bbox_type:
            for f_i in range(start_frame, -1, -1): # recent frame to the first frame per win
                x_c, y_c, d_c = seq_in_BBX5_r[0, f_i, 0, 0], seq_in_BBX5_r[0, f_i, 0, 1],  seq_in_BBX5_r[0, f_i, 0, 2]
                w_c, h_c = seq_in_BBX5_r[0, f_i, 0, 3], seq_in_BBX5_r[0, f_i, 0, 4]
                # x_c, y_c, d_c = C.test_seq_in_BBX5_dict[(win_i - C.len_win, subj_i_r)][0, f_i, 0, 0], C.test_seq_in_BBX5_dict[(win_i, subj_i_r)][0, f_i, 0, 1],  C.test_seq_in_BBX5_dict[(win_i, subj_i_r)][0, f_i, 0, 2]
                # w_c, h_c = C.test_seq_in_BBX5_dict[(win_i - C.len_win, subj_i_r)][0, f_i, 0, 3], C.test_seq_in_BBX5_dict[(win_i, subj_i_r)][0, f_i, 0, 4]

                top_left = (int(x_c - (w_c / 2)), int(y_c - (h_c / 2)))
                bottom_right = (int(x_c + (w_c / 2)), int(y_c + (h_c / 2)))
                # if C.vis_bbox_type_idx == 0:
                #     img = cv2.rectangle(img, top_left, bottom_right, subj_color, C.border_size)
                # elif C.vis_bbox_type_idx == 2:
                if 'PRED' in C.vis_bbox_type: border_size = C.border_size + 7
                else: border_size = C.border_size
                center = (int(x_c), int(y_c))
                
                # if C.vis_bbox_type_idx == 0:
                #     img = cv2.circle(img, center, 2, subj_color, C.border_size)
                # elif C.vis_bbox_type_idx == 2:
                # if 'PRED' in C.vis_bbox_type: border_size = C.border_size + 7
                # else: border_size = C.border_size

                if f_i == 0:
                    thickness = 10; subj_color_ = (240, 240, 240)
                    img = cv2.circle(img, center, thickness, subj_color_, border_size)
                    img = cv2.rectangle(img, top_left, bottom_right, subj_color_, border_size)
                else:
                    thickness = 2
                    img = cv2.circle(img, center, thickness, subj_color, border_size)
                    img = cv2.rectangle(img, top_left, bottom_right, subj_color, border_size)
        # GND BBX <<<

        # PRED BBX >>>
        if 'PRED' in C.vis_bbox_type:
            for f_i in range(start_frame, -1, -1): # recent frame to the first frame per win
                x_c, y_c, d_c = pred_BBX5_c[0, f_i, 0, 0], pred_BBX5_c[0, f_i, 0, 1],  pred_BBX5_c[0, f_i, 0, 2]
                w_c, h_c = pred_BBX5_c[0, f_i, 0, 3], pred_BBX5_c[0, f_i, 0, 4]

                top_left = (int(x_c - (w_c / 2)), int(y_c - (h_c / 2)))
                bottom_right = (int(x_c + (w_c / 2)), int(y_c + (h_c / 2)))
                center = (int(x_c), int(y_c))

                if f_i == 0:
                    thickness = 10; subj_color_PRED_ = (240, 240, 240)
                    img = cv2.circle(img, center, thickness, subj_color_PRED_, C.border_size)
                    img = cv2.rectangle(img, top_left, bottom_right, subj_color_PRED_, C.border_size)
                else:
                    thickness = 2
                    img = cv2.circle(img, center, thickness, subj_color_PRED, C.border_size)
                    img = cv2.rectangle(img, top_left, bottom_right, subj_color_PRED, C.border_size)
        # PRED BBX <<<
        return img

    # ----------
    #  Log Eval
    # ----------
    # Log >>>>>>>>>>>>>>>>
    eucl_dist_log_ls = []
    d_err_PRED_log_ls = []
    d_err_FTM_log_ls = []
    d_err_improvement_log_ls = []
    dtw_log_ls = []
    if C.minimum_required_frames: mfr_log_ls = []
    diou_loss_log_ls = []
    iou_log_ls = []
    AP_log_ls2 = [] # [.05:.95] with step .05 # 19
    threds = list(np.array(range(5, 100, 5)) / 100)
    print('\n threds: ', threds)
    # e.g. [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    for thred in threds: AP_log_ls2.append([])
    # AP_0p1_log_ls = []
    # AP_0p5_log_ls = []
    turning_tracklets_cnt, tracklets_total = 0, 0
    total_infer_time_sec = 0

    if C.model_cfg['model_type'] == 'kalman_filter' or C.model_cfg['model_type'] == 'kalman_filter_v2': fst_seq_in_BBX5_r_prev = None

    n_out_of_view = 0
    if C.minimum_required_frames:
        prev_pred_BBX5_c_ls = [[] for _ in range(C.n_subjects)]
        iou_per_frame = 1
        cumu_iou_ls_per_win = []
    MRF = 1
    proc_n_frames, proc_n_wins = 0, 0 # actual processed number of frames and windows
    MRF_win_i_ls, MRF_win_1st_ts16s = [], []
    # Log <<<<<<<<<<<<<<<<



    # ------------------------
    #  Iterate Over the Video
    # ------------------------
    # for win_i in range(C.n_wins):
    for win_i in range(0, C.n_wins, C.win_stride):
        
        ts16_dfv4 = C.RGBg_ts16_dfv4_ls[win_i] # C.RGB_ts16_dfv4_valid_ls[win_i]
        if ts16_dfv4 < C.seq_id_to_start_end_ts16_dfv4[C.seq_id]['start']:
            # phone_win_offset += 1
            continue
        elif ts16_dfv4 >= C.seq_id_to_start_end_ts16_dfv4[C.seq_id]['end']: break
        print('\n\n win_i: ', win_i, ', ts16_dfv4: ', ts16_dfv4)
        phone_win_i = win_i # + phone_win_offset
        if ts16_dfv4 in C.test_seq_subj_i_in_view_dict.keys():
            img_path = C.img_path + '/' + ts16_dfv4 + '_anonymized.jpg'
            C.img = cv2.imread(img_path)
            # print('utils.py - eval_recon_core - img.size: ', img.size)
            # print('utils.py - eval_recon_core - img.shape: ', img.shape) # e.g. (720, 1280, 3)
            if C.imu_ned_pos2:
                if win_i == 0 and C.imu_ned_pos2:
                    # >>> Initialize IMU_NED_imgs >>>
                    if win_i == 0:
                        for subj_i in range(C.n_subjects):
                            rows, cols = int(np.shape(C.img)[0]), int(np.shape(C.img)[1])
                            channels = np.shape(C.img)[2]
                            IMU_NED_img = np.zeros((rows, cols, channels), dtype=np.uint8)
                            C.IMU_NED_imgs.append(IMU_NED_img)
                            if C.vis_win_NED: C.IMU_NED_imgs_ORI.append(copy.deepcopy(C.IMU_NED_imgs[subj_i]))
                            C.IMU_NED_origins.append((int(cols / 2), int(rows / 2)))
                            C.IMU_NED_pos_scales.append(16)
                    # <<< Initialize IMU_NED_imgs <<<


                pre_BEV = copy.deepcopy(C.img)
                # >>> BEV >>>
                C.BEV, C.BEV_M, C.img_f_BEV, C.BEV_M_iv = psp_tf(pre_BEV, C) # BEV, BEV_M, img_f_BEV, BEV_M_iv
                print('\n C.BEV_M: ', C.BEV_M)
                # <<< BEV <<<

            if C.vis_eval and win_i > C.len_win:
                if C.imu_ned_pos2:
                    # >>> src 4 pnts >>>
                    for pnt_i in range(4):
                        # print(); print()
                        # print('C.BEV_src_pnts[pnt_i]: ', C.BEV_src_pnts[pnt_i])
                        C.img = cv2.circle(C.img, C.BEV_src_pnts[pnt_i], 4, C.color_dict[C.color_ls[-2]], 4)
                        C.img = cv2.putText(C.img, str(pnt_i), C.BEV_src_pnts[pnt_i], cv2.FONT_HERSHEY_SIMPLEX, 1, C.color_dict[C.color_ls[-2]], 2, cv2.LINE_AA)
                    # <<< src 4 pnts <<<

                    # >>> dst 4 pnts >>>
                    for pnt_i in range(4):
                        # print(); print()
                        # print('C.BEV_dst_pnts[pnt_i]: ', C.BEV_dst_pnts[pnt_i])
                        C.BEV = cv2.circle(C.BEV, C.BEV_dst_pnts[pnt_i], 4, C.color_dict[C.color_ls[-2]], 4)
                        C.BEV = cv2.putText(C.BEV, str(pnt_i), C.BEV_dst_pnts[pnt_i], cv2.FONT_HERSHEY_SIMPLEX, 1, C.color_dict[C.color_ls[-2]], 2, cv2.LINE_AA)
                    # <<< dst 4 pnts <<<

                # cv2.imshow('img', C.img); cv2.waitKey(0)
                # cv2.imshow('BEV', C.BEV); cv2.waitKey(0)
                # img = Image.open(img_path)
                print('\n\n img_path: ', img_path)

            test_seq_subj_i_in_view_ls_ = C.test_seq_subj_i_in_view_dict[ts16_dfv4]
            print('\n\n test_seq_subj_i_in_view_ls_: ', test_seq_subj_i_in_view_ls_) # e.g. [1, 2, 3, 4]

            n = len(test_seq_subj_i_in_view_ls_)
            if n > 0:
                # C.ts16_dfv4_subj_i_to_BBX5_prime[ts16_dfv4] = defaultdict()

                # MRF
                if C.minimum_required_frames and iou_per_frame < C.iou_per_frame_thred:
                    MRF += 1
                    MRF_win_i_ls.append(win_i)
                    MRF_win_1st_ts16s.append(ts16_dfv4)

                # Some initializations
                cumu_iou_ls_per_win = []
                
                # --------------------------------------------------------
                #  Iterate Over All Sujects Present in the Current Window
                # --------------------------------------------------------
                for r_i, subj_i_r in enumerate(test_seq_subj_i_in_view_ls_):
                    tracklets_total += 1
                    dist_Cam_row, dist_Phone_row = [], []

                    print('\n np.shape(C.test_seq_in_BBX5_dict[(phone_win_i, subj_i_r)]): ', np.shape(C.test_seq_in_BBX5_dict[(phone_win_i, subj_i_r)]))
                    # e.g. (1, 10, 1, 5) # ORI

                    seq_in_BBX5_r = torch.from_numpy(C.test_seq_in_BBX5_dict[(phone_win_i, subj_i_r)])
                    print('\n np.shape(seq_in_BBX5_r): ', np.shape(seq_in_BBX5_r))
                    # e.g. torch.Size([1, 10, 1, 5])
                    # seq_in_BBX5_r = torch.unsqueeze(seq_in_BBX5_video[win_i, :, :, :], 0)
                    # seq_in_BBX5_r = torch.unsqueeze(seq_in_BBX5_video[phone_win_i, :, :, :], 0)
                    # print('\n\n utils.py - eval_recon_core() - np.shape(seq_in_BBX5_r): ', np.shape(seq_in_BBX5_r))
                    # e.g.utils.py - eval_recon_core() - np.shape(seq_in_BBX5_r):  torch.Size([1, 10, 1, 5])

                    # if subj_i_r in range(C.n_subjects):
                    #     seq_in_BBX5_r = C.test_seq_in_BBX5_dict[(win_i, subj_i_r)]
                    # else:
                    #     seq_in_BBX5_r = C.seq_in_BBX5_Others_dict[(win_i, subj_i_r)]
                    #     # print(); print() # debug
                    #     # print('win_i, subj_i_r: ', win_i, subj_i_r)
                    #     # print('seq_in_BBX5_r: ', seq_in_BBX5_r)

                    seq_in_FTM2_c = torch.from_numpy(C.test_seq_in_FTM2_dict[(phone_win_i, subj_i_r)])
                    seq_in_IMU19_c = torch.from_numpy(C.test_seq_in_IMU19_dict[(phone_win_i, subj_i_r)])
                    print('\n\n utils.py - eval_recon_core() - np.shape(seq_in_FTM2_c): ', np.shape(seq_in_FTM2_c))
                    print('\n\n utils.py - eval_recon_core() - np.shape(seq_in_IMU19_c): ', np.shape(seq_in_IMU19_c))
                    '''
                    utils.py - eval_recon_core() - np.shape(seq_in_FTM2_c):  torch.Size([1, 10, 1, 2])
                    utils.py - eval_recon_core() - np.shape(seq_in_IMU19_c):  torch.Size([1, 10, 1, 19])
                    '''

                    # print('\n sys.getsizeof(C.test_seq_in_IMU19_dict[(phone_win_i, subj_i_r)]): ', sys.getsizeof(C.test_seq_in_IMU19_dict[(phone_win_i, subj_i_r)]]))

                    # Filter Tc5 by cumulative_turning_angle >>>
                    if C.eval_with_cumulative_turning_angle:
                        seq_in_BBX5_r_angles = seq_in_BBX5_r.clone()
                        seq_in_BBX5_r_angles[:, :, :, 0] /= C.cols
                        seq_in_BBX5_r_angles[:, :, :, 1] /= C.rows
                        seq_in_BBX5_r_angles[:, :, :, 2] /= C.max_depths[2]
                        prev_angle, prev_vec, cumu_angles = 0, None, 0
                        for f_i in range(C.len_win):
                            vec = torch.squeeze(torch.squeeze(seq_in_BBX5_r_angles[:, f_i, :, :3] - seq_in_BBX5_r_angles[:, f_i - 1, :, :3]))
                            if f_i > 1:
                                # Ref: https://stackoverflow.com/questions/2827393/angles-between-two-n-dimensional-vectors-in-python
                                c = dot(vec, prev_vec) / norm(vec) / norm(prev_vec) # -> cosine of the angle
                                angle_ = arccos(clip(c, -1, 1))
                                angle = angle_ * 180 / math.pi
                                cumu_angles += angle
                                print('angle: ', angle)
                                prev_angle = angle
                            prev_vec = vec
                        print('cumu_angles: ', cumu_angles)
                        print('C.cumu_turning_angle: ', C.cumu_turning_angle)
                        if cumu_angles > C.cumu_turning_angle:
                            turning_tracklets_cnt += 1
                            print('cumu_angles: ', cumu_angles)
                        else:
                            continue

                    # Filter Tc5 by cumulative_turning_angle <<<
                    if C.imu_ned_pos2:
                        # seq_in_BBX5_r, seq_in_FTM2_c, seq_in_IMU_NED_pos2 = test_seq_in_correspondence
                        seq_in_IMU_NED_pos2_c = C.test_seq_in_IMU_NED_pos2_dict[(phone_win_i, subj_i_r)]
                        seq_in_IMU_NED_pos2_c = np.array([[-pos[0], pos[1]] for pos in seq_in_IMU_NED_pos2_c]) # negate x
                        if len(seq_in_IMU_NED_pos2_c) == C.len_win:
                            C.seq_in_IMU_NED_pos2_dfv4bl_ls_dfv3_ls.append(seq_in_IMU_NED_pos2_c)
                            C.seq_in_IMU_NED_pos2_dict[(win_i, subj_i)] = np.expand_dims(seq_in_IMU_NED_pos2_c, axis=0)
                        seq_in_IMU_NED_pos2_c = torch.from_numpy(C.test_seq_in_IMU_NED_pos2_dict[(phone_win_i, subj_i_r)])
                        # DEBUG:
                        print('\n\n utils.py - eval_recon_core() - np.shape(seq_in_BBX5_r): ', np.shape(seq_in_BBX5_r))
                        print('\n\n utils.py - eval_recon_core() - np.shape(seq_in_FTM2_c): ', np.shape(seq_in_FTM2_c))
                        print('\n\n utils.py - eval_recon_core() - np.shape(seq_in_IMU_NED_pos2_c): ', np.shape(seq_in_IMU_NED_pos2_c))
                        '''
                        e.g.
                        utils.py - eval_recon_core() - np.shape(seq_in_BBX5_r):  torch.Size([1, 10, 1, 5])
                        utils.py - eval_recon_core() - np.shape(seq_in_FTM2_c):  torch.Size([1, 10, 1, 2])
                        utils.py - eval_recon_core() - np.shape(seq_in_IMU_NED_pos2_c):  torch.Size([1, 10, 1, 2])
                        '''
                        # shape(torch.squeeze(torch.squeeze(seq_in_BBX5_, dim=0), dim=1)): (10, 5)
                        seq_in_BEV_Feet_Center_ = [psp_tf_pnt((int(BBX5_[0]), int(BBX5_[1] + int(BBX5_[4] / 2))), C, 'Cam2BEV') \
                                                    for BBX5_ in torch.squeeze(torch.squeeze(seq_in_BBX5_r, dim=0), dim=1)]

                        print('\n\n np.shape(seq_in_BEV_Feet_Center_): ', np.shape(seq_in_BEV_Feet_Center_)) # e.g. (10, 2)
                        seq_in_IMU_NED_pos2_c_sqz = torch.squeeze(torch.squeeze(seq_in_IMU_NED_pos2_c, dim=0), dim=1).numpy()
                        print('\n\n np.shape(seq_in_IMU_NED_pos2_c_sqz): ', np.shape(seq_in_IMU_NED_pos2_c_sqz)) # e.g. (10, 2)
                        seq_in_IMU_NED_pos2_c_calib = np.zeros(np.shape(seq_in_BEV_Feet_Center_))
                        for i in range(C.len_win):
                            if i == 0:
                                seq_in_IMU_NED_pos2_c_calib[i] = seq_in_BEV_Feet_Center_[0]
                            else:
                                # DEBUG:
                                # print('\n seq_in_BEV_Feet_Center_[0]: ', seq_in_BEV_Feet_Center_[0])
                                # print('\n C.BEV_px2meters_ratio: ', C.BEV_px2meters_ratio)
                                # print('\n diff: ', seq_in_IMU_NED_pos2_c_sqz[i] - seq_in_IMU_NED_pos2_c_sqz[i - 1])
                                # print('\n scaled diff: ', (seq_in_IMU_NED_pos2_c_sqz[i] - seq_in_IMU_NED_pos2_c_sqz[i - 1]) * C.BEV_px2meters_ratio)
                                seq_in_IMU_NED_pos2_c_calib_diff = (seq_in_IMU_NED_pos2_c_sqz[i] - seq_in_IMU_NED_pos2_c_sqz[i - 1]) * C.BEV_px2meters_ratio
                                seq_in_IMU_NED_pos2_c_calib[i] = seq_in_IMU_NED_pos2_c_calib[i - 1] + seq_in_IMU_NED_pos2_c_calib_diff
                        print('\n\n np.shape(seq_in_IMU_NED_pos2_c_calib): ', np.shape(seq_in_IMU_NED_pos2_c_calib)) # e.g. torch.Size([10, 2])
                        print('\n\n seq_in_IMU_NED_pos2_c_calib: ', seq_in_IMU_NED_pos2_c_calib)

                        # Debug >>>
                        # img_f_BEV_pos2_ = [psp_tf_pnt(IMU_NED_pos2_, C, 'BEV2Cam') \
                        #                             for IMU_NED_pos2_ in torch.squeeze(torch.squeeze(seq_in_IMU_NED_pos2_c, dim=0), dim=1)]
                        img_f_BEV_pos2_ = [psp_tf_pnt(IMU_NED_pos2_, C, 'BEV2Cam') \
                                                    for IMU_NED_pos2_ in seq_in_IMU_NED_pos2_c_calib]
                        # Debug <<<

                        # Visualize BBX >>>
                        if C.vis and C.vis_eval and win_i > C.len_win:
                            C.img = vis_tracklet('Cam', C.img, seq_in_BBX5_r, subj_i_r, seq_in_IMU_NED_pos2_c, C)
                            C.BEV = vis_tracklet('BEV', C.BEV, seq_in_BBX5_r, subj_i_r, seq_in_BEV_Feet_Center_, C)
                            C.img_f_BEV = vis_tracklet('img_f_BEV', C.img_f_BEV, seq_in_BBX5_r, subj_i_r, img_f_BEV_pos2_, C)
                            if C.vis_PDR_BEV:
                                C.BEV = vis_tracklet('PDR_BEV', C.BEV, seq_in_IMU_NED_pos2_c, subj_i_r, seq_in_BEV_Feet_Center_, C) # DEBUG

                            print('\n\n len(C.IMU_NED_imgs): ', len(C.IMU_NED_imgs))
                            if C.vis_win_NED:
                                C.IMU_NED_imgs[subj_i_r] = copy.deepcopy(C.IMU_NED_imgs_ORI[subj_i_r])
                            IMU_NED_win, C.IMU_NED_imgs[subj_i_r], C = vis_tracklet('IMU_NED', C.IMU_NED_imgs[subj_i_r], seq_in_IMU_NED_pos2_c, subj_i_r, seq_in_FTM2_c, C)
                            C.IMU_NED_range_imgs.append(IMU_NED_win)
                        # Visualize BBX <<<

                    else:
                        if C.imu200: seq_in_correspondence = [seq_in_BBX5_r, seq_in_FTM2_c, seq_in_IMUaccgym9_200_c]
                        else: seq_in_correspondence = [seq_in_BBX5_r, seq_in_FTM2_c, seq_in_IMU19_c]
                        test_seq_in_correspondence = prepare_XT_data(C, seq_in_correspondence, 'test')
                        seq_in_IMU_XT_c = test_seq_in_correspondence[-1]

                        if C.normalization:
                            pipeline = [Preprocess4Normalization(C.IMU_XT_dim)] #, Preprocess4Mask(C.mask_cfg)]
                        else: pipeline = None
                        test_data_set = LIBERTDataset4Pretrain(test_seq_in_correspondence, pipeline=pipeline, correspondence=True).proc_dataset()

                        seq_in_BBX5_r = np.expand_dims(test_data_set[0][0], axis=0)
                        seq_in_FTM2_c = np.expand_dims(test_data_set[0][1], axis=0)
                        seq_in_IMU_XT_c = np.expand_dims(test_data_set[0][2], axis=0)
                        print('\n\n utils.py - eval_recon_core() - after pipeline - np.shape(seq_in_BBX5_r): ', np.shape(seq_in_BBX5_r))
                        print('\n\n utils.py - eval_recon_core() - after pipeline - np.shape(seq_in_FTM2_c): ', np.shape(seq_in_FTM2_c))
                        print('\n\n utils.py - eval_recon_core() - after pipeline - np.shape(seq_in_IMU_XT_c): ', np.shape(seq_in_IMU_XT_c))
                        # e.g. np.shape(seq_in_BBX5_r): (1, len_win, 1, 5)

                        print('\n\n utils.py - eval_recon_core() - after pipeline - getsizeof(seq_in_BBX5_r): ', getsizeof(seq_in_BBX5_r))
                        print('\n\n utils.py - eval_recon_core() - after pipeline - getsizeof(seq_in_FTM2_c): ', getsizeof(seq_in_FTM2_c))
                        print('\n\n utils.py - eval_recon_core() - after pipeline - getsizeof(seq_in_IMU_XT_c): ', getsizeof(seq_in_IMU_XT_c))
                        print('\n\n utils.py - eval_recon_core() - after pipeline - seq_in_BBX5_r.__sizeof__(): ', seq_in_BBX5_r.__sizeof__())
                        print('\n\n utils.py - eval_recon_core() - after pipeline - seq_in_FTM2_c.__sizeof__(): ', seq_in_FTM2_c.__sizeof__())
                        print('\n\n utils.py - eval_recon_core() - after pipeline - seq_in_IMU_XT_c.__sizeof__(): ', seq_in_IMU_XT_c.__sizeof__())

                    if C.vis_eval and win_i > C.len_win:
                        print('\n\n utils.py - eval_recon_core() - after pipeline - getsizeof(img): ', getsizeof(C.img))
                        print('\n\n utils.py - eval_recon_core() - after pipeline - img.__sizeof__(): ', C.img.__sizeof__())
                    '''
                    utils.py - eval_recon_core() - after pipeline - np.shape(seq_in_BBX5_r):  torch.Size([1, 10, 1, 5])
                    utils.py - eval_recon_core() - after pipeline - np.shape(seq_in_FTM2_c):  torch.Size([1, 10, 1, 2])
                    utils.py - eval_recon_core() - after pipeline - np.shape(seq_in_IMU_XT_c):  torch.Size([1, 10, 1, 9])
                    '''
                    print('\n\n utils.py - eval_recon_core() - after pipeline - seq_in_FTM2_c: ', seq_in_FTM2_c)
                    # print('\n\n utils.py - eval_recon_core() - after pipeline - seq_in_IMU_XT_c: ', seq_in_IMU_XT_c)
                    # print('\n\n C.model_cfg[model_type]: ', C.model_cfg['model_type'])

                    # ==================
                    #  Mask Out-of-view
                    if C.mask_out_of_view: M = C.M * (C.M // 2)
                    else: M = 1
                    print('\n C.seen_left_borders: ', C.seen_left_borders)
                    print('\n C.seen_right_borders: ', C.seen_right_borders)
                    print('\n C.seen_top_borders: ', C.seen_top_borders)
                    print('\n C.seen_bottom_borders: ', C.seen_bottom_borders)

                    # First BBX
                    x, y = seq_in_BBX5_r[:, 0, :, 0][0][0], seq_in_BBX5_r[:, 0, :, 1][0][0]
                    # ==========================================
                    #  Iterate over all masked out-of-views >>>
                    for m_i in range(M):
                        # print('\n np.shape(seq_in_BBX5_r): ', np.shape(seq_in_BBX5_r)) # e.g. (1, W, 1, 5)
                        # print('\n seq_in_BBX5_r[:, :, :, 0]: ', seq_in_BBX5_r[:, :, :, 0]) # x, col
                        # print('\n seq_in_BBX5_r[:, :, :, 1]: ', seq_in_BBX5_r[:, :, :, 1]) # y, row
                        # print('\n seq_in_BBX5_r[:, 0, :, 0][0][0]: ', seq_in_BBX5_r[:, 0, :, 0][0][0])

                        print('\n M :', M, ', m_i: ', m_i, ', x: ', x, ', y: ', y)
                        print('\n C.seen_left_borders[m_i]: ', C.seen_left_borders[m_i])
                        print('\n C.seen_right_borders[m_i]: ', C.seen_right_borders[m_i])
                        print('\n C.seen_top_borders[m_i]: ', C.seen_top_borders[m_i])
                        print('\n C.seen_bottom_borders[m_i]: ', C.seen_bottom_borders[m_i])
                        print('\n win_i: ', win_i, ', C.n_wins: ', C.n_wins)

                        # Out of view
                        if not (x >= C.seen_left_borders[m_i] and x < C.seen_right_borders[m_i] and y >= C.seen_top_borders[m_i] and y < C.seen_bottom_borders[m_i]):
                            continue
                        # ================
                        # In the view >>>
                        else:
                            n_out_of_view += 1
                            if C.broken_trajectories:
                                # ===================================
                                # Has next frame and in the view >>>
                                if phone_win_i + C.len_win < C.n_wins and \
                                (phone_win_i + C.len_win, subj_i_r) in C.test_seq_in_BBX5_dict:
                                    seq_in_BBX5_next_r = torch.from_numpy(C.test_seq_in_BBX5_dict[(phone_win_i + C.len_win, subj_i_r)])
                                    # print('\n\n np.shape(seq_in_BBX5_next_r): ', np.shape(seq_in_BBX5_next_r)) # e.g. torch.Size([1, 10, 1, 5])
                                    x_next, y_next = seq_in_BBX5_next_r[:, 0, :, 0][0][0], seq_in_BBX5_next_r[:, 0, :, 1][0][0]

                                    # In the view
                                    if x_next >= C.seen_left_borders[m_i] and x_next < C.seen_right_borders[m_i] and y_next >= C.seen_top_borders[m_i] and y_next < C.seen_bottom_borders[m_i]:
                                        proceed = False
                                    # Has next frame and in the view <<<
                                    # ===================================
                                    else: proceed = True # Has next frame but out of view
                                else: continue # No next frame. End of the video.
                            else: proceed = True

                            if proceed:
                                # Our model >>>
                                if C.model_cfg['model_type'] == 'transformer' or C.model_cfg['model_type'] == 'lstm':
                                    if not torch.is_tensor(seq_in_BBX5_r): seq_in_BBX5_r = torch.from_numpy(seq_in_BBX5_r)
                                    if not torch.is_tensor(seq_in_FTM2_c): seq_in_FTM2_c = torch.from_numpy(seq_in_FTM2_c)
                                    if not torch.is_tensor(seq_in_IMU_XT_c): seq_in_IMU_XT_c = torch.from_numpy(seq_in_IMU_XT_c)
                                    # seq_in_BBX5_r = seq_in_BBX5_r.fillna(seq_in_BBX5_r.mean())
                                    # seq_in_FTM2_c = seq_in_FTM2_c.fillna(seq_in_FTM2_c.mean())
                                    # seq_in_IMU_XT_c = seq_in_IMU_XT_c.fillna(seq_in_IMU_XT_c.mean())

                                    with torch.no_grad(): # evaluation without gradient calculation
                                        start_time = time.time()
                                        if reconstruction_path == 'IemcC2C' or reconstruction_path == 'IFcC2C':
                                            BBX5 = seq_in_BBX5_r.clone(); BBX5.to(device)
                                            
                                            if C.model_cfg['transformer_Tc_in'] == 'fstBBX5':
                                                # MRF >>>
                                                if C.minimum_required_frames:
                                                    if iou_per_frame >= C.iou_per_frame_thred and \
                                                        win_i > 0 and prev_pred_BBX5_c_ls[subj_i_r] is not None and len(prev_pred_BBX5_c_ls[subj_i_r]) != 0:
                                                        # print('\n prev_pred_BBX5_c_ls[subj_i_r]: ', prev_pred_BBX5_c_ls[subj_i_r])
                                                        print('\n\n np.shape(prev_pred_BBX5_c_ls[subj_i_r]): ', np.shape(prev_pred_BBX5_c_ls[subj_i_r]))
                                                        prev_pred_BBX = prev_pred_BBX5_c_ls[subj_i_r].clone(); prev_pred_BBX5_c_ls[subj_i_r].to(device) # torch.Size([1, 10, 1, 5])
                                                        BBX5[:, 0, :, :] = prev_pred_BBX[:, -1, :, :] # The 1st frame in the current window should be the last frame from prev_pred_BBX5_c_ls[subj_i_r]
                                                        batch = (BBX5, seq_in_FTM2_c, seq_in_IMU_XT_c) # to feed the model
                                                    else: batch = (seq_in_BBX5_r, seq_in_FTM2_c, seq_in_IMU_XT_c)
                                                # MRF <<<
                                                else: batch = (seq_in_BBX5_r, seq_in_FTM2_c, seq_in_IMU_XT_c)
                                                
                                            elif C.model_cfg['transformer_Tc_in'] == 'flBBX5':
                                                if phone_win_i + C.len_win < C.n_wins:
                                                    if (phone_win_i + C.len_win, subj_i_r) in C.test_seq_in_BBX5_dict:
                                                        seq_in_BBX5_next_r = torch.from_numpy(C.test_seq_in_BBX5_dict[(phone_win_i + C.len_win, subj_i_r)])
                                                        # print('\n\n np.shape(seq_in_BBX5_next_r): ', np.shape(seq_in_BBX5_next_r)) # e.g. torch.Size([1, 10, 1, 5])
                                                    else: seq_in_BBX5_next_r = seq_in_BBX5_r.clone()
                                                else: seq_in_BBX5_next_r = seq_in_BBX5_r.clone()

                                                seq_in_BBX5_r_batch = seq_in_BBX5_r.clone()
                                                # Replace the last frame of the current BBX5 with the first frame in the next window
                                                seq_in_BBX5_r_batch[:, -1, :, :] = seq_in_BBX5_next_r[:, 0, :, :]
                                                batch = (seq_in_BBX5_r_batch, seq_in_FTM2_c, seq_in_IMU_XT_c)

                                            batch = [t.to(device) for t in batch]
                                            print('\n utils.py - eval_recon_core() - with torch.no_grad() - np.shape(batch): ', np.shape(batch))
                                            print('\n utils.py - eval_recon_core() - with torch.no_grad() - np.shape(batch[0]): ', np.shape(batch[0]))
                                            print('\n utils.py - eval_recon_core() - with torch.no_grad() - np.shape(batch[1]): ', np.shape(batch[1]))
                                            print('\n utils.py - eval_recon_core() - with torch.no_grad() - np.shape(batch[2]): ', np.shape(batch[2]))
                                            start_time = time.time()
                                            BBX5_PRED = func_forward(model, batch, C.model_cfg['transformer_Tc_in'])
                                            end_time = time.time()
                                            inference_time = end_time - start_time
                                            print('\n utils.py - eval_recon_core() - with torch.no_grad() - inference_time: ', inference_time)
                                            total_infer_time_sec += inference_time
                                            # print('\n utils.py - eval_recon_core() - with torch.no_grad() - np.shape(BBX5): ', np.shape(BBX5))
                                            # print('\n utils.py - eval_recon_core() - with torch.no_grad() - BBX5: ', BBX5)
                                            # print('\n utils.py - eval_recon_core() - with torch.no_grad() - np.shape(BBX5_PRED): ', np.shape(BBX5_PRED))
                                            # print('\n utils.py - eval_recon_core() - with torch.no_grad() - BBX5_PRED: ', BBX5_PRED)

                                            # print('\n utils.py - eval_recon_core() - with torch.no_grad() - np.shape(seq_in_IMU_XT_c): ', np.shape(seq_in_IMU_XT_c))
                                            # print('\n utils.py - eval_recon_core() - with torch.no_grad() - seq_in_IMU_XT_c: ', seq_in_IMU_XT_c)
                                            # print('------------------------------------------------------------------------')

                                    # seq_in_BBX5_r = BBX5.cpu()
                                    pred_BBX5_c = BBX5_PRED.cpu()
                                    print('\n\n utils.py - eval_recon_core() - after squeezing for baseline  - np.shape(pred_BBX5_c): ', np.shape(pred_BBX5_c))
                                    # utils.py - eval_recon_core() - after squeezing for baseline  - np.shape(pred_BBX5_c):  torch.Size([1, 10, 1, 5])

                                    # dBBX5 >>>
                                    d_pred_BBX5_c = BBX5_to_dBBX5(C, pred_BBX5_c, 1)
                                    if C.minimum_required_frames:
                                        if iou_per_frame >= C.iou_per_frame_thred and \
                                            win_i > 0 and prev_pred_BBX5_c_ls[subj_i_r] is not None and len(prev_pred_BBX5_c_ls[subj_i_r]) != 0:
                                            pred_BBX5_c[0, 0, :, :] = BBX5[0, 0, :, :] # equivalent to prev_pred_BBX5_c_ls[subj_i_r][0, -1, :, :] # 1st frame bbox in the current is the bbox in the last frame of the last window
                                        else: pred_BBX5_c[0, 0, :, :] = seq_in_BBX5_r[0, 0, :, :]
                                    else: pred_BBX5_c[0, 0, :, :] = seq_in_BBX5_r[0, 0, :, :]
                                    for f_i in range(1, C.len_win):
                                        pred_BBX5_c[0, f_i, :, :] = pred_BBX5_c[0, f_i - 1, :, :] + d_pred_BBX5_c[0, f_i, :, :]
                                    # dBBX5 <<<

                                    # print('\n\n img_path: ', img_path)
                                    print('\n\n ts16_dfv4: ', ts16_dfv4)
                                    print('\n\n C.vis_eval: ', C.vis_eval)
                                    print('\n\n C.vis_bbox_type: ', C.vis_bbox_type)
                                    print('\n\n C.vis_bbox_len: ', C.vis_bbox_len)
                                    print('\n\n np.shape(seq_in_BBX5_r[0, 0, :, :]): ', np.shape(seq_in_BBX5_r[0, 0, :, :]))
                                    print('\n\n seq_in_BBX5_r[0, 0, :, :]: ', seq_in_BBX5_r[0, 0, :, :])
                                    print()

                                    fst_frame_nonzero = False
                                    for i in range(5):
                                        if torch.is_nonzero(seq_in_BBX5_r[0, 0, :, i]):
                                            fst_frame_nonzero = True
                                            break

                                    vis_len_win = min(C.len_win, C.vis_bbox_len)
                                    if C.vis_eval and win_i > C.len_win and fst_frame_nonzero:
                                        # print('\n\n np.shape(seq_in_BBX5_r): ', np.shape(seq_in_BBX5_r))
                                        # print('\n\n np.shape(BBX5): ', np.shape(BBX5))
                                        # print('\n\n np.shape(pred_BBX5_c): ', np.shape(pred_BBX5_c))
                                        # All three: torch.Size([1, 10, 1, 5])
                                        # if C.minimum_required_frames and iou_per_frame >= C.iou_per_frame_thred and \
                                        #     win_i > 0 and prev_pred_BBX5_c_ls[subj_i_r] is not None and len(prev_pred_BBX5_c_ls[subj_i_r]) != 0:
                                        #     prev_pred_BBX5_c = prev_pred_BBX5_c_ls[subj_i_r]
                                        #     img = print_bbox_win(C, C.img, BBX5, pred_BBX5_c, vis_len_win)
                                        # else:
                                        #     img = print_bbox_win(C, img, seq_in_BBX5_r, pred_BBX5_c, vis_len_win)
                                        img = print_bbox_win(C, C.img, seq_in_BBX5_r, pred_BBX5_c, vis_len_win)

                                    # Update prev_pred_BBX5_c_ls
                                    if C.minimum_required_frames: prev_pred_BBX5_c_ls[subj_i_r] = copy.deepcopy(pred_BBX5_c)
                                    # (win, frame, sample, feature)

                                    # -----------------------
                                    #  Evaluation in one win
                                    eucl_dist = np.linalg.norm(seq_in_BBX5_r.clone().numpy() - pred_BBX5_c.clone().numpy())
                                    eucl_dist_log_ls.append(eucl_dist)

                                    # Depth >>>
                                    d_GND = torch.squeeze(seq_in_BBX5_r[:,:,:,2])
                                    zero_indices_GND = torch.tensor(np.where(d_GND == 0)[0])
                                    zeros = torch.zeros(len(zero_indices_GND))
                                    d_PRED = torch.squeeze(pred_BBX5_c[:,:,:,2])
                                    d_FTM = torch.squeeze(seq_in_FTM2_c[:,:,:,0] / 1000).float()
                                    # DEBUG:
                                    # print('\n\n utils.py - eval_recon_core - zero_indices_GND: ', zero_indices_GND)
                                    # print('\n\n utils.py - eval_recon_core - zeros: ', zeros)
                                    # print('\n\n utils.py - eval_recon_core - np.shape(d_GND): ', np.shape(d_GND))
                                    # print('\n\n utils.py - eval_recon_core - np.shape(d_PRED): ', np.shape(d_PRED))
                                    # print('\n\n utils.py - eval_recon_core - np.shape(d_FTM): ', np.shape(d_FTM))
                                    # print('\n\n utils.py - eval_recon_core - d_GND: ', d_GND)
                                    # print('\n\n utils.py - eval_recon_core - d_PRED: ', d_PRED)
                                    # print('\n\n utils.py - eval_recon_core - d_FTM: ', d_FTM)
                                    # print('\n\n utils.py - eval_recon_core - d_PRED.put_(zero_indices_GND, zeros): ', d_PRED.put_(zero_indices_GND, zeros))
                                    # print('\n\n utils.py - eval_recon_core - d_FTM.put_(zero_indices_GND, zeros): ', d_FTM.put_(zero_indices_GND, zeros))
                                    # e.g. torch.Size([50])
                                    d_err_PRED = abs(d_PRED.put_(zero_indices_GND, zeros) - d_GND)
                                    d_err_FTM = abs(d_FTM.put_(zero_indices_GND, zeros) - d_GND)
                                    d_err_improvement = d_err_FTM - d_err_PRED
                                    # print('\n\n utils.py - eval_recon_core - d_err_PRED: ', d_err_PRED)
                                    # print('\n\n utils.py - eval_recon_core - d_err_FTM: ', d_err_FTM)
                                    print('\n\n utils.py - eval_recon_core - d_err_improvement: ', d_err_improvement)

                                    # DTW
                                    # alignment = dtw(torch.squeeze(seq_in_BBX5_r[:, :, :, :2]), torch.squeeze(pred_BBX5_c[:, :, :, :2]), keep_internals=True) # edited
                                    # print('\n\n dir(alignment): ', dir(alignment))
                                    # print('\n\n alignment: ', alignment)
                                    # print('\n\n alignment.distance: ', alignment.distance)

                                    # Log
                                    d_err_PRED_log_ls.append(list(d_err_PRED))
                                    d_err_FTM_log_ls.append(list(d_err_FTM))
                                    d_err_improvement_log_ls.append(list(d_err_improvement))
                                    # Depth <<<

                                    # DTW >>>
                                    # alignment = dtw(torch.squeeze(seq_in_BBX5_r[:, :, :, :2]), torch.squeeze(pred_BBX5_c[:, :, :, :2]), keep_internals=True) # edited
                                    # print('\n\n dir(alignment): ', dir(alignment))
                                    # print('\n\n alignment: ', alignment)
                                    # dtw_log_ls.append(alignment.distance) # edited
                                    dtw_log_ls.append(0)
                                    # print('\n\n alignment.distance: ', alignment.distance)
                                    # DTW <<<

                                    # BBX >>>
                                    seq_in_BBX5_r_squeezed = torch.squeeze(seq_in_BBX5_r)
                                    pred_BBX5_c_squeezed = torch.squeeze(pred_BBX5_c)
                                    # print('\n\n utils.py - eval_recon_core - np.shape(seq_in_BBX5_r): ', np.shape(seq_in_BBX5_r))
                                    # print('\n\n utils.py - eval_recon_core - np.shape(pred_BBX5_c): ', np.shape(pred_BBX5_c))
                                    # print('\n\n utils.py - eval_recon_core - np.shape(seq_in_BBX5_r_squeezed): ', np.shape(seq_in_BBX5_r_squeezed))
                                    # print('\n\n utils.py - eval_recon_core - np.shape(pred_BBX5_c_squeezed): ', np.shape(pred_BBX5_c_squeezed))
                                    # e.g. utils.py - eval_recon_core - np.shape(pred_BBX5_c):  torch.Size([1, 50, 1, 5])
                                    # e.g. utils.py - eval_recon_core - np.shape(pred_BBX5_c_squeezed):  torch.Size([50, 5])

                                    pred_BBX5_c_squeezed = pred_BBX5_c_squeezed.put_(zero_indices_GND, zeros)
                                    # print('\n\n utils.py - eval_recon_core - pred_BBX5_c_squeezed: ', pred_BBX5_c_squeezed)

                                    # seq_in_BBX5_r_a, pred_BBX5_c_b = labels_to_boxes(seq_in_BBX5_r, pred_BBX5_c)
                                    # print('\n\n utils.py - eval_recon_core - np.shape(seq_in_BBX5_r_a): ', np.shape(seq_in_BBX5_r_a))
                                    # print('\n\n utils.py - eval_recon_core - np.shape(pred_BBX5_c_b): ', np.shape(pred_BBX5_c_b))
                                    # # e.g. torch.Size([50, 4])
                                    #
                                    # diou_loss, iou = _diou_iou_loss(seq_in_BBX5_r_a, pred_BBX5_c_b, C.eps_denominator_iou)
                                    # print('\n\n utils.py - eval_recon_core - diou_loss: ', diou_loss)
                                    # print('\n\n utils.py - eval_recon_core - iou: ', iou)
                                    # # print('\n\n utils.py - eval_recon_core - np.shape(diou_loss): ', np.shape(diou_loss))
                                    # # print('\n\n utils.py - eval_recon_core - np.shape(iou): ', np.shape(iou))
                                    # '''
                                    # e.g.
                                    # utils.py - eval_recon_core - np.shape(loss_):  torch.Size([50])
                                    # utils.py - eval_recon_core - np.shape(iou):  torch.Size([50])
                                    # '''
                                    # # mse = mean_squared_error(seq_in_BBX5_r_a, pred_BBX5_c_b)
                                    # # print('\n\n utils.py - eval_recon_core - mse: ', mse)
                                # Our model <<<
                                # Baseline >>>
                                elif C.model_cfg['model_type'] == 'translator':
                                    # seq_in_BBX5_r = np.squeeze(seq_in_BBX5_r, axis=2) # Only squeeze it when feeding the model; Note expand_dim() later.
                                    seq_in_FTM2_c = np.squeeze(seq_in_FTM2_c, axis=2)
                                    seq_in_IMU_XT_c = np.squeeze(seq_in_IMU_XT_c, axis=2)
                                    # print('\n\n utils.py - eval_recon_core() - after squeezing for baseline - np.shape(seq_in_BBX5_r): ', np.shape(seq_in_BBX5_r))
                                    # print('\n\n utils.py - eval_recon_core() - after squeezing for baseline  - np.shape(seq_in_FTM2_c): ', np.shape(seq_in_FTM2_c))
                                    # print('\n\n utils.py - eval_recon_core() - after squeezing for baseline  - np.shape(seq_in_IMU_XT_c): ', np.shape(seq_in_IMU_XT_c))
                                    # e.g. (1, 10, 5) # The correct shape to feed ViTag Xtranslator

                                    start_time = time.time()
                                    if reconstruction_path == 'IemcC2C' or reconstruction_path == 'IFcC2C':
                                        batch = (seq_in_BBX5_r, seq_in_FTM2_c, seq_in_IMU_XT_c)
                                        BBX5 = batch[0]
                                        print('\n\n np.shape(BBX5): ', np.shape(BBX5))
                                        # np.shape(BBX5):  (1, 10, 1, 5)
                                        print('\n\n np.shape(seq_in_BBX5_r): ', np.shape(seq_in_BBX5_r))
                                        # np.shape(seq_in_BBX5_r):  (1, 10, 1, 5)
                                        # MRF >>>
                                        if C.minimum_required_frames: 
                                            if iou_per_frame >= C.iou_per_frame_thred and \
                                                win_i > 0 and prev_pred_BBX5_c_ls[subj_i_r] is not None and len(prev_pred_BBX5_c_ls[subj_i_r]) != 0:
                                                prev_pred_BBX = copy.deepcopy(prev_pred_BBX5_c_ls[subj_i_r]) # prev_pred_BBX5_c_ls[subj_i_r].to(device) # torch.Size([1, 10, 1, 5])
                                                print('\n\n np.shape(BBX5): ', np.shape(BBX5))
                                                print('\n\n np.shape(prev_pred_BBX5_c_ls[subj_i_r]): ', np.shape(prev_pred_BBX5_c_ls[subj_i_r])) # np.shape(prev_pred_BBX5_c_ls[subj_i_r]):  (1, 10, 1, 5)
                                                BBX5[:, 0, :, :] = prev_pred_BBX[:, -1, :, :] # # The 1st frame in the current window should be the last frame from prev_pred_BBX5_c_ls[subj_i_r]
                                                print('\n\n np.shape(BBX5): ', np.shape(BBX5))
                                                batch = (np.squeeze(BBX5, 2), seq_in_FTM2_c, seq_in_IMU_XT_c) # to feed the model
                                            else: batch = (np.squeeze(seq_in_BBX5_r, 2), seq_in_FTM2_c, seq_in_IMU_XT_c)
                                        # MRF <<<
                                        else: batch = (np.squeeze(seq_in_BBX5_r, 2), seq_in_FTM2_c, seq_in_IMU_XT_c)
                                        BBX5_PRED = func_forward(model, batch)

                                    pred_BBX5_c = BBX5_PRED
                                    
                                    # (win, frame, sample, feature)
                                    print('\n\n utils.py - eval_recon_core() - after squeezing for baseline  - np.shape(pred_BBX5_c): ', np.shape(pred_BBX5_c))
                                    # e.g. np.shape(pred_BBX5_c):  (1, 10, 1, 5)
                                    # seq_in_BBX5_r = np.expand_dims(seq_in_BBX5_r, axis=2) # No squeeze before
                                    seq_in_FTM2_c = np.expand_dims(seq_in_FTM2_c, axis=2)
                                    seq_in_IMU_XT_c = np.expand_dims(seq_in_IMU_XT_c, axis=2)
                                    # print('\n\n utils.py - eval_recon_core() - reshape  - np.shape(seq_in_BBX5_r): ', np.shape(seq_in_BBX5_r))

                                    # dBBX5 >>>
                                    d_pred_BBX5_c = BBX5_to_dBBX5(C, pred_BBX5_c, 1)
                                    if C.minimum_required_frames:
                                        if iou_per_frame >= C.iou_per_frame_thred and \
                                            win_i > 0 and prev_pred_BBX5_c_ls[subj_i_r] is not None and len(prev_pred_BBX5_c_ls[subj_i_r]) != 0:
                                            pred_BBX5_c[0, 0, :, :] = BBX5[0, 0, :, :] # equivalent to prev_pred_BBX5_c_ls[subj_i_r][0, -1, :, :] # 1st frame bbox in the current is the bbox in the last frame of the last window
                                        else: pred_BBX5_c[0, 0, :, :] = seq_in_BBX5_r[0, 0, :, :]
                                    else: pred_BBX5_c[0, 0, :, :] = seq_in_BBX5_r[0, 0, :, :]
                                    for f_i in range(1, C.len_win):
                                        pred_BBX5_c[0, f_i, :, :] = pred_BBX5_c[0, f_i - 1, :, :] + d_pred_BBX5_c[0, f_i, :, :]
                                    # dBBX5 <<<

                                    # print('\n\n img_path: ', img_path)
                                    print('\n\n ts16_dfv4: ', ts16_dfv4)
                                    print('\n\n C.vis_eval: ', C.vis_eval)
                                    print('\n\n C.vis_bbox_type: ', C.vis_bbox_type)
                                    print('\n\n C.vis_bbox_len: ', C.vis_bbox_len)
                                    # print('\n\n np.shape(seq_in_BBX5_r[0, 0, :, :]): ', np.shape(seq_in_BBX5_r[0, 0, :, :]))
                                    # print('\n\n seq_in_BBX5_r[0, 0, :, :]: ', seq_in_BBX5_r[0, 0, :, :])
                                    # print()

                                    fst_frame_nonzero = False
                                    for i in range(5):
                                        if not np.all(seq_in_BBX5_r[0, 0, :, i] == 0):
                                            fst_frame_nonzero = True
                                            break

                                    vis_len_win = min(C.len_win, C.vis_bbox_len)
                                    if C.vis_eval and win_i > C.len_win and fst_frame_nonzero:
                                        # img = print_bbox_win(C, img, seq_in_BBX5_r, pred_BBX5_c, vis_len_win)
                                        img = print_bbox_win(C, C.img, seq_in_BBX5_r, pred_BBX5_c, vis_len_win)

                                    # Update prev_pred_BBX5_c_ls
                                    if C.minimum_required_frames: prev_pred_BBX5_c_ls[subj_i_r] = copy.deepcopy(pred_BBX5_c)

                                    # -----------------------
                                    #  Evaluation in one win
                                    eucl_dist = np.linalg.norm(seq_in_BBX5_r - pred_BBX5_c)
                                    eucl_dist_log_ls.append(eucl_dist)

                                    # Depth >>>
                                    print('\n\n np.shape(seq_in_BBX5_r): ', np.shape(seq_in_BBX5_r))
                                    print('\n\n np.shape(pred_BBX5_c): ', np.shape(pred_BBX5_c))
                                    d_GND = np.squeeze(seq_in_BBX5_r[:,:,:,2])
                                    zero_indices_GND = np.array(np.where(d_GND == 0)[0])
                                    zeros = np.zeros(len(zero_indices_GND))
                                    d_PRED = np.squeeze(pred_BBX5_c[:,:,:,2])
                                    d_FTM = np.squeeze(seq_in_FTM2_c[:,:,:,0] / 1000)
                                    # DEBUG:
                                    # print('\n\n utils.py - eval_recon_core - zero_indices_GND: ', zero_indices_GND)
                                    # print('\n\n utils.py - eval_recon_core - zeros: ', zeros)
                                    # print('\n\n utils.py - eval_recon_core - np.shape(d_GND): ', np.shape(d_GND))
                                    # print('\n\n utils.py - eval_recon_core - np.shape(d_PRED): ', np.shape(d_PRED))
                                    # print('\n\n utils.py - eval_recon_core - np.shape(d_FTM): ', np.shape(d_FTM))
                                    # print('\n\n utils.py - eval_recon_core - d_GND: ', d_GND)
                                    # print('\n\n utils.py - eval_recon_core - d_PRED: ', d_PRED)
                                    # print('\n\n utils.py - eval_recon_core - d_FTM: ', d_FTM)
                                    # print('len(zero_indices_GND): ', len(zero_indices_GND))
                                    if len(zero_indices_GND) > 0:
                                        np.put(d_PRED, zero_indices_GND, zeros)
                                        np.put(d_FTM, zero_indices_GND, zeros)
                                        # print('\n\n utils.py - eval_recon_core - after np.put zeros - d_PRED: ', d_PRED)
                                        # print('\n\n utils.py - eval_recon_core - after np.put zeros - d_FTM: ', d_FTM)
                                    d_err_PRED = abs(d_PRED - d_GND)
                                    d_err_FTM = abs(d_FTM - d_GND)
                                    d_err_improvement = d_err_FTM - d_err_PRED
                                    # print('\n\n utils.py - eval_recon_core - d_err_PRED: ', d_err_PRED)
                                    # print('\n\n utils.py - eval_recon_core - d_err_FTM: ', d_err_FTM)
                                    # print('\n\n utils.py - eval_recon_core - d_err_improvement: ', d_err_improvement)

                                    # Log
                                    d_err_PRED_log_ls.append(list(d_err_PRED))
                                    d_err_FTM_log_ls.append(list(d_err_FTM))
                                    d_err_improvement_log_ls.append(list(d_err_improvement))
                                    # Depth <<<

                                    # BBX >>>
                                    seq_in_BBX5_r_squeezed = np.squeeze(seq_in_BBX5_r)
                                    pred_BBX5_c_squeezed = np.squeeze(pred_BBX5_c)
                                    # print('\n\n utils.py - eval_recon_core - np.shape(seq_in_BBX5_r): ', np.shape(seq_in_BBX5_r))
                                    # print('\n\n utils.py - eval_recon_core - np.shape(pred_BBX5_c): ', np.shape(pred_BBX5_c))
                                    # print('\n\n utils.py - eval_recon_core - np.shape(seq_in_BBX5_r_squeezed): ', np.shape(seq_in_BBX5_r_squeezed))
                                    # print('\n\n utils.py - eval_recon_core - np.shape(pred_BBX5_c_squeezed): ', np.shape(pred_BBX5_c_squeezed))
                                    # e.g. utils.py - eval_recon_core - np.shape(pred_BBX5_c):  torch.Size([1, 50, 1, 5])
                                    # e.g. utils.py - eval_recon_core - np.shape(pred_BBX5_c_squeezed):  torch.Size([50, 5])

                                    if len(zero_indices_GND) > 0:
                                        np.put(pred_BBX5_c_squeezed, zero_indices_GND, zeros)
                                    # print('\n\n utils.py - eval_recon_core - pred_BBX5_c_squeezed: ', pred_BBX5_c_squeezed)

                                    seq_in_BBX5_r = torch.from_numpy(seq_in_BBX5_r)
                                    pred_BBX5_c = torch.from_numpy(pred_BBX5_c)

                                # Baseline <<<
                                # PDR Baseline >>>
                                elif C.model_cfg['model_type'] == 'pdr':
                                    if not torch.is_tensor(seq_in_BBX5_r): seq_in_BBX5_r = torch.from_numpy(seq_in_BBX5_r)
                                    if not torch.is_tensor(seq_in_FTM2_c): seq_in_FTM2_c = torch.from_numpy(seq_in_FTM2_c)
                                    if not torch.is_tensor(seq_in_IMU_NED_pos2_c): seq_in_IMU_NED_pos2_c = torch.from_numpy(seq_in_IMU_NED_pos2_c)
                                    print('utilspy - eval_recon_core - np.shape(seq_in_BBX5_r): ', np.shape(seq_in_BBX5_r))
                                    print('utilspy - eval_recon_core - np.shape(seq_in_FTM2_c): ', np.shape(seq_in_FTM2_c))
                                    print('utilspy - eval_recon_core - np.shape(seq_in_IMU_NED_pos2_c): ', np.shape(seq_in_IMU_NED_pos2_c))
                                    '''
                                    e.g.
                                    utilspy - eval_recon_core - np.shape(seq_in_BBX5_r):  (1, 10, 1, 5)
                                    utilspy - eval_recon_core - np.shape(seq_in_FTM2_c):  (1, 10, 1, 2)
                                    utilspy - eval_recon_core - np.shape(seq_in_IMU_NED_pos2_c):  (1, 10, 1, 2)
                                    '''
                                    with torch.no_grad(): # evaluation without gradient calculation
                                        start_time = time.time()
                                        if reconstruction_path == 'IemcC2C' or reconstruction_path == 'IFcC2C':
                                            batch = (seq_in_BBX5_r, seq_in_FTM2_c, seq_in_IMU_NED_pos2_c)
                                            batch = [t.to(device) for t in batch]
                                            BBX5 = batch[0]
                                            # print('\n\n GND np.shape(BBX5): ', np.shape(BBX5)) # GND np.shape(BBX5):  torch.Size([1, 10, 1, 5])
                                            
                                            # MRF >>>
                                            if C.minimum_required_frames:
                                                if iou_per_frame >= C.iou_per_frame_thred and \
                                                    win_i > 0 and prev_pred_BBX5_c_ls[subj_i_r] is not None and len(prev_pred_BBX5_c_ls[subj_i_r]) != 0:
                                                    # print('\n prev_pred_BBX5_c_ls[subj_i_r]: ', prev_pred_BBX5_c_ls[subj_i_r])
                                                    print('\n\n np.shape(prev_pred_BBX5_c_ls[subj_i_r]): ', np.shape(prev_pred_BBX5_c_ls[subj_i_r]))
                                                    prev_pred_BBX = prev_pred_BBX5_c_ls[subj_i_r].clone(); prev_pred_BBX5_c_ls[subj_i_r].to(device) # torch.Size([1, 10, 1, 5])
                                                    BBX5[:, 0, :, :] = prev_pred_BBX[:, -1, :, :] # The 1st frame in the current window should be the last frame from prev_pred_BBX5_c_ls[subj_i_r]
                                                    batch = (BBX5, seq_in_FTM2_c, seq_in_IMU_NED_pos2_c) # to feed the model
                                                else: batch = (seq_in_BBX5_r, seq_in_FTM2_c, seq_in_IMU_NED_pos2_c)
                                            # MRF <<<
                                            else: batch = (seq_in_BBX5_r, seq_in_FTM2_c, seq_in_IMU_NED_pos2_c)
                                            # print('\n\n np.shape(img_f_BEV_pos2_): ', np.shape(img_f_BEV_pos2_)) # e.g. (10, 2)

                                            # DEBUG_HERE
                                            # print('\n utils.py - eval_recon_core() - with torch.no_grad() - np.shape(batch[0]): ', np.shape(batch[0]))
                                            # print('\n utils.py - eval_recon_core() - with torch.no_grad() - np.shape(batch[1]): ', np.shape(batch[1]))
                                            # print('\n utils.py - eval_recon_core() - with torch.no_grad() - np.shape(batch[2]): ', np.shape(batch[2]))
                                            print('\n utils.py - eval_recon_core() - with torch.no_grad() - np.shape(BBX5): ', np.shape(BBX5))
                                            # print('\n utils.py - eval_recon_core() - with torch.no_grad() - BBX5: ', BBX5)
                                            '''
                                            e.g.
                                            utils.py - eval_recon_core() - with torch.no_grad() - np.shape(BBX5):  torch.Size([1, 10, 1, 5])
                                            '''
                                            # BBX5_PRED = func_forward(model, BBX5) # batch)
                                            BBX5_PRED = func_forward(model, img_f_BEV_pos2_, BBX5, device) # batch)
                                            print('\n utils.py - eval_recon_core() - with torch.no_grad() - np.shape(BBX5_PRED): ', np.shape(BBX5_PRED))
                                            print('\n utils.py - eval_recon_core() - with torch.no_grad() - BBX5_PRED: ', BBX5_PRED)
                                            # utils.py - eval_recon_core() - with torch.no_grad() - np.shape(BBX5_PRED):  torch.Size([1, 10, 1, 5])
                                            #
                                            # print('\n utils.py - eval_recon_core() - with torch.no_grad() - np.shape(seq_in_IMU_XT_c): ', np.shape(seq_in_IMU_NED_pos2))
                                            # print('\n utils.py - eval_recon_core() - with torch.no_grad() - seq_in_IMU_XT_c: ', seq_in_IMU_NED_pos2)
                                            # print('------------------------------------------------------------------------')

                                    # seq_in_BBX5_r = BBX5.cpu()
                                    # pred_BBX5_c = BBX5_PRED.cpu() # DEBUG_HERE

                                    # Temporary solution >>>
                                    if BBX5_PRED is None:
                                        pred_BBX5_c = copy.deepcopy(BBX5)
                                    else: pred_BBX5_c = BBX5_PRED.cpu()
                                    # Temporary solution <<<

                                    # (win, frame, sample, feature)
                                    print('\n\n utils.py - eval_recon_core() - after squeezing for baseline  - np.shape(pred_BBX5_c): ', np.shape(pred_BBX5_c))

                                    # dBBX5 >>>
                                    d_pred_BBX5_c = BBX5_to_dBBX5(C, pred_BBX5_c, 1)
                                    if C.minimum_required_frames:
                                        if iou_per_frame >= C.iou_per_frame_thred and \
                                            win_i > 0 and prev_pred_BBX5_c_ls[subj_i_r] is not None and len(prev_pred_BBX5_c_ls[subj_i_r]) != 0:
                                            pred_BBX5_c[0, 0, :, :] = BBX5[0, 0, :, :] # equivalent to prev_pred_BBX5_c_ls[subj_i_r][0, -1, :, :] # 1st frame bbox in the current is the bbox in the last frame of the last window
                                        else: pred_BBX5_c[0, 0, :, :] = seq_in_BBX5_r[0, 0, :, :]
                                    else: pred_BBX5_c[0, 0, :, :] = seq_in_BBX5_r[0, 0, :, :]
                                    for f_i in range(1, C.len_win):
                                        pred_BBX5_c[0, f_i, :, :] = pred_BBX5_c[0, f_i - 1, :, :] + d_pred_BBX5_c[0, f_i, :, :]
                                    # dBBX5 <<<

                                    # print('\n\n img_path: ', img_path)
                                    print('\n\n ts16_dfv4: ', ts16_dfv4)
                                    print('\n\n C.vis_eval: ', C.vis_eval)
                                    print('\n\n C.vis_bbox_type: ', C.vis_bbox_type)
                                    print('\n\n C.vis_bbox_len: ', C.vis_bbox_len)
                                    print('\n\n np.shape(seq_in_BBX5_r[0, 0, :, :]): ', np.shape(seq_in_BBX5_r[0, 0, :, :]))
                                    print('\n\n seq_in_BBX5_r[0, 0, :, :]: ', seq_in_BBX5_r[0, 0, :, :])
                                    print()

                                    fst_frame_nonzero = False
                                    for i in range(5):
                                        if torch.is_nonzero(seq_in_BBX5_r[0, 0, :, i]):
                                            fst_frame_nonzero = True
                                            break

                                    vis_len_win = min(C.len_win, C.vis_bbox_len)
                                    if C.vis_eval and win_i > C.len_win and fst_frame_nonzero:
                                        img = print_bbox_win(C, C.img, seq_in_BBX5_r, pred_BBX5_c, vis_len_win)

                                    # Update prev_pred_BBX5_c_ls
                                    if C.minimum_required_frames: prev_pred_BBX5_c_ls[subj_i_r] = copy.deepcopy(pred_BBX5_c)

                                    # -----------------------
                                    #  Evaluation in one win
                                    eucl_dist = np.linalg.norm(seq_in_BBX5_r.clone().numpy() - pred_BBX5_c.clone().numpy())
                                    eucl_dist_log_ls.append(eucl_dist)

                                    # Depth >>>
                                    d_GND = torch.squeeze(seq_in_BBX5_r[:,:,:,2]).float()
                                    zero_indices_GND = torch.tensor(np.where(d_GND == 0)[0])
                                    zeros = torch.zeros(len(zero_indices_GND))
                                    d_PRED = torch.squeeze(pred_BBX5_c[:,:,:,2]).float()
                                    d_FTM = torch.squeeze(seq_in_FTM2_c[:,:,:,0] / 1000).float()
                                    # DEBUG:
                                    # print('\n\n utils.py - eval_recon_core - zero_indices_GND: ', zero_indices_GND)
                                    # print('\n\n utils.py - eval_recon_core - zeros: ', zeros)
                                    # print('\n\n utils.py - eval_recon_core - np.shape(d_GND): ', np.shape(d_GND))
                                    # print('\n\n utils.py - eval_recon_core - np.shape(d_PRED): ', np.shape(d_PRED))
                                    # print('\n\n utils.py - eval_recon_core - np.shape(d_FTM): ', np.shape(d_FTM))
                                    # print('\n\n utils.py - eval_recon_core - d_GND: ', d_GND)
                                    # print('\n\n utils.py - eval_recon_core - d_PRED: ', d_PRED)
                                    # print('\n\n utils.py - eval_recon_core - d_FTM: ', d_FTM)
                                    # print('\n\n utils.py - eval_recon_core - d_PRED.put_(zero_indices_GND, zeros): ', d_PRED.put_(zero_indices_GND, zeros))
                                    # print('\n\n utils.py - eval_recon_core - d_FTM.put_(zero_indices_GND, zeros): ', d_FTM.put_(zero_indices_GND, zeros))
                                    # e.g. torch.Size([50])
                                    d_err_PRED = abs(d_PRED.put_(zero_indices_GND, zeros) - d_GND)
                                    d_err_FTM = abs(d_FTM.put_(zero_indices_GND, zeros) - d_GND)
                                    d_err_improvement = d_err_FTM - d_err_PRED
                                    # print('\n\n utils.py - eval_recon_core - d_err_PRED: ', d_err_PRED)
                                    # print('\n\n utils.py - eval_recon_core - d_err_FTM: ', d_err_FTM)
                                    print('\n\n utils.py - eval_recon_core - d_err_improvement: ', d_err_improvement)                                    

                                    # Log
                                    d_err_PRED_log_ls.append(list(d_err_PRED))
                                    d_err_FTM_log_ls.append(list(d_err_FTM))
                                    d_err_improvement_log_ls.append(list(d_err_improvement))
                                    # Depth <<<

                                    # BBX >>>
                                    seq_in_BBX5_r_squeezed = torch.squeeze(seq_in_BBX5_r).float()
                                    pred_BBX5_c_squeezed = torch.squeeze(pred_BBX5_c).float()
                                    # print('\n\n utils.py - eval_recon_core - np.shape(seq_in_BBX5_r): ', np.shape(seq_in_BBX5_r))
                                    # print('\n\n utils.py - eval_recon_core - np.shape(pred_BBX5_c): ', np.shape(pred_BBX5_c))
                                    # print('\n\n utils.py - eval_recon_core - np.shape(seq_in_BBX5_r_squeezed): ', np.shape(seq_in_BBX5_r_squeezed))
                                    # print('\n\n utils.py - eval_recon_core - np.shape(pred_BBX5_c_squeezed): ', np.shape(pred_BBX5_c_squeezed))
                                    # e.g. utils.py - eval_recon_core - np.shape(pred_BBX5_c):  torch.Size([1, 50, 1, 5])
                                    # e.g. utils.py - eval_recon_core - np.shape(pred_BBX5_c_squeezed):  torch.Size([50, 5])

                                    pred_BBX5_c_squeezed = pred_BBX5_c_squeezed.put_(zero_indices_GND, zeros)
                                    # print('\n\n utils.py - eval_recon_core - pred_BBX5_c_squeezed: ', pred_BBX5_c_squeezed)

                                    # seq_in_BBX5_r_a, pred_BBX5_c_b = labels_to_boxes(seq_in_BBX5_r, pred_BBX5_c)
                                    # print('\n\n utils.py - eval_recon_core - np.shape(seq_in_BBX5_r_a): ', np.shape(seq_in_BBX5_r_a))
                                    # print('\n\n utils.py - eval_recon_core - np.shape(pred_BBX5_c_b): ', np.shape(pred_BBX5_c_b))
                                    # # e.g. torch.Size([50, 4])
                                    #
                                    # diou_loss, iou = _diou_iou_loss(seq_in_BBX5_r_a, pred_BBX5_c_b, C.eps_denominator_iou)
                                    # print('\n\n utils.py - eval_recon_core - diou_loss: ', diou_loss)
                                    # print('\n\n utils.py - eval_recon_core - iou: ', iou)
                                    # # print('\n\n utils.py - eval_recon_core - np.shape(diou_loss): ', np.shape(diou_loss))
                                    # # print('\n\n utils.py - eval_recon_core - np.shape(iou): ', np.shape(iou))
                                    # '''
                                    # e.g.
                                    # utils.py - eval_recon_core - np.shape(loss_):  torch.Size([50])
                                    # utils.py - eval_recon_core - np.shape(iou):  torch.Size([50])
                                    # '''
                                    # # mse = mean_squared_error(seq_in_BBX5_r_a, pred_BBX5_c_b)
                                    # # print('\n\n utils.py - eval_recon_core - mse: ', mse)
                                # PDR Baseline <<<
                                # Tc_interp Baseline >>> Archived Y2022M12D06Tue
                                # elif 'Tc_interp' in C.model_cfg['model_type']:
                                #     if not torch.is_tensor(seq_in_BBX5_r): seq_in_BBX5_r = torch.from_numpy(seq_in_BBX5_r)
                                #     print('\n\n np.shape(seq_in_BBX5_r): ', np.shape(seq_in_BBX5_r)) # e.g. torch.Size([1, 10, 1, 5])
                                #     if phone_win_i + C.len_win < C.n_wins:
                                #         if (phone_win_i + C.len_win, subj_i_r) in C.test_seq_in_BBX5_dict:
                                #             seq_in_BBX5_next_r = torch.from_numpy(C.test_seq_in_BBX5_dict[(phone_win_i + C.len_win, subj_i_r)])
                                #             # print('\n\n np.shape(seq_in_BBX5_next_r): ', np.shape(seq_in_BBX5_next_r)) # e.g. torch.Size([1, 10, 1, 5])
                                #
                                #             # Linearly interpolate bounding box values in two of the first frame bbox in two consecutive windows
                                #             fst_seq_in_BBX5_r, fst_seq_in_BBX5_next_r = seq_in_BBX5_r[:, 0, :, :], seq_in_BBX5_next_r[:, 0, :, :]
                                #             diff = (fst_seq_in_BBX5_next_r - fst_seq_in_BBX5_r) / C.len_win
                                #             # print('\n\n np.shape(fst_seq_in_BBX5_r): ', np.shape(fst_seq_in_BBX5_r)) # e.g. torch.Size([1, 1, 5])
                                #             # print('\n\n diff: ', diff)
                                #             # print('\n\n np.shape(diff): ', np.shape(diff)) # e.g. torch.Size([1, 1, 5])
                                #             pred_BBX5_c = seq_in_BBX5_r.clone()
                                #             pred_BBX5_c[:, 0, :, :] = seq_in_BBX5_r[:, 0, :, :] # First frame
                                #             # Predict rest of the frames by interpolation
                                #             for i_ in range(1, C.len_win):
                                #                 pred_BBX5_c[:, i_, :, :] = pred_BBX5_c[:, i_ - 1, :, :] + diff
                                #             # print('\n\n pred_BBX5_c: ', pred_BBX5_c)
                                #         else:
                                #             # No next window exists. Broadcast the first frames.
                                #             pred_BBX5_c = seq_in_BBX5_r.clone()
                                #             pred_BBX5_c[:, 0, :, :] = seq_in_BBX5_r[:, 0, :, :] # First frame
                                #             # Predict rest of the frames by interpolation
                                #             for i_ in range(1, C.len_win):
                                #                 pred_BBX5_c[:, i_, :, :] = pred_BBX5_c[:, 0, :, :]
                                #     else: break
                                #
                                #     # -----------------------
                                #     #  Evaluation in one win
                                #     eucl_dist = np.linalg.norm(seq_in_BBX5_r - pred_BBX5_c)
                                #     eucl_dist_log_ls.append(eucl_dist)
                                #
                                #     # Depth >>>
                                #     d_GND = np.squeeze(seq_in_BBX5_r[:,:,:,2])
                                #     zero_indices_GND = np.array(np.where(d_GND == 0)[0])
                                #     zeros = np.zeros(len(zero_indices_GND))
                                #     d_PRED = np.squeeze(pred_BBX5_c[:,:,:,2])
                                #     d_FTM = np.squeeze(seq_in_FTM2_c[:,:,:,0] / 1000)
                                #     # DEBUG:
                                #     # print('\n\n utils.py - eval_recon_core - zero_indices_GND: ', zero_indices_GND)
                                #     # print('\n\n utils.py - eval_recon_core - zeros: ', zeros)
                                #     # print('\n\n utils.py - eval_recon_core - np.shape(d_GND): ', np.shape(d_GND))
                                #     # print('\n\n utils.py - eval_recon_core - np.shape(d_PRED): ', np.shape(d_PRED))
                                #     # print('\n\n utils.py - eval_recon_core - np.shape(d_FTM): ', np.shape(d_FTM))
                                #     # print('\n\n utils.py - eval_recon_core - d_GND: ', d_GND)
                                #     # print('\n\n utils.py - eval_recon_core - d_PRED: ', d_PRED)
                                #     # print('\n\n utils.py - eval_recon_core - d_FTM: ', d_FTM)
                                #     # print('len(zero_indices_GND): ', len(zero_indices_GND))
                                #     if len(zero_indices_GND) > 0:
                                #         np.put(d_PRED, zero_indices_GND, zeros)
                                #         np.put(d_FTM, zero_indices_GND, zeros)
                                #         # print('\n\n utils.py - eval_recon_core - after np.put zeros - d_PRED: ', d_PRED)
                                #         # print('\n\n utils.py - eval_recon_core - after np.put zeros - d_FTM: ', d_FTM)
                                #     d_err_PRED = abs(d_PRED - d_GND)
                                #     d_err_FTM = abs(d_FTM - d_GND.detach().cpu().numpy())
                                #     d_err_improvement = d_err_FTM - d_err_PRED.detach().cpu().numpy()
                                #     # print('\n\n utils.py - eval_recon_core - d_err_PRED: ', d_err_PRED)
                                #     # print('\n\n utils.py - eval_recon_core - d_err_FTM: ', d_err_FTM)
                                #     # print('\n\n utils.py - eval_recon_core - d_err_improvement: ', d_err_improvement)
                                #
                                #     # Log
                                #     d_err_PRED_log_ls.append(list(d_err_PRED))
                                #     d_err_FTM_log_ls.append(list(d_err_FTM))
                                #     d_err_improvement_log_ls.append(list(d_err_improvement))
                                #     # Depth <<<
                                #
                                # # Tc_interp Baseline <<< Archived Y2022M12D06Tue
                                elif 'Tc_interp' in C.model_cfg['model_type']:
                                    if not torch.is_tensor(seq_in_BBX5_r): seq_in_BBX5_r = torch.from_numpy(seq_in_BBX5_r)
                                    print('\n\n np.shape(seq_in_BBX5_r): ', np.shape(seq_in_BBX5_r)) # e.g. torch.Size([1, 10, 1, 5])
                                    BBX5 = copy.deepcopy(seq_in_BBX5_r)
                                    if phone_win_i + C.len_win < C.n_wins:
                                        # MRF >>>
                                        if C.minimum_required_frames:
                                            if iou_per_frame >= C.iou_per_frame_thred and \
                                                win_i > 0 and prev_pred_BBX5_c_ls[subj_i_r] is not None and len(prev_pred_BBX5_c_ls[subj_i_r]) != 0:
                                                # print('\n prev_pred_BBX5_c_ls[subj_i_r]: ', prev_pred_BBX5_c_ls[subj_i_r])
                                                print('\n\n np.shape(prev_pred_BBX5_c_ls[subj_i_r]): ', np.shape(prev_pred_BBX5_c_ls[subj_i_r]))
                                                prev_pred_BBX = prev_pred_BBX5_c_ls[subj_i_r].clone(); prev_pred_BBX5_c_ls[subj_i_r].to(device) # torch.Size([1, 10, 1, 5])
                                                BBX5[:, 0, :, :] = prev_pred_BBX[:, -1, :, :] # The 1st frame in the current window should be the last frame from prev_pred_BBX5_c_ls[subj_i_r]
                                                batch = (BBX5, seq_in_FTM2_c, seq_in_IMU_XT_c) # to feed the model
                                            else: batch = (seq_in_BBX5_r, seq_in_FTM2_c, seq_in_IMU_XT_c)
                                        # MRF <<<
                                        else: batch = (seq_in_BBX5_r, seq_in_FTM2_c, seq_in_IMU_XT_c)
                                        has_next_first_BBOX = False
                                        if (phone_win_i + C.len_win, subj_i_r) in C.test_seq_in_BBX5_dict:
                                            seq_in_BBX5_next_r = torch.from_numpy(C.test_seq_in_BBX5_dict[(phone_win_i + C.len_win, subj_i_r)])
                                            # print('\n\n np.shape(seq_in_BBX5_next_r): ', np.shape(seq_in_BBX5_next_r)) # e.g. torch.Size([1, 10, 1, 5])
                                            x_next, y_next = seq_in_BBX5_next_r[:, 0, :, 0][0][0], seq_in_BBX5_next_r[:, 0, :, 1][0][0]

                                            # In the view
                                            if x_next >= C.seen_left_borders[m_i] and x_next < C.seen_right_borders[m_i] and y_next >= C.seen_top_borders[m_i] and y_next < C.seen_bottom_borders[m_i]:
                                                has_next_first_BBOX = True

                                        if C.first_frame_only:
                                            # Broadcast the first frames.
                                            pred_BBX5_c = BBX5.clone()
                                            pred_BBX5_c[:, 0, :, :] = BBX5[:, 0, :, :] # First frame
                                            # Predict rest of the frames by interpolation
                                            for i_ in range(1, C.len_win):
                                                pred_BBX5_c[:, i_, :, :] = pred_BBX5_c[:, 0, :, :]
                                        elif has_next_first_BBOX:
                                            # Linearly interpolate bounding box values in two of the first frame bbox in two consecutive windows
                                            fst_BBX5, fst_seq_in_BBX5_next_r = BBX5[:, 0, :, :], seq_in_BBX5_next_r[:, 0, :, :]
                                            diff = (fst_seq_in_BBX5_next_r - fst_BBX5) / C.len_win
                                            # print('\n\n np.shape(fst_BBX5): ', np.shape(fst_BBX5)) # e.g. torch.Size([1, 1, 5])
                                            # print('\n\n diff: ', diff)
                                            # print('\n\n np.shape(diff): ', np.shape(diff)) # e.g. torch.Size([1, 1, 5])
                                            pred_BBX5_c = BBX5.clone()
                                            pred_BBX5_c[:, 0, :, :] = BBX5[:, 0, :, :] # First frame
                                            # Predict rest of the frames by interpolation
                                            for i_ in range(1, C.len_win):
                                                pred_BBX5_c[:, i_, :, :] = pred_BBX5_c[:, i_ - 1, :, :] + diff
                                            # print('\n\n pred_BBX5_c: ', pred_BBX5_c)
                                        else:
                                            # No next window exists. Broadcast the first frames.
                                            pred_BBX5_c = BBX5.clone()
                                            pred_BBX5_c[:, 0, :, :] = BBX5[:, 0, :, :] # First frame
                                            # Predict rest of the frames by interpolation
                                            for i_ in range(1, C.len_win):
                                                pred_BBX5_c[:, i_, :, :] = pred_BBX5_c[:, 0, :, :]
                                        if C.minimum_required_frames: prev_pred_BBX5_c_ls[subj_i_r] = copy.deepcopy(pred_BBX5_c)
                                    else: break
                                    

                                    # -----------------------
                                    #  Evaluation in one win
                                    eucl_dist = np.linalg.norm(seq_in_BBX5_r - pred_BBX5_c)
                                    eucl_dist_log_ls.append(eucl_dist)

                                    # Depth >>>
                                    d_GND = np.squeeze(seq_in_BBX5_r[:,:,:,2])
                                    zero_indices_GND = np.array(np.where(d_GND == 0)[0])
                                    zeros = np.zeros(len(zero_indices_GND))
                                    d_PRED = np.squeeze(pred_BBX5_c[:,:,:,2])
                                    d_FTM = np.squeeze(seq_in_FTM2_c[:,:,:,0] / 1000)
                                    # DEBUG:
                                    # print('\n\n utils.py - eval_recon_core - zero_indices_GND: ', zero_indices_GND)
                                    # print('\n\n utils.py - eval_recon_core - zeros: ', zeros)
                                    # print('\n\n utils.py - eval_recon_core - np.shape(d_GND): ', np.shape(d_GND))
                                    # print('\n\n utils.py - eval_recon_core - np.shape(d_PRED): ', np.shape(d_PRED))
                                    # print('\n\n utils.py - eval_recon_core - np.shape(d_FTM): ', np.shape(d_FTM))
                                    # print('\n\n utils.py - eval_recon_core - d_GND: ', d_GND)
                                    # print('\n\n utils.py - eval_recon_core - d_PRED: ', d_PRED)
                                    # print('\n\n utils.py - eval_recon_core - d_FTM: ', d_FTM)
                                    # print('len(zero_indices_GND): ', len(zero_indices_GND))
                                    if len(zero_indices_GND) > 0:
                                        np.put(d_PRED, zero_indices_GND, zeros)
                                        np.put(d_FTM, zero_indices_GND, zeros)
                                        # print('\n\n utils.py - eval_recon_core - after np.put zeros - d_PRED: ', d_PRED)
                                        # print('\n\n utils.py - eval_recon_core - after np.put zeros - d_FTM: ', d_FTM)
                                    d_err_PRED = abs(d_PRED - d_GND)
                                    d_err_FTM = abs(d_FTM - d_GND.detach().cpu().numpy())
                                    d_err_improvement = d_err_FTM - d_err_PRED.detach().cpu().numpy()
                                    # print('\n\n utils.py - eval_recon_core - d_err_PRED: ', d_err_PRED)
                                    # print('\n\n utils.py - eval_recon_core - d_err_FTM: ', d_err_FTM)
                                    # print('\n\n utils.py - eval_recon_core - d_err_improvement: ', d_err_improvement)

                                    # DTW >>>
                                    # alignment = dtw(torch.squeeze(seq_in_BBX5_r[:, :, :, :2]), torch.squeeze(pred_BBX5_c[:, :, :, :2]), keep_internals=True) # edited
                                    # print('\n\n dir(alignment): ', dir(alignment))
                                    # print('\n\n alignment: ', alignment)
                                    # print('\n\n alignment.distance: ', alignment.distance)
                                    # dtw_log_ls.append(alignment.distance) # edited
                                    dtw_log_ls.append(0)
                                    # DTW <<<

                                    # Log
                                    d_err_PRED_log_ls.append(list(d_err_PRED))
                                    d_err_FTM_log_ls.append(list(d_err_FTM))
                                    d_err_improvement_log_ls.append(list(d_err_improvement))
                                    # Depth <<<

                                # Tc_interp Baseline <<<
                                # kalman_filter Baseline >>>
                                elif C.model_cfg['train_model_id'] == 'kalman_filter': # C.model_cfg['model_type'] == 'kalman_filter':
                                    # print('\n\n model.kf_ls: ', model.kf_ls)
                                    print('\n\n subj_i_r: ', subj_i_r)
                                    if not torch.is_tensor(seq_in_BBX5_r): seq_in_BBX5_r = torch.from_numpy(seq_in_BBX5_r)
                                    print('\n\n np.shape(seq_in_BBX5_r): ', np.shape(seq_in_BBX5_r)) # e.g. torch.Size([1, 10, 1, 5])

                                    pred_BBX5_c = seq_in_BBX5_r.clone()

                                    if phone_win_i == 0: fst_WIN = True
                                    else: fst_WIN = False

                                    fst_seq_in_BBX5_r = seq_in_BBX5_r[:, 0, :, :]
                                    # fst_seq_in_BBX5_r = torch.squeeze((torch.squeeze(fst_seq_in_BBX5_r)))
                                    
                                    print('\n utils.py - np.shape(fst_seq_in_BBX5_r): ', np.shape(fst_seq_in_BBX5_r))
                                    # e.g. utils.py - np.shape(fst_seq_in_BBX5_r):  torch.Size([1, 1, 5])
                                    
                                    # v1 beforeY2023M07D09Sun >>>
                                    model.kf_ls[subj_i_r], fst_seq_in_BBX5_next_r_pred = func_forward(model.kf_ls[subj_i_r], fst_seq_in_BBX5_r, fst_WIN, device)
                                    # v1 beforeY2023M07D09Sun <<<

                                    print('\n utils.py - np.shape(fst_seq_in_BBX5_next_r_pred): ', np.shape(fst_seq_in_BBX5_next_r_pred))
                                    # e.g. utils.py - np.shape(fst_seq_in_BBX5_next_r_pred):  torch.Size([1, 1, 5])

                                    # fst_seq_in_BBX5_r = torch.unsqueeze(torch.unsqueeze(fst_seq_in_BBX5_r,0), 0)
                                    # print('\n\n np.shape(fst_seq_in_BBX5_r): ', np.shape(fst_seq_in_BBX5_r)) # e.g. torch.Size([1, 1, 5])
                                    # print('\n\n np.shape(fst_seq_in_BBX5_next_r_pred): ', np.shape(fst_seq_in_BBX5_next_r_pred))
                                    diff = (fst_seq_in_BBX5_next_r_pred - fst_seq_in_BBX5_r) / C.len_win
                                    # print('\n\n diff: ', diff)
                                    # print('\n\n np.shape(diff): ', np.shape(diff)) # e.g. torch.Size([1, 1, 5])

                                    pred_BBX5_c[:, 0, :, :] = seq_in_BBX5_r[:, 0, :, :] # First frame
                                    # Predict rest of the frames by interpolation
                                    for i_ in range(1, C.len_win):
                                        pred_BBX5_c[:, i_, :, :] = pred_BBX5_c[:, i_ - 1, :, :] + diff
                                    # print('\n\n pred_BBX5_c: ', pred_BBX5_c)

                                    if phone_win_i + C.len_win >= C.n_wins: break

                                    # -----------------------
                                    #  Evaluation in one win
                                    eucl_dist = np.linalg.norm(seq_in_BBX5_r - pred_BBX5_c)
                                    eucl_dist_log_ls.append(eucl_dist)

                                    # Depth >>>
                                    d_GND = np.squeeze(seq_in_BBX5_r[:,:,:,2])
                                    zero_indices_GND = np.array(np.where(d_GND == 0)[0])
                                    zeros = np.zeros(len(zero_indices_GND))
                                    d_PRED = np.squeeze(pred_BBX5_c[:,:,:,2])
                                    d_FTM = np.squeeze(seq_in_FTM2_c[:,:,:,0] / 1000)
                                    # DEBUG:
                                    # print('\n\n utils.py - eval_recon_core - zero_indices_GND: ', zero_indices_GND)
                                    # print('\n\n utils.py - eval_recon_core - zeros: ', zeros)
                                    # print('\n\n utils.py - eval_recon_core - np.shape(d_GND): ', np.shape(d_GND))
                                    # print('\n\n utils.py - eval_recon_core - np.shape(d_PRED): ', np.shape(d_PRED))
                                    # print('\n\n utils.py - eval_recon_core - np.shape(d_FTM): ', np.shape(d_FTM))
                                    # print('\n\n utils.py - eval_recon_core - d_GND: ', d_GND)
                                    # print('\n\n utils.py - eval_recon_core - d_PRED: ', d_PRED)
                                    # print('\n\n utils.py - eval_recon_core - d_FTM: ', d_FTM)
                                    # print('len(zero_indices_GND): ', len(zero_indices_GND))
                                    if len(zero_indices_GND) > 0:
                                        np.put(d_PRED, zero_indices_GND, zeros)
                                        np.put(d_FTM, zero_indices_GND, zeros)
                                        # print('\n\n utils.py - eval_recon_core - after np.put zeros - d_PRED: ', d_PRED)
                                        # print('\n\n utils.py - eval_recon_core - after np.put zeros - d_FTM: ', d_FTM)
                                    d_err_PRED = abs(d_PRED - d_GND)
                                    d_err_FTM = abs(d_FTM - d_GND.detach().cpu().numpy())
                                    d_err_improvement = d_err_FTM - d_err_PRED.detach().cpu().numpy()
                                    # print('\n\n utils.py - eval_recon_core - d_err_PRED: ', d_err_PRED)
                                    # print('\n\n utils.py - eval_recon_core - d_err_FTM: ', d_err_FTM)
                                    # print('\n\n utils.py - eval_recon_core - d_err_improvement: ', d_err_improvement)

                                    # Log
                                    d_err_PRED_log_ls.append(list(d_err_PRED))
                                    d_err_FTM_log_ls.append(list(d_err_FTM))
                                    d_err_improvement_log_ls.append(list(d_err_improvement))
                                    # Depth <<<
                                # kalman_filter Baseline <<<
                                elif C.model_cfg['train_model_id'] == 'kalman_filter_v2': # C.model_cfg['model_type'] == 'kalman_filter_v2':
                                    # print('\n\n model.kf_ls: ', model.kf_ls)
                                    print('\n\n subj_i_r: ', subj_i_r)
                                    if not torch.is_tensor(seq_in_BBX5_r): seq_in_BBX5_r = torch.from_numpy(seq_in_BBX5_r)
                                    print('\n\n np.shape(seq_in_BBX5_r): ', np.shape(seq_in_BBX5_r)) # e.g. torch.Size([1, 10, 1, 5])

                                    pred_BBX5_c = seq_in_BBX5_r.clone()

                                    if phone_win_i == 0: fst_WIN = True
                                    else: fst_WIN = False

                                    fst_seq_in_BBX5_r = seq_in_BBX5_r[:, 0, :, :]
                                    # fst_seq_in_BBX5_r = torch.squeeze((torch.squeeze(fst_seq_in_BBX5_r)))
                                    
                                    print('\n utils.py - np.shape(fst_seq_in_BBX5_r): ', np.shape(fst_seq_in_BBX5_r))
                                    # e.g. utils.py - np.shape(fst_seq_in_BBX5_r):  torch.Size([1, 1, 5])
                                    
                                    # v2 Y2023M07D09Sun >>>
                                    # velocity of bbx
                                    if fst_WIN:
                                        v_BBX = torch.tensor([[[0, 0, 0, 0, 0]]])
                                        print('\n np.shape(v_BBX): ', np.shape(v_BBX))
                                        # e.g. np.shape(v_BBX):  torch.Size([1, 1, 5])
                                    else:
                                        v_BBX = fst_seq_in_BBX5_r - fst_seq_in_BBX5_r_prev 
                                    fst_seq_in_BBX5_r_prev = copy.deepcopy(fst_seq_in_BBX5_r)
                                    fst_seq_in_BBX5_r = torch.cat((fst_seq_in_BBX5_r, v_BBX), dim=2)
                                    print('\n np.shape(fst_seq_in_BBX5_r): ', np.shape(fst_seq_in_BBX5_r))
                                    # np.shape(fst_seq_in_BBX5_r):  torch.Size([1, 1, 10])
                                    model.kf_ls[subj_i_r], fst_seq_in_BBX5_next_r_pred = func_forward(model.kf_ls[subj_i_r], fst_seq_in_BBX5_r, fst_WIN, device)
                                    # v2 Y2023M07D09Sun <<<

                                    print('\n utils.py - np.shape(fst_seq_in_BBX5_next_r_pred): ', np.shape(fst_seq_in_BBX5_next_r_pred))
                                    # e.g. utils.py - np.shape(fst_seq_in_BBX5_next_r_pred):  torch.Size([1, 1, 5])
                                    
                                    # fst_seq_in_BBX5_r = torch.unsqueeze(torch.unsqueeze(fst_seq_in_BBX5_r,0), 0)
                                    # print('\n\n np.shape(fst_seq_in_BBX5_r): ', np.shape(fst_seq_in_BBX5_r)) # e.g. torch.Size([1, 1, 5])
                                    # print('\n\n np.shape(fst_seq_in_BBX5_next_r_pred): ', np.shape(fst_seq_in_BBX5_next_r_pred))
                                    diff = (fst_seq_in_BBX5_next_r_pred - fst_seq_in_BBX5_r[:,:,:5]) / C.len_win
                                    # print('\n\n diff: ', diff)
                                    # print('\n\n np.shape(diff): ', np.shape(diff)) # e.g. torch.Size([1, 1, 5])

                                    pred_BBX5_c[:, 0, :, :] = seq_in_BBX5_r[:, 0, :, :] # First frame
                                    # Predict rest of the frames by interpolation
                                    for i_ in range(1, C.len_win):
                                        pred_BBX5_c[:, i_, :, :] = pred_BBX5_c[:, i_ - 1, :, :] + diff
                                    # print('\n\n pred_BBX5_c: ', pred_BBX5_c)

                                    if phone_win_i + C.len_win >= C.n_wins: break

                                    # -----------------------
                                    #  Evaluation in one win
                                    eucl_dist = np.linalg.norm(seq_in_BBX5_r - pred_BBX5_c)
                                    eucl_dist_log_ls.append(eucl_dist)

                                    # Depth >>>
                                    d_GND = np.squeeze(seq_in_BBX5_r[:,:,:,2])
                                    zero_indices_GND = np.array(np.where(d_GND == 0)[0])
                                    zeros = np.zeros(len(zero_indices_GND))
                                    d_PRED = np.squeeze(pred_BBX5_c[:,:,:,2])
                                    d_FTM = np.squeeze(seq_in_FTM2_c[:,:,:,0] / 1000)
                                    # DEBUG:
                                    # print('\n\n utils.py - eval_recon_core - zero_indices_GND: ', zero_indices_GND)
                                    # print('\n\n utils.py - eval_recon_core - zeros: ', zeros)
                                    # print('\n\n utils.py - eval_recon_core - np.shape(d_GND): ', np.shape(d_GND))
                                    # print('\n\n utils.py - eval_recon_core - np.shape(d_PRED): ', np.shape(d_PRED))
                                    # print('\n\n utils.py - eval_recon_core - np.shape(d_FTM): ', np.shape(d_FTM))
                                    # print('\n\n utils.py - eval_recon_core - d_GND: ', d_GND)
                                    # print('\n\n utils.py - eval_recon_core - d_PRED: ', d_PRED)
                                    # print('\n\n utils.py - eval_recon_core - d_FTM: ', d_FTM)
                                    # print('len(zero_indices_GND): ', len(zero_indices_GND))
                                    if len(zero_indices_GND) > 0:
                                        np.put(d_PRED, zero_indices_GND, zeros)
                                        np.put(d_FTM, zero_indices_GND, zeros)
                                        # print('\n\n utils.py - eval_recon_core - after np.put zeros - d_PRED: ', d_PRED)
                                        # print('\n\n utils.py - eval_recon_core - after np.put zeros - d_FTM: ', d_FTM)
                                    d_err_PRED = abs(d_PRED - d_GND)
                                    d_err_FTM = abs(d_FTM - d_GND.detach().cpu().numpy())
                                    d_err_improvement = d_err_FTM - d_err_PRED.detach().cpu().numpy()
                                    # print('\n\n utils.py - eval_recon_core - d_err_PRED: ', d_err_PRED)
                                    # print('\n\n utils.py - eval_recon_core - d_err_FTM: ', d_err_FTM)
                                    # print('\n\n utils.py - eval_recon_core - d_err_improvement: ', d_err_improvement)

                                    # Log
                                    d_err_PRED_log_ls.append(list(d_err_PRED))
                                    d_err_FTM_log_ls.append(list(d_err_FTM))
                                    d_err_improvement_log_ls.append(list(d_err_improvement))
                                    # Depth <<<
                                # kalman_filter Baseline <<<


                                seq_in_BBX5_r_a, pred_BBX5_c_b = labels_to_boxes(seq_in_BBX5_r, pred_BBX5_c)
                                # print('\n\n utils.py - eval_recon_core - np.shape(seq_in_BBX5_r_a): ', np.shape(seq_in_BBX5_r_a))
                                # print('\n\n utils.py - eval_recon_core - np.shape(pred_BBX5_c_b): ', np.shape(pred_BBX5_c_b))

                                diou_loss, iou = _diou_iou_loss(seq_in_BBX5_r_a, pred_BBX5_c_b, C.eps_denominator_iou)
                                print('\n\n utils.py - eval_recon_core - iou: ', iou)
                                print('\n\n utils.py - eval_recon_core - iou[0]: ', iou[0])
                                print('\n\n utils.py - eval_recon_core - iou[0].clone().numpy(): ', iou[0].clone().numpy())
                                # e.g. utils.py - eval_recon_core - iou:  tensor([1.0000, 0.3575, 0.1619, 0.1613, 0.7688, 0.4466, 0.3799, 0.3389, 0.4956, 0.3979])

                                if C.minimum_required_frames:
                                    # print('\n\n utils.py - eval_recon_core - iou: ', iou)
                                    cumu_iou_ls_per_win.extend(iou.numpy())
                                    print('\n\n utils.py - eval_recon_core - if C.minimum_required_frames - iou_per_frame: ', iou_per_frame)
                                    print('\n\n utils.py - eval_recon_core - if C.minimum_required_frames - cumu_iou_ls_per_win: ', cumu_iou_ls_per_win)

                                if iou[0].clone().numpy() >= 0.99: # Note there could be some numerical numerical instability when converting a Tensor to numpy array from GPU to CPU memory like 1.0 -> 0.99999999
                                    print('\n\n utils.py - eval_recon_core - np.shape(diou_loss): ', np.shape(diou_loss))
                                    print('\n\n utils.py - eval_recon_core - np.shape(iou): ', np.shape(iou))
                                    '''
                                    e.g.
                                    utils.py - eval_recon_core - np.shape(loss_):  torch.Size([50])
                                    utils.py - eval_recon_core - np.shape(iou):  torch.Size([50])
                                    '''
                                    # mse = mean_squared_error(seq_in_BBX5_r_a, pred_BBX5_c_b)
                                    # print('\n\n utils.py - eval_recon_core - mse: ', mse)

                                    # >>>
                                    # AP_0p5 = np.zeros(len(iou)); AP_ones_indices = np.where(iou >= 0.5)[0]
                                    # np.put(AP_0p5, AP_ones_indices, 1)
                                    # print('\n\n utils.py - eval_recon_core - AP_0p5: ', AP_0p5)
                                    #
                                    # AP_0p1 = np.zeros(len(iou)); AP_ones_indices = np.where(iou >= 0.1)[0]
                                    # np.put(AP_0p1, AP_ones_indices, 1)
                                    # print('\n\n utils.py - eval_recon_core - AP_0p1: ', AP_0p1)
                                    # <<<
                                    AP_ls = []
                                    for thred in threds:
                                        AP = np.zeros(len(iou)); AP_ones_indices = np.where(iou >= thred)[0]
                                        np.put(AP, AP_ones_indices, 1)
                                        print('\n\n utils.py - eval_recon_core - AP: ', AP, ', thred: ', thred)
                                        AP_ls.append(list(AP))

                                    # Log
                                    diou_loss_log_ls.append(list(diou_loss))
                                    iou_log_ls.append(list(iou))

                                    # >>>
                                    # AP_0p5_log_ls.append(list(AP_0p5))
                                    # AP_0p1_log_ls.append(list(AP_0p1))
                                    # <<<
                                    for thred_i, thred in enumerate(threds):
                                        AP_log_ls2[thred_i].append(AP_ls[thred_i])

                                    # BBX <<<
                                    #  Evaluation in one win
                                    # -----------------------
                                    # dist_BBX5 = np.linalg.norm(seq_in_BBX5_r - pred_BBX5_c)
                                    # C.ts16_dfv4_subj_i_to_BBX5_prime[ts16_dfv4][subj_i_r] = pred_BBX5_c
                        # In the view <<<
                        # ================

                # End of all subjects
                print('\n\n cumu_iou_ls_per_win: ', cumu_iou_ls_per_win)
                print('\n\n len(cumu_iou_ls_per_win): ', len(cumu_iou_ls_per_win))
                iou_per_frame = np.mean(cumu_iou_ls_per_win[:C.win_stride])
                print('\n\n iou_per_frame: ', iou_per_frame)

                # Log
                C.test_log_per_frame_file.write('{},{},{}\n'.format(ts16_dfv4,iou_per_frame,MRF))

                # Some countings in the current window
                proc_n_frames += C.len_win - 1
                proc_n_wins += 1

                # Visualize BBX >>>
                if C.imu_ned_pos2 and C.vis and C.vis_eval and win_i > C.len_win and not C.use_docker:
                    cv2.imshow('img', C.img); cv2.waitKey(0)
                    cv2.imshow('BEV', C.BEV); cv2.waitKey(0)
                    cv2.imshow('img_f_BEV', C.img_f_BEV); cv2.waitKey(0)
                    print('\n len(C.IMU_NED_imgs): ', len(C.IMU_NED_imgs))
                    print('\n len(C.IMU_NED_range_imgs): ', len(C.IMU_NED_range_imgs))
                    for subj_i in range(C.n_subjects):
                        cv2.imshow('IMU_NED_img ' + str(subj_i), C.IMU_NED_imgs[subj_i])
                        # cv2.imshow('IMU_NED_range_img ' + str(subj_i), C.IMU_NED_range_imgs[subj_i])
                    cv2.waitKey(0)
                # Visualize BBX <<<

                # la_res_dict = {'ts16_dfv4': ts16_dfv4, 'win_i' : win_i, 'C.n_wins' : C.n_wins, \
                #     'test_seq_subj_i_in_view_ls_' : test_seq_subj_i_in_view_ls_}
                # print(la_res_dict)
                # C.ts16_dfv4_to_test_stats[ts16_dfv4] = la_res_dict
                # C.test_log_file.write(str(la_res_dict) + '\n\n')
                # C.test_log_file.flush()
                # e.g. shape(A):  (3, 5) , test_seq_subj_i_in_view_ls_:  [0, 1, 4] , row_ind:  [0 1 2] , col_ind:  [3 2 4]

        if C.vis_eval and win_i > C.len_win:
            # img = cv2.putText(img, 'GND: darker bbox, PRED: lighter bbox.', (30, 30), \
            #                     cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA)

            img_save_folder = C.test_log_path + '/vis_' + C.vis_bbox_type + '_' + str(C.vis_bbox_len)
            if not os.path.exists(img_save_folder): os.makedirs(img_save_folder)
            img_save_path = img_save_folder + '/' + ts16_dfv4 + '_anonymized_Tc5.jpg'

            cv2.imwrite(img_save_path, C.img) # img)
            print('\n\n ', img_save_path, 'saved')

    print()

    # Post process
    print('\n n_out_of_view: ', n_out_of_view)
    infer_time_sec_per_tracklet = float(total_infer_time_sec) / tracklets_total # sec
    infer_time_sec_per_win = float(total_infer_time_sec) / proc_n_wins # sec
    infer_time_sec_per_frame = float(total_infer_time_sec) / proc_n_frames # sec
    MRF_win_intervals, MRF_frame_intervals = [], []
    for i, _ in enumerate(MRF_win_i_ls):
        if i > 0:
            MRF_win_intervals.append(int((MRF_win_i_ls[i] - MRF_win_i_ls[i - 1]) / C.win_stride))
            MRF_frame_intervals.append((MRF_win_i_ls[i] - MRF_win_i_ls[i - 1]))
    print('\n MRF_win_i_ls: ', MRF_win_i_ls)
    MRF_win_interv_max, MRF_win_interv_min = np.max(MRF_win_intervals), np.min(MRF_win_intervals)
    MRF_win_interv_mean, MRF_win_interv_stdev = np.mean(MRF_win_intervals), np.std(MRF_win_intervals)
    MRF_frame_interv_max, MRF_frame_interv_min = np.max(MRF_frame_intervals), np.min(MRF_frame_intervals)
    MRF_frame_interv_mean, MRF_frame_interv_stdev = np.mean(MRF_frame_intervals), np.std(MRF_frame_intervals)

    # ---------------
    #  Log Test Data
    # ---------------
    eucl_dist_log_arr = np.array(eucl_dist_log_ls)
    d_err_PRED_log_arr = np.array(d_err_PRED_log_ls)
    d_err_FTM_log_arr = np.array(d_err_FTM_log_ls)
    d_err_improvement_log_arr = np.array(d_err_improvement_log_ls)
    dtw_log_arr = np.array(dtw_log_ls)
    diou_loss_log_arr = np.array(diou_loss_log_ls)
    iou_log_arr = np.array(iou_log_ls)
    # AP_0p5_log_arr = np.array(AP_0p5_log_ls)
    # AP_0p1_log_arr = np.array(AP_0p1_log_ls)
    AP_log_ls2_arr = np.array(AP_log_ls2)
    MRF_log = MRF
    MRF_win_i_ls_log = MRF_win_i_ls
    MRF_win_1st_ts16s_log = MRF_win_1st_ts16s
    proc_n_frames_log = proc_n_frames
    proc_n_wins_log = proc_n_wins
    infer_time_sec_per_tracklet_log = infer_time_sec_per_tracklet
    infer_time_sec_per_win_log = infer_time_sec_per_win
    infer_time_sec_per_frame_log = infer_time_sec_per_frame
    total_infer_time_sec_log = total_infer_time_sec
    MRF_win_intervals_log = MRF_win_intervals
    MRF_frame_intervals_log = MRF_frame_intervals

    eucl_dist_log_arr[eucl_dist_log_arr==0] = np.nan
    d_err_PRED_log_arr[d_err_PRED_log_arr==0] = np.nan
    d_err_FTM_log_arr[d_err_FTM_log_arr==0] = np.nan
    d_err_improvement_log_arr[d_err_improvement_log_arr==0] = np.nan
    dtw_log_arr[dtw_log_arr==0] = 0 # np.nan # edited
    diou_loss_log_arr[diou_loss_log_arr==0] = np.nan
    iou_log_arr[iou_log_arr==0] = np.nan
    # AP_0p5_log_arr[AP_0p5_log_arr==0] = np.nan
    # AP_0p1_log_arr[AP_0p1_log_arr==0] = np.nan

    eucl_dist_log_arr_mean = repr(np.array(np.nanmean(eucl_dist_log_arr))) # np.mean(d_err_PRED_log_arr)
    eucl_dist_log_arr_mean_0 = repr(np.array(np.nanmean(eucl_dist_log_arr, axis=0))) # np.mean(d_err_PRED_log_arr, axis=0)
    d_err_PRED_log_arr_mean_0 = repr(np.array(np.nanmean(d_err_PRED_log_arr, axis=0))) # np.mean(d_err_PRED_log_arr, axis=0)
    d_err_FTM_log_arr_mean_0 = repr(np.array(np.nanmean(d_err_FTM_log_arr, axis=0))) # np.mean(d_err_FTM_log_arr, axis=0)
    d_err_improvement_log_arr_mean_0 = repr(np.array(np.nanmean(d_err_improvement_log_arr, axis=0))) # np.mean(d_err_improvement_log_arr, axis=0)
    dtw_log_arr_mean_0 = repr(np.array(np.nanmean(dtw_log_arr, axis=0)))
    diou_loss_log_arr_mean_0 = repr(np.array(np.nanmean(diou_loss_log_arr, axis=0))) # np.mean(diou_loss_log_arr, axis=0)
    iou_log_arr_mean_0 = repr(np.array(np.nanmean(iou_log_arr, axis=0))) # np.mean(iou_log_arr, axis=0)
    iou_log_arr_mean = np.mean(np.nanmean(iou_log_arr, axis=0))
    # AP_0p5_log_arr_mean_0 = np.array(np.nanmean(AP_0p5_log_arr, axis=0)) # np.mean(iou_log_arr, axis=0)
    # AP_0p1_log_arr_mean_0 = np.array(np.nanmean(AP_0p1_log_arr, axis=0)) # np.mean(iou_log_arr, axis=0)
    AP_log_ls2_arr_mean_0_ls, AP_log_ls2_arr_mean_ls = [], []
    for thred_i, thred in enumerate(threds):
        AP_log_ls2_arr_mean_0_ls.append(repr(np.array(np.nanmean(AP_log_ls2[thred_i], axis=0)))) # np.mean(iou_log_arr, axis=0)
        AP_log_ls2_arr_mean_ls.append(np.mean(np.nanmean(AP_log_ls2[thred_i], axis=0))) # np.mean(iou_log_arr, axis=0)
    AP = np.mean(AP_log_ls2_arr_mean_ls)
    AP_p5_to_p95 = np.mean(AP_log_ls2_arr_mean_ls[-10:])
    
    AP_log_ls2_arr_mean_0_arr = repr(np.array(AP_log_ls2_arr_mean_0_ls))
    print('\n\n utils.py - eval_recon_core - np.shape(eucl_dist_log_arr_mean): ', np.shape(eucl_dist_log_arr_mean))
    print('\n\n utils.py - eval_recon_core - np.shape(eucl_dist_log_arr_mean_0): ', np.shape(eucl_dist_log_arr_mean_0))
    print('\n\n utils.py - eval_recon_core - np.shape(d_err_PRED_log_arr_mean_0): ', np.shape(d_err_PRED_log_arr_mean_0))
    print('\n\n utils.py - eval_recon_core - np.shape(d_err_FTM_log_arr_mean_0): ', np.shape(d_err_FTM_log_arr_mean_0))
    print('\n\n utils.py - eval_recon_core - np.shape(d_err_improvement_log_arr_mean_0): ', np.shape(d_err_improvement_log_arr_mean_0))
    print('\n\n utils.py - eval_recon_core - np.shape(dtw_log_arr_mean_0): ', np.shape(dtw_log_arr_mean_0))
    print('\n\n utils.py - eval_recon_core - np.shape(diou_loss_log_arr_mean_0): ', np.shape(diou_loss_log_arr_mean_0))
    print('\n\n utils.py - eval_recon_core - np.shape(iou_log_arr_mean_0): ', np.shape(iou_log_arr_mean_0))
    # print('\n\n utils.py - eval_recon_core - np.shape(AP_0p5_log_arr_mean_0): ', np.shape(AP_0p5_log_arr_mean_0))
    # print('\n\n utils.py - eval_recon_core - np.shape(AP_0p1_log_arr_mean_0): ', np.shape(AP_0p1_log_arr_mean_0))
    print('\n\n utils.py - eval_recon_core - np.shape(AP_log_ls2_arr_mean_0_arr): ', np.shape(AP_log_ls2_arr_mean_0_arr))

    test_log_pkl_dict = defaultdict()
    if C.eval_with_cumulative_turning_angle:
        C.test_log_file.write('turning_tracklets_cnt\n'); C.test_log_file.write('{}\n'.format(turning_tracklets_cnt))
        print('\n\n utils.py - eval_recon_core - turning_tracklets_cnt: ', turning_tracklets_cnt)
        test_log_pkl_dict['turning_tracklets_cnt'] = turning_tracklets_cnt

        C.test_log_file.write('tracklets_total\n'); C.test_log_file.write('{}\n'.format(tracklets_total))
        print('\n\n utils.py - eval_recon_core - tracklets_total: ', tracklets_total)
        test_log_pkl_dict['tracklets_total'] = tracklets_total

        turning_tracklets_rate = float(turning_tracklets_cnt) / float(tracklets_total)
        C.test_log_file.write('turning_tracklets_rate\n'); C.test_log_file.write('{}\n'.format(turning_tracklets_rate))
        print('\n\n utils.py - eval_recon_core - turning_tracklets_rate: ', turning_tracklets_rate)
        test_log_pkl_dict['turning_tracklets_rate'] = turning_tracklets_rate

    # print('\n\n utils.py - eval_recon_core - AP_0p5_log_arr_mean_0: ', AP_0p5_log_arr_mean_0)
    # print('\n\n utils.py - eval_recon_core - AP_0p1_log_arr_mean_0: ', AP_0p1_log_arr_mean_0)
    print('\n ============================================')
    for thred_i, thred in enumerate(threds):
        C.test_log_file.write('AP_log_ls2_arr_mean_0_ls[thred_i]\n'); C.test_log_file.write('{}\n'.format(AP_log_ls2_arr_mean_0_ls[thred_i]))
        print('\n utils.py - eval_recon_core - thred: ', thred, ', AP_log_ls2_arr_mean_0_ls[thred_i]: ', AP_log_ls2_arr_mean_0_ls[thred_i])
        test_log_pkl_dict['AP_log_ls2_arr_mean_0_ls[{}]'.format(thred_i)] = AP_log_ls2_arr_mean_0_ls[thred_i]

    print('\n --------------------------------------------')
    for thred_i, thred in enumerate(threds):
        C.test_log_file.write('AP_log_ls2_arr_mean_ls[thred_i]\n'); C.test_log_file.write('{}\n'.format(AP_log_ls2_arr_mean_ls[thred_i]))
        print('\n utils.py - eval_recon_core - thred: ', thred, ', AP_log_ls2_arr_mean_ls[thred_i]: ', AP_log_ls2_arr_mean_ls[thred_i])
        test_log_pkl_dict['AP_log_ls2_arr_mean_ls[{}]'.format(thred_i)] = AP_log_ls2_arr_mean_ls[thred_i]

    C.test_log_file.write('eucl_dist_log_arr_mean\n'); C.test_log_file.write('{}\n'.format(eucl_dist_log_arr_mean))
    print('\n utils.py - eval_recon_core - eucl_dist_log_arr_mean: ', eucl_dist_log_arr_mean)
    test_log_pkl_dict['eucl_dist_log_arr_mean'] = eucl_dist_log_arr_mean

    C.test_log_file.write('eucl_dist_log_arr_mean_0\n'); C.test_log_file.write('{}\n'.format(eucl_dist_log_arr_mean_0))
    print('\n utils.py - eval_recon_core - eucl_dist_log_arr_mean_0: ', eucl_dist_log_arr_mean_0)
    test_log_pkl_dict['eucl_dist_log_arr_mean_0'] = eucl_dist_log_arr_mean_0

    C.test_log_file.write('d_err_PRED_log_arr_mean_0\n'); C.test_log_file.write('{}\n'.format(d_err_PRED_log_arr_mean_0))
    print('\n utils.py - eval_recon_core - d_err_PRED_log_arr_mean_0: ', d_err_PRED_log_arr_mean_0)
    test_log_pkl_dict['d_err_PRED_log_arr_mean_0'] = d_err_PRED_log_arr_mean_0

    C.test_log_file.write('d_err_FTM_log_arr_mean_0\n'); C.test_log_file.write('{}\n'.format(d_err_FTM_log_arr_mean_0))
    print('\n utils.py - eval_recon_core - d_err_FTM_log_arr_mean_0: ', d_err_FTM_log_arr_mean_0)
    test_log_pkl_dict['d_err_FTM_log_arr_mean_0'] = d_err_FTM_log_arr_mean_0

    C.test_log_file.write('d_err_improvement_log_arr_mean_0\n'); C.test_log_file.write('{}\n'.format(d_err_improvement_log_arr_mean_0))
    print('\n utils.py - eval_recon_core - d_err_improvement_log_arr_mean_0: ', d_err_improvement_log_arr_mean_0)
    test_log_pkl_dict['d_err_improvement_log_arr_mean_0'] = d_err_improvement_log_arr_mean_0

    C.test_log_file.write('dtw_log_arr_mean_0\n'); C.test_log_file.write('{}\n'.format(dtw_log_arr_mean_0))
    print('\n utils.py - eval_recon_core - dtw_log_arr_mean_0: ', dtw_log_arr_mean_0)
    test_log_pkl_dict['dtw_log_arr_mean_0'] = dtw_log_arr_mean_0

    C.test_log_file.write('diou_loss_log_arr_mean_0\n'); C.test_log_file.write('{}\n'.format(diou_loss_log_arr_mean_0))
    print('\n utils.py - eval_recon_core - diou_loss_log_arr_mean_0: ', diou_loss_log_arr_mean_0)
    test_log_pkl_dict['diou_loss_log_arr_mean_0'] = diou_loss_log_arr_mean_0

    C.test_log_file.write('iou_log_arr_mean_0\n'); C.test_log_file.write('{}\n'.format(iou_log_arr_mean_0))
    print('\n utils.py - eval_recon_core - iou_log_arr_mean_0: ', iou_log_arr_mean_0)
    test_log_pkl_dict['iou_log_arr_mean_0'] = iou_log_arr_mean_0

    print('\n >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')

    C.test_log_file.write('MRF_log\n'); C.test_log_file.write('{}\n'.format(MRF_log))
    print('\n utils.py - eval_recon_core - MRF_log: ', MRF_log); test_log_pkl_dict['MRF_log'] = MRF_log

    C.test_log_file.write('MRF_win_i_ls_log\n'); C.test_log_file.write('{}\n'.format(MRF_win_i_ls_log))
    print('\n utils.py - eval_recon_core - MRF_win_i_ls_log: ', MRF_win_i_ls_log); test_log_pkl_dict['MRF_win_i_ls_log'] = MRF_win_i_ls_log

    C.test_log_file.write('MRF_win_1st_ts16s_log\n'); C.test_log_file.write('{}\n'.format(MRF_win_1st_ts16s_log))
    print('\n utils.py - eval_recon_core - MRF_win_1st_ts16s_log: ', MRF_win_1st_ts16s_log); test_log_pkl_dict['MRF_win_1st_ts16s_log'] = MRF_win_1st_ts16s_log

    C.test_log_file.write('MRF_win_intervals_log\n'); C.test_log_file.write('{}\n'.format(MRF_win_intervals_log))
    print('\n utils.py - eval_recon_core - MRF_win_intervals_log: ', MRF_win_intervals_log); test_log_pkl_dict['MRF_win_intervals_log'] = MRF_win_intervals_log

    C.test_log_file.write('MRF_frame_intervals_log\n'); C.test_log_file.write('{}\n'.format(MRF_frame_intervals_log))
    print('\n utils.py - eval_recon_core - MRF_frame_intervals_log: ', MRF_frame_intervals_log); test_log_pkl_dict['MRF_frame_intervals_log'] = MRF_frame_intervals_log

    C.test_log_file.write('iou_log_arr_mean\n'); C.test_log_file.write('{}\n'.format(iou_log_arr_mean))
    print('\n utils.py - eval_recon_core - iou_log_arr_mean: ', iou_log_arr_mean); test_log_pkl_dict['iou_log_arr_mean'] = iou_log_arr_mean

    # win stats
    C.test_log_file.write('MRF_win_interv_max\n'); C.test_log_file.write('{}\n'.format(MRF_win_interv_max))
    print('\n utils.py - eval_recon_core - MRF_win_interv_max: ', MRF_win_interv_max); test_log_pkl_dict['MRF_win_interv_max'] = MRF_win_interv_max

    C.test_log_file.write('MRF_win_interv_min\n'); C.test_log_file.write('{}\n'.format(MRF_win_interv_min))
    print('\n utils.py - eval_recon_core - MRF_win_interv_min: ', MRF_win_interv_min); test_log_pkl_dict['MRF_win_interv_min'] = MRF_win_interv_min

    C.test_log_file.write('MRF_win_interv_mean\n'); C.test_log_file.write('{}\n'.format(MRF_win_interv_mean))
    print('\n utils.py - eval_recon_core - MRF_win_interv_mean: ', MRF_win_interv_mean); test_log_pkl_dict['MRF_win_interv_mean'] = MRF_win_interv_mean

    C.test_log_file.write('MRF_win_interv_stdev\n'); C.test_log_file.write('{}\n'.format(MRF_win_interv_stdev))
    print('\n utils.py - eval_recon_core - MRF_win_interv_stdev: ', MRF_win_interv_stdev); test_log_pkl_dict['MRF_win_interv_stdev'] = MRF_win_interv_stdev

    # frame stats
    C.test_log_file.write('MRF_frame_interv_max\n'); C.test_log_file.write('{}\n'.format(MRF_frame_interv_max))
    print('\n utils.py - eval_recon_core - MRF_frame_interv_max: ', MRF_frame_interv_max); test_log_pkl_dict['MRF_frame_interv_max'] = MRF_frame_interv_max

    C.test_log_file.write('MRF_frame_interv_min\n'); C.test_log_file.write('{}\n'.format(MRF_frame_interv_min))
    print('\n utils.py - eval_recon_core - MRF_frame_interv_min: ', MRF_frame_interv_min); test_log_pkl_dict['MRF_frame_interv_min'] = MRF_frame_interv_min

    C.test_log_file.write('MRF_frame_interv_mean\n'); C.test_log_file.write('{}\n'.format(MRF_frame_interv_mean))
    print('\n utils.py - eval_recon_core - MRF_frame_interv_mean: ', MRF_frame_interv_mean); test_log_pkl_dict['MRF_frame_interv_mean'] = MRF_frame_interv_mean

    C.test_log_file.write('MRF_frame_interv_stdev\n'); C.test_log_file.write('{}\n'.format(MRF_frame_interv_stdev))
    print('\n utils.py - eval_recon_core - MRF_frame_interv_stdev: ', MRF_frame_interv_stdev); test_log_pkl_dict['MRF_frame_interv_stdev'] = MRF_frame_interv_stdev

    C.test_log_file.write('AP\n'); C.test_log_file.write('{}\n'.format(AP))
    print('\n utils.py - eval_recon_core - AP: ', AP); test_log_pkl_dict['AP'] = AP

    print('\n utils.py - eval_recon_core - thred: 0.5, AP_log_ls2_arr_mean_ls[0.5]: ', AP_log_ls2_arr_mean_ls[9])

    C.test_log_file.write('AP_p5_to_p95\n'); C.test_log_file.write('{}\n'.format(AP_p5_to_p95))
    print('\n utils.py - eval_recon_core - AP_p5_to_p95: ', AP_p5_to_p95); test_log_pkl_dict['AP_p5_to_p95'] = AP_p5_to_p95

    C.test_log_file.write('proc_n_frames_log\n'); C.test_log_file.write('{}\n'.format(proc_n_frames_log))
    print('\n utils.py - eval_recon_core - proc_n_frames_log: ', proc_n_frames_log); test_log_pkl_dict['proc_n_frames_log'] = proc_n_frames_log

    C.test_log_file.write('proc_n_wins_log\n'); C.test_log_file.write('{}\n'.format(proc_n_wins_log))
    print('\n utils.py - eval_recon_core - proc_n_wins_log: ', proc_n_wins_log); test_log_pkl_dict['proc_n_wins_log'] = proc_n_wins_log

    C.test_log_file.write('infer_time_sec_per_tracklet_log\n'); C.test_log_file.write('{}\n'.format(infer_time_sec_per_tracklet_log))
    print('\n utils.py - eval_recon_core - infer_time_sec_per_tracklet_log: ', infer_time_sec_per_tracklet_log)
    test_log_pkl_dict['infer_time_sec_per_tracklet_log'] = infer_time_sec_per_tracklet_log

    C.test_log_file.write('infer_time_sec_per_win_log\n'); C.test_log_file.write('{}\n'.format(infer_time_sec_per_win_log))
    print('\n utils.py - eval_recon_core - infer_time_sec_per_win_log: ', infer_time_sec_per_win_log)
    test_log_pkl_dict['infer_time_sec_per_win_log'] = infer_time_sec_per_win_log

    C.test_log_file.write('infer_time_sec_per_frame_log\n'); C.test_log_file.write('{}\n'.format(infer_time_sec_per_frame_log))
    print('\n utils.py - eval_recon_core - infer_time_sec_per_frame_log: ', infer_time_sec_per_frame_log)
    test_log_pkl_dict['infer_time_sec_per_frame_log'] = infer_time_sec_per_frame_log

    C.test_log_file.write('total_infer_time_sec_log\n'); C.test_log_file.write('{}\n'.format(total_infer_time_sec_log))
    print('\n utils.py - eval_recon_core - total_infer_time_sec_log: ', total_infer_time_sec_log)
    test_log_pkl_dict['total_infer_time_sec_log'] = total_infer_time_sec_log

    print('\n <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')

    print('\n n_wins: ', C.n_wins, ', win_stride: ', C.win_stride)
    

    # d_err_PRED_log_arr_mean_1 = np.mean(d_err_PRED_log_arr, axis=1)
    # d_err_FTM_log_arr_mean_1 = np.mean(d_err_FTM_log_arr, axis=1)
    # d_err_improvement_log_arr_mean_1 = np.mean(d_err_improvement_log_arr, axis=1)
    # diou_loss_log_arr_mean_1 = np.mean(diou_loss_log_arr, axis=1)
    # iou_log_arr_mean_1 = np.mean(iou_log_arr, axis=1)
    # print('\n\n utils.py - eval_recon_core - np.shape(d_err_PRED_log_arr_mean_1): ', np.shape(d_err_PRED_log_arr_mean_1))
    # print('\n\n utils.py - eval_recon_core - np.shape(d_err_FTM_log_arr_mean_1): ', np.shape(d_err_FTM_log_arr_mean_1))
    # print('\n\n utils.py - eval_recon_core - np.shape(d_err_improvement_log_arr_mean_1): ', np.shape(d_err_improvement_log_arr_mean_1))
    # print('\n\n utils.py - eval_recon_core - np.shape(diou_loss_log_arr_mean_1): ', np.shape(diou_loss_log_arr_mean_1))
    # print('\n\n utils.py - eval_recon_core - np.shape(iou_log_arr_mean_1): ', np.shape(iou_log_arr_mean_1))

    # DEBUG:
    # print('\n utils.py - eval_recon_core - np.shape(AP_0p5_log_arr): ', np.shape(AP_0p5_log_arr))
    # print('\n utils.py - eval_recon_core - np.shape(AP_0p1_log_arr): ', np.shape(AP_0p1_log_arr))
    # e.g. (2445, 50)
    C.test_log_file.write('np.shape(eucl_dist_log_arr)\n'); C.test_log_file.write('{}\n'.format(np.shape(eucl_dist_log_arr)))
    print('\n utils.py - eval_recon_core - np.shape(eucl_dist_log_arr): ', np.shape(eucl_dist_log_arr))
    test_log_pkl_dict['np_shape_eucl_dist_log_arr'] = np.shape(eucl_dist_log_arr)

    C.test_log_file.write('np.shape(d_err_PRED_log_arr)\n'); C.test_log_file.write('{}\n'.format(np.shape(d_err_PRED_log_arr)))
    print('\n utils.py - eval_recon_core - np.shape(d_err_PRED_log_arr): ', np.shape(d_err_PRED_log_arr))
    test_log_pkl_dict['np_shape_d_err_PRED_log_arr'] = np.shape(d_err_PRED_log_arr)

    C.test_log_file.write('np.shape(d_err_FTM_log_arr)\n'); C.test_log_file.write('{}\n'.format(np.shape(d_err_FTM_log_arr)))
    print('\n utils.py - eval_recon_core - np.shape(d_err_FTM_log_arr): ', np.shape(d_err_FTM_log_arr))
    test_log_pkl_dict['np_shape_d_err_FTM_log_arr'] = np.shape(d_err_FTM_log_arr)

    C.test_log_file.write('np.shape(d_err_improvement_log_arr)\n'); C.test_log_file.write('{}\n'.format(np.shape(d_err_improvement_log_arr)))
    print('\n utils.py - eval_recon_core - np.shape(d_err_improvement_log_arr): ', np.shape(d_err_improvement_log_arr))
    test_log_pkl_dict['np_shape_d_err_improvement_log_arr'] = np.shape(d_err_improvement_log_arr)

    C.test_log_file.write('np.shape(dtw_log_arr)\n'); C.test_log_file.write('{}\n'.format(np.shape(dtw_log_arr)))
    print('\n utils.py - eval_recon_core - np.shape(dtw_log_arr): ', np.shape(dtw_log_arr))
    test_log_pkl_dict['dtw_log_arr'] = np.shape(dtw_log_arr)

    C.test_log_file.write('np.shape(diou_loss_log_arr)\n'); C.test_log_file.write('{}\n'.format(np.shape(diou_loss_log_arr)))
    print('\n utils.py - eval_recon_core - np.shape(diou_loss_log_arr): ', np.shape(diou_loss_log_arr))
    test_log_pkl_dict['diou_loss_log_arr'] = np.shape(diou_loss_log_arr)

    C.test_log_file.write('np.shape(iou_log_arr)\n'); C.test_log_file.write('{}\n'.format(np.shape(iou_log_arr)))
    print('\n utils.py - eval_recon_core - np.shape(iou_log_arr): ', np.shape(iou_log_arr))
    test_log_pkl_dict['np_shape_iou_log_arr'] = np.shape(iou_log_arr)

    C.test_log_file.write('np.shape(AP_log_ls2_arr)\n'); C.test_log_file.write('{}\n'.format(np.shape(AP_log_ls2_arr)))
    print('\n utils.py - eval_recon_core - np.shape(AP_log_ls2_arr): ', np.shape(AP_log_ls2_arr))
    test_log_pkl_dict['np_shape_AP_log_ls2_arr'] = np.shape(AP_log_ls2_arr)

    C.test_log_file.flush(); print('\n ', C.test_log_file_path, ' saved!')
    with open(C.test_log_pkl_file_path, 'wb') as f:
        pickle.dump(test_log_pkl_dict, f); print('\n ', C.test_log_pkl_file_path, ' saved!')
    # END_OF_EVAL
    '''
    C.model_weights_path_to_save = C.checkpoint_path_dict[C.running_mode] + '/' + C.test_log_id + '_' + log_str + '_w_dfv4.ckpt'
    C.model.save_weights(C.model_weights_path_to_save)
    print(C.model_weights_path_to_save, 'saved!')

    C.ts16_dfv4_subj_i_to_BBX5_prime_path_to_save = C.checkpoint_path_dict[C.running_mode] + '/' + \
        C.test_log_id + '_' + log_str + '_dfv4_ts16_dfv4_subj_i_to_BBX5_prime.pkl'
    pickle.dump(C.ts16_dfv4_subj_i_to_BBX5_prime, open(C.ts16_dfv4_subj_i_to_BBX5_prime_path_to_save, 'wb'))
    print(C.ts16_dfv4_subj_i_to_BBX5_prime_path_to_save, 'saved!')

    C.ts16_dfv4_to_pred_BBX5_labels_path_to_save = C.checkpoint_path_dict[C.running_mode] + '/' + \
        C.test_log_id + '_' + log_str + '_dfv4_ts16_dfv4_to_pred_BBX5_labels.pkl'
    pickle.dump(C.ts16_dfv4_to_pred_BBX5_labels, open(C.ts16_dfv4_to_pred_BBX5_labels_path_to_save, 'wb'))
    print(C.ts16_dfv4_to_pred_BBX5_labels_path_to_save, 'saved!')

    C.scene_test_stats_path_to_save = C.checkpoint_path_dict[C.running_mode] + '/' + \
        C.test_log_id + '_' + log_str + '_dfv4_scene_test_stats.pkl'
    # pickle.dump(C.scene_test_stats_path_to_save, open(C.scene_test_stats_path_to_save, 'wb'))
    # print(C.scene_test_stats_path_to_save, 'saved!')
    '''

def BBX5_to_dBBX5(C: Config, BBX5, version, norm=False, img_rows=720, img_cols=1280):
    if BBX5 is None: return None
    len_win = np.shape(BBX5)[1]
    if version == 1:
        BBX5[:, 1:, :, :] = BBX5[:, 1:, :, :] - BBX5[:, :len_win - 1, :, :]
        # BBX5[:, :len_win - 1, :, :] = BBX5[:, 1:, :, :] - BBX5[:, :len_win - 1, :, :] # BugY2022M06D08Wed
    elif version == 2:
        for i in range(1, len_win):
            BBX5[:, i, :, :] = BBX5[:, i, :, :] - BBX5[:, 0, :, :]
    elif version == 3:
        BBX5[:, len_win - 1, :, :] = BBX5[:, len_win - 1, :, :] - BBX5[:, 0, :, :]
        BBX5[:, :len_win - 1, :, :] = 0
    BBX5[:, 0, :, :] = 0
    dBBX5 = BBX5

    if norm:
        dBBX5[:,:,:,0] = dBBX5[:,:,:,0] / img_cols
        dBBX5[:,:,:,1] = dBBX5[:,:,:,1] / img_rows
        dBBX5[:,:,:,2] = dBBX5[:,:,:,2] / C.max_depth[0]
    return dBBX5

def BBX5_helper(loss_func, BBX5):
    # print('\n\n utils.py - BBX5_helper - np.shape(BBX5): ', np.shape(BBX5))
    if BBX5 is not None and np.shape(BBX5)[-1] < 5:
        if 'MSE_xydwh' in loss_func: BBX5 = BBX5
        elif 'MSE_xy' in loss_func: BBX5 = BBX5[:,:,:,:2]
        elif 'MSE_xyd' in loss_func: BBX5 = BBX5[:,:,:,:3]
    return BBX5

def intersection(x0, y0, w0, h0, x1, y1, w1, h1):
    # debugging
    x0_left, x0_right, y0_top, y0_bottom = int(x0 - w0 / 2), int(x0 + w0 / 2), int(y0 - h0 / 2), int(y0 + h0 / 2)
    x1_left, x1_right, y1_top, y1_bottom = int(x1 - w1 / 2), int(x1 + w1 / 2), int(y1 - h1 / 2), int(y1 + h1 / 2)
    # x0_left, x0_right, y0_top, y0_bottom = x0 - w0 / 2, x0 + w0 / 2, y0 - h0 / 2, y0 + h0 / 2
    # x1_left, x1_right, y1_top, y1_bottom = x1 - w1 / 2, x1 + w1 / 2, y1 - h1 / 2, y1 + h1 / 2

    if x0_left > x1_right or x0_right < x1_left or y0_top > y1_bottom or y0_bottom < y1_top:
        return 0

    inter_x_left, inter_x_right = max(x0_left, x1_left), min(x0_right, x1_right)
    inter_y_top, inter_y_bottom = max(y0_top, y1_top), min(y0_bottom, y1_bottom)
    return (inter_x_right - inter_x_left) * (inter_y_bottom - inter_y_top)

def iou(BBX0_, BBX1_):
    # debugging
    # torch.Size([32, 10, 1, 5])
    x0, y0, w0, h0 = BBX0_[0], BBX0_[1], BBX0_[3], BBX0_[4]
    x1, y1, w1, h1 = BBX1_[0], BBX1_[1], BBX1_[3], BBX1_[4]
    return float(intersection(x0, y0, w0, h0, x1, y1, w1, h1)) / (w0 * h0 + w1 * h1)

def IOU_loss(loss_func, BBX0, BBX1):
    # print('\n\n utils.py - IOU_loss() - np.shape(BBX0): ', np.shape(BBX0))
    # print('\n\n utils.py - IOU_loss() - np.shape(BBX1): ', np.shape(BBX1))
    # e.g. utils.py - IOU_loss() - np.shape(BBX0):  torch.Size([32, 10, 1, 5])
    iou_loss_sum = 0
    for batch_i in range(np.shape(BBX0)[0]):
        for frame_i in range(np.shape(BBX0)[1]):
            BBX0_ = torch.squeeze(BBX0[batch_i, frame_i, 0, :])
            BBX1_ = torch.squeeze(BBX1[batch_i, frame_i, 0, :])
            # print('\n\n utils.py - IOU_loss() - np.shape(BBX0_): ', np.shape(BBX0_))
            # print('\n\n utils.py - IOU_loss() - np.shape(BBX1_): ', np.shape(BBX1_))
            # e.g. torch.Size([5])
            iou_loss_sum += iou(BBX0_, BBX1_)
    d_loss_sum = torch.mean((BBX0[:,:,:,2] - BBX1[:,:,:,2]) ** 2) if loss_func == 'IOU_d' else 0 # depth
    return iou_loss_sum * 3 + d_loss_sum

def labels_to_boxes(labels_a, labels_b):
    '''
    https://pytorch.org/vision/main/generated/torchvision.ops.distance_box_iou_loss.html
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format with 0 <= x1 < x2 and 0 <= y1 < y2, and
    The two boxes should have the same dimensions.
    boxes1 (Tensor[N, 4]) – first set of boxes
    boxes2 (Tensor[N, 4]) – second set of boxes
    '''
    # print('\n\n utils.py - labels_to_boxes np.shape(labels_a): ', np.shape(labels_a))
    # e.g. torch.Size([32, 30, 1, 5]) one window

    # boxes1 - a >>>
    x1_a, x2_a = labels_a[:,:,:,0] - labels_a[:,:,:,3] / 2, labels_a[:,:,:,0] + labels_a[:,:,:,3] / 2
    y1_a, y2_a = labels_a[:,:,:,1] - labels_a[:,:,:,4] / 2, labels_a[:,:,:,1] + labels_a[:,:,:,4] / 2
    # print('\n\n utils - label_to_boxes - np.shape(x1_a): ', np.shape(x1_a))
    # e.g. torch.Size([32, 10, 1])
    # print('\n\n utils - label_to_boxes - x1_a: ', x1_a)
    # shape_ = np.shape(labels_a)
    x1_a, x2_a = torch.unsqueeze(torch.flatten(x1_a), 1), torch.unsqueeze(torch.flatten(x2_a), 1)
    y1_a, y2_a = torch.unsqueeze(torch.flatten(y1_a), 1), torch.unsqueeze(torch.flatten(y2_a), 1)
    # print('\n\n utils - label_to_boxes - after flatten - np.shape(x1_a): ', np.shape(x1_a))
    # e.g. torch.Size([320, 1])

    boxes_a = torch.cat((x1_a, y1_a, x2_a, y2_a), 1)
    # print('\n\n utils - label_to_boxes - np.shape(boxes_a): ', np.shape(boxes_a))
    # e.g. torch.Size([320, 4])
    # boxes1 - a<<<

    # boxes2 - b >>>
    x1_b, x2_b = labels_b[:,:,:,0] - labels_b[:,:,:,3] / 2, labels_b[:,:,:,0] + labels_b[:,:,:,3] / 2
    y1_b, y2_b = labels_b[:,:,:,1] - labels_b[:,:,:,4] / 2, labels_b[:,:,:,1] + labels_b[:,:,:,4] / 2
    x1_b, x2_b = torch.unsqueeze(torch.flatten(x1_b), 1), torch.unsqueeze(torch.flatten(x2_b), 1)
    y1_b, y2_b = torch.unsqueeze(torch.flatten(y1_b), 1), torch.unsqueeze(torch.flatten(y2_b), 1)
    boxes_b = torch.cat((x1_b, y1_b, x2_b, y2_b), 1)
    # boxes2 - b <<<
    return boxes_a, boxes_b

def np_labels_to_boxes(labels_a, labels_b):
    '''
    https://pytorch.org/vision/main/generated/torchvision.ops.distance_box_iou_loss.html
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format with 0 <= x1 < x2 and 0 <= y1 < y2, and
    The two boxes should have the same dimensions.
    boxes1 (Tensor[N, 4]) – first set of boxes
    boxes2 (Tensor[N, 4]) – second set of boxes
    '''
    # print('\n\n utils.py - labels_to_boxes np.shape(labels_a): ', np.shape(labels_a))
    # e.g. torch.Size([32, 30, 1, 5]) one window

    # boxes1 - a >>>
    x1_a, x2_a = labels_a[:,:,:,0] - labels_a[:,:,:,3] / 2, labels_a[:,:,:,0] + labels_a[:,:,:,3] / 2
    y1_a, y2_a = labels_a[:,:,:,1] - labels_a[:,:,:,4] / 2, labels_a[:,:,:,1] + labels_a[:,:,:,4] / 2
    # print('\n\n utils - label_to_boxes - np.shape(x1_a): ', np.shape(x1_a))
    # e.g. ([32, 10, 1])
    # print('\n\n utils - label_to_boxes - x1_a: ', x1_a)
    # shape_ = np.shape(labels_a)
    x1_a, x2_a = np.expand_dims(x1_a.flatten(), axis=1), np.expand_dims(x2_a.flatten(), axis=1)
    y1_a, y2_a = np.expand_dims(y1_a.flatten(), axis=1), np.expand_dims(y2_a.flatten(), axis=1)
    print('\n\n utils - label_to_boxes - after flatten - np.shape(x1_a): ', np.shape(x1_a))
    # e.g. ([320, 1])

    boxes_a = np.concatenate((x1_a, y1_a, x2_a, y2_a), axis=1)
    # print('\n\n utils - label_to_boxes - np.shape(boxes_a): ', np.shape(boxes_a))
    # e.g. ([320, 4])
    # boxes1 - a<<<

    # boxes2 - b >>>
    x1_b, x2_b = labels_b[:,:,:,0] - labels_b[:,:,:,3] / 2, labels_b[:,:,:,0] + labels_b[:,:,:,3] / 2
    y1_b, y2_b = labels_b[:,:,:,1] - labels_b[:,:,:,4] / 2, labels_b[:,:,:,1] + labels_b[:,:,:,4] / 2
    x1_b, x2_b = np.expand_dims(x1_b.flatten(), axis=1), np.expand_dims(x2_b.flatten(), axis=1)
    y1_b, y2_b = np.expand_dims(y1_b.flatten(), axis=1), np.expand_dims(y2_b.flatten(), axis=1)
    boxes_b = np.concatenate((x1_b, y1_b, x2_b, y2_b), axis=1)
    # boxes2 - b <<<
    return boxes_a, boxes_b

def proc_distance_box_iou_loss(labels_a, labels_b):
    boxes_a, boxes_b = labels_to_boxes(labels_a, labels_b)
    return distance_box_iou_loss(boxes_a, boxes_b)

def proc_generalized_box_iou_loss(labels_a, labels_b):
    boxes_a, boxes_b = labels_to_boxes(labels_a, labels_b)
    return generalized_box_iou_loss(boxes_a, boxes_b)

def combined_NMSE_IOU_loss(C: Config, labels_a, labels_b, labels, labels_PRED):
    # print('\n\n utils.py - combined_NMSE_IOU_loss - C.loss_func: ', C.loss_func)
    # print('\n\n utils.py - combined_NMSE_IOU_loss - np.shape(labels_a): ', np.shape(labels_a))
    # print('\n\n utils.py - combined_NMSE_IOU_loss - np.shape(labels_b): ', np.shape(labels_b))
    # e.g. shape(labels_a): torch.Size([32, 10, 1, 5])
    labels_a = labels_a.float()
    labels_b = labels_b.float()

    if C.loss_func == 'DIOU_depth': # Best Loss: Train: 34.673, Val: 39.899
        depth_a, depth_b = torch.flatten(labels_a[:,:,:,2]), torch.flatten(labels_b[:,:,:,2])
        mse_loss = nn.MSELoss(reduction='none')
        depth_loss = mse_loss(depth_a, depth_b)
        # depth_loss = torch.tensor(depth_loss, dtype=torch.float)
        depth_loss = depth_loss.clone().detach().requires_grad_(True).float()
        return proc_distance_box_iou_loss(labels_a, labels_b) * 30 / (30 + 1) + depth_loss / (30 + 1)
    elif C.loss_func == 'GIOU_depth':
        depth_a, depth_b = torch.flatten(labels_a[:,:,:,2]), torch.flatten(labels_b[:,:,:,2])
        mse_loss = nn.MSELoss(reduction='none')
        depth_loss = mse_loss(depth_a, depth_b)
        # depth_loss = torch.tensor(depth_loss, dtype=torch.float)
        depth_loss = depth_loss.clone().detach().requires_grad_(True).float()
        return proc_generalized_box_iou_loss(labels_a, labels_b) * 30 / (30 + 1) + depth_loss / (30 + 1)
    elif C.loss_func == 'NMSE_xydwh_DIOU':
        # print('\n\n utils.py - combined_NMSE_IOU_loss - labels:', labels)
        # print('\n\n utils.py - combined_NMSE_IOU_loss - labels_PRED:', labels_PRED)
        # labels_ = BBX5_to_dBBX5(C, labels, 0, True).float()
        # labels_PRED_ = BBX5_to_dBBX5(C, labels_PRED, 0, True).float()
        labels_ = labels
        labels_PRED_ = labels_PRED
        # print('\n\n utils.py - combined_NMSE_IOU_loss - after normed - labels_:', labels_)
        # print('\n\n utils.py - combined_NMSE_IOU_loss - after normed - labels_PRED_:', labels_PRED_)

        labels_x_, labels_PRED_x_ = torch.flatten(labels_[:,:,:,0]), torch.flatten(labels_PRED_[:,:,:,0])
        labels_y_, labels_PRED_y_ = torch.flatten(labels_[:,:,:,1]), torch.flatten(labels_PRED_[:,:,:,1])
        labels_d_, labels_PRED_d_ = torch.flatten(labels_[:,:,:,2]), torch.flatten(labels_PRED_[:,:,:,2])
        # labels_d_ = labels_d_.cpu() / C.max_depth
        # labels_PRED_d_ = labels_PRED_d_.cpu() / C.max_depth
        labels_w_, labels_PRED_w_ = torch.flatten(labels_[:,:,:,3]), torch.flatten(labels_PRED_[:,:,:,3])
        labels_h_, labels_PRED_h_ = torch.flatten(labels_[:,:,:,4]), torch.flatten(labels_PRED_[:,:,:,4])
        mse_loss = nn.MSELoss(reduction='none')
        mse_x_loss = mse_loss(labels_x_, labels_PRED_x_)
        mse_y_loss = mse_loss(labels_y_, labels_PRED_y_)
        mse_d_loss = mse_loss(labels_d_, labels_PRED_d_)
        mse_w_loss = mse_loss(labels_w_, labels_PRED_w_)
        mse_h_loss = mse_loss(labels_h_, labels_PRED_h_)
        # mse_x_loss = torch.tensor(mse_x_loss, dtype=torch.float)
        # mse_y_loss = torch.tensor(mse_y_loss, dtype=torch.float)
        # mse_d_loss = torch.tensor(mse_d_loss, dtype=torch.float)
        # mse_w_loss = torch.tensor(mse_w_loss, dtype=torch.float)
        # mse_h_loss = torch.tensor(mse_h_loss, dtype=torch.float)
        mse_x_loss = mse_x_loss.clone().detach().requires_grad_(True).float()
        mse_y_loss = mse_x_loss.clone().detach().requires_grad_(True).float()
        mse_d_loss = mse_x_loss.clone().detach().requires_grad_(True).float()
        mse_w_loss = mse_x_loss.clone().detach().requires_grad_(True).float()
        mse_h_loss = mse_x_loss.clone().detach().requires_grad_(True).float()

        # DEBUG: check range
        # print('\n\n utils.py - mse_x_loss[:10]:', mse_x_loss[:10])
        # print('\n\n utils.py - mse_y_loss[:10]:', mse_y_loss[:10])
        # print('\n\n utils.py - mse_d_loss[:10]:', mse_d_loss[:10])
        # print('\n\n utils.py - mse_w_loss[:10]:', mse_w_loss[:10])
        # print('\n\n utils.py - mse_h_loss[:10]:', mse_h_loss[:10])

        return proc_distance_box_iou_loss(labels_a, labels_b) + mse_x_loss + mse_y_loss + mse_d_loss + mse_w_loss + mse_h_loss
    elif C.loss_func == 'NMSE_xydwh_GIOU':
        pass
    # Best Loss: Train: 1.626, Val: 1.644

#  ViFiT
# ===========
