import copy
import os
import time
import json
import glob
import numpy as np
import torch
import torch.nn as nn
from utils import count_model_parameters, update_test_log_config, eval_recon_core
import datetime
import pickle

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

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

class Trainer(object):
    ''' Training Helper Class '''
    def __init__(self, C, model, optimizer, save_path, device):
        self.C = C # Config() object
        self.all_cfg = C.__dict__ # cfg['all_cfg'] # config for training : see class Config
        if self.all_cfg['model_cfg']['model_type'] == 'pdr':
            self.model_w = model[0]; self.model_h = model[1]; self.model_d = model[2]
            self.optimizer_w = optimizer[0]; self.optimizer_h = optimizer[1]; self.optimizer_d = optimizer[2]
        else:
            self.model = model
            self.optimizer = optimizer
        self.save_path = save_path
        self.device = device # device name

    def make_exp_dir(self, running_mode):
        all_exp_paths = glob.glob(self.all_cfg['checkpoint_path_dict'][running_mode]\
            [:self.all_cfg['checkpoint_path_dict'][running_mode].index('exps') + len('exps')] + '/*')
        print('\n\n all_exp_paths: ', all_exp_paths)

        if len(all_exp_paths) > 0:
            all_exp_paths = sorted(all_exp_paths)
            all_exp_id_ls = [int(exp_path[exp_path.index('exps') + len('exps') + 1:]) for exp_path in all_exp_paths]
            # print(all_exp_paths)
            # print('\n\n all_exp_id_ls: ', all_exp_id_ls)
            self.all_cfg['exp_id_dict'][running_mode] = max(all_exp_id_ls) + 1
            print('\n\n self.all_cfg[exp_id_dict][running_mode]: ', self.all_cfg['exp_id_dict'][running_mode])
            self.all_cfg['checkpoint_path_dict'][running_mode] = \
                self.all_cfg['checkpoint_path_dict'][running_mode][:self.all_cfg['checkpoint_path_dict'][running_mode].\
                index('exps') + len('exps')] + '/' + str(self.all_cfg['exp_id_dict'][running_mode])
            # print('\n\n self.all_cfg[checkpoint_path_dict][running_mode]: ', self.all_cfg['checkpoint_path_dict'][running_mode])

        self.save_path = self.all_cfg['checkpoint_path_dict'][running_mode]
        print('\n self.save_path: ', self.save_path)

        if not os.path.exists(self.all_cfg['checkpoint_path_dict'][running_mode]):
            os.makedirs(self.all_cfg['checkpoint_path_dict'][running_mode])

    def pretrain(self, func_loss, func_forward, func_evaluate, \
        train_data_loader, val_data_loader, model_file=None, data_parallel=False):
        self.all_cfg['train_cfg']['type'] = 'pretrain'
        self.make_exp_dir(self.all_cfg['train_cfg']['type'])

        ''' Train Loop '''
        self.load(model_file)
        model = self.model.to(self.device)
        if data_parallel: # use Data Parallelism with Multi-GPU
            model = nn.DataParallel(model)

        global_step = 0 # global iteration steps regardless of epochs
        best_val_loss = 1e6
        model_best = model.state_dict()

        for epoch in range(self.all_cfg['train_cfg']['n_epochs']):
            sum_loss = 0. # the sum of iteration losses to get average loss in every epoch
            time_sum = 0.0
            self.model.train()
            for i, batch in enumerate(train_data_loader):
                batch = [t.to(self.device) for t in batch] # ORI
                # batch = [t.to(self.device).float() for t in batch] # Don't use as the second array should be in type Long, which are masked_pos

                # DEBUG:
                # print('\n\n')
                # print('train.py - Trainer - pretrain() - batch[0]: ', batch[0])
                # print('train.py - Trainer - pretrain() - batch[1]: ', batch[1]) # masked_pos
                # print('train.py - Trainer - pretrain() - batch[2]: ', batch[2])
                # print('train.py - Trainer - pretrain() - np.shape(batch): ', np.shape(batch)) # e.g. (3, )
                # print('train.py - Trainer - pretrain() - np.shape(batch[0]): ', np.shape(batch[0])) # e.g. torch.Size([32, 10, 1, 9], prev torch.Size([32, 10, 9]
                # print('train.py - Trainer - pretrain() - np.shape(batch[1]): ', np.shape(batch[1])) # e.g. torch.Size([32, 2])
                # print('train.py - Trainer - pretrain() - np.shape(batch[2]): ', np.shape(batch[2])) # e.g. torch.Size([32, 2, 1, 9]), prev torch.Size([32, 2, 9])

                batch[0], batch[2] = batch[0].float(), batch[2].float() # convert data type except masked_pos in [1]

                start_time = time.time()
                self.optimizer.zero_grad()
                loss = func_loss(model, batch)
                # DEBUG:
                # print('\n\n train.py - Trainer() - pretrain() - model: ', model)
                # print('\n\n train.py - Trainer() - pretrain() - batch: ', batch) # e.g. torch.float64
                # batch = batch.float()

                loss = loss.mean() # mean() for Data Parallelism
                loss.backward()
                self.optimizer.step()
                time_sum += time.time() - start_time
                global_step += 1
                sum_loss += loss.item()

                if self.all_cfg['train_cfg']['total_steps'] and self.all_cfg['train_cfg']['total_steps'] < global_step:
                    print('The Total Steps have been reached.')
                    return
                # print(i)

            val_loss = self.run(func_forward, func_evaluate, val_data_loader)
            avg_train_loss = sum_loss / len(train_data_loader)
            print('Epoch %d/%d: Sum Loss: %5.4f, Average Train Loss: %5.4f, Train Loss: %0.3f, Val Loss: %0.3f, # Training Samples: %d'
                  % (epoch + 1, self.all_cfg['train_cfg']['n_epochs'], sum_loss, avg_train_loss, loss, val_loss, len(train_data_loader)))
            print('train_model_id: ', self.all_cfg['args_dict']['train_model_id'], ', delta_BBX5_version: ', self.all_cfg['delta_BBX5_version'])
            print('len_win: ', self.all_cfg['args_dict']['len_win'])
            self.all_cfg['train_cfg']['last_avg_train_loss'] = avg_train_loss
            self.all_cfg['train_cfg']['last_sum_train_loss'] = sum_loss
            self.all_cfg['train_cfg']['len_train_data_loader'] = len(train_data_loader)
            self.all_cfg['train_cfg']['last_val_loss'] = val_loss
            # print("Train execution time: %.5f seconds" % (time_sum / len(self.data_loader)))

            if global_step == 0 or epoch % self.all_cfg['train_cfg']['save_model_epoch_interval'] == 0:
                # Save all_cfg and model
                self.all_cfg['train_cfg']['global_epochs_done'] = epoch
                self.save(epoch, self.all_cfg)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                # Save all_cfg and model
                self.all_cfg['train_cfg']['best_val_loss'] = best_val_loss
                self.all_cfg['train_cfg']['best_avg_train_loss'] = avg_train_loss
                self.all_cfg['train_cfg']['best_sum_train_loss'] = sum_loss
                self.all_cfg['train_cfg']['global_epochs_done'] = epoch
                model_best = copy.deepcopy(model.state_dict());
                self.save('best', self.all_cfg)

        model.load_state_dict(model_best)
        print('Training complete!.')
        # Save all_cfg and model
        self.all_cfg['train_cfg']['global_epochs_done'] = self.all_cfg['train_cfg']['n_epochs']; self.save(self.all_cfg['train_cfg']['n_epochs'], self.all_cfg)

    def run(self, func_forward, func_evaluate, data_loader, \
        model_file=None, data_parallel=False, load_self=False, output_embed=False, \
        reconstruction_path=None):
        ''' Evaluation Loop '''
        # DEBUG:
        print('\n\n train.py - Trainer() - run() - output_embed: ', output_embed)
        self.model.eval() # evaluation mode
        self.load(model_file, load_self=load_self)
        print('\n\n train.py - Trainer() - run() - model_file: ', model_file)
        # print(count_model_parameters(self.model))
        model = self.model.to(self.device)
        if data_parallel: # use Data Parallelism with Multi-GPU
            model = nn.DataParallel(model)

        # embeds = [] # prediction results
        embed_ls, sub_seq_REC_ls, masked_frames_ORI_ls, BBX5_ls, BBX5_PRED_ls = [], [], [], [], []
        GT_ls, PRED_ls = [], []
        time_sum = 0.0
        for batch in data_loader:
            if self.all_cfg['model_cfg']['train_model_id'] == 'Xformer_IFcC2C' and \
                np.shape(batch[0])[0] != self.all_cfg['train_cfg']['batch_size']: continue
            elif ('Width_regressor' in self.all_cfg['model_cfg']['train_model_id'] or \
                    'Height_regressor' in self.all_cfg['model_cfg']['train_model_id'] or \
                    'Depth_regressor' in self.all_cfg['model_cfg']['train_model_id']) and \
                np.shape(batch)[0] != self.all_cfg['train_cfg']['batch_size']: continue
            # if output_embed: batch = batch[-1] # -1: seq_in_IMU_TX from [seq_in_BBX5, seq_in_FTM2, seq_in_IMU19, seq_in_IMU_TX]
            if reconstruction_path != 'y2w' and reconstruction_path != 'y2h' and reconstruction_path != 'y2d': batch = [t.to(self.device) for t in batch]
            # print('train.py - Trainer - run() - np.shape(batch): ', np.shape(batch)) # e.g. (3,)

            # print('\n\n train.py - Trainer - run() - for batch in data_loader - reconstruction_path: ', reconstruction_path) # e.g.
            if reconstruction_path == 'IemcC2C':
                embeds, BBX5 = batch
            elif reconstruction_path == 'IFcC2C':
                # print('\n train.py - Trainer - run - reconstruction_path == Xformer_IFcC2C - np.shape(batch): ', np.shape(batch))
                # print('\n train.py - Trainer - run - reconstruction_path == Xformer_IFcC2C - np.shape(batch[0]): ', np.shape(batch[0]))
                # print('\n train.py - Trainer - run - reconstruction_path == Xformer_IFcC2C - np.shape(batch[1]): ', np.shape(batch[1]))
                # print('\n train.py - Trainer - run - reconstruction_path == Xformer_IFcC2C - np.shape(batch[2]): ', np.shape(batch[2]))
                '''
                e.g.
                train.py - Trainer - run - reconstruction_path == Xformer_IFcC2C - np.shape(batch):  (3,)
                train.py - Trainer - run - reconstruction_path == Xformer_IFcC2C - np.shape(batch[0]):  torch.Size([32, 10, 1, 5])
                train.py - Trainer - run - reconstruction_path == Xformer_IFcC2C - np.shape(batch[1]):  torch.Size([32, 10, 1, 2])
                train.py - Trainer - run - reconstruction_path == Xformer_IFcC2C - np.shape(batch[2]):  torch.Size([32, 10, 1, 9])
                '''
                BBX5 = batch[0]
            elif reconstruction_path == 'y2w':
                # print('\n\n run() np.shape(batch): ', np.shape(batch))
                # e.g. torch.Size([32, 1, 2])
                batch_y = batch[:, :, 0].to(self.device).float() #; batch_y = [t.to(self.device) for t in batch_y]
                batch_w = batch[:, :, 1].to(self.device).float() #; batch_y = [t.to(self.device) for t in batch_w]
            elif reconstruction_path == 'y2h':
                # print('\n\n run() np.shape(batch): ', np.shape(batch))
                # e.g. torch.Size([32, 1, 2])
                batch_y = batch[:, :, 0].to(self.device).float() #; batch_y = [t.to(self.device) for t in batch_y]
                batch_h = batch[:, :, 1].to(self.device).float() #; batch_y = [t.to(self.device) for t in batch_h]
            elif reconstruction_path == 'y2d':
                # print('\n\n run() np.shape(batch): ', np.shape(batch))
                # e.g. torch.Size([32, 1, 2])
                batch_y = batch[:, :, 0].to(self.device).float() #; batch_y = [t.to(self.device) for t in batch_y]
                batch_d = batch[:, :, 1].to(self.device).float() #; batch_y = [t.to(self.device) for t in batch_h]
            elif output_embed:
                seq_in_BBX5, seq_in_FTM2, seq_in_IMU_TX = batch
                # DEBUG:
                # print('train.py - Trainer() - run() - np.shape(seq_in_BBX5): ', np.shape(seq_in_BBX5))
                # print('train.py - Trainer() - run() - np.shape(seq_in_FTM2): ', np.shape(seq_in_FTM2))
                # print('train.py - Trainer() - run() - np.shape(seq_in_IMU_TX): ', np.shape(seq_in_IMU_TX))
                '''
                e.g.
                train.py - Trainer() - run() - np.shape(seq_in_BBX5):  torch.Size([32, 10, 1, 5])
                train.py - Trainer() - run() - np.shape(seq_in_FTM2):  torch.Size([32, 10, 1, 2])
                train.py - Trainer() - run() - np.shape(seq_in_IMU_TX):  torch.Size([32, 10, 1, 9])
                '''
            else:
                full_win_seq, masked_pos, masked_frames_ORI = batch # New # Note that only full_win_seq is used to generate embeddings

            with torch.no_grad(): # evaluation without gradient calculation
                start_time = time.time()
                if reconstruction_path == 'IemcC2C' or reconstruction_path == 'IFcC2C':
                    BBX5_PRED = func_forward(model, batch, self.C.model_cfg['transformer_Tc_in'])
                    BBX5_PRED_ls.append(BBX5_PRED)
                    BBX5_ls.append(BBX5)
                    # print('\n train.py - Trainer - run - with torch.no_grad() - np.shape(batch): ', np.shape(batch))
                    # print('\n train.py - Trainer - run - with torch.no_grad() - len(BBX5_PRED_ls): ', np.shape(BBX5_PRED_ls))
                    # print('\n train.py - Trainer - run - with torch.no_grad() - len(BBX5_ls): ', np.shape(BBX5_ls))
                # el:
                #     BBX5_PRED = func_forward(model, batch)
                #     # print('\n train.py - Trainer - run - reconstruction_path == Xformer_IFcC2C - np.shape(batch): ', np.shape(batch))
                #     # print('\n train.py - Trainer - run - reconstruction_path == Xformer_IFcC2C - np.shape(batch[0]): ', np.shape(batch[0]))
                #     # print('\n train.py - Trainer - run - reconstruction_path == Xformer_IFcC2C - np.shape(batch[1]): ', np.shape(batch[1]))
                #     # print('\n train.py - Trainer - run - reconstruction_path == Xformer_IFcC2C - np.shape(batch[2]): ', np.shape(batch[2]))
                #     # print('\n train.py - Trainer - run - reconstruction_path == Xformer_IFcC2C - np.shape(BBX5_PRED): ', np.shape(BBX5_PRED))
                #     '''
                #     e.g.
                #     train.py - Trainer - run - reconstruction_path == Xformer_IFcC2C - np.shape(batch):  (3,)
                #     train.py - Trainer - run - reconstruction_path == Xformer_IFcC2C - np.shape(output):  torch.Size([32, 10, 1, 5])
                #     '''
                #     BBX5_PRED_ls.append(BBX5_PRED)
                #     BBX5_ls.append(BBX5)
                elif reconstruction_path == 'y2w':
                    # print('\n model: ', model)
                    w_PRED = func_forward(model, batch_y)
                    PRED_ls.append(w_PRED)
                    GT_ls.append(batch_w) # batch_y)
                    # print('w_PRED: ', w_PRED, ', \n batch_w: ', batch_w)
                    # print('np.shape(w_PRED): ', np.shape(w_PRED)) # torch.Size([1])
                    # print('np.shape(batch_w): ', np.shape(batch_w)) # torch.Size([7000, 1])
                    # print('len(PRED_ls): ', len(PRED_ls)) # 1
                    # print('len(GT_ls): ', len(GT_ls)) # 1
                elif reconstruction_path == 'y2h':
                    h_PRED = func_forward(model, batch_y)
                    PRED_ls.append(h_PRED)
                    GT_ls.append(batch_h) # batch_y)
                elif reconstruction_path == 'y2d':
                    d_PRED = func_forward(model, batch_y)
                    PRED_ls.append(d_PRED)
                    GT_ls.append(batch_d) # batch_y)
                elif output_embed:
                    # DEBUG:
                    # print('\n train.py - Trainer - run - if output_embed: np.shape(batch): ', np.shape(batch))
                    # e.g. (3,)
                    # print('\n train.py - Trainer - run - if output_embed: np.shape(seq_in_IMU_TX): ', np.shape(seq_in_IMU_TX)) \
                    # e.g. torch.Size([32, 10, 1, 9])
                    embed = func_forward(model, seq_in_IMU_TX) # batch) # full_win_seq) # batch) # BUG HERE
                    embed_ls.append(embed)
                else:
                    sub_seq_REC, masked_frames_ORI = func_forward(model, batch) # full_win_seq) # batch) # BUG HERE
                    # print('train.py - Trainer - run() - np.shape(sub_seq_REC): ', np.shape(sub_seq_REC))
                    # print('train.py - Trainer - run() - np.shape(masked_frames_ORI): ', np.shape(masked_frames_ORI))
                    sub_seq_REC_ls.append(sub_seq_REC)
                    masked_frames_ORI_ls.append(masked_frames_ORI)

                time_sum += time.time() - start_time
        # print("Eval execution time: %.5f seconds" % (time_sum / len(dt)))

        if output_embed:
            if reconstruction_path == None or reconstruction_path == '':
                # DEBUG:
                # print('train.py - Trainer - run() - if func_evaluate: np.shape(np.array(embed_ls)): ', np.shape(np.array(embed_ls)))
                # print(HERE)
                return torch.cat(embed_ls, 0).cpu().numpy()
            elif reconstruction_path == 'IemcC2C' or reconstruction_path == 'IFcC2C':
                return torch.cat(BBX5_PRED_ls, 0).cpu().numpy()
            else:
                return None
        elif func_evaluate:
            if reconstruction_path == None or reconstruction_path == '':
                # full_win_seq, masked_pos, masked_frames_ORI = batch # batch should be in shape of (3,) at this point
                # DEBUG:
                print('train.py - Trainer - run() - if func_evaluate: np.shape(masked_frames_ORI_ls): ', np.shape(masked_frames_ORI_ls))
                print('train.py - Trainer - run() - if func_evaluate: np.shape(sub_seq_REC_ls): ', np.shape(sub_seq_REC_ls))
                return func_evaluate(torch.cat(masked_frames_ORI_ls, 0), torch.cat(sub_seq_REC_ls, 0))
                # return torch.cat(embeds, 0)
                # return func_evaluate(torch.cat(labels, 0), torch.cat(results, 0)) # ORI
            elif reconstruction_path == 'IemcC2C' or reconstruction_path == 'IFcC2C':
                return func_evaluate(torch.cat(BBX5_ls, 0), torch.cat(BBX5_PRED_ls, 0))
            elif reconstruction_path == 'y2w' or reconstruction_path == 'y2h' or reconstruction_path == 'y2d':
                # print('np.shape(GT_ls): ', np.shape(GT_ls), ', np.shape(PRED_ls): ', np.shape(PRED_ls))
                print('\n GT_ls: ', GT_ls); print('\n PRED_ls: ', PRED_ls)
                # print(np.shape(PRED_ls[0])) # torch.Size([6000, 1])
                # for i in range(np.shape(PRED_ls[0])[0]):
                #     if PRED_ls[0][i] > 500:
                #         print(PRED_ls[0][i])
                #         hhh
                return func_evaluate(torch.cat(GT_ls, 0), torch.cat(PRED_ls, 0))
            else:
                return None
        else:
            return None

    def eval_recon(self, C, func_forward, func_evaluate, test_seq_in_correspondence, \
        model_file=None, data_parallel=False, load_self=False, output_embed=False):
        ''' Evaluation Loop '''
        self.make_exp_dir('test')
        self.C = update_test_log_config(self.C)
        reconstruction_path = C.reconstruction_path

        # DEBUG:
        # print('\n\n self.C.model_cfg[model_type]: ', self.C.model_cfg['model_type']); HERE
        # translator

        if self.C.model_cfg['model_type'] == 'transformer' or self.C.model_cfg['model_type'] == 'lstm':
            # DEBUG:
            print('\n\n train.py - Trainer() - eval_recon() - output_embed: ', output_embed)
            self.model.eval() # evaluation mode
            print('\n\n train.py - Trainer() - eval_recon() - C.load_train_exp_id_model: ', C.load_train_exp_id_model)
            if C.load_train_exp_id_model:
                self.load(model_file, load_self=load_self)

            print('\n\n train.py - Trainer() - eval_recon() - model_file: ', model_file)
            # print(count_model_parameters(self.model))
            model = self.model.to(self.device)
            if data_parallel: # use Data Parallelism with Multi-GPU
                model = nn.DataParallel(model)
        elif self.C.model_cfg['model_type'] == 'translator':
            self.model.load_weights(model_file); print(model_file, ' loaded!')
            model = self.model
        elif self.C.model_cfg['model_type'] == 'pdr':
            # DEBUG:
            print('\n\n train.py - Trainer() - eval_recon() - output_embed: ', output_embed)
            self.model_w.eval(); self.model_h.eval() # evaluation mode
            print('\n\n train.py - Trainer() - eval_recon() - C.load_train_exp_id_model: ', C.load_train_exp_id_model)
            if C.load_train_exp_id_model:
                self.load(model_file, load_self=load_self)

            print('\n\n train.py - Trainer() - eval_recon() - model_file: ', model_file)
            # print(count_model_parameters(self.model))
            model = (self.model_w.to(self.device), self.model_h.to(self.device), self.model_d.to(self.device))
        elif self.C.model_cfg['model_type'] == 'kalman_filter':
            model = self.model
        else: model = None

        # embeds = [] # prediction results
        time_sum = 0.0

        # ===================================
        #  >>> Iterate Over The Whole Video

        # Main method for reconstruction evaluation
        eval_recon_core(self.C, func_forward, model, test_seq_in_correspondence, self.device)

        # =================================
        #  May not be used for now yet >>>
        # if C.scene_id == 0: C.RGBg_ts16_dfv4_ls_path = C.seq_path_for_model + '/RGBh_ts16_dfv4_ls.json'
        # else: C.RGBg_ts16_dfv4_ls_path = C.seq_path_for_model + '/RGB_ts16_dfv4_ls.json'
        # print('C.RGBg_ts16_dfv4_ls_path: ', C.RGBg_ts16_dfv4_ls_path)
        # with open(C.RGBg_ts16_dfv4_ls_path, 'r') as f:
        #     C.RGBg_ts16_dfv4_ls = json.load(f)
        #     print(C.RGBg_ts16_dfv4_ls_path, 'loaded!')
        #     print('C.RGBg_ts16_dfv4_ls[:5]: ', C.RGBg_ts16_dfv4_ls[:5]) # verification
        # print('\n\n train.py - eval_recon - C.test_seq_id: ', C.test_seq_id)
        # print('\n\n train.py - eval_recon - C.img_path: ', C.img_path)
        # for img_path_ in glob.glob(C.img_path + '/*.jpg'):
        #     print('img_path_: ', img_path_)

        # for batch in data_loader: # TODO
        #     # if output_embed: batch = batch[-1] # -1: seq_in_IMU_TX from [seq_in_BBX5, seq_in_FTM2, seq_in_IMU19, seq_in_IMU_TX]
        #     batch = [t.to(self.device) for t in batch]
        #     # print('train.py - Trainer - run() - np.shape(batch): ', np.shape(batch)) # e.g. (3,)
        #
        #     # DEBUG:
        #     # print('\n\n train.py - Trainer - run() - for batch in data_loader - reconstruction_path: ', reconstruction_path) # e.g.
        #     if reconstruction_path == 'IemcC2C':
        #         embeds, BBX5 = batch
        #     elif reconstruction_path == 'IFcC2C':
        #         # print('\n train.py - Trainer - run - reconstruction_path == Xformer_IFcC2C - np.shape(batch): ', np.shape(batch))
        #         # print('\n train.py - Trainer - run - reconstruction_path == Xformer_IFcC2C - np.shape(batch[0]): ', np.shape(batch[0]))
        #         # print('\n train.py - Trainer - run - reconstruction_path == Xformer_IFcC2C - np.shape(batch[1]): ', np.shape(batch[1]))
        #         # print('\n train.py - Trainer - run - reconstruction_path == Xformer_IFcC2C - np.shape(batch[2]): ', np.shape(batch[2]))
        #         '''
        #         e.g.
        #         train.py - Trainer - run - reconstruction_path == Xformer_IFcC2C - np.shape(batch):  (3,)
        #         train.py - Trainer - run - reconstruction_path == Xformer_IFcC2C - np.shape(batch[0]):  torch.Size([32, 10, 1, 5])
        #         train.py - Trainer - run - reconstruction_path == Xformer_IFcC2C - np.shape(batch[1]):  torch.Size([32, 10, 1, 2])
        #         train.py - Trainer - run - reconstruction_path == Xformer_IFcC2C - np.shape(batch[2]):  torch.Size([32, 10, 1, 9])
        #         '''
        #         BBX5 = batch[0]
        #     elif output_embed:
        #         seq_in_BBX5, seq_in_FTM2, seq_in_IMU_TX = batch
        #         # DEBUG:
        #         # print('train.py - Trainer() - run() - np.shape(seq_in_BBX5): ', np.shape(seq_in_BBX5))
        #         # print('train.py - Trainer() - run() - np.shape(seq_in_FTM2): ', np.shape(seq_in_FTM2))
        #         # print('train.py - Trainer() - run() - np.shape(seq_in_IMU_TX): ', np.shape(seq_in_IMU_TX))
        #         '''
        #         e.g.
        #         train.py - Trainer() - run() - np.shape(seq_in_BBX5):  torch.Size([32, 10, 1, 5])
        #         train.py - Trainer() - run() - np.shape(seq_in_FTM2):  torch.Size([32, 10, 1, 2])
        #         train.py - Trainer() - run() - np.shape(seq_in_IMU_TX):  torch.Size([32, 10, 1, 9])
        #         '''
        #     else:
        #         full_win_seq, masked_pos, masked_frames_ORI = batch # New # Note that only full_win_seq is used to generate embeddings
        #
        #     with torch.no_grad(): # evaluation without gradient calculation
        #         start_time = time.time()
        #         # ============
        #         #  Eval Recon
        #         # ============
        #         if reconstruction_path == 'IemcC2C' or reconstruction_path == 'IFcC2C':
        #             BBX5_PRED = func_forward(model, batch)
        #             print('\n train.py - Trainer - run() - with torch.no_grad() - np.shape(batch): ', np.shape(batch))
        #             print('\n train.py - Trainer - run() - with torch.no_grad() - np.shape(batch[0]): ', np.shape(batch[0]))
        #             print('\n train.py - Trainer - run() - with torch.no_grad() - np.shape(batch[1]): ', np.shape(batch[1]))
        #             print('\n train.py - Trainer - run() - with torch.no_grad() - np.shape(batch[2]): ', np.shape(batch[2]))
        #             print('\n train.py - Trainer - run() - with torch.no_grad() - np.shape(BBX5): ', np.shape(BBX5))
        #             print('\n train.py - Trainer - run() - with torch.no_grad() - BBX5: ', BBX5)
        #             print('\n train.py - Trainer - run() - with torch.no_grad() - np.shape(BBX5_PRED): ', np.shape(BBX5_PRED))
        #             print('\n train.py - Trainer - run() - with torch.no_grad() - BBX5_PRED: ', BBX5_PRED)
        #             print('------------------------------------------------------------------------')
        #
        #         elif output_embed:
        #             # DEBUG:
        #             # print('\n train.py - Trainer - run - if output_embed: np.shape(batch): ', np.shape(batch))
        #             # e.g. (3,)
        #             # print('\n train.py - Trainer - run - if output_embed: np.shape(seq_in_IMU_TX): ', np.shape(seq_in_IMU_TX)) \
        #             # e.g. torch.Size([32, 10, 1, 9])
        #             embed = func_forward(model, seq_in_IMU_TX) # batch) # full_win_seq) # batch) # BUG HERE
        #         else:
        #             sub_seq_REC, masked_frames_ORI = func_forward(model, batch) # full_win_seq) # batch) # BUG HERE
        #             # print('train.py - Trainer - run() - np.shape(sub_seq_REC): ', np.shape(sub_seq_REC))
        #             # print('train.py - Trainer - run() - np.shape(masked_frames_ORI): ', np.shape(masked_frames_ORI))
        #
        #         time_sum += time.time() - start_time
        # # print("Eval execution time: %.5f seconds" % (time_sum / len(dt)))
        # else:
        #     return None
        #  May not be used for now yet <<<
        # =================================

    def train(self, func_loss, func_forward, func_evaluate, train_data_loader, val_data_loader, \
        model_file=None, data_parallel=False, load_self=False, reconstruction_path=None):
        self.all_cfg['train_cfg']['type'] = 'train'
        # self.all_cfg['train_cfg']['loss_func'] = 'MSE'
        self.make_exp_dir(self.all_cfg['train_cfg']['type'])

        if self.all_cfg['model_cfg']['train_model_id'] == 'Xformer_IFcC2C':
            ''' Train Loop for xformer'''
            self.load(model_file, load_self)
            model = self.model.to(self.device)
            if data_parallel: # use Data Parallelism with Multi-GPU
                model = nn.DataParallel(model)

            global_step = 0 # global iteration steps regardless of epochs
            best_val_loss = 0.0
            best_stat = None
            model_best = model.state_dict()

            # Iterate over all training samples
            for epoch in range(self.all_cfg['train_cfg']['n_epochs']):
                sum_loss = 0.0 # the sum of iteration losses to get average loss in every epoch
                time_sum = 0.0
                self.model.train()
                for i, batch in enumerate(train_data_loader):
                    batch = [t.to(self.device) for t in batch]
                    # print('\n\n train.py - Trainer() - train() - np.shape(batch): ', np.shape(batch)) # e.g. (3,)
                    # print('\n\n train.py - Trainer() - train() - np.shape(batch[0]): ', np.shape(batch[0])) # e.g. torch.Size([32, 30, 1, 5])
                    # print('\n\n train.py - Trainer() - train() - np.shape(batch[1]): ', np.shape(batch[1])) # e.g. torch.Size([32, 30, 1, 2])
                    # print('\n\n train.py - Trainer() - train() - np.shape(batch[2]): ', np.shape(batch[2])) # e.g. torch.Size([32, 30, 1, 9])
                    # print('\n\n train.py - Trainer() - train() - reconstruction_path: ', reconstruction_path)
                    if self.all_cfg['model_cfg']['train_model_id'] == 'Xformer_IFcC2C' and \
                        np.shape(batch[0])[0] != self.all_cfg['train_cfg']['batch_size']: continue

                    start_time = time.time()
                    self.optimizer.zero_grad()
                    # print('\n\n self.all_cfg[model_cfg][transformer_Tc_in]: ', self.all_cfg['model_cfg']['transformer_Tc_in'])
                    # print('\n\n len(batch): ', len(batch))
                    loss = func_loss(model, batch, self.all_cfg['model_cfg']['transformer_Tc_in'])
                    # print('\n\n train.py - Trainer() - train() - loss): ', loss)

                    loss = loss.mean() # mean() for Data Parallelism
                    # print('\n\n train.py - Trainer() - train() - type(loss)): ', type(loss))
                    loss.backward()
                    self.optimizer.step()

                    global_step += 1
                    sum_loss += loss.item()
                    time_sum += time.time() - start_time

                    if self.all_cfg['train_cfg']['total_steps'] and self.all_cfg['train_cfg']['total_steps'] < global_step:
                        print('The Total Steps have been reached.')
                        return

                train_loss = self.run(func_forward, func_evaluate, train_data_loader, \
                    None, False, False, False, reconstruction_path)
                # test_loss, test_loss = self.run(func_forward, func_evaluate, data_loader_test)
                val_loss = self.run(func_forward, func_evaluate, val_data_loader, \
                    None, False, False, False, reconstruction_path)
                avg_train_loss = sum_loss / len(train_data_loader)
                print('Epoch %d/%d: Sum Loss: %5.4f, Average Train Loss: %5.4f, Train Loss: %0.3f, Val Loss: %0.3f, # Training Samples: %d'
                      % (epoch + 1, self.all_cfg['train_cfg']['n_epochs'], sum_loss, avg_train_loss, train_loss, val_loss, len(train_data_loader)))
                print('train_model_id: ', self.all_cfg['args_dict']['train_model_id'], ', delta_BBX5_version: ', self.all_cfg['delta_BBX5_version'], \
                    ', scene id: ', self.all_cfg['scene_id'], ', len_win: ', self.all_cfg['len_win'])
                self.all_cfg['train_cfg']['last_avg_train_loss'] = avg_train_loss
                self.all_cfg['train_cfg']['last_sum_train_loss'] = sum_loss
                self.all_cfg['train_cfg']['len_train_data_loader'] = len(train_data_loader)
                self.all_cfg['train_cfg']['last_val_loss'] = val_loss
                # print("Train execution time: %.5f seconds" % (time_sum / len(self.data_loader)))

                if global_step == 0 or epoch % self.all_cfg['train_cfg']['save_model_epoch_interval'] == 0:
                    # Save all_cfg and model
                    self.all_cfg['train_cfg']['global_epochs_done'] = epoch
                    self.save(epoch, self.all_cfg)

                if epoch == 0 or val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_stat = (train_loss, val_loss)
                    # Save all_cfg and model
                    self.all_cfg['train_cfg']['best_val_loss'] = best_val_loss
                    self.all_cfg['train_cfg']['best_avg_train_loss'] = avg_train_loss
                    self.all_cfg['train_cfg']['best_sum_train_loss'] = sum_loss
                    self.all_cfg['train_cfg']['global_epochs_done'] = epoch
                    model_best = copy.deepcopy(model.state_dict())
                    self.save('best', self.all_cfg)
                print('Best Loss: Train: %0.3f, Val: %0.3f' % best_stat)

            self.model.load_state_dict(model_best)
            print('Training complete!.')
            print('Best Loss: Train: %0.3f, Val: %0.3f' % best_stat)

        elif self.all_cfg['model_cfg']['train_model_id'] == 'Xtranslator_IFcC2C':
            train_seq_in_correspondence, val_seq_in_correspondence = train_data_loader, val_data_loader

            # DEBUG:
            print('train.py - Trainer - train() - np.shape(train_seq_in_correspondence[0]): ', np.shape(train_seq_in_correspondence[0]))
            print('train.py - Trainer - train() - np.shape(train_seq_in_correspondence[1]): ', np.shape(train_seq_in_correspondence[1]))
            print('train.py - Trainer - train() - np.shape(train_seq_in_correspondence[2]): ', np.shape(train_seq_in_correspondence[2]))
            print('train.py - Trainer - train() - np.shape(train_seq_in_correspondence[3]): ', np.shape(train_seq_in_correspondence[3]))
            print('train.py - Trainer - train() - np.shape(val_seq_in_correspondence[0]): ', np.shape(val_seq_in_correspondence[0]))
            print('train.py - Trainer - train() - np.shape(val_seq_in_correspondence[1]): ', np.shape(val_seq_in_correspondence[1]))
            print('train.py - Trainer - train() - np.shape(val_seq_in_correspondence[2]): ', np.shape(val_seq_in_correspondence[2]))
            print('train.py - Trainer - train() - np.shape(val_seq_in_correspondence[3]): ', np.shape(val_seq_in_correspondence[3]))
            '''
            e.g.
            train.py - Trainer - train() - np.shape(train_seq_in_correspondence[0]):  (31761, 30, 1, 5)
            train.py - Trainer - train() - np.shape(train_seq_in_correspondence[1]):  (31761, 30, 1, 2)
            train.py - Trainer - train() - np.shape(train_seq_in_correspondence[2]):  (31761, 30, 1, 19)
            train.py - Trainer - train() - np.shape(train_seq_in_correspondence[3]):  (31761, 30, 1, 9)
            train.py - Trainer - train() - np.shape(val_seq_in_correspondence[0]):  (3529, 30, 1, 5)
            train.py - Trainer - train() - np.shape(val_seq_in_correspondence[1]):  (3529, 30, 1, 2)
            train.py - Trainer - train() - np.shape(val_seq_in_correspondence[2]):  (3529, 30, 1, 19)
            train.py - Trainer - train() - np.shape(val_seq_in_correspondence[3]):  (3529, 30, 1, 9)
            '''

            # Reshape to feed ViTag Xtranslator
            seq_in_BBX5 = np.squeeze(train_seq_in_correspondence[0], axis=2)
            seq_in_FTM2 = np.squeeze(train_seq_in_correspondence[1], axis=2)
            seq_in_IMU_TX = np.squeeze(train_seq_in_correspondence[3], axis=2)
            print('train.py - Trainer - train() - np.shape(seq_in_BBX5): ', np.shape(seq_in_BBX5))
            print('train.py - Trainer - train() - np.shape(seq_in_FTM2): ', np.shape(seq_in_FTM2))
            print('train.py - Trainer - train() - np.shape(seq_in_IMU_TX): ', np.shape(seq_in_IMU_TX))
            '''
            e.g.
            train.py - Trainer - train() - np.shape(seq_in_BBX5):  (31761, 30, 5)
            train.py - Trainer - train() - np.shape(seq_in_FTM2):  (31761, 30, 2)
            train.py - Trainer - train() - np.shape(seq_in_IMU_TX):  (31761, 30, 9)
            '''

            # DEBUG:
            print('\n\n train.py - train() - Xtranslator_IFcC2C - self.save_path: ', self.save_path)
            # print('\n\n cfg: ', cfg)
            if not os.path.exists(self.save_path): os.makedirs(self.save_path)
            save_all_cfg_file_path = self.save_path + '/all_cfg.json'
            with open(save_all_cfg_file_path, 'w') as f:
                # json.dump(self.all_cfg, f, cls=NpEncoder)
                json.dump(str(self.all_cfg).replace('\'', '\"'), f, cls=NpEncoder)
                print(save_all_cfg_file_path, 'saved!')

            save_path_ = self.save('best', self.all_cfg)
            print('train.py - Trainer - train() - save_path_: ', save_path_)

            model_checkpoint = ModelCheckpoint(save_path_, \
                monitor='loss', verbose=1, \
                save_weights_only=True, \
                save_best_only=True, mode='auto', \
                period=self.all_cfg['train_cfg']['save_model_epoch_interval'])

            # ------------------------------
            #  Load model weights if exists
            # ------------------------------
            if self.all_cfg['train_cfg']['resume_training']:
                self.model.load_weights(save_path_)
                print(save_path_, 'loaded!')

            start_training_time = str(datetime.datetime.now())
            start_training_time_ckpt_path = self.save_path + '/start_training_time_' \
                + str(datetime.datetime.now().strftime('Y%Y_Mth%m_D%d_H%H_Mn%M'))
            pickle.dump(None, open(start_training_time_ckpt_path, 'wb'))
            print(start_training_time_ckpt_path, 'logged!')

            if self.all_cfg['train_cfg']['loss_func'] == 'MSE_xydwh':
                history_callback = self.model.fit([seq_in_BBX5, seq_in_FTM2, seq_in_IMU_TX], \
                    [seq_in_BBX5, seq_in_FTM2, seq_in_IMU_TX, \
                    seq_in_FTM2, seq_in_IMU_TX, seq_in_BBX5, \
                    seq_in_IMU_TX, seq_in_BBX5, seq_in_FTM2, \
                    seq_in_BBX5, seq_in_FTM2, seq_in_IMU_TX, \
                    seq_in_BBX5, seq_in_FTM2, seq_in_IMU_TX, \
                    seq_in_BBX5, seq_in_FTM2, seq_in_IMU_TX, \
                    seq_in_BBX5, seq_in_FTM2, seq_in_IMU_TX, \
                    seq_in_FTM2, seq_in_IMU_TX, seq_in_BBX5, \
                    seq_in_BBX5, seq_in_FTM2, seq_in_IMU_TX, \
                    seq_in_BBX5, seq_in_FTM2, seq_in_IMU_TX, \
                    seq_in_BBX5, seq_in_FTM2, seq_in_IMU_TX], \
                    validation_split=0.1, batch_size=self.all_cfg['train_cfg']['batch_size'], \
                        epochs=self.all_cfg['train_cfg']['n_epochs'], verbose=2, \
                    callbacks=[model_checkpoint])
            elif self.all_cfg['train_cfg']['loss_func'] == 'b':
                history_callback = self.model.fit([seq_in_BBX5, seq_in_FTM2, seq_in_IMU_TX], \
                    [seq_in_BBX5, seq_in_FTM2, seq_in_IMU_TX, \
                    seq_in_FTM2, seq_in_IMU_TX, seq_in_BBX5, \
                    seq_in_IMU_TX, seq_in_BBX5, seq_in_FTM2, \
                    seq_in_BBX5, seq_in_FTM2, seq_in_IMU_TX, \
                    seq_in_BBX5, seq_in_FTM2, seq_in_IMU_TX, \
                    seq_in_BBX5, seq_in_FTM2, seq_in_IMU_TX, \
                    seq_in_BBX5, seq_in_FTM2, seq_in_IMU_TX, \
                    seq_in_FTM2, seq_in_IMU_TX, seq_in_BBX5, \
                    seq_in_BBX5, seq_in_FTM2, seq_in_IMU_TX, \
                    seq_in_BBX5, seq_in_FTM2, seq_in_IMU_TX, \
                    seq_in_BBX5, seq_in_FTM2, seq_in_IMU_TX, \
                    seq_in_FTM2, seq_in_FTM2, seq_in_FTM2, \
                    seq_in_FTM2, seq_in_FTM2, seq_in_FTM2, \
                    seq_in_FTM2, seq_in_FTM2, seq_in_FTM2, \
                    seq_in_FTM2, seq_in_FTM2, seq_in_FTM2], \
                    validation_split=0.1, batch_size=self.all_cfg['train_cfg']['batch_size'], \
                        epochs=self.all_cfg['train_cfg']['n_epochs'], verbose=2, \
                    callbacks=[model_checkpoint])



            # # Iterate over all training samples
            # for epoch in range(self.all_cfg['train_cfg']['n_epochs']):
            #     sum_loss = 0.0 # the sum of iteration losses to get average loss in every epoch
            #     time_sum = 0.0
            #     self.model.train()
            #     for i, batch in enumerate(train_data_loader):
            #         batch = [t.to(self.device) for t in batch]
            #         # print('\n\n train.py - Trainer() - train() - np.shape(batch): ', np.shape(batch)) # e.g. (3,)
            #         # print('\n\n train.py - Trainer() - train() - np.shape(batch[0]): ', np.shape(batch[0])) # e.g. torch.Size([32, 30, 1, 5])
            #         # print('\n\n train.py - Trainer() - train() - np.shape(batch[1]): ', np.shape(batch[1])) # e.g. torch.Size([32, 30, 1, 2])
            #         # print('\n\n train.py - Trainer() - train() - np.shape(batch[2]): ', np.shape(batch[2])) # e.g. torch.Size([32, 30, 1, 9])
            #         # print('\n\n train.py - Trainer() - train() - reconstruction_path: ', reconstruction_path)
            #         if np.shape(batch[0])[0] != self.all_cfg['train_cfg']['batch_size']: continue
            #
            #         start_time = time.time()
            #         self.optimizer.zero_grad()
            #         loss = func_loss(model, batch)
            #         # print('\n\n train.py - Trainer() - train() - loss): ', loss)
            #
            #         loss = loss.mean() # mean() for Data Parallelism
            #         # print('\n\n train.py - Trainer() - train() - type(loss)): ', type(loss))
            #         loss.backward()
            #         self.optimizer.step()
            #
            #         global_step += 1
            #         sum_loss += loss.item()
            #         time_sum += time.time() - start_time
            #
            #         if self.all_cfg['train_cfg']['total_steps'] and self.all_cfg['train_cfg']['total_steps'] < global_step:
            #             print('The Total Steps have been reached.')
            #             return
            #
            #     train_loss = self.run(func_forward, func_evaluate, train_data_loader, \
            #         None, False, False, False, reconstruction_path)
            #     # test_loss, test_loss = self.run(func_forward, func_evaluate, data_loader_test)
            #     val_loss = self.run(func_forward, func_evaluate, val_data_loader, \
            #         None, False, False, False, reconstruction_path)
            #     avg_train_loss = sum_loss / len(train_data_loader)
            #     print('Epoch %d/%d: Sum Loss: %5.4f, Average Train Loss: %5.4f, Train Loss: %0.3f, Val Loss: %0.3f, # Training Samples: %d'
            #           % (epoch + 1, self.all_cfg['train_cfg']['n_epochs'], sum_loss, avg_train_loss, train_loss, val_loss, len(train_data_loader)))
            #     print('train_model_id: ', self.all_cfg['args_dict']['train_model_id'], ', delta_BBX5_version: ', self.all_cfg['delta_BBX5_version'])
            #     self.all_cfg['train_cfg']['last_avg_train_loss'] = avg_train_loss
            #     self.all_cfg['train_cfg']['last_sum_train_loss'] = sum_loss
            #     self.all_cfg['train_cfg']['len_train_data_loader'] = len(train_data_loader)
            #     self.all_cfg['train_cfg']['last_val_loss'] = val_loss
            #     # print("Train execution time: %.5f seconds" % (time_sum / len(self.data_loader)))
            #
            #     if global_step == 0 or epoch % self.all_cfg['train_cfg']['save_model_epoch_interval'] == 0:
            #         # Save all_cfg and model
            #         self.all_cfg['train_cfg']['global_epochs_done'] = epoch
            #         self.save(epoch, self.all_cfg)
            #
            #     if epoch == 0 or val_loss < best_val_loss:
            #         best_val_loss = val_loss
            #         best_stat = (train_loss, val_loss)
            #         # Save all_cfg and model
            #         self.all_cfg['train_cfg']['best_val_loss'] = best_val_loss
            #         self.all_cfg['train_cfg']['best_avg_train_loss'] = avg_train_loss
            #         self.all_cfg['train_cfg']['best_sum_train_loss'] = sum_loss
            #         self.all_cfg['train_cfg']['global_epochs_done'] = epoch
            #         model_best = copy.deepcopy(model.state_dict())
            #         self.save('best', self.all_cfg)
            #     print('Best Loss: Train: %0.3f, Val: %0.3f' % best_stat)
            #
            # self.model.load_state_dict(model_best)
            # print('Training complete!.')
            # print('Best Loss: Train: %0.3f, Val: %0.3f' % best_stat)
        elif self.all_cfg['model_cfg']['train_model_id'] == 'Width_regressor':
            ''' Train Loop for Width_regressor'''
            self.load(model_file, load_self)
            model = self.model.to(self.device)
            if data_parallel: # use Data Parallelism with Multi-GPU
                model = nn.DataParallel(model)

            global_step = 0 # global iteration steps regardless of epochs
            best_val_loss = 0.0
            best_stat = None
            model_best = model.state_dict()

            # Iterate over all training samples
            for epoch in range(self.all_cfg['train_cfg']['n_epochs']):
                sum_loss = 0.0 # the sum of iteration losses to get average loss in every epoch
                time_sum = 0.0
                self.model.train()
                for i, batch in enumerate(train_data_loader):
                    # print('\n\n train.py - Trainer() - train() - np.shape(batch): ', np.shape(batch)) # e.g. (32,)
                    # e.g. torch.Size([32, 1, 2])

                    # batch = [t.to(self.device) for t in batch]
                    # print('\n\n train.py - Trainer() - train() - np.shape(batch): ', np.shape(batch)) # e.g. (32,)
                    # print('\n\n train.py - Trainer() - train() - np.shape(batch[0]): ', np.shape(batch[0])) # e.g. torch.Size([1, 2])
                    # print('\n\n train.py - Trainer() - train() - np.shape(batch[1]): ', np.shape(batch[1])) # e.g. torch.Size([1, 2])
                    # print('\n\n train.py - Trainer() - train() - np.shape(batch[2]): ', np.shape(batch[2])) # e.g. torch.Size([1, 2])
                    # print('\n\n train.py - Trainer() - train() - reconstruction_path: ', reconstruction_path)
                    if np.shape(batch)[0] != self.all_cfg['train_cfg']['batch_size']: continue
                    # batch_y = [b[:0] for b in batch]
                    # batch_w = [b[:1] for b in batch]

                    # print('batch: ', batch)
                    batch_y = batch[:, :, 0].to(self.device).float() #; batch_y = [t.to(self.device) for t in batch_y]
                    batch_w = batch[:, :, 1].to(self.device).float() #; batch_w = [t.to(self.device) for t in batch_w]
                    # print('\n\n train.py - Trainer() - train() - np.shape(batch_y): ', np.shape(batch_y))
                    # print('\n\n train.py - Trainer() - train() - np.shape(batch_w): ', np.shape(batch_w))
                    '''
                    e.g.
                     train.py - Trainer() - train() - np.shape(batch_y):  (32,)
                     train.py - Trainer() - train() - np.shape(batch_w):  (32,)
                    '''
                    # print('batch_y: ', batch_y)
                    # print('batch_w: ', batch_w)

                    batch = (batch_y, batch_w)
                    start_time = time.time()
                    self.optimizer.zero_grad()
                    loss = func_loss(model, batch)
                    # print('\n\n train.py - Trainer() - train() - loss): ', loss)

                    loss = loss.mean() # mean() for Data Parallelism
                    # print('\n\n train.py - Trainer() - train() - type(loss)): ', type(loss))
                    loss.backward()
                    self.optimizer.step()

                    global_step += 1
                    sum_loss += loss.item()
                    time_sum += time.time() - start_time

                    if self.all_cfg['train_cfg']['total_steps'] and self.all_cfg['train_cfg']['total_steps'] < global_step:
                        print('The Total Steps have been reached.')
                        return

                train_loss = self.run(func_forward, func_evaluate, train_data_loader, \
                    None, False, False, False, reconstruction_path)
                # test_loss, test_loss = self.run(func_forward, func_evaluate, data_loader_test)
                # print('\n\n train.py - train() - train_loss: ', train_loss)
                val_loss = self.run(func_forward, func_evaluate, val_data_loader, \
                    None, False, False, False, reconstruction_path)
                # print('\n\n train.py - train() - val_loss: ', val_loss)
                avg_train_loss = sum_loss / len(train_data_loader)
                print('Epoch %d/%d: Sum Loss: %5.4f, Average Train Loss: %5.4f, Train Loss: %0.3f, Val Loss: %0.3f, # Training Samples: %d'
                      % (epoch + 1, self.all_cfg['train_cfg']['n_epochs'], sum_loss, avg_train_loss, train_loss, val_loss, len(train_data_loader)))
                print('train_model_id: ', self.all_cfg['args_dict']['train_model_id'], ', delta_BBX5_version: ', self.all_cfg['delta_BBX5_version'], \
                    ', scene id: ', self.all_cfg['scene_id'], ', len_win: ', self.all_cfg['len_win'])
                self.all_cfg['train_cfg']['last_avg_train_loss'] = avg_train_loss
                self.all_cfg['train_cfg']['last_sum_train_loss'] = sum_loss
                self.all_cfg['train_cfg']['len_train_data_loader'] = len(train_data_loader)
                self.all_cfg['train_cfg']['last_val_loss'] = val_loss
                # print("Train execution time: %.5f seconds" % (time_sum / len(self.data_loader)))

                if global_step == 0 or epoch % self.all_cfg['train_cfg']['save_model_epoch_interval'] == 0:
                    # Save all_cfg and model
                    self.all_cfg['train_cfg']['global_epochs_done'] = epoch
                    self.save(epoch, self.all_cfg)

                if epoch == 0 or val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_stat = (train_loss, val_loss)
                    # Save all_cfg and model
                    self.all_cfg['train_cfg']['best_val_loss'] = best_val_loss
                    self.all_cfg['train_cfg']['best_avg_train_loss'] = avg_train_loss
                    self.all_cfg['train_cfg']['best_sum_train_loss'] = sum_loss
                    self.all_cfg['train_cfg']['global_epochs_done'] = epoch
                    model_best = copy.deepcopy(model.state_dict())
                    self.save('best', self.all_cfg)
                print('Best Loss: Train: %0.3f, Val: %0.3f' % best_stat)

            self.model.load_state_dict(model_best)
            print('Training complete!.')
            print('Best Loss: Train: %0.3f, Val: %0.3f' % best_stat)
        elif self.all_cfg['model_cfg']['train_model_id'] == 'Height_regressor':
            ''' Train Loop for Height_regressor'''
            self.load(model_file, load_self)
            model = self.model.to(self.device)
            if data_parallel: # use Data Parallelism with Multi-GPU
                model = nn.DataParallel(model)

            global_step = 0 # global iteration steps regardless of epochs
            best_val_loss = 0.0
            best_stat = None
            model_best = model.state_dict()

            # Iterate over all training samples
            for epoch in range(self.all_cfg['train_cfg']['n_epochs']):
                sum_loss = 0.0 # the sum of iteration losses to get average loss in every epoch
                time_sum = 0.0
                self.model.train()
                for i, batch in enumerate(train_data_loader):
                    if np.shape(batch)[0] != self.all_cfg['train_cfg']['batch_size']: continue
                    batch_y = batch[:, :, 0].to(self.device).float() #; batch_y = [t.to(self.device) for t in batch_y]
                    batch_h = batch[:, :, 1].to(self.device).float() #; batch_h = [t.to(self.device) for t in batch_h]
                    batch = (batch_y, batch_h)
                    start_time = time.time()
                    self.optimizer.zero_grad()
                    loss = func_loss(model, batch) # batch)
                    loss = loss.mean() # mean() for Data Parallelism
                    loss.backward()
                    self.optimizer.step()

                    global_step += 1
                    sum_loss += loss.item()
                    time_sum += time.time() - start_time

                    if self.all_cfg['train_cfg']['total_steps'] and self.all_cfg['train_cfg']['total_steps'] < global_step:
                        print('The Total Steps have been reached.')
                        return

                train_loss = self.run(func_forward, func_evaluate, train_data_loader, \
                    None, False, False, False, reconstruction_path)
                # test_loss, test_loss = self.run(func_forward, func_evaluate, data_loader_test)
                # print('\n\n train.py - train() - train_loss: ', train_loss)
                val_loss = self.run(func_forward, func_evaluate, val_data_loader, \
                    None, False, False, False, reconstruction_path)
                # print('\n\n train.py - train() - val_loss: ', val_loss)
                avg_train_loss = sum_loss / len(train_data_loader)
                print('Epoch %d/%d: Sum Loss: %5.4f, Average Train Loss: %5.4f, Train Loss: %0.3f, Val Loss: %0.3f, # Training Samples: %d'
                      % (epoch + 1, self.all_cfg['train_cfg']['n_epochs'], sum_loss, avg_train_loss, train_loss, val_loss, len(train_data_loader)))
                print('train_model_id: ', self.all_cfg['args_dict']['train_model_id'], ', delta_BBX5_version: ', self.all_cfg['delta_BBX5_version'], \
                    ', scene id: ', self.all_cfg['scene_id'], ', len_win: ', self.all_cfg['len_win'])
                self.all_cfg['train_cfg']['last_avg_train_loss'] = avg_train_loss
                self.all_cfg['train_cfg']['last_sum_train_loss'] = sum_loss
                self.all_cfg['train_cfg']['len_train_data_loader'] = len(train_data_loader)
                self.all_cfg['train_cfg']['last_val_loss'] = val_loss
                # print("Train execution time: %.5f seconds" % (time_sum / len(self.data_loader)))

                if global_step == 0 or epoch % self.all_cfg['train_cfg']['save_model_epoch_interval'] == 0:
                    # Save all_cfg and model
                    self.all_cfg['train_cfg']['global_epochs_done'] = epoch
                    self.save(epoch, self.all_cfg)

                if epoch == 0 or val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_stat = (train_loss, val_loss)
                    # Save all_cfg and model
                    self.all_cfg['train_cfg']['best_val_loss'] = best_val_loss
                    self.all_cfg['train_cfg']['best_avg_train_loss'] = avg_train_loss
                    self.all_cfg['train_cfg']['best_sum_train_loss'] = sum_loss
                    self.all_cfg['train_cfg']['global_epochs_done'] = epoch
                    model_best = copy.deepcopy(model.state_dict())
                    self.save('best', self.all_cfg)
                print('Best Loss: Train: %0.3f, Val: %0.3f' % best_stat)

            self.model.load_state_dict(model_best)
            print('Training complete!.')
            print('Best Loss: Train: %0.3f, Val: %0.3f' % best_stat)
        elif self.all_cfg['model_cfg']['train_model_id'] == 'Depth_regressor':
            ''' Train Loop for Depth_regressor'''
            self.load(model_file, load_self)
            model = self.model.to(self.device)
            if data_parallel: # use Data Parallelism with Multi-GPU
                model = nn.DataParallel(model)

            global_step = 0 # global iteration steps regardless of epochs
            best_val_loss = 0.0
            best_stat = None
            model_best = model.state_dict()

            # Iterate over all training samples
            for epoch in range(self.all_cfg['train_cfg']['n_epochs']):
                sum_loss = 0.0 # the sum of iteration losses to get average loss in every epoch
                time_sum = 0.0
                self.model.train()
                for i, batch in enumerate(train_data_loader):
                    if np.shape(batch)[0] != self.all_cfg['train_cfg']['batch_size']: continue
                    batch_y = batch[:, :, 0].to(self.device).float() #; batch_y = [t.to(self.device) for t in batch_y]
                    batch_d = batch[:, :, 1].to(self.device).float() #; batch_d = [t.to(self.device) for t in batch_d]
                    batch = (batch_y, batch_d)
                    start_time = time.time()
                    self.optimizer.zero_grad()
                    loss = func_loss(model, batch) # batch)
                    loss = loss.mean() # mean() for Data Parallelism
                    loss.backward()
                    self.optimizer.step()

                    global_step += 1
                    sum_loss += loss.item()
                    time_sum += time.time() - start_time

                    if self.all_cfg['train_cfg']['total_steps'] and self.all_cfg['train_cfg']['total_steps'] < global_step:
                        print('The Total Steps have been reached.')
                        return

                train_loss = self.run(func_forward, func_evaluate, train_data_loader, \
                    None, False, False, False, reconstruction_path)
                # test_loss, test_loss = self.run(func_forward, func_evaluate, data_loader_test)
                # print('\n\n train.py - train() - train_loss: ', train_loss)
                val_loss = self.run(func_forward, func_evaluate, val_data_loader, \
                    None, False, False, False, reconstruction_path)
                # print('\n\n train.py - train() - val_loss: ', val_loss)
                avg_train_loss = sum_loss / len(train_data_loader)
                print('Epoch %d/%d: Sum Loss: %5.4f, Average Train Loss: %5.4f, Train Loss: %0.3f, Val Loss: %0.3f, # Training Samples: %d'
                      % (epoch + 1, self.all_cfg['train_cfg']['n_epochs'], sum_loss, avg_train_loss, train_loss, val_loss, len(train_data_loader)))
                print('train_model_id: ', self.all_cfg['args_dict']['train_model_id'], ', delta_BBX5_version: ', self.all_cfg['delta_BBX5_version'], \
                    ', scene id: ', self.all_cfg['scene_id'], ', len_win: ', self.all_cfg['len_win'])
                self.all_cfg['train_cfg']['last_avg_train_loss'] = avg_train_loss
                self.all_cfg['train_cfg']['last_sum_train_loss'] = sum_loss
                self.all_cfg['train_cfg']['len_train_data_loader'] = len(train_data_loader)
                self.all_cfg['train_cfg']['last_val_loss'] = val_loss
                # print("Train execution time: %.5f seconds" % (time_sum / len(self.data_loader)))

                if global_step == 0 or epoch % self.all_cfg['train_cfg']['save_model_epoch_interval'] == 0:
                    # Save all_cfg and model
                    self.all_cfg['train_cfg']['global_epochs_done'] = epoch
                    self.save(epoch, self.all_cfg)

                if epoch == 0 or val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_stat = (train_loss, val_loss)
                    # Save all_cfg and model
                    self.all_cfg['train_cfg']['best_val_loss'] = best_val_loss
                    self.all_cfg['train_cfg']['best_avg_train_loss'] = avg_train_loss
                    self.all_cfg['train_cfg']['best_sum_train_loss'] = sum_loss
                    self.all_cfg['train_cfg']['global_epochs_done'] = epoch
                    model_best = copy.deepcopy(model.state_dict())
                    self.save('best', self.all_cfg)
                print('Best Loss: Train: %0.3f, Val: %0.3f' % best_stat)

            self.model.load_state_dict(model_best)
            print('Training complete!.')
            print('Best Loss: Train: %0.3f, Val: %0.3f' % best_stat)

    def load(self, model_file, load_self=False):
        ''' load saved model or pretrained transformer (a part of model) '''
        if model_file:
            if self.all_cfg['model_cfg']['model_type'] == 'pdr':
                print('Loading the model from', model_file)
                if load_self:
                    self.model_w.load_self(model_file[0], map_location=self.device)
                    self.model_h.load_self(model_file[1], map_location=self.device)
                else:
                    self.model_w.load_state_dict(torch.load(model_file[0], map_location=self.device))
                    self.model_h.load_state_dict(torch.load(model_file[1], map_location=self.device))
                print('\n', model_file, 'loaded!')
            else:
                print('Loading the model from', model_file)
                if load_self:
                    self.model.load_self(model_file, map_location=self.device)
                else:
                    self.model.load_state_dict(torch.load(model_file, map_location=self.device))
                print('\n', model_file, 'loaded!')

    def save(self, epoch, cfg):
        ''' save current model '''
        # DEBUG:
        print('\n\n train.py - save() - self.save_path: ', self.save_path)
        # print('\n\n cfg: ', cfg)
        if not os.path.exists(self.save_path): os.makedirs(self.save_path)
        save_all_cfg_file_path = self.save_path + '/all_cfg.json'
        with open(save_all_cfg_file_path, 'w') as f:
            # json.dump(self.all_cfg, f, cls=NpEncoder)
            json.dump(str(self.all_cfg).replace('\'', '\"'), f, cls=NpEncoder)
            print(save_all_cfg_file_path, 'saved!')

        if 'transformer' in cfg['model_cfg']['model_type'] or 'lstm' in cfg['model_cfg']['model_type'] \
            or 'regressor' in cfg['model_cfg']['model_type']:
            save_path_ = self.save_path + '/epoch_' + str(epoch) + '.pt'
            torch.save(self.model.state_dict(), save_path_)
            print(save_path_, 'saved!')
        elif 'translator' in cfg['model_cfg']['model_type']:
            save_path_ = self.save_path + '/epoch_' + str(epoch) + '_w.ckpt'
        return save_path_
