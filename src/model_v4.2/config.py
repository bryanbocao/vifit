import argparse
import sys

import numpy as np
import random
import matplotlib
import matplotlib.pyplot as plt
import math
import glob
import io
import base64
import os
from os import path
import copy

from collections import deque, namedtuple
from itertools import count
from PIL import Image
import shutil
import gc
import statistics
import csv
from scipy import signal
import json
from itertools import permutations
from queue import Queue
import pickle
import datetime
from collections import defaultdict
from csv import reader
from PIL import Image
import argparse

# -----------------------------------
#  Configurations for One Experiment
# -----------------------------------
class Config:
    def __init__(self):
        # --------------------------
        #  Paramaters of experiment
        # --------------------------
        parser = argparse.ArgumentParser()
        parser.add_argument('-m', '--machine', type=str, default='snow', help='snow | night | dawn | aw | AI') # edit
        parser.add_argument('-ud' '--use_docker', action='store_true', default=False) # edit
        parser.add_argument('-rm', '--running_mode', type=str, default='pretrain', help='pretrain | train | test | embed | investigate') # edit
        parser.add_argument('-pt_eid', '--pretrain_exp_id', type=int, default=-1)
        parser.add_argument('-tr_eid', '--train_exp_id', type=int, default=-1)
        parser.add_argument('-tr_w_eid', '--train_w_exp_id', type=int, default=-1)
        parser.add_argument('-tr_h_eid', '--train_h_exp_id', type=int, default=-1)
        parser.add_argument('-tr_d_eid', '--train_d_exp_id', type=int, default=-1)
        parser.add_argument('-ts_eid', '--test_exp_id', type=int, default=-1)
        parser.add_argument('-eb_eid', '--embed_exp_id', type=int, default=-1)
        parser.add_argument('-iv_eid', '--investigate_exp_id', type=int, default=-1)
        parser.add_argument('-iv_vt', '--investigate_vis_type', type=str, default='2D', help='2D | 3D')
        parser.add_argument('-ld_tr_eid', '--load_train_exp_id_model', action='store_true', default=False)
        parser.add_argument('-tsid_idx', '--test_seq_id_idx', type=int, default='0', help='0-14') # edit
        parser.add_argument('-tr_md_id', '--train_model_id', type=str, default='Xformer_IFcC2C') # edit
        parser.add_argument('-lw', '--len_win', type=int, default=60, help='Window length') # edit
        parser.add_argument('-rt', '--resume_training', action='store_true', default=False)
        parser.add_argument('-tt', '--test_type', type=str, default='random', help='random | crowded') # edit
        parser.add_argument('-fps', '--fps', type=int, default=10, help='10 | 3 | 1') # edit
        parser.add_argument('-v', '--vis', action='store_true', default=False, help='True | False') # edit
        parser.add_argument('-ve', '--vis_eval', action='store_true', default=False, help='True | False') # edit
        parser.add_argument('-vbt', '--vis_bbox_type', type=str, default='PRED', help='PRED | GND | GT | GND_PRED | GT_PRED')
        parser.add_argument('-vcc', '--vis_color_change', type=str, default='same', help='same | gradient')
        parser.add_argument('-vbl', '--vis_bbox_len', type=int, default=10)
        parser.add_argument('-vwned', '--vis_win_NED', action='store_true', default=False)
        parser.add_argument('-vpdrbev', '--vis_PDR_BEV', action='store_true', default=False)
        parser.add_argument('-vt', '--vis_text', action='store_true', default=False)
        # parser.add_argument('-l', '--loss', type=str, default='mse', help='mse: Mean Squared Error | b: Bhattacharyya Loss') # edit
        parser.add_argument('-nmd', '--not_makedirs', action='store_true', default=False)
        parser.add_argument('-n', '--normalization', action='store_true', default=False)
        parser.add_argument('-smiv', '--save_model_epoch_interval', type=int, default=50)
        parser.add_argument('-f_d', '--FTM_dist', type=str, default='eucl', \
            help='eucl: Euclidean Distane;' \
                 'sb: Simplified Bhattacharyya Distance'
                 'b: Bhattacharyya Distance;' \
                 'b_exp1: Bhattacharyya Distance with second term to be exponential.') # edit
        parser.add_argument(
            '-w_ls',
            "--weight_of_distance_list",  # name on the CLI - drop the `--` for positional/required parameters
            nargs="*",  # 0 or more values expected => creates a list
            type=int,
            default=[1, 1, 1, 0],  # default if nothing is provided
            help='Weight of Distance for Different Modalities in this order -- ' \
                    '0: BBX5, 1: IMU, 2: FTM, 3: D_FTM'
        ) # edit
        parser.add_argument('-nl', '--noise_level', type=float, default=0.0, help='0.0, 0.1, 0.3, 0.5') # edit
        parser.add_argument('-bm', '--best_model', action='store_true', default=False)
        parser.add_argument('-sc', '--scene_id', type=int, default=0, help='[0-5]')
        parser.add_argument('-i200', '--imu200', action='store_true', default=False)
        parser.add_argument('-ip2', '--imu_ned_pos2', action='store_true', default=False)
        parser.add_argument('-dBv', '--delta_BBX5_version', type=int, default=0) # 0 means the original BBX5 without delta.
        parser.add_argument('-lf', '--loss_func', type=str, default='MSE_xydwh', \
            help='MSE_xydwh, MSE_xyd, MSE_xy, DIOU_depth, DIOU, GIOU_depth, GIOU, NMSE_xydwh_DIOU, NMSE_xydwh_GIOU')
        parser.add_argument('-otm', '--optimizer', type=str, default='Adam')
        parser.add_argument('-nan', '--nan_fill', type=str, help='0 | linear_interp')
        parser.add_argument('-w_s', '--win_stride', type=int, default=1, help='Sliding window step')
        parser.add_argument('-te', '--training_epochs', type=int, default=100000000000000000)
        parser.add_argument('-sm_f', '--sm_filter', type=str, default=None, help='savgol_filter')
        parser.add_argument('-bl', '--baseline', action='store_true', default=False)
        parser.add_argument('-hf', '--hidden', type=int, default=72, help='# of hidden features')
        parser.add_argument('-mf', '--mask_features', type=str, default='', help='a: accel, g: gyro, m: magnetometer')
        parser.add_argument('-bs', '--batch_size', type=int, default=32)
        parser.add_argument('-pa', '--plot_alpha', type=float, default=0.5)
        parser.add_argument('-pss', '--plot_slicing_steps', type=int, default=50)
        parser.add_argument('-pxywhd', '--plot_xywhd', type=str, default='xw', help='xw | xh | yw | yh')
        parser.add_argument('-lr', '--learning_rate', type=float, default=0.001, help='learning rate')
        parser.add_argument('-st', '--slicing_step', type=int, default=10)
        parser.add_argument('-kfm', '--kalman_filter_modality', type=str, default='C', help='C, C_Ph')
        parser.add_argument('-kfxd', '--kf_x_dim', type=int, default=3)
        parser.add_argument('-kfzd', '--kf_z_dim', type=int, default=3)
        parser.add_argument('-tftcin', '--transformer_Tc_in', type=str, default='fstBBX5', help='fstBBX5 | flBBX5')
        parser.add_argument('-ecta', '--eval_with_cumulative_turning_angle', action='store_true', default=False)
        parser.add_argument('-ata', '--average_turning_angle', type=int, default=45, help='degrees')
        parser.add_argument('-mo', '--mask_out_of_view', action='store_true', default=False)
        parser.add_argument('-n_mo', '--number_of_mask_view', type=int, default=3, help='Number of Masked Views')
        parser.add_argument('-bt', '--broken_trajectories', action='store_true', default=False)
        parser.add_argument('-ffo', '--first_frame_only', action='store_true', default=False)
        parser.add_argument('-mrf', '--minimum_required_frames', action='store_true', default=False)
        parser.add_argument('-iou_pft', '--iou_per_frame_thred', type=float, default=0.5)
        parser.add_argument('-std', '--std', type=float, default=1.)
        parser.add_argument('-sd', '--seed', type=int, default=-1)
        parser.add_argument('-ns', '--noise', action='store_true', default=False)
        parser.add_argument('-rp', '--root_path', type=str, default='None')
        args = parser.parse_args()
        self.args_dict = vars(args)

        # ------------------------------------------------------------
        #  ATTENTIONS - Need to anonymize the code before submission.
        # ------------------------------------------------------------
        self.machine = args.machine # 'snow' # {'snow', 'night, 'dawn', 'aw', 'AI'} # edit
        if args.root_path == 'None':
            if self.machine == 'snow':
                self.user = 'brcao'
                self.root_path = '/media/' + self.user + '/eData21'
                # self.root_path = '/home/' + self.user
            elif self.machine == 'night':
                self.user = 'brcao'
                self.root_path = '/media/' + self.user + '/eData'
            elif self.machine == 'dawn':
                self.user = 'brcao'
                self.root_path = '/home/' + self.user
            elif self.machine == 'AI':
                self.user = 'boccao'
                self.root_path = '/home/' + self.user
            elif self.machine == 'lab':
                self.user = 'brcao'
                self.root_path = '/home/' + self.user
            elif self.machine == 'aw2':
                self.user = 'brcao'
                self.root_path = '/media/' + self.user + '/eData2'
        else:
            self.user = 'None'
            self.root_path = args.root_path

        print(args)
        self.use_docker = args.ud__use_docker # edit
        print('self.use_docker: ', self.use_docker)
        if self.use_docker:
            self.root_path = '/share' + self.root_path

        # Update parameters from command line >>>
        self.seq_id_idx = self.test_seq_id_idx = args.test_seq_id_idx
        self.running_mode = args.running_mode
        self.noise_level = args.noise_level
        self.FTM_dist = args.FTM_dist
        self.delta_BBX5_version = args.delta_BBX5_version
        self.imu200 = args.imu200
        self.imu_ned_pos2 = args.imu_ned_pos2
        self.makedirs = not args.not_makedirs
        self.normalization = args.normalization
        self.load_train_exp_id_model = args.load_train_exp_id_model
        self.train_w_exp_id = args.train_w_exp_id; self.train_h_exp_id = args.train_h_exp_id; self.train_d_exp_id = args.train_d_exp_id
        self.vis_bbox_type = args.vis_bbox_type
        self.vis_bbox_len = args.vis_bbox_len
        self.resume_training = args.resume_training
        self.loss_func = args.loss_func
        self.optimizer = args.optimizer
        self.save_model_epoch_interval = args.save_model_epoch_interval
        self.nan_fill = args.nan_fill
        self.win_stride = args.win_stride
        self.training_epochs = args.training_epochs
        self.sm_filter = args.sm_filter
        self.baseline = args.baseline
        self.hidden = args.hidden
        self.mask_features = args.mask_features
        self.batch_size = args.batch_size
        self.vis_color_change = args.vis_color_change
        self.vis_win_NED = args.vis_win_NED
        self.vis_PDR_BEV = args.vis_PDR_BEV
        self.vis_text = args.vis_text
        self.investigate_vis_type = args.investigate_vis_type
        self.plot_alpha = args.plot_alpha
        self.plot_slicing_steps = args.plot_slicing_steps
        self.plot_xywhd = args.plot_xywhd
        self.learning_rate = args.learning_rate
        self.slicing_step = args.slicing_step
        self.kalman_filter_modality = args.kalman_filter_modality
        self.kf_x_dim = args.kf_x_dim
        self.kf_z_dim = args.kf_z_dim
        self.transformer_Tc_in = args.transformer_Tc_in
        self.eval_with_cumulative_turning_angle = args.eval_with_cumulative_turning_angle
        self.average_turning_angle = args.average_turning_angle
        self.mask_out_of_view = args.mask_out_of_view
        self.M = args.number_of_mask_view
        self.broken_trajectories = args.broken_trajectories
        self.first_frame_only = args.first_frame_only
        self.minimum_required_frames = args.minimum_required_frames
        self.iou_per_frame_thred = args.iou_per_frame_thred
        self.std = args.std
        if args.seed == -1: self.seed = 3431
        else: self.seed = args.seed
        self.noise = args.noise
        # Update parameters from command line <<<

        # -----------------------------------
        #  To be updated in prepare_*_data()
        self.model_id = 'XT_' # edit
        # Prev Implementation >>>
        # if args.loss == 'mse': self.model_id += 'test_idx_' + str(self.test_seq_id_idx)
        # elif args.loss == 'b': self.model_id = self.model_id[:self.model_id.index('FTM2_') + len('FTM_2')] + \
        #     'Bloss_test_idx_' + str(self.test_seq_id_idx)
        # Prev Implementation <<<
        self.model_id += 'test_idx_' + str(self.test_seq_id_idx)
        print('self.model_id: ', self.model_id)
        self.scene_id = args.scene_id
        self.macro_scene = 'indoor' if self.scene_id == 0 else 'outdoor'
        self.seq_root_path = self.root_path + '/Data/datasets/RAN_All/RAN/seqs/' + self.macro_scene
        self.seq_root_path_for_model = self.root_path + '/Data/datasets/RAN_All/RAN4model_dfv4.2/seqs/' + self.macro_scene + \
            '/scene' + str(self.scene_id)

        print('\n')
        print('self.seq_root_path: ', self.seq_root_path)
        self.dataset4model_root_path = self.root_path + '/Data/datasets/RAN_All/RAN4model_dfv4.2'
        # if self.baseline:
        self.dataset4modelbl_root_path = self.root_path + '/Data/datasets/RAN_All/RAN4model_dfv4.2bl'

        # OBS >>>
        # self.seq4model_root_path = self.dataset4model_root_path + '/seqs/' + self.macro_scene
        # if not os.path.exists(self.seq4model_root_path): os.makedirs(self.seq4model_root_path)
        # OBS <<<
        self.seq4model_root_path = self.root_path + '/Data/datasets/RAN_All/RAN4model_dfv4.2/seqs/' + self.macro_scene + \
            '/scene' + str(self.scene_id)
        if not os.path.exists(self.seq4model_root_path): os.makedirs(self.seq4model_root_path)
        print('self.seq_root_path: ', self.seq_root_path)
        print('self.seq4model_root_path: ', self.seq4model_root_path)
        print('glob.glob(self.seq4model_root_path: ', glob.glob(self.seq4model_root_path + '/*'))
        self.seq_id_path_ls = sorted(glob.glob(self.seq4model_root_path + '/*'))
        print('\n')
        print('self.seq_id_path_ls: ', self.seq_id_path_ls)
        self.seq_id_ls = sorted([seq_id_path[-15:] for seq_id_path in self.seq_id_path_ls])
        print('\n')
        print('self.seq_id_ls: ', self.seq_id_ls)
        self.seq_id = self.seq_id_ls[0]
        self.test_seq_id = self.seq_id_ls[self.test_seq_id_idx]
        self.train_seq_id_ls = []

        # Normalize weights to sum up to 1 >>>
        self.w_dct = {'Cam': defaultdict(), 'Phone': defaultdict()}
        self.w_ls = args.weight_of_distance_list

        Cam_deno = float(self.w_ls[0] + self.w_ls[3])
        self.w_dct['Cam'] = {'BBX5': float(self.w_ls[0]) / float(Cam_deno), 'D_FTM': float(self.w_ls[3]) / float(Cam_deno)}
        Phone_deno = float(self.w_ls[1] + self.w_ls[2] + self.w_ls[3])
        self.w_dct['Phone'] = {'IMU19': float(self.w_ls[1]) / float(Phone_deno),
                                'FTM2': float(self.w_ls[2]) / float(Phone_deno),
                                'D_FTM': float(self.w_ls[3]) / float(Phone_deno)}
        # Normalize weights to sum up to 1 <<<

        # --------------------------------------
        #  To be updated in update_parameters()
        self.seq_path = self.seq_id_path_ls[0]
        print(); print() # debug
        print('self.seq_id_path_ls: ', self.seq_id_path_ls)
        print('self,seq_id_ls: ', self.seq_id_ls)
        print('len(self,seq_id_ls): ', len(self.seq_id_ls))

        self.img_type = 'RGBh_ts16_dfv4' if self.macro_scene == 'indoor' else 'RGB_ts16_dfv4'
        self.img_type += '_anonymized'
        self.img_path = self.seq_path + '/' + self.img_type
        self.img_path = self.img_path.replace('RAN4model_dfv4.2', 'RAN')
        # self.RGBh_ts16_dfv4_ls = []
        # global RGBh_ts16_dfv4_ls

        self.n_subjects = 5 # dummy one
        if self.scene_id == 0: self.n_subjects = 5
        # elif self.scene_id == 1 or self.scene_id == 2: self.n_subjects = 2
        elif self.scene_id == 1: self.n_subjects = 2
        # elif self.scene_id in list(range(3, 6)): self.n_subjects = 3
        elif self.scene_id in list(range(2, 5)): self.n_subjects = 3
        self.phone_time_offsets = [0] * self.n_subjects

        print(); print() # debug
        print('self.n_subjects: ', self.n_subjects)

        # Baseline >>>
        # if self.baseline:
        # self.seq4model_root_path = self.root_path + '/Data/datasets/RAN_All/RAN4model_dfv4.2/seqs/' + self.macro_scene + \
        #     '/scene' + str(self.scene_id)
        # if not os.path.exists(self.seq4model_root_path): os.makedirs(self.seq4model_root_path)
        self.seq4modelbl_root_path = self.dataset4modelbl_root_path + '/seqs'
        if not os.path.exists(self.seq4modelbl_root_path): os.makedirs(self.seq4modelbl_root_path)
        self.seq_path_for_modelbl = self.seq4modelbl_root_path + '/' + self.macro_scene + \
            '/scene' + str(self.scene_id) + '/' + self.seq_id # bl: baseline
        self.sync_dfv4bl_path = self.seq_path_for_modelbl + '/sync_ts16_dfv4bl'
        if not os.path.exists(self.sync_dfv4bl_path): os.makedirs(self.sync_dfv4bl_path)
        # Baseline <<<

        # ------------------------------------------------------
        #  Synchronized data: BBXC3,BBX5,IMU,_sync_dfv4 to load
        # ------------------------------------------------------
        self.BBXC3_sync_dfv4, self.BBX5_sync_dfv4 = [], []
        self.IMU19_sync_dfv4, self.IMUlgyq10_sync_dfv4 = [], []
        if self.imu200: self.IMU19_200_sync_dfv4, self.IMUaccgym9_200_sync_dfv4 = [], []
        elif self.imu_ned_pos2: self.IMU_NED_pos2_sync_dfv4bl = []

        self.IMUlgyqm13_sync_dfv4 = []
        self.FTM_sync_dfv4 = []
        self.seq_path_for_model = self.seq4model_root_path + '/' + self.seq_id
        # self.RGBg_ts16_dfv4_ls_path = self.seq_path_for_model + '/RGBh_ts16_dfv4_ls.json' \
        #     if self.macro_scene == 'indoor' \
        #     else self.seq_path_for_model + '/RGB_ts16_dfv4_ls.json'
        # 'g' in RGBg_ts16_dfv4 stands for general, referring to either RGB_ts16_dfv4 or RGBh_ts16_dfv4.
        self.RGBg_ts16_dfv4_ls_path = self.seq_path_for_model + '/RGBg_ts16_dfv4_ls.json'
        self.RGBg_ts16_dfv4_valid_ls = []
        # self.RGB_ts16_dfv4_ls = [img_path[-21:-4] for img_path in glob.glob(self.img_path + '/*')]
        # print(); print() # debug
        # print('self.RGB_ts16_dfv4_ls: ', self.RGB_ts16_dfv4_ls)
        with open(self.RGBg_ts16_dfv4_ls_path, 'r') as f:
            self.RGBg_ts16_dfv4_ls = json.load(f)
            print(self.RGBg_ts16_dfv4_ls_path, 'loaded!')

        self.dataset4model_root_path = self.root_path + '/Data/datasets/RAN_All/RAN4model_dfv4.2'
        self.exp = 1 # edit
        if self.macro_scene == 'indoor':
            self.start_end_ts16_indoor_dfv4_path = self.dataset4model_root_path + '/exps/exp' + \
                str(self.exp) + '/start_end_ts16_dfv4_indoor.json'
        elif self.macro_scene == 'outdoor':
            self.start_end_ts16_indoor_dfv4_path = self.dataset4model_root_path + '/exps/exp' + \
                str(self.exp) + '/start_end_ts16_dfv4_outdoor.json'
        with open(self.start_end_ts16_indoor_dfv4_path, 'r') as f:
            self.seq_id_to_start_end_ts16_dfv4 = json.load(f)
            print(self.start_end_ts16_indoor_dfv4_path, 'loaded!')

        self.sync_dfv4_path = self.seq_path_for_model + '/sync_ts16_dfv4'
        if not os.path.exists(self.sync_dfv4_path): os.makedirs(self.sync_dfv4_path)
        self.BBXC3_sync_dfv4_path = self.sync_dfv4_path + '/BBXC3_sync_dfv4.pkl'
        self.BBX5_sync_dfv4_path = self.sync_dfv4_path + '/BBX5_sync_dfv4.pkl'
        self.FTM_sync_dfv4_path = self.sync_dfv4_path + '/FTM_sync_dfv4.pkl'

        self.IMU19_sync_dfv4_path = self.sync_dfv4_path + '/IMU19_sync_dfv4.pkl'
        if self.imu200:
            # self.IMU19_200_sync_dfv4_path = self.sync_dfv4_path + '/IMU19_200_sync_dfv4.pkl'
            self.IMUaccgym9_200_sync_dfv4_path = self.sync_dfv4_path + '/IMUaccgym9_200_sync_dfv4.pkl'
        elif self.imu_ned_pos2:
            self.IMU_NED_pos2_sync_dfv4bl_path = self.sync_dfv4bl_path + '/IMU_NED_pos2_sync_dfv4bl.pkl' # to be added
        if not os.path.exists(self.sync_dfv4_path): os.makedirs(self.sync_dfv4_path)

        # ------
        #  BBX5
        # ------
        self.BBX5_dim = 5
        self.BBX5_dummy = [0] * self.BBX5_dim
        self.max_depth = 0
        '''
        Empirical measurements
        scene0: 18.00
        scene1: 11.35
        scene2: 14.18
        scene3: 13.48
        scene4: 16.03
        '''
        self.max_depths = [18.00, 11.35, 14.18, 13.48, 16.03]
        self.rows, self.cols = 720, 1280

        # ------------
        #  FTM2
        # ------------
        self.FTM2_dim = 2
        self.FTM2_dummy = [0] * self.FTM2_dim

        # -----
        #  IMU
        # -----
        self.IMU_path = self.seq_path + '/IMU'
        self.IMU_dfv4_path = self.seq_path + '/IMU_dfv4' # ts13_dfv4 with offsets (os)
        if not os.path.exists(self.IMU_dfv4_path): os.makedirs(self.IMU_dfv4_path)

        # -------
        #  IMU19
        # -------
        self.IMU19_data_types = ['ACCEL', 'GYRO', 'MAG', 'GRAV', 'LINEAR', 'Quaternion'] # ['ACCEL', 'GRAV', 'LINEAR', 'Quaternion', 'MAG', 'GYRO']
        self.IMU19_data_type_dict = {'ACCEL' : 3, 'GYRO' : 3, 'MAG' : 3, \
                                        'GRAV' : 3, 'LINEAR' : 3, 'Quaternion' : 4}

        self.IMU19_dim = np.sum(list(self.IMU19_data_type_dict.values())) # 19
        print('self.IMU19_dim: ', self.IMU19_dim)
        self.IMU19_dummy = [0] * self.IMU19_dim
        # 'XT' stands for Cross Transformer
        self.IMU_XT_data_types = ['ACCEL', 'GYRO',  'MAG']
        self.IMU_XT_data_type_dict = {'ACCEL' : 3, 'GYRO' : 3, 'MAG' : 3}
        self.IMU_XT_dim = np.sum(list(self.IMU_XT_data_type_dict.values()))
        print('self.IMU_XT_dim: ', self.IMU_XT_dim)
        # if self.imu200: self.IMU_XTaccgym_200_dim = (self.IMU_XT_dim, 200)
        if self.imu_ned_pos2: self.IMU_NED_pos2_dim = 2
        self.IMU_NED_origins = [] # DEBUG
        self.IMU_NED_pos_scales = [] # DEBUG

        # ------
        #  Mask
        # ------
        self.mask_cfg = {
            'mask_ratio': 0.15,
            'mask_alpha': 6, # May not be used
            'max_gram': 10,
            'mask_prob': 0.8,
            'replace_prob': 0.0,
            'imu200': self.imu200
        }
        self.img_left_border, self.img_right_border, self.img_top_border, self.img_bottom_border = 0, 1280, 0, 720
        if self.scene_id == 0:
            self.img_left_border, self.img_right_border, self.img_top_border, self.img_bottom_border = 0, 2180, 0, 1420
        self.seen_left_borders, self.seen_right_borders = [], []
        self.seen_top_borders, self.seen_bottom_borders = [], []

        # self.a_width, self.a_height = 1280, 720
        if self.scene_id == 0:
            self.a_left_border, self.a_right_border, self.a_top_border, self.a_bottom_border = 450, 1730, 350, 1070
            self.a_width, self.a_height = self.a_right_border - self.a_left_border, self.a_bottom_border - self.a_top_border
            self.m_width, self.m_height = int(self.a_width / self.M), int(self.a_height / (self.M // 2))
            if self.mask_out_of_view:
                for m_i in range(0, self.M // 2):
                    seen_top_border_ = self.a_top_border + self.m_height * m_i
                    if m_i < self.M // 2 - 1: seen_bottom_border_ = self.a_top_border + self.m_height * (m_i + 1)
                    elif m_i == self.M // 2 - 1: seen_bottom_border_ = self.a_bottom_border
                    for m_j in range(0, self.M):
                        self.seen_left_borders.append(self.a_left_border + self.m_width * m_j)
                        self.seen_right_borders.append(self.a_left_border + self.m_width * (m_j + 1))
                        self.seen_top_borders.append(seen_top_border_)
                        self.seen_bottom_borders.append(seen_bottom_border_)
            else:
                self.seen_left_borders.append(self.a_left_border)
                self.seen_right_borders.append(self.a_right_border)
                self.seen_top_borders.append(self.a_top_border)
                self.seen_bottom_borders.append(self.a_bottom_border)

        elif self.scene_id in [1, 2, 3]:
            self.a_left_border, self.a_right_border, self.a_top_border, self.a_bottom_border = \
                self.img_left_border, self.img_right_border, self.img_top_border, self.img_bottom_border
            self.a_width, self.a_height = self.a_right_border - self.a_left_border, self.a_bottom_border - self.a_top_border
            self.m_width, self.m_height = int(self.a_width / self.M), int(self.a_height / (self.M // 2))
            if self.mask_out_of_view:
                for m_i in range(0, self.M // 2):
                    seen_top_border_ = self.a_top_border + self.m_height * m_i
                    if m_i < self.M // 2 - 1: seen_bottom_border_ = self.a_top_border + self.m_height * (m_i + 1)
                    elif m_i == self.M // 2 - 1: seen_bottom_border_ = self.a_bottom_border
                    for m_j in range(0, self.M):
                        self.seen_left_borders.append(self.a_left_border + self.m_width * m_j)
                        self.seen_right_borders.append(self.a_left_border + self.m_width * (m_j + 1))
                        self.seen_top_borders.append(seen_top_border_)
                        self.seen_bottom_borders.append(seen_bottom_border_)
            else:
                self.seen_left_borders.append(self.a_left_border)
                self.seen_right_borders.append(self.a_right_border)
                self.seen_top_borders.append(self.a_top_border)
                self.seen_bottom_borders.append(self.a_bottom_border)
        elif self.scene_id == 4:
            self.a_left_border, self.a_right_border = 200, 1280
            self.a_top_border, self.a_bottom_border = self.img_top_border, self.img_bottom_border
            self.a_width, self.a_height = self.a_right_border - self.a_left_border, self.a_bottom_border - self.a_top_border
            self.m_width, self.m_height = int(self.a_width / self.M), int(self.a_height / (self.M // 2))
            if self.mask_out_of_view:
                for m_i in range(0, self.M // 2):
                    seen_top_border_ = self.a_top_border + self.m_height * m_i
                    if m_i < self.M // 2 - 1: seen_bottom_border_ = self.a_top_border + self.m_height * (m_i + 1)
                    elif m_i == self.M // 2 - 1: seen_bottom_border_ = self.a_bottom_border
                    for m_j in range(0, self.M):
                        self.seen_left_borders.append(self.a_left_border + self.m_width * m_j)
                        self.seen_right_borders.append(self.a_left_border + self.m_width * (m_j + 1))
                        self.seen_top_borders.append(seen_top_border_)
                        self.seen_bottom_borders.append(seen_bottom_border_)
            else:
                self.seen_left_borders.append(self.a_left_border)
                self.seen_right_borders.append(self.a_right_border)
                self.seen_top_borders.append(self.a_top_border)
                self.seen_bottom_borders.append(self.a_bottom_border)


        # --------------
        #  Video Window
        # --------------
        self.crr_ts16_dfv4_ls_all_i = 0
        self.video_len = 0 # len(self.ts12_BBX5_all)
        self.len_win = args.len_win
        self.cumu_turning_angle = self.average_turning_angle * self.len_win
        self.n_wins = 0

        self.sub_tracklets = None # (win_i, subj_i, first_f_i_in_win_in_view, last_f_i_in_win_in_view) with len <= K

        self.seq_subj_i_in_view_dict = defaultdict()
        self.test_seq_subj_i_in_view_dict = defaultdict()
        self.seq_in_BBX5_dict = defaultdict() # key: (win_i, subj_i)
        # PDR >>>
        self.seq_in_BEV_Feet_Center_dict = defaultdict() # key: (win_i, subj_i)
        self.seq_in_BEV_Feet_Center_Others_dict = defaultdict() # key: (win_i, subj_i)
        # PDR <<<
        self.seq_in_BBXC3_dict = defaultdict() # key: (win_i, subj_i)
        self.seq_in_FTM2_dict = defaultdict() # key: (win_i, subj_i)
        self.seq_in_IMU19_dict = defaultdict() # key: (win_i, subj_i)
        self.seq_in_BBX5_Others_dict = defaultdict() # key: (win_i, subj_)
        self.seq_in_BBXC3_Others_dict = defaultdict() # key: (win_i, subj_i)
        self.test_seq_in_BBX5_dict = defaultdict() # key: (win_i, subj_i)
        self.test_seq_in_BBXC3_dict = defaultdict() # key: (win_i, subj_i)
        self.test_seq_in_FTM2_dict = defaultdict() # key: (win_i, subj_i)
        self.test_seq_in_IMU19_dict = defaultdict() # key: (win_i, subj_i)
        self.test_seq_in_BBX5_Others_dict = defaultdict() # key: (win_i, subj_)
        self.test_seq_in_BBXC3_Others_dict = defaultdict() # key: (win_i, subj_i)
        self.seq_in_BBX5_r_shape = None
        self.seq_in_BBXC3_r_shape = None
        self.seq_in_FTM2_c_shape = None
        self.seq_in_IMU19_c_shape = None
        self.seq_in_BBX5_Others_r_shape = None
        self.seq_in_BBXC3_Others_r_shape = None
        self.test_seq_in_BBX5_r_shape = None
        self.test_seq_in_BBXC3_r_shape = None
        self.test_seq_in_FTM2_c_shape = None
        self.test_seq_in_IMU19_c_shape = None

        self.test_seq_in_IMUaccgym9_200_c_shape = None
        self.test_seq_in_BBX5_Others_r_shape = None
        self.test_seq_in_BBXC3_Others_r_shape = None
        self.eps_denominator_iou = 1e-7

        if self.imu200:
            self.seq_in_IMUaccgym9_200_dict = defaultdict() # key: (win_i, subj_i)
            self.test_seq_in_IMUaccgym9_200_dict = defaultdict() # key: (win_i, subj_i)
            self.seq_in_IMU19accgym9_200_c_shape = None
        elif self.imu_ned_pos2:
            self.seq_in_IMU_NED_pos2_dict = defaultdict() # key: (win_i, subj_i)
            self.test_seq_in_IMU_NED_pos2_dict = defaultdict() # key: (win_i, subj_i)
            self.seq_in_IMU_NED_pos2_c_shape = None

        # XT - Transformer model <<<
        self.h_BBX5_dim = 32 # X8: 32
        self.h_FTM2_dim = 32
        self.h_IMU19_dim = 32 # X8: 32
        self.h_IMU_XT_dim = 32
        self.h_fused_dim = 32 # X8: 32
        self.n_filters = 32 # X8: 32
        self.kernel_size = 16 # X8: 16
        # self.seq_in_BBX5, self.seq_in_IMU19, self.seq_in_IMU_XT, self.seq_in_FTM2 = None, None, None, None
        # global seq_in_BBX5; global seq_in_IMU19; global seq_in_IMU_XT; global seq_in_FTM2
        # self.seq_out_BBX5, self.seq_out_IMU19, self.seq_in_IMU_XT, self.seq_out_FTM2 = None, None, None, None
        # global seq_out_BBX5; global seq_out_IMU19; global seq_in_IMU_XT; global seq_out_FTM2
        self.train_val_rate = 0.9

        self.model = None
        self.exp_id_dict = {'pretrain': 0, 'train': 0, 'test': 0, 'embed': 0, 'investigate': 0} # Default id; edit

        # Update exp_id_dict if specified in the command line >>>
        # -1: unspecified
        if args.pretrain_exp_id != -1: self.exp_id_dict['pretrain'] = args.pretrain_exp_id
        if args.train_exp_id != -1: self.exp_id_dict['train'] = args.train_exp_id
        if args.test_exp_id != -1: self.exp_id_dict['test'] = args.test_exp_id
        if args.embed_exp_id != -1: self.exp_id_dict['embed'] = args.embed_exp_id
        if args.investigate_exp_id != -1: self.exp_id_dict['investigate'] = args.investigate_exp_id
        # Update exp_id_dict if specified in the command line <<<

        self.checkpoint_path_dict = {'pretrain': '', 'train': '', 'test': '', 'embed': '', 'investigate': ''}
        for running_mode, path in self.checkpoint_path_dict.items():
            self.checkpoint_path_dict[running_mode] = self.root_path + '/Data/checkpoints/XT/scene' + \
                str(self.scene_id) + '/' + self.model_id + '/' + running_mode + '/exps/' + str(self.exp_id_dict[running_mode])
        # all_exp_paths = glob.glob(self.checkpoint_path_dict[self.running_mode][:self.checkpoint_path_dict[self.running_mode].index('exps') + len('exps')] + '/*')
        # # DEBUG:
        # print('\n\n all_exp_paths: ', all_exp_paths)
        # if args.pretrain_exp_id == -1 and len(all_exp_paths) > 0:
        #     all_exp_paths = sorted(all_exp_paths)
        #     all_exp_id_ls = [int(exp_path[exp_path.index('exps') + len('exps') + 1:]) for exp_path in all_exp_paths]
        #     print(all_exp_paths)
        #     print('\n\n all_exp_id_ls: ', all_exp_id_ls)
        #     self.exp_id_dict[self.running_mode] = max(all_exp_id_ls) + 1
        # self.checkpoint_path_dict[self.running_mode] = \
        #     self.checkpoint_path_dict[self.running_mode][:self.checkpoint_path_dict[self.running_mode].\
        #     index('exps') + len('exps')] + '/' + str(self.exp_id_dict[self.running_mode])
        # DEBUG:
        # print('\n\n all_exp_paths[-1][all_exp_paths[-1].index(exps) + len(exps)]: ', all_exp_paths[-1][all_exp_paths[-1].index('exps') + len('exps') + 1:])
        print('\n\n self.checkpoint_path_dict[self.running_mode]:', self.checkpoint_path_dict[self.running_mode])

        if not os.path.exists(self.checkpoint_path_dict[self.running_mode]) and self.makedirs:
            os.makedirs(self.checkpoint_path_dict[self.running_mode])
        # self.model_path_to_save = self.checkpoint_root_path + '/model.h5' # UPDATE
        # self.model_weights_path_to_save = self.checkpoint_root_path + '/w.pt' # '/w.ckpt' # UPDATE
        # self.start_training_time = ''
        # self.start_training_time_ckpt_path = ''
        # self.history_callback_path_to_save = self.checkpoint_root_path + '/history_callback.p' # self.seq_path + '/' + self.model_id + '_history_callback.p'
        # self.history_callback = None
        # self.loss_lambda = 1
        # self.opt = None

        # -------
        #  Train
        # -------
        self.train_cfg = {
            'type': 'pretrain',
            'global_epochs_done': 0,
            'last_sum_train_loss': np.nan,
            'best_sum_train_loss': np.nan,
            'len_train_data_loader': 0,
            'last_avg_train_loss': np.nan,
            'best_avg_train_loss': np.nan,
            'last_val_loss': np.nan,
            'best_val_loss': np.nan,
            'batch_size': self.batch_size, # 32, # 128
            'lr': self.learning_rate, # 0.001,
            'n_epochs': self.training_epochs, # 1000000000000000, # 3200 # 200 # 100000 # 100000
            'warmup': 0.1,
            'save_steps': 0, # check self.save_model_epoch_interval
            'total_steps': 200000000,
            'lambda1': 0,
            'lambda2': 0,
            'global_epochs_done': 0,
            'save_model_epoch_interval': self.save_model_epoch_interval, # edit
            'label_rate': 0.9, # 0.01 # labeled sample / total sample with correspondences # xdecoder
            'loss_func': self.loss_func,
            'resume_training': self.resume_training,
            'seed': self.seed,
            'noise': self.noise
        }
        # ------
        #  Test
        # ------
        self.test_cfg = {
            'type': 'test',
            'loss_func': '',
            'batch_size': 1,
            'label_rate': 0.9, # 0.01 # labeled sample / total sample with correspondences # xdecoder
            'seed': self.seed,
            'noise': self.noise
        }

        # -------
        #  Model
        # -------
        # XT - Transformer (and other models) >>>
        self.model_cfg = {
            'train_model_id': args.train_model_id,
            'model_type': 'transformer',
            'hidden': self.hidden, # 72 by default
            'hidden_imu200': 144, # to be updated
            'hidden_ff': 144,
            'hidden_ff_imu200':144, # to be updated
            'feature': 'IMU_TX', # to be updated
            'BBX5_feature_num': self.BBX5_dim,
            'FTM2_feature_num': self.FTM2_dim,
            'IMU_XT_feature_num': self.IMU_XT_dim,
            'IMU_XT_feature_num_imu200': self.IMU_XT_dim * 200,
            'n_layers': 4,
            'n_layers_lstm': 2,
            'hidden_lstm': 8,
            'hidden_lstm_imu200': 12, # to be updated
            'n_heads': 4,
            'len_win': self.len_win,
            'seq_len': self.len_win,
            'emb_norm': True,
            'BBX5_dim': self.BBX5_dim,
            # Xtranslator >>>
            'FTM2_dim': self.FTM2_dim,
            'IMU_XT_dim': self.IMU_XT_dim,
            'h_BBX5_dim': self.h_BBX5_dim,
            'h_FTM2_dim': self.h_FTM2_dim,
            'h_IMU19_dim': self.h_IMU19_dim,
            'h_IMU_XT_dim': self.h_IMU_XT_dim,
            'h_fused_dim': self.h_fused_dim,
            'n_filters': self.n_filters,
            'kernel_size': self.kernel_size,
            # Xtranslator <<<
            'imu200': self.imu200,
            'label_rate': 0.9, # 0.01 # labeled sample / total sample with correspondences # xdecoder
            'dropout_rate': 0.5,
            'bidirectional': False,
            'batch_size': self.train_cfg['batch_size'],
            'optimizer': self.optimizer,
            'loss_func': self.loss_func,
            'n_subjects': self.n_subjects,
            'kalman_filter_modality': self.kalman_filter_modality,
            'kf_x_dim': self.kf_x_dim,
            'kf_z_dim': self.kf_z_dim,
            'transformer_Tc_in': self.transformer_Tc_in,
            'std': self.std,
            'seed': self.seed,
            'noise': self.noise
        }

        self.reconstruction_path = None
        if self.model_cfg['train_model_id'] == 'Xformer_IemcC2C':
            self.reconstruction_path = 'IemcC2C' # 'I2C' # I: IMU; Iem: IMU embeddings; C: Camera
        elif 'IFcC2C' in self.model_cfg['train_model_id']:
            self.reconstruction_path = 'IFcC2C' # 'I2C' # I: IMU; Iem: IMU embeddings; C: Camera
        elif self.model_cfg['train_model_id'] == 'Width_regressor':
            self.reconstruction_path = 'y2w'
        elif self.model_cfg['train_model_id'] == 'Height_regressor':
            self.reconstruction_path = 'y2h'
        elif self.model_cfg['train_model_id'] == 'Depth_regressor':
            self.reconstruction_path = 'y2d'
        if 'former' in self.model_cfg['train_model_id']: self.model_cfg['model_type'] = 'transformer'
        elif 'lstm' in self.model_cfg['train_model_id']: self.model_cfg['model_type'] = 'lstm'
        elif 'translator' in self.model_cfg['train_model_id']: self.model_cfg['model_type'] = 'translator' # Baseline of ViTag Xtranslator
        elif 'regressor' in self.model_cfg['train_model_id']: self.model_cfg['model_type'] = 'regressor'
        elif 'pdr' in self.model_cfg['train_model_id'] or \
            self.model_cfg['train_model_id'] == 'Width_regressor' or \
            self.model_cfg['train_model_id'] == 'Height_regressor' or \
            self.model_cfg['train_model_id'] == 'Depth_regressor': self.model_cfg['model_type'] = 'pdr'
        elif 'Tc_interp' in self.model_cfg['train_model_id']: self.model_cfg['model_type'] = 'Tc_interp'
        elif 'kalman_filter' in self.model_cfg['train_model_id']: self.model_cfg['model_type'] = 'kalman_filter'
        elif 'kalman_filter_v2' in self.model_cfg['train_model_id']: self.model_cfg['model_type'] = 'kalman_filter_v2'
        #  To be updated in prepare_*_data()
        # -----------------------------------

        # ---------------
        #  Visualization
        # ---------------
        self.vis = args.vis # edit
        self.vis_eval = args.vis_eval # edit
        self.img_rows = 720
        self.img_cols = 1280

        # PDR >>>
        # -----------------
        #  Imgs to Display
        # -----------------
        self.img, self.BEV, self.BEV_ORI, self.img_f_BEV = None, None, None, None
        self.IMU_NED_range_imgs, self.IMU_NED_imgs, self.IMU_NED_imgs_ORI = [], [], []

        # -----
        #  BEV
        # -----
        self.BEV_src_pnts_ls = [((830, 662), (996, 599), (1117, 651), (949, 746)), \
                                ((440, 432), (542, 427), (570, 463), (454, 471)), \
                                # ((551, 296), (541, 323), (612, 315), (613, 293)), \ # OBS scene2
                                ((611, 284), (612, 315), (686, 311), (668, 280)), \
                                ((52, 522), (165, 573), (452, 464), (313, 450)), \
                                ((480, 298), (801, 266), (661, 190), (472, 196))]
        self.BEV_src_pnts = ()
        self.BEV_dst_pnts_ls = [((500, 500), (550, 550), (500, 600), (450, 550)), \
                                # ((500, 500), (600, 500), (600, 600), (500, 600)), \ # OBS scene0
                                # ((847, 608), (835, 600), (855, 601), (850, 590)), \ # OBS
                                # ((754, 471), (740, 432), (842, 427), (870, 463)), \ # OBS
                                ((754, 471), (735 - 11, 425 - 27), (842, 427), (870, 463)), \
                                ((612, 565 + 20 - 180), (686 - 50, 560 + 20 - 180), (668, 530 - 20 - 180), (611, 534 - 180)), \
                                ((50 + 500, 522 - 100), (165 + 500, 573 - 100), (452 + 500, 464 - 100), (313 + 500, 450 - 100)), \
                                ((480 + 500, 298), (755 + 500, 274), (653 + 500, 188), (472 + 500, 196))]
                                # (point0, point1, point3, point2)
        self.BEV_dst_pnts = ()
        self.BEV_M = None # Bird's Eye View Perspective Transformation Matrix
        self.BEV_scale = 2
        self.BEV_M_iv = None

        # PDR <<<

        # -------
        #  Color
        # -------
        self.color_ls = ['crimson', 'lime green', 'royal blue', 'chocolate', 'purple', 'lemon']
        self.color_dict = {
            'crimson': (60,20,220),
            'lime green': (50,205,50),
            'royal blue': (225,105,65),
            'chocolate': (30,105,210),
            'purple': (128,0,128),
            'lemon': (0,247,255)
        }
        self.border_size = 2

        self.gpu = None

        # ----------
        #  Test Log
        # ----------
        self.log_time = datetime.datetime.now().strftime("%D_%H_%M_%S")
        self.log_time = self.log_time.replace('/', '_')
        print('self.log_time: ', self.log_time)
        self.test_log_id = ''
        self.test_log_root_path = self.checkpoint_path_dict[self.running_mode] + '/log'
        self.test_log_path = self.test_log_root_path + '/' + self.model_id

        self.test_log_file_path = ''
        self.test_log_file = None
        self.test_log_pkl_file_path = ''
        self.test_log_pkl_file = None

        self.test_log_per_frame_file_path =''
        self.test_log_per_frame_file = None

        self.ts16_dfv4_subj_i_to_BBX5_prime = defaultdict()
        self.ts16_dfv4_subj_i_to_BBX5_prime_path_to_save = self.checkpoint_path_dict[self.running_mode] + '/ts16_dfv4_subj_i_to_BBX5_prime.pkl'
        self.ts16_dfv4_to_pred_BBX5_labels = defaultdict()
        self.ts16_dfv4_to_pred_BBX5_labels_path_to_save = self.checkpoint_path_dict[self.running_mode] + '/ts16_dfv4_to_pred_BBX5_labels.pkl'

        self.ts16_dfv4_to_test_stats = defaultdict()
        self.ts16_dfv4_to_test_stats_path_to_save = self.checkpoint_path_dict[self.running_mode] + '/ts16_dfv4_to_test_stats.pkl'

        self.prev_gd_pred_phone_i_BBX_ls, self.prev_gd_pred_phone_i_BBXwDelta_ls, self.prev_gd_pred_phone_i_IMU_ls = [], [], []
        self.prev_hg_pred_phone_i_BBX_ls, self.prev_hg_pred_phone_i_BBXwDelta_ls, self.prev_hg_pred_phone_i_IMU_ls = [], [], []
        self.scene_test_stats = {'gd': {'BBX': {'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0, 'correct_num': 0, 'total_num': 0, \
                                            'IDSWITCH': np.nan, 'ts16_dfv4_BBX_IDP': 0.0, 'cumu_BBX_IDP': 0.0}, \
                                        'IMU': {'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0, 'correct_num': 0, 'total_num': 0, \
                                            'IDSWITCH': np.nan, 'ts16_dfv4_IMU_IDP': 0.0, 'cumu_IMU_IDP': 0.0}}, \
                                 'hg': {'BBX': {'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0, 'correct_num': 0, 'total_num': 0, \
                                            'IDSWITCH': np.nan, 'ts16_dfv4_BBX_IDP': 0.0, 'cumu_BBX_IDP': 0.0}, \
                                        'IMU': {'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0, 'correct_num': 0, 'total_num': 0, \
                                            'IDSWITCH': np.nan, 'ts16_dfv4_IMU_IDP': 0.0, 'cumu_IMU_IDP': 0.0}}} # hg: Hungarian, gd: greedy-matching
        # FN: misses of Phone Holders, TN: Others -> None
        self.scene_test_stats_path_to_save = self.checkpoint_path_dict[self.running_mode] + '/scene_test_stats.pkl'
        #  To de updated in prepare_testing_data()
        # -----------------------------------------
