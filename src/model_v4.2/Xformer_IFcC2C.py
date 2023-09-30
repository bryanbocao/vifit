'''
Usage:
Train:
    python3 Xformer_IFcC2C.py -ud -n -rm train -te 500 -nan linear_interp -tr_md_id Xformer_IFcC2C -m <MACHINE_NAME> -sc 0 -tsid_idx 5 -lw 30
    python3 Xformer_IFcC2C.py -ud -n -rm train -te 500 -nan linear_interp -tr_md_id Xformer_IFcC2C -m <MACHINE_NAME> -sc 1 -tsid_idx 0 -lw 30
    python3 Xformer_IFcC2C.py -ud -n -rm train -te 500 -nan linear_interp -tr_md_id Xformer_IFcC2C -m <MACHINE_NAME> -sc 2 -tsid_idx 13 -lw 30
    python3 Xformer_IFcC2C.py -ud -n -rm train -te 500 -nan linear_interp -tr_md_id Xformer_IFcC2C -m <MACHINE_NAME> -sc 3 -tsid_idx 8 -lw 30
    python3 Xformer_IFcC2C.py -ud -n -rm train -te 500 -nan linear_interp -tr_md_id Xformer_IFcC2C -m <MACHINE_NAME> -sc 4 -tsid_idx 4 -lw 30

Test:
    python3 Xformer_IFcC2C.py -ud -n -rm test -nan linear_interp -tr_md_id Xformer_IFcC2C -m <MACHINE_NAME> -sc 0 -tsid_idx 5 -lw 30 -ld_tr_eid -tr_eid <MODEL_TRAIN_ID> -ve -vbt GND_PRED -vbl 10
    python3 Xformer_IFcC2C.py -ud -n -rm test -nan linear_interp -tr_md_id Xformer_IFcC2C -m <MACHINE_NAME> -sc 1 -tsid_idx 0 -lw 30 -ld_tr_eid -tr_eid <MODEL_TRAIN_ID> -ve -vbt GND_PRED -vbl 10
    python3 Xformer_IFcC2C.py -ud -n -rm test -nan linear_interp -tr_md_id Xformer_IFcC2C -m <MACHINE_NAME> -sc 2 -tsid_idx 13 -lw 30 -ld_tr_eid -tr_eid <MODEL_TRAIN_ID> -ve -vbt GND_PRED -vbl 10
    python3 Xformer_IFcC2C.py -ud -n -rm test -nan linear_interp -tr_md_id Xformer_IFcC2C -m <MACHINE_NAME> -sc 3 -tsid_idx 8 -lw 30 -ld_tr_eid -tr_eid <MODEL_TRAIN_ID> -ve -vbt GND_PRED -vbl 10
    python3 Xformer_IFcC2C.py -ud -n -rm test -nan linear_interp -tr_md_id Xformer_IFcC2C -m <MACHINE_NAME> -sc 4 -tsid_idx 4 -lw 30  -ld_tr_eid -tr_eid <MODEL_TRAIN_ID> -ve -vbt GND_PRED -vbl 10


'''

import argparse

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from sklearn.metrics.pairwise import cosine_similarity
from scipy import stats
from embeddings import load_embeddings
from utils import Config, preprocess_config, EmbedDataset, BBX5_to_dBBX5, \
    BBX5_helper, labels_to_boxes
import train
from models import Xformer_IFcC2C
from embeddings import generate_embeddings

def proc_Xformer_IFcC2C(C: Config):
    if C.running_mode == 'train':
        update_running_mode = True
        load_pretrained_embeds = False
        trainer, train_data_loader, val_data_loader, _, \
            train_seq_in_correspondence, val_seq_in_correspondence, _, \
            criterion, optimizer, C = preprocess_config(C, 'train', update_running_mode, load_pretrained_embeds)
        # [seq_in_BBX5, seq_in_FTM2, seq_in_IMU19]

        # print('Xformer.py - proc_Xformer_IFcC2C - elif C.running_mode == train: np.shape(train_data_loader): ', np.shape(train_data_loader))
        # print('Xformer.py - proc_Xformer_IFcC2C - elif C.running_mode == train: np.shape(val_data_loader): ', np.shape(val_data_loader))

        # Generate embeddings for train and val sets
        # if C.reconstruction_path == 'IFcC2C':
        #     trainer_embed, train_data_loader_embed, val_data_loader_embed, _, \
        #         train_seq_in_correspondence, val_seq_in_correspondence, _, \
        #         criterion_embed, optimizer_embed, C = preprocess_config(C, 'embed')
        #     train_embeds = generate_embeddings(C, trainer_embed, train_data_loader, 'IMU', save=False)
        #     val_embeds = generate_embeddings(C, trainer_embed, val_data_loader, 'IMU', save=False)
        #     # DEBUG:
        #     print('\n\n np.shape(train_embeds): ', np.shape(train_embeds)) # (27475, 30, 72)
        #     print('\n\n np.shape(val_embeds): ', np.shape(val_embeds)) # (3053, 30, 72)
        #     print('\n\n np.shape(train_seq_in_correspondence[0]): ', np.shape(train_seq_in_correspondence[0])) # (27475, 30, 5)
        #     print('\n\n np.shape(train_seq_in_correspondence[1]): ', np.shape(train_seq_in_correspondence[1])) # (27475, 30, 2)
        #     print('\n\n np.shape(train_seq_in_correspondence[2]): ', np.shape(train_seq_in_correspondence[2])) # (27475, 30, 19)
        #     print('\n\n np.shape(val_seq_in_correspondence[0]): ', np.shape(val_seq_in_correspondence[0])) # (3053, 30, 5)
        #     print('\n\n np.shape(val_seq_in_correspondence[1]): ', np.shape(val_seq_in_correspondence[1])) # (3053, 30, 2)
        #     print('\n\n np.shape(val_seq_in_correspondence[2]): ', np.shape(val_seq_in_correspondence[2])) # (3053, 30, 19)
        #     # print('\n\n (train_embeds): ', (train_embeds))
        #
        # train_data_set = EmbedDataset(train_embeds, train_seq_in_correspondence[0]) # Iem2C
        # train_data_loader = DataLoader(train_data_set, shuffle=True, batch_size=C.batch_size)
        # val_data_set = EmbedDataset(val_embeds, val_seq_in_correspondence[0]) # Iem2C
        # val_data_loader = DataLoader(val_data_set, shuffle=False, batch_size=C.batch_size)
    elif C.running_mode == 'test':
        update_running_mode = True
        load_pretrained_embeds = False
        # flag = None
        trainer, _, val_data_loader, test_data_loader, \
            _, val_seq_in_correspondence, test_seq_in_correspondence, \
            criterion, optimizer, C = preprocess_config(C, 'test', update_running_mode, load_pretrained_embeds) #, flag)
        # [seq_in_BBX5, seq_in_FTM2, seq_in_IMU19]
        # print('Xformer.py - proc_Xformer_IFcC2C - elif C.running_mode == test: test_data_loader.__len__(): ', test_data_loader.__len__())
        # print('Xformer.py - proc_Xformer_IFcC2C - elif C.running_mode == test: len(test_seq_in_correspondence): ', len(test_seq_in_correspondence))
        # print('Xformer.py - proc_Xformer_IFcC2C - elif C.running_mode == test: len(val_seq_in_correspondence): ', len(val_seq_in_correspondence))
        '''
        e.g.
        Xformer.py - proc_Xformer_IFcC2C - elif C.running_mode == test: test_data_loader.__len__():  2685
        Xformer.py - proc_Xformer_IFcC2C - elif C.running_mode == test: len(test_seq_in_correspondence):  4
        Xformer.py - proc_Xformer_IFcC2C - elif C.running_mode == test: len(val_seq_in_correspondence):  4
        '''

    def func_loss(model, batch, transformer_Tc_in):
        # print('Xformer_IFcC2C - func_loss() - np.shape(batch): ', np.shape(batch))
        # print('Xformer_IFcC2C - func_loss() - np.shape(batch[0]): ', np.shape(batch[0]))
        # print('Xformer_IFcC2C - func_loss() - np.shape(batch[1]): ', np.shape(batch[1]))
        # print('Xformer_IFcC2C - func_loss() - np.shape(batch[2]): ', np.shape(batch[2]))
        '''
        e.g.
        Xformer_IFcC2C - func_loss() - np.shape(batch):  (3,)
        Xformer_IFcC2C - func_loss() - np.shape(batch[0]):  torch.Size([32, 10, 1, 5])
        Xformer_IFcC2C - func_loss() - np.shape(batch[1]):  torch.Size([32, 10, 1, 2])
        Xformer_IFcC2C - func_loss() - np.shape(batch[2]):  torch.Size([32, 10, 1, 9])
        '''
        labels = batch[0]
        labels_PRED = model(batch, transformer_Tc_in)

        # >>> Move to prepare_sync_training_data() and prepare_sync_testing_data to solve the inplace operation issue
        # if C.delta_BBX5_version > 0:
        #     labels = BBX5_to_dBBX5(C, labels, C.delta_BBX5_version)
        #     labels_PRED = BBX5_to_dBBX5(C, labels_PRED, C.delta_BBX5_version)
        # print('Xformer_IFcC2C - func_loss() - C.loss_func: ', C.loss_func)
        # print('Xformer_IFcC2C - func_loss() - np.shape(labels): ', np.shape(labels))
        # print('Xformer_IFcC2C - func_loss() - np.shape(labels_PRED): ', np.shape(labels_PRED))
        '''
        Xformer_IFcC2C - func_loss() - C.loss_func:  MSE_xyd
        Xformer_IFcC2C - func_loss() - np.shape(labels):  torch.Size([32, 10, 1, 5])
        Xformer_IFcC2C - func_loss() - np.shape(labels_PRED):  torch.Size([32, 10, 1, 5])
        '''
        labels_ = BBX5_helper(C.loss_func, labels).float(); labels_PRED_ = BBX5_helper(C.loss_func, labels_PRED).float()
        # <<< Move to prepare_sync_training_data() and prepare_sync_testing_data to solve the inplace operation issue

        # print('Xformer_IFcC2C - func_loss() - after loss_func selection - np.shape(labels): ', np.shape(labels))
        # print('Xformer_IFcC2C - func_loss() - after loss_func selection - np.shape(labels_PRED): ', np.shape(labels_PRED))

        # DEBUG:
        # if 'MSE' in C.loss_func: loss = criterion(labels_, labels_PRED_)
        # elif 'IOU' in C.loss_func:
        #     boxes_a, boxes_b = labels_to_boxes(labels_, labels_PRED_)
        #     # loss = criterion(boxes_a, boxes_b)
        #     if 'DIOU' in C.loss_func:
        #         if 'depth' in C.loss_func:
        #             loss = criterion(C.loss_func, boxes_a, boxes_b)
        #         else: loss = criterion(boxes_a, boxes_b)
        #     if 'GIOU' in C.loss_func:
        #         if 'depth' in C.loss_func:
        #             loss = criterion(C.loss_func, boxes_a, boxes_b)
        #         else: loss = criterion(boxes_a, boxes_b)

        # if 'MSE' in C.loss_func: loss = criterion(labels_, labels_PRED_)
        # elif 'IOU' in C.loss_func:
        #     if 'DIOU' in C.loss_func:
        #         if 'depth' in C.loss_func: loss = criterion(C.loss_func, labels_, labels_PRED_)
        #         else: loss = criterion(labels_, labels_PRED_)
        #     if 'GIOU' in C.loss_func:
        #         if 'depth' in C.loss_func: loss = criterion(C.loss_func, labels_, labels_PRED_)
        #         else: loss = criterion(labels_, labels_PRED_)

        if C.loss_func in ['MSE_xydwh', 'MSE_xyd', 'MSE_xy']:
            loss = criterion(labels_, labels_PRED_)
        elif C.loss_func in ['DIOU_depth', 'GIOU_depth']:
            loss = criterion(C, labels_, labels_PRED_, None, None)
        elif C.loss_func in ['DIOU', 'GIOU']:
            loss = criterion(labels_, labels_PRED_)
        elif C.loss_func in['NMSE_xydwh_DIOU', 'NMSE_xydwh_GIOU']:
            loss = criterion(C, labels_, labels_PRED_, labels, labels_PRED)

        # print('Xformer_IFcC2C - func_loss() - loss: ', loss)

        # Add penalty if predictions are not moving
        # moving_thred = 5 * C.len_win
        # dlabels_PRED_ = BBX5_to_dBBX5(C, labels_PRED_, 1)
        # # print('dlabels_PRED_: ', dlabels_PRED_)
        # for batch_i in range(np.shape(dlabels_PRED_)[0]):
        #     # if torch.equal(dlabels_PRED_[batch_i, :, :, :], torch.zeros(np.shape(dlabels_PRED_))):
        #     # print('\n\n dlabels_PRED_[batch_i, :, :, :]: ', dlabels_PRED_[batch_i, :, :, :])
        #     # if dlabels_PRED_[batch_i, :, :, :] == np.zeros(np.shape(dlabels_PRED_)):
        #     if torch.sum(dlabels_PRED_[batch_i, :, :, :]) < moving_thred:
        #         loss += moving_thred
        #         # print('\n\n Xformer_IFcC2C.py - proc_Xformer_IFcC2C - func_loss - not moving - loss: ', loss)
        #         # print('\n\n Xformer_IFcC2C.py - proc_Xformer_IFcC2C - func_loss - not moving - torch.sum(dlabels_PRED_[batch_i, :, :, :]): ', torch.sum(dlabels_PRED_[batch_i, :, :, :]))

        return loss

    def func_forward(model, batch, transformer_Tc_in):
        # print('Xformer_IFcC2C - func_forward() - np.shape(batch): ', np.shape(batch))
        # print('Xformer_IFcC2C - func_forward() - np.shape(batch[0]): ', np.shape(batch[0]))
        # print('Xformer_IFcC2C - func_forward() - np.shape(batch[1]): ', np.shape(batch[1]))
        # print('Xformer_IFcC2C - func_forward() - np.shape(batch[2]): ', np.shape(batch[2]))
        '''
        e.g.
        Xformer_IFcC2C - func_forward() - np.shape(batch):  (3,)
        Xformer_IFcC2C - func_forward() - np.shape(batch[0]):  torch.Size([32, 10, 1, 5])
        Xformer_IFcC2C - func_forward() - np.shape(batch[1]):  torch.Size([32, 10, 1, 2])
        Xformer_IFcC2C - func_forward() - np.shape(batch[2]):  torch.Size([32, 10, 1, 9])
        '''
        # labels = batch[0]
        labels_PRED = model(batch, transformer_Tc_in)

        # >>> Move to prepare_sync_training_data() and prepare_sync_testing_data to solve the inplace operation issue
        # if C.delta_BBX5_version > 0:
        #     # labels = BBX5_to_dBBX5(C, labels, C.delta_BBX5_version) # OBS
        #     labels_PRED = BBX5_to_dBBX5(C, labels_PRED, C.delta_BBX5_version)
        # labels_ = BBX5_helper(C.loss_func, labels)
        labels_PRED_ = BBX5_helper(C.loss_func, labels_PRED)
        # <<< Move to prepare_sync_training_data() and prepare_sync_testing_data to solve the inplace operation issue
        return labels_PRED_

    def func_evaluate(labels, labels_PRED):
        # print('\n\n Xformer_IFcC2C - func_evaluate() - np.shape(labels): ', np.shape(labels))
        # print('\n\n Xformer_IFcC2C - func_evaluate() - np.shape(labels_PRED): ', np.shape(labels_PRED))
        labels_ = BBX5_helper(C.loss_func, labels); labels_PRED_ = BBX5_helper(C.loss_func, labels_PRED)
        # if 'MSE' in C.loss_func: loss_lm = criterion(labels_, labels_PRED_)
        # elif 'IOU' in C.loss_func:
        #     labels_ = torch.tensor(labels_, dtype=torch.float)
        #     labels_PRED_ = torch.tensor(labels_PRED_, dtype=torch.float)
        #     if 'DIOU' in C.loss_func:
        #         if 'depth' in C.loss_func: loss_lm = criterion(C.loss_func, labels_, labels_PRED_)
        #         else: loss_lm = criterion(labels_, labels_PRED_)
        #     if 'GIOU' in C.loss_func:
        #         if 'depth' in C.loss_func: loss_lm = criterion(C.loss_func, labels_, labels_PRED_)
        #         else: loss_lm = criterion(labels_, labels_PRED_)
        if C.loss_func in ['MSE_xydwh', 'MSE_xyd', 'MSE_xy']:
            loss_lm = criterion(labels_, labels_PRED_)
        elif C.loss_func in ['DIOU_depth', 'GIOU_depth']:
            loss_lm = criterion(C, labels_, labels_PRED_, None, None)
        elif C.loss_func in ['DIOU', 'GIOU']:
            loss_lm = criterion(labels_, labels_PRED_)
        elif C.loss_func in['NMSE_xydwh_DIOU', 'NMSE_xydwh_GIOU']:
            loss_lm = criterion(C, labels_, labels_PRED_, labels, labels_PRED)

        # return loss_lm.mean().cpu().numpy()
        return loss_lm.mean().cpu().detach().numpy()

    if C.running_mode == 'train':
        trainer.train(func_loss, func_forward, func_evaluate, \
            train_data_loader, val_data_loader, None, False, False, \
            C.reconstruction_path)
        print('Training done!')
    elif C.running_mode == 'test':
        if C.load_train_exp_id_model and C.exp_id_dict['train'] != -1:
            model_file = C.checkpoint_path_dict['train'] + '/epoch_best.pt'
        else: model_file = None
        # print('\n\n proc_Xformer_IFcC2C.py - proc_Xformer_IFcC2C() - C.running_mode == test - model_file: ', model_file)
        # trainer.eval_recon(C, func_forward, func_evaluate, test_data_loader, \
        #     model_file, None, False, False)
        trainer.eval_recon(C, func_forward, func_evaluate, test_seq_in_correspondence, \
            model_file, None, False, False)

if __name__ == "__main__":
    C = Config()
    # print('\n\n C.__dict__: ', C.__dict__)
    # print('\n\n C: ', C)

    # Label here refers to decoded modality
    proc_Xformer_IFcC2C(C)
