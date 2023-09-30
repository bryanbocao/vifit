import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from utils import Config

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

from filterpy.kalman import KalmanFilter

# >>>
# Code Base: LIMU-BERT
# https://github.com/dapowan/LIMU-BERT-Public/blob/master/models.py

# Model Utilities >>>
def gelu(x):
    '''Implementation of the gelu activation function by Hugging Face'''
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

def split_last(x, shape):
    'spit the last dimension with the given shape'
    shape = list(shape)
    if - 1 in shape:
        shape[shape.index(-1)] = x.size(-1) // -np.prod(shape)
    return x.view(*x.size()[:-1], *shape)

def merge_last(x, n_dims):
    'merge the last n_dims to a given number of dimensions'
    s = x.size()
    assert n_dims > 1 and n_dims < len(s)
    return x.view(*s[:-n_dims], -1)
# Model Utilities <<<

class LayerNorm(nn.Module):
    '''A layernorm module in the TF style (epsilon inside the square root).'''
    def __init__(self, cfg, variance_epsilon=1e-12):
        super().__init__()
        self.cfg = cfg
        # =============
        #  transformer
        if self.cfg['model_type'] == 'transformer':
            if self.cfg['imu200']:
                # cfg['hidden_imu200'] = cfg['hidden'] * 200
                self.gamma = nn.Parameter(torch.ones(cfg['hidden_imu200']), requires_grad=True)
                self.beta = nn.Parameter(torch.zeros(cfg['hidden_imu200']), requires_grad=True)
            else:
                self.gamma = nn.Parameter(torch.ones(cfg['hidden']), requires_grad=True)
                self.beta = nn.Parameter(torch.zeros(cfg['hidden']), requires_grad=True)
        #  transformer
        # =============

        # ======
        #  lstm
        if self.cfg['model_type'] == 'lstm':
            if self.cfg['imu200']:
                # cfg['hidden_imu200'] = cfg['hidden'] * 200
                self.gamma = nn.Parameter(torch.ones(cfg['hidden_lstm_mu200']), requires_grad=True)
                self.beta = nn.Parameter(torch.zeros(cfg['hidden_lstm_imu200']), requires_grad=True)
            else:
                self.gamma = nn.Parameter(torch.ones(cfg['hidden_lstm']), requires_grad=True)
                self.beta = nn.Parameter(torch.zeros(cfg['hidden_lstm']), requires_grad=True)
        #  lstm
        # ======
        self.variance_epsilon = variance_epsilon

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        # DEBUG:
        # print('\n\n')
        # print('models.py - LayerNorm - forward - before reshape - np.shape(u): ', np.shape(u))
        # print('models.py - LayerNorm - forward - before reshape - np.shape(s): ', np.shape(s))
        # print('models.py - LayerNorm - forward - before reshape - np.shape(x): ', np.shape(x))
        # print('models.py - LayerNorm - forward - np.shape(self.gamma): ', np.shape(self.gamma))
        # print('models.py - LayerNorm - forward - np.shape(self.beta): ', np.shape(self.beta))
        '''
        e.g.
        models.py - LayerNorm - forward - before reshape - np.shape(u):  torch.Size([32, 10, 200, 1]), prev torch.Size([32, 10, 1])
        models.py - LayerNorm - forward - before reshape - np.shape(s):  torch.Size([32, 10, 200, 1]), prev torch.Size([32, 10, 1])
        models.py - LayerNorm - forward - before reshape - np.shape(x):  torch.Size([32, 10, 200, 72]), prev torch.Size([32, 10, 72])
        models.py - LayerNorm - forward - np.shape(self.gamma):  torch.Size([14400])
        models.py - LayerNorm - forward - np.shape(self.beta):  torch.Size([14400])
        '''
        if x.ndim == 4:
            shape = np.shape(x)
            x = x.reshape((shape[0], shape[1], shape[2] * shape[3]))
            # print('models.py - LayerNorm - forward - before reshape - np.shape(x): ', np.shape(x))
        return self.gamma * x + self.beta

class Embeddings(nn.Module):
    def __init__(self, cfg, pos_embed=None):
        super().__init__()

        self.cfg = cfg
        print('\n models.py - Embeddings - self.cfg[feature]: ', self.cfg['feature'])

        # =============
        #  transformer
        # factorzide embedding
        if self.cfg['model_type'] == 'transformer':
            if 'BBX5' in self.cfg['feature']:
                self.lin = nn.Linear(cfg['BBX5_feature_num'], cfg['hidden'])
            elif self.cfg['feature'] == 'FTM2':
                self.lin = nn.Linear(cfg['FTM2_feature_num'], cfg['hidden'])
            elif self.cfg['feature'] == 'IMU_TX':
                if self.cfg['imu200']:
                    cfg['hidden_imu200'] = cfg['hidden'] * 200
                    self.lin = nn.Linear(cfg['IMU_XT_feature_num_imu200'], cfg['hidden_imu200'])
                else: self.lin = nn.Linear(cfg['IMU_XT_feature_num'], cfg['hidden'])

            if pos_embed is None:
                if self.cfg['imu200']: self.pos_embed = nn.Embedding(cfg['seq_len'], cfg['hidden_imu200'])
                else: self.pos_embed = nn.Embedding(cfg['seq_len'], cfg['hidden'])
            else:
                self.pos_embed = pos_embed
        #  transformer
        # =============

        # ======
        #  lstm
        # factorzide embedding
        if self.cfg['model_type'] == 'lstm':
            if self.cfg['feature'] == 'fstBBX5':
                self.lin = nn.Linear(cfg['BBX5_feature_num'], cfg['hidden_lstm'])
            elif self.cfg['feature'] == 'FTM2':
                self.lin = nn.Linear(cfg['FTM2_feature_num'], cfg['hidden_lstm'])
            elif self.cfg['feature'] == 'IMU_TX':
                if self.cfg['imu200']:
                    cfg['hidden_lstm_imu200'] = cfg['hidden_lstm'] * 200
                    self.lin = nn.Linear(cfg['IMU_XT_feature_num_imu200'], cfg['hidden_lstm_imu200'])
                else: self.lin = nn.Linear(cfg['IMU_XT_feature_num'], cfg['hidden_lstm'])

            if pos_embed is None:
                if self.cfg['imu200']: self.pos_embed = nn.Embedding(cfg['seq_len'], cfg['hidden_lstm_imu200'])
                else: self.pos_embed = nn.Embedding(cfg['seq_len'], cfg['hidden_lstm'])
            else:
                self.pos_embed = pos_embed
        #  lstm
        # ======

        self.norm = LayerNorm(cfg)
        self.emb_norm = cfg['emb_norm']

    def forward(self, x):
        # DEBUG:
        # print('\n\n models.py - Embeddings() - forward() - x: ', x)
        # print('\n\n models.py - Embeddings() - forward() - np.shape(x): ', np.shape(x))
        # e.g. torch.Size([32, 10, 1, 9]), prev torch.Size([32, 10, 9])
        seq_len = x.size(1)
        pos = torch.arange(seq_len, dtype=torch.long, device=x.device)
        pos = pos.unsqueeze(0).expand(x.size(0), seq_len) # (s,) -> (B, S)

        # factorized embedding
        # DEBUG:
        # print('\n\n  models.py - Embeddings - pos.type(): ', pos.type())
        # e.g. torch.cuda.LongTensor
        # print('\n\n  models.py - Embeddings - x.type(): ', x.type())
        # e.g. torch.cuda.DoubleTensor # Originally torch.cuda.FloatTensor
        x = x.float()
        # print('\n\n  models.py - Embeddings - after conversion - x.type(): ', x.type())
        e = self.lin(x)
        if self.emb_norm:
            e = self.norm(e)
        # print('\n\n np.shape(pos): ', np.shape(pos))
        # e.g. np.shape(pos):  torch.Size([32, 10])
        # print('\n\n Before reshape np.shape(e): ', np.shape(e))
        # e.g. np.shape(pos):  torch.Size([32, 10, 1, 72]), prev torch.Size([32, 10, 72])
        e_shape = np.shape(e)

        # =============
        #  transformer
        if self.cfg['model_type'] == 'transformer':
            if self.cfg['imu200']: e = e.reshape((e_shape[0], e_shape[1], self.cfg['hidden_imu200']))
            else: e = e.reshape((e_shape[0], e_shape[1], self.cfg['hidden']))
        #  transformer
        # =============

        # ======
        #  lstm
        if self.cfg['model_type'] == 'lstm':
            if self.cfg['imu200']: e = e.reshape((e_shape[0], e_shape[1], self.cfg['hidden_lstm_imu200']))
            else: e = e.reshape((e_shape[0], e_shape[1], self.cfg['hidden_lstm']))
        #  lstm
        # ======

        # print('\n\n After reshape np.shape(e): ', np.shape(e))
        # e.g. torch.Size([32, 10, 72])
        # print('\n\n np.shape(self.pos_embed(pos)): ', np.shape(self.pos_embed(pos)))
        # e.g. prev np.shape(self.pos_embed(pos)):  torch.Size([32, 10, 72])
        e = e + self.pos_embed(pos)
        return self.norm(e)

class MultiProjection(nn.Module):
    '''Multi-Headed Dot Product Attention'''
    def __init__(self, cfg):
        super().__init__()
        self.proj_q = nn.Linear(cfg['hidden'], cfg['hidden'])
        self.proj_k = nn.Linear(cfg['hidden'], cfg['hidden'])
        self.proj_v = nn.Linear(cfg['hidden'], cfg['hidden'])

    def forward(self, x):
        '''
        x, q(query), k(key), v(value) : (B(batch_size), S(seq_len), D(dim))
        * split D(dim) into (H(n_heads), W(width of head)); D = H * W
        '''
        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        q, k, v = self.proj_q(x), self.proj_k(x), self.proj_v(x)
        return q, k, v

class MultiHeadedSelfAttention(nn.Module):
    '''Multi-Headed Dot Product Attention'''
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        if self.cfg['imu200']:
            self.proj_q = nn.Linear(cfg['hidden_imu200'], cfg['hidden_imu200'])
            self.proj_k = nn.Linear(cfg['hidden_imu200'], cfg['hidden_imu200'])
            self.proj_v = nn.Linear(cfg['hidden_imu200'], cfg['hidden_imu200'])
        else:
            self.proj_q = nn.Linear(cfg['hidden'], cfg['hidden'])
            self.proj_k = nn.Linear(cfg['hidden'], cfg['hidden'])
            self.proj_v = nn.Linear(cfg['hidden'], cfg['hidden'])
        # self.drop = nn.Dropout(cfg['p_drop_attn'])
        self.scores = None # for visualization
        self.n_heads = cfg['n_heads']

    def forward(self, x):
        '''
        x, q(query), k(key), v(value) : (B(batch_size), S(seq_len), D(dim))
        * split D(dim) into (H(n_heads), W(width of head)); D = H * W
        '''
        # (B, S, D) -proj-> (B, S, D) -split->(B, S, H, W) -trans-> (B, H, S, W)
        q, k, v = self.proj_q(x), self.proj_k(x), self.proj_v(x)
        q, k, v = (split_last(x, (self.n_heads, -1)).transpose(1, 2)
                    for x in [q, k, v])
        # (B, H, S, W) @ (B, H, W, S) -> (B, H, S, S) -softmax-> (B, H, S, S)
        scores = q @ k.transpose(-2, -1) / np.sqrt(k.size(-1))
        # scores = self.drop(F.softmax(scores, dim=-1))
        scores = F.softmax(scores, dim=-1)
        # (B, H, S, S) @ (B, H, S, W) -> (B, H, S, W) -trans-> (B, S, H, W)
        h = (scores @ v).transpose(1, 2).contiguous()
        # -merge-> (B, S, D)
        h = merge_last(h, 2)
        self.scores = scores
        return h

class PositionWiseFeedForward(nn.Module):
    '''Feedforward Neural Networks for each position'''
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        # self.fc1 = nn.Linear(cfg['hidden'], cfg['hidden_ff']) # Original Code
        # self.fc2 = nn.Linear(cfg['hidden_ff'], cfg['hidden']) # Original Code
        if self.cfg['imu200']:
            self.fc0 = nn.Linear(cfg['hidden_imu200'], cfg['hidden_ff_imu200'])
            self.fc1 = nn.Linear(cfg['hidden_ff_imu200'], cfg['hidden_imu200'])
        else:
            self.fc0 = nn.Linear(cfg['hidden'], cfg['hidden_ff'])
            self.fc1 = nn.Linear(cfg['hidden_ff'], cfg['hidden'])

        # self.activ = lambda x: activ_fn(cfg['activ_fn'], x)

    def forward(self, x):
        # (B, S, D) -> (B, S, D_ff) -> (B, S, D)
        # return self.fc2(gelu(self.fc1(x))) # Original Code
        return self.fc1(gelu(self.fc0(x)))

class Transformer(nn.Module):
    '''Transformer with Self-Attention Blocks'''
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.embed = Embeddings(cfg)
        # Original BERT does not use parameter-sharing strategy
        # self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg['n_layers'])])

        # Use parameter-sharing strategy
        self.n_layers = cfg['n_layers']
        self.attn = MultiHeadedSelfAttention(cfg)
        if self.cfg['imu200']: self.proj = nn.Linear(cfg['hidden_imu200'], cfg['hidden_imu200'])
        else: self.proj = nn.Linear(cfg['hidden'], cfg['hidden'])
        # self.norm1 = LayerNorm(cfg) # Original Code
        self.norm0 = LayerNorm(cfg)
        self.pwff = PositionWiseFeedForward(cfg)
        # self.norm2 = LayerNorm(cfg) # Original Code
        self.norm1 = LayerNorm(cfg)
        # self.drop = nn.Dropout(cfg['p_drop_hidden'])

    def forward(self, x):
        # DEBUG:
        # print('\n\n')
        # print('models.py - Transformer - forward - np.shape(x): ', np.shape(x))
        # e.g. models.py - Transformer - forward - np.shape(x):  torch.Size([32, 10, 1, 9])
        h = self.embed(x)
        # DEBUG:
        # print('\n\n')
        # print('models.py - Transformer - forward - np.shape(h): ', np.shape(h))
        # e.g. models.py - Transformer - forward - np.shape(h):  torch.Size([32, 10, 14400])

        for _ in range(self.n_layers):
            # h = block(h, mask)
            h = self.attn(h)
            # h = self.norm1(h + self.proj(h)) # Original Code
            # h = self.norm2(h + self.pwff(h)) # Original Code
            h = self.norm0(h + self.proj(h))
            h = self.norm1(h + self.pwff(h))
        return h

class LIMUBertModel4Pretrain(nn.Module):
    def __init__(self, cfg, output_embed=False):
        super().__init__()
        self.cfg = cfg
        self.transformer = Transformer(cfg) # encoder
        if self.cfg['feature'] == 'BBX5':
            self.fc = nn.Linear(cfg['hidden'], cfg['hidden'])
            self.linear = nn.Linear(cfg['hidden'], cfg['hidden'])
            self.activ = gelu
            self.norm = LayerNorm(cfg)
            self.decoder = nn.Linear(cfg['hidden'], cfg['BBX5_feature_num'])
        elif self.cfg['feature'] == 'FTM2':
            self.fc = nn.Linear(cfg['hidden'], cfg['hidden'])
            self.linear = nn.Linear(cfg['hidden'], cfg['hidden'])
            self.activ = gelu
            self.norm = LayerNorm(cfg)
            self.decoder = nn.Linear(cfg['hidden'], cfg['FTM2_feature_num'])
        elif self.cfg['feature'] == 'IMU_TX':
            if self.cfg['imu200']:
                self.fc = nn.Linear(cfg['hidden_imu200'], cfg['hidden_imu200'])
                self.linear = nn.Linear(cfg['hidden_imu200'], cfg['hidden_imu200'])
                self.activ = gelu
                self.norm = LayerNorm(cfg)
                self.decoder = nn.Linear(cfg['hidden_imu200'], cfg['IMU_XT_feature_num_imu200'])
            else:
                self.fc = nn.Linear(cfg['hidden'], cfg['hidden'])
                self.linear = nn.Linear(cfg['hidden'], cfg['hidden'])
                self.activ = gelu
                self.norm = LayerNorm(cfg)
                self.decoder = nn.Linear(cfg['hidden'], cfg['IMU_XT_feature_num'])
            self.output_embed = output_embed

    def forward(self, input_seqs, masked_pos=None):
        # DEBUG:
        # print('\n\n')
        # print('models.py - LIMUBertModel4Pretrain - np.shape(input_seqs): ', np.shape(input_seqs))
        '''
        e.g.
        models.py - LIMUBertModel4Pretrain - np.shape(input_seqs):  torch.Size([32, 10, 1800])
            prev torch.Size([32, 10, 9])
        '''
        h = self.transformer(input_seqs)
        # DEBUG:
        # print('\n\n')
        # print('models.py - LIMUBertModel4Pretrain - np.shape(h): ', np.shape(h))
        # h = self.transformer(input_seqs.astype(np.float32))
        if self.output_embed:
            return h
        if masked_pos is not None:
            masked_pos = masked_pos[:, :, None].expand(-1, -1, h.size(-1))
            h_masked = torch.gather(h, 1, masked_pos)
            h_masked = self.activ(self.linear(h_masked))
            h_masked = self.norm(h_masked)
            logits_lm = self.decoder(h_masked)
            return logits_lm
        return h
# Code Base: LIMU-BERT
# <<<

# >>>
# Classifier from pytorchic-bert
# https://github.com/dhlee347/pytorchic-bert/blob/master/classify.py
class Classifier(nn.Module):
    ''' Classifier with Transformer '''
    def __init__(self, cfg, n_labels):
        super().__init__()
        self.transformer = models.Transformer(cfg)
        self.fc = nn.Linear(cfg.dim, cfg.dim)
        self.activ = nn.Tanh()
        self.drop = nn.Dropout(cfg.p_drop_hidden)
        self.classifier = nn.Linear(cfg.dim, n_labels)

    def forward(self, input_ids, segment_ids, input_mask):
        h = self.transformer(input_ids, segment_ids, input_mask)
        # only use the first h in the sequence
        pooled_h = self.activ(self.fc(h[:, 0]))
        logits = self.classifier(self.drop(pooled_h))
        return logits
# Classifier from pytorchic-bert
# <<<

# >>>
# New Code
class Xformer_IemcC2C(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        # self.transformer = Transformer(cfg) # encoder

        # encoder
        # self.cfg['feature'] = 'IMU_TX'; self.transformer_IMU_TX = Transformer(cfg)
        self.cfg['feature'] = 'FTM2'; self.transformer_FTM2 = Transformer(cfg)
        self.cfg['feature'] = 'fstBBX5'; self.transformer_fstBBX5 = Transformer(cfg)

        if self.cfg['imu200']:
            self.linear = nn.Linear(cfg['hidden_imu200'], cfg['hidden_imu200'])
            self.activ = gelu
            self.norm = LayerNorm(cfg)
            self.linear_out = nn.Linear(cfg['hidden_imu200'], cfg['BBX5_dim'])
        else:
            self.linear = nn.Linear(cfg['hidden'], cfg['hidden'])
            self.activ = gelu
            self.norm = LayerNorm(cfg)
            self.linear_out = nn.Linear(cfg['hidden'], cfg['BBX5_dim'])

    def forward(self, embeds):
        # h = self.transformer(input_seqs)
        h = self.activ(self.linear(embeds))
        h = self.norm(h)
        out = self.linear_out(h)
        # print('np.shape(out): ', np.shape(out))
        if self.cfg['imu200']:
            out = torch.reshape(out, (np.shape(out)[0], self.cfg['seq_len'], 200, self.cfg['BBX5_dim']))
        else:
            out = torch.reshape(out, (np.shape(out)[0], self.cfg['seq_len'], 1, self.cfg['BBX5_dim']))
        return out

class Xformer_IFcC2C(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        # encoder
        self.cfg['feature'] = 'IMU_TX'; self.transformer_IMU_TX = Transformer(cfg)
        self.cfg['feature'] = 'FTM2'; self.transformer_FTM2 = Transformer(cfg)
        if self.cfg['transformer_Tc_in'] == 'fstBBX5':
            self.cfg['feature'] = 'fstBBX5'; self.transformer_fstBBX5 = Transformer(cfg)
        elif self.cfg['transformer_Tc_in'] == 'flBBX5':
            self.cfg['feature'] = 'flBBX5'; self.transformer_flBBX5 = Transformer(cfg)

        # decoder
        if self.cfg['imu200']:
            self.linear = nn.Linear(cfg['hidden'] * 3, cfg['hidden_imu200'])
            self.activ = gelu
            self.norm = LayerNorm(cfg)
            self.linear_out = nn.Linear(cfg['hidden_imu200'], cfg['BBX5_dim'])
        else:
            self.linear = nn.Linear(cfg['hidden'] * 3, cfg['hidden'])
            self.activ = gelu
            self.norm = LayerNorm(cfg)
            self.linear_out = nn.Linear(cfg['hidden'], cfg['BBX5_dim'])

    def forward(self, input_seqs, transformer_Tc_in):
        # decoder
        seq_in_BBX5, seq_in_FTM2, seq_in_IMUaccgym9 = input_seqs[0], input_seqs[1], input_seqs[2]
        # fstBBX5 = seq_in_BBX5[:, 0, :, :] # first frame
        seq_in_BBX5_ = copy.deepcopy(seq_in_BBX5) # first frame

        if transformer_Tc_in == 'fstBBX5':
            # >>> v1
            for i in range(self.cfg['seq_len']):
                seq_in_BBX5_[:,i,:,:] = seq_in_BBX5[:,0,:,:]
            # <<< v1

            # >>> v2
            # shp = np.shape(seq_in_BBX5)
            # for i in range(1, self.cfg['seq_len']):
            #     seq_in_BBX5_[:,i,:,:] = torch.zeros(shp[0], 1, shp[2], shp[3]) # np.zeros((shp[0], 1, shp[2], shp[3]))
            # <<< v2
            fstBBX5 = seq_in_BBX5_
            h_IMUaccgym9 = self.transformer_IMU_TX(seq_in_IMUaccgym9)
            h_FTM2 = self.transformer_FTM2(seq_in_FTM2)
            h_fstBBX5 = self.transformer_fstBBX5(fstBBX5)

            # DEBUG:
            # print('\n models.py - Xformer_IFcC2C - forward - np.shape(h_IMUaccgym9): ', np.shape(h_IMUaccgym9))
            # print('\n models.py - Xformer_IFcC2C - forward - np.shape(h_FTM2): ', np.shape(h_FTM2))
            # print('\n models.py - Xformer_IFcC2C - forward - np.shape(h_fstBBX5): ', np.shape(h_fstBBX5))
            '''
            e.g.
            models.py - Xformer_IFcC2C - forward - np.shape(h_IMUaccgym9):  torch.Size([32, 10, 72])
            models.py - Xformer_IFcC2C - forward - np.shape(h_FTM2):  torch.Size([32, 10, 72])
            models.py - Xformer_IFcC2C - forward - np.shape(h_fstBBX5):  torch.Size([32, 10, 72])
            '''
            # print('\n models.py - Xformer_IFcC2C - forward - h_fstBBX5: ', h_fstBBX5)
            h = torch.cat((h_IMUaccgym9, h_FTM2, h_fstBBX5), 2)

        elif transformer_Tc_in == 'flBBX5':
            for i in range(self.cfg['seq_len'] // 2):
                seq_in_BBX5_[:,i,:,:] = seq_in_BBX5[:,0,:,:]
            for i in range(self.cfg['seq_len'] // 2, self.cfg['seq_len']):
                seq_in_BBX5_[:,i,:,:] = seq_in_BBX5[:,-1,:,:]
            flBBX5 = seq_in_BBX5_

            # print('\n\n self.cfg[seq_len]: ', self.cfg['seq_len'])
            # print('\n models.py - Xformer_IFcC2C - forward - np.shape(flBBX5): ', np.shape(flBBX5))
            # print('\n models.py - Xformer_IFcC2C - forward - flBBX5: ', flBBX5)
            h_IMUaccgym9 = self.transformer_IMU_TX(seq_in_IMUaccgym9)
            h_FTM2 = self.transformer_FTM2(seq_in_FTM2)
            h_flBBX5 = self.transformer_flBBX5(flBBX5)
            # print('\n models.py - Xformer_IFcC2C - forward - h_flBBX5: ', h_flBBX5)
            h = torch.cat((h_IMUaccgym9, h_FTM2, h_flBBX5), 2)

        h = self.activ(self.linear(h))
        h = self.norm(h)
        out = self.linear_out(h)
        # print('np.shape(out): ', np.shape(out))
        if self.cfg['imu200']:
            out = torch.reshape(out, (np.shape(out)[0], self.cfg['seq_len'], 200, self.cfg['BBX5_dim']))
        else:
            out = torch.reshape(out, (np.shape(out)[0], self.cfg['seq_len'], 1, self.cfg['BBX5_dim']))
        return out

# Ref: https://github.com/bentrevett/pytorch-seq2seq
class LSTMEncoder(nn.Module):
    '''LSTMEncoder Blocks that has similar architecture to Transformer except attention layers'''
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.embed = Embeddings(cfg)
        # Original BERT does not use parameter-sharing strategy
        # self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg['n_layers'])])

        # Use parameter-sharing strategy
        self.n_layers_lstm = cfg['n_layers_lstm']
        self.dropout_rate = cfg['dropout_rate']
        self.bidirectional = cfg['bidirectional']

        # if self.cfg['imu200']:
        #     self.lstm = nn.LSTM(cfg['hidden_imu200'], cfg['hidden_imu200'], self.n_layers_lstm, \
        #         dropout=self.dropout_rate, bidirectional=self.bidirectional)
        # else:
        #     self.lstm = nn.LSTM(cfg['hidden'], cfg['hidden'], self.n_layers_lstm, \
        #         dropout=self.dropout_rate, bidirectional=self.bidirectional)
        if self.cfg['imu200']:
            self.lstm = nn.LSTM(cfg['hidden_lstm_imu200'] * cfg['seq_len'], cfg['hidden_lstm_imu200'] * cfg['seq_len'], self.n_layers_lstm, \
                dropout=self.dropout_rate, bidirectional=self.bidirectional)
        else:
            self.lstm = nn.LSTM(cfg['hidden_lstm'] * cfg['seq_len'], cfg['hidden_lstm'] * cfg['seq_len'], self.n_layers_lstm, \
                dropout=self.dropout_rate, bidirectional=self.bidirectional)
        self.dropout = nn.Dropout(self.dropout_rate)

    def forward(self, x):
        # DEBUG:
        # print('\n\n')
        # print('models.py - LSTMEncoder - forward - before reshape: np.shape(x): ', np.shape(x))
        # e.g. models.py - LSTMEncoder - forward - np.shape(x): torch.Size([1, 32, 10, 1, 9])
        shape_ = np.shape(x)
        # x = x.reshape((shape_[0], shape_[1], shape_[2], shape_[3] * shape_[4]))
        x = x.reshape((shape_[0], shape_[1], shape_[2] * shape_[3]))
        # print('models.py - LSTMEncoder - forward - after reshape: np.shape(x): ', np.shape(x))
        h = self.embed(x)
        # DEBUG:
        # print('\n\n')
        # print('models.py - LSTMEncoder - forward - before unsqueeze: np.shape(h): ', np.shape(h))
        # e.g. models.py - LSTMEncoder - forward - np.shape(h):

        h = h.unsqueeze(0)
        # print('models.py - LSTMEncoder - forward - after unsqueeze: np.shape(h): ', np.shape(h))

        shape_ = np.shape(h)
        h = h.reshape((shape_[0], shape_[1], shape_[2] * shape_[3]))
        # print('models.py - LSTMEncoder - forward - after reshape: np.shape(h): ', np.shape(h))
        # for _ in range(self.n_layers_lstm):
        # o: outputs, h: hidden_lstm states, c: cell
        o, (h, c) = self.lstm(h)
        # print('models.py - LSTMEncoder - forward - before return: np.shape(h): ', np.shape(h))
        return o, (h, c)

class Xlstm_IFcC2C(nn.Module):
    '''LSTMDecoder Blocks that has similar architecture to Transformer except attention layers'''
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.n_layers_lstm = cfg['n_layers_lstm']
        self.dropout_rate = cfg['dropout_rate']
        self.bidirectional = cfg['bidirectional']

        # encoder
        self.cfg['feature'] = 'IMU_TX'; self.lstmencoder_IMU_TX = LSTMEncoder(cfg)
        self.cfg['feature'] = 'FTM2'; self.lstmencoder_FTM2 = LSTMEncoder(cfg)
        self.cfg['feature'] = 'fstBBX5'; self.lstmencoder_fstBBX5 = LSTMEncoder(cfg)
        # TODO: flBBX5

        # decoder
        if self.cfg['imu200']:
            self.lstm = nn.LSTM(cfg['hidden_lstm'] * 3, cfg['hidden_lstm_imu200'], \
                self.n_layers_lstm, dropout=self.dropout_rate)
            self.activ = gelu
            self.linear_out = nn.Linear(cfg['hidden_lstm_imu200'], cfg['BBX5_dim'])
        else:
            self.lstm = nn.LSTM(cfg['hidden_lstm'] * 3, cfg['hidden_lstm'], \
                self.n_layers_lstm, dropout=self.dropout_rate)
            self.activ = gelu
            self.linear_out = nn.Linear(cfg['hidden_lstm'], cfg['BBX5_dim'])
        self.dropout = nn.Dropout(self.dropout_rate)

    def forward(self, input_seqs):
        # decoder
        # print('\n\n ')
        seq_in_BBX5, seq_in_FTM2, seq_in_IMUaccgym9 = input_seqs[0], input_seqs[1], input_seqs[2]
        # seq_in_BBX5, seq_in_FTM2, seq_in_IMUaccgym9 = input_seqs[0].squeeze(2), input_seqs[1].squeeze(2), input_seqs[2].squeeze(2)
        # seq_in_BBX5, seq_in_FTM2, seq_in_IMUaccgym9 = input_seqs[0].unsqueeze(0), input_seqs[1].unsqueeze(0), input_seqs[2].unsqueeze(0)

        # print('\n models.py - Xlstm_IFcC2C - forward - np.shape(seq_in_BBX5): ', np.shape(seq_in_BBX5))
        # print('\n models.py - Xlstm_IFcC2C - forward - np.shape(seq_in_FTM2): ', np.shape(seq_in_FTM2))
        # print('\n models.py - Xlstm_IFcC2C - forward - np.shape(seq_in_IMUaccgym9): ', np.shape(seq_in_IMUaccgym9))

        # fstBBX5 = seq_in_BBX5[:, 0, :, :] # first frame

        seq_in_BBX5_ = copy.deepcopy(seq_in_BBX5) # first frame

        # >>> v1
        for i in range(self.cfg['seq_len']):
            seq_in_BBX5_[:,i,:,:] = seq_in_BBX5[:,0,:,:]
        # <<< v1

        # >>> v2
        # shp = np.shape(seq_in_BBX5)
        # for i in range(1, self.cfg['seq_len']):
        #     seq_in_BBX5_[:,i,:,:] = torch.zeros(shp[0], 1, shp[2], shp[3]) # np.zeros((shp[0], 1, shp[2], shp[3]))
        # <<< v2
        fstBBX5 = seq_in_BBX5_

        o_IMUaccgym9, (h_IMUaccgym9, c_IMUaccgym9) = self.lstmencoder_IMU_TX(seq_in_IMUaccgym9)
        o_FTM2, (h_FTM2, c_FTM2) = self.lstmencoder_FTM2(seq_in_FTM2)
        o_fstBBX5, (h_fstBBX5, c_fstBBX5) = self.lstmencoder_fstBBX5(fstBBX5)

        # Only use the recent layer
        if self.cfg['n_layers_lstm'] > 1:
            h_IMUaccgym9 = h_IMUaccgym9[-1]
            h_FTM2 = h_FTM2[-1]
            h_fstBBX5 = h_fstBBX5[-1]

        # DEBUG:
        # print('\n models.py - Xlstm_IFcC2C - forward - np.shape(h_IMUaccgym9): ', np.shape(h_IMUaccgym9))
        # print('\n models.py - Xlstm_IFcC2C - forward - np.shape(h_FTM2): ', np.shape(h_FTM2))
        # print('\n models.py - Xlstm_IFcC2C - forward - np.shape(h_fstBBX5): ', np.shape(h_fstBBX5))
        '''
        e.g.
        models.py - Xlstm_IFcC2C - forward - np.shape(h_IMUaccgym9):  torch.Size([32, 10, 72])
        models.py - Xlstm_IFcC2C - forward - np.shape(h_FTM2):  torch.Size([32, 10, 72])
        models.py - Xlstm_IFcC2C - forward - np.shape(h_fstBBX5):  torch.Size([32, 10, 72])
        '''
        # print('self.cfg[batch_size]: ', self.cfg['batch_size'])

        shape_ = np.shape(h_IMUaccgym9)
        h_IMUaccgym9 = h_IMUaccgym9.reshape((self.cfg['batch_size'], self.cfg['seq_len'], self.cfg['hidden_lstm']))
        h_FTM2 = h_FTM2.reshape((self.cfg['batch_size'], self.cfg['seq_len'], self.cfg['hidden_lstm']))
        h_fstBBX5 = h_fstBBX5.reshape((self.cfg['batch_size'], self.cfg['seq_len'], self.cfg['hidden_lstm']))

        # print('\n models.py - LSTMDecoder - forward - h_fstBBX5: ', h_fstBBX5)
        h = torch.cat((h_IMUaccgym9, h_FTM2, h_fstBBX5), 2)
        o, (h, c) = self.lstm(h)
        # pred = self.activ()
        # h = self.norm(h)
        # TODO: flBBX5

        # print('\n models.py - Xlstm_IFcC2C - forward - np.shape(o): ', np.shape(o))
        # print('\n models.py - Xlstm_IFcC2C - forward - np.shape(o.squeeze(0)): ', np.shape(o.squeeze(0)))
        out = self.linear_out(o.squeeze(0))
        if self.cfg['batch_size'] == 1: out = torch.unsqueeze(out, 0)
        # print('np.shape(out): ', np.shape(out))
        if self.cfg['imu200']:
            out = torch.reshape(out, (np.shape(out)[0], self.cfg['seq_len'], 200, self.cfg['BBX5_dim']))
        else:
            out = torch.reshape(out, (np.shape(out)[0], self.cfg['seq_len'], 1, self.cfg['BBX5_dim']))
        # print('np.shape(out) before return: ', np.shape(out))
        return out

# --------------------
#  Baseline Vitag >>>
class ZeroPadding(Layer):
     def __init__(self, **kwargs):
          super(ZeroPadding, self).__init__(**kwargs)
     def call(self, x, mask=None):
          return K.zeros_like(x)
     def get_output_shape_for(self, input_shape):
          return input_shape

class Xtranslator_IFcC2C(Model):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg # model_cfg
        self.net = None

        # ---------------
        #  BBX5 Encoder
        # ---------------
        in_BBX5 = Input(shape=(self.cfg['len_win'], self.cfg['BBX5_dim'],))
        conv1_BBX5 = Conv1D(filters=self.cfg['n_filters'], kernel_size=self.cfg['kernel_size'], strides=1,
                           activation='relu', padding='same')(in_BBX5)
        en_BBX5 = Bidirectional(LSTM(self.cfg['h_BBX5_dim'], activation='relu', \
                batch_input_shape=(self.cfg['batch_size'], ), return_sequences=True))(conv1_BBX5) # in_BBX5)

        # --------------------
        #  FTM2 Encoder
        # --------------------
        in_FTM2 = Input(shape=(self.cfg['len_win'], self.cfg['FTM2_dim'],))
        conv1_FTM2 = Conv1D(filters=self.cfg['n_filters'], kernel_size=self.cfg['kernel_size'], strides=1,
                           activation='relu', padding='same')(in_FTM2)
        en_FTM2 = Bidirectional(LSTM(self.cfg['h_FTM2_dim'], activation='relu', \
                batch_input_shape=(self.cfg['batch_size'], ), return_sequences=True))(conv1_FTM2)


        # ----------------
        #  IMU_XT Encoder
        # ----------------
        in_IMU_XT = Input(shape=(self.cfg['len_win'], self.cfg['IMU_XT_dim'],))
        conv1_IMU_XT = Conv1D(filters=self.cfg['n_filters'], kernel_size=self.cfg['kernel_size'], strides=1,
                           activation='relu', padding='same')(in_IMU_XT)
        en_IMU_XT = Bidirectional(LSTM(self.cfg['h_IMU_XT_dim'], activation='relu', \
                batch_input_shape=(self.cfg['batch_size'], ), return_sequences=True))(conv1_IMU_XT) # in_IMU_XT)

        # ----------------------
        #  Joint Representation
        # ----------------------
        in_fused_add = Add()([en_BBX5, en_IMU_XT])
        # in_fused_concat = Concatenate()([en_BBX5, en_IMU_XT])
        # in_fused = RepeatVector(self.cfg['len_win'])(in_fused_concat)

        # --------------
        #  BBX5 Decoder
        # --------------
        # de_BBX5 = RepeatVector(self.cfg['len_win'])(in_fused_add)
        de_BBX5 = Conv1D(filters=self.cfg['n_filters'], kernel_size=self.cfg['kernel_size'], strides=1,
                           activation='relu', padding='same')(in_fused_add)
        de_BBX5 = Bidirectional(LSTM(self.cfg['h_fused_dim'], activation='relu', \
                batch_input_shape=(self.cfg['batch_size'], ), return_sequences=True))(de_BBX5) # in_fused_add) # in_fused)
        de_BBX5 = Bidirectional(LSTM(self.cfg['BBX5_dim'], activation='relu', \
                batch_input_shape=(self.cfg['batch_size'], ), return_sequences=True))(de_BBX5) # in_fused_add) # de_BBX5)
        de_BBX5 = TimeDistributed(Dense(self.cfg['BBX5_dim']))(de_BBX5)

        # --------------------
        #  FTM2 Decoder
        # --------------------
        # de_FTM2 = RepeatVector(self.cfg['len_win'])(in_fused_add)
        de_FTM2 = Conv1D(filters=self.cfg['n_filters'], kernel_size=self.cfg['kernel_size'], strides=1,
                           activation='relu', padding='same')(in_fused_add)
        de_FTM2 = Bidirectional(LSTM(self.cfg['h_fused_dim'], activation='relu', \
                batch_input_shape=(self.cfg['batch_size'], ), return_sequences=True))(de_FTM2)
        de_FTM2 = Bidirectional(LSTM(self.cfg['FTM2_dim'], activation='relu', \
                batch_input_shape=(self.cfg['batch_size'], ), return_sequences=True))(de_FTM2)
        de_FTM2 = TimeDistributed(Dense(self.cfg['FTM2_dim']))(de_FTM2)

        # ----------------
        #  IMU_XT Decoder
        # ----------------
        # de_IMU_XT = RepeatVector(self.cfg['len_win'])(in_fused_add)
        de_IMU_XT = Conv1D(filters=self.cfg['n_filters'], kernel_size=self.cfg['kernel_size'], strides=1,
                           activation='relu', padding='same')(in_fused_add)
        de_IMU_XT = Bidirectional(LSTM(self.cfg['h_fused_dim'], activation='relu', \
                batch_input_shape=(self.cfg['batch_size'], ), return_sequences=True))(de_IMU_XT) # in_fused_add) # in_fused)
        de_IMU_XT = Bidirectional(LSTM(self.cfg['IMU_XT_dim'], activation='relu', \
                batch_input_shape=(self.cfg['batch_size'], ), return_sequences=True))(de_IMU_XT) # de_IMU_XT)
        de_IMU_XT = TimeDistributed(Dense(self.cfg['IMU_XT_dim']))(de_IMU_XT)

        # ------------
        #  Base Model
        # ------------
        BaseNet = Model([in_BBX5, in_FTM2, in_IMU_XT], [de_BBX5, de_FTM2, de_IMU_XT])
        print('BaseNet.summary(): ', BaseNet.summary())

        # Env0: Encoder for BBX5, Dec0: Decoder for BBX5
        # Env1: Encoder for FTM2, Dec1: Decoder for FTM2
        # Env2: Encoder for IMU_XT, Dec2: Decoder for IMU_XT
        # -------------------------------------------
        #  Self-Reconstruction
        #      BBX5 = Dec0(Env0(BBX5))
        #      FTM2 = Dec1(Env1(FTM2))
        #      IMU_XT = Dec2(Env2(IMU_XT))
        # -------------------------------------------
        [sl_rec_BBX5_0, _, _] = BaseNet([in_BBX5, ZeroPadding()(in_FTM2), ZeroPadding()(in_IMU_XT)])
        [_, sl_rec_FTM2_1, _] = BaseNet([ZeroPadding()(in_BBX5), in_FTM2, ZeroPadding()(in_IMU_XT)])
        [_, _, sl_rec_IMU_XT_2] = BaseNet([ZeroPadding()(in_BBX5), ZeroPadding()(in_FTM2), in_IMU_XT])

        # -------------------------------------------
        #  Cross-Reconstruction
        #      FTM2 = Dec1(Env0(BBX5))
        #      IMU_XT = Dec2(Env0(BBX5))
        #      BBX5 = Dec0(Env1(FTM2))
        #      IMU_XT = Dec2(Env1(FTM2))
        #      BBX5 = Dec0(Env2(IMU_XT))
        #      FTM2 = Dec1(Env2(IMU_XT))
        # -------------------------------------------
        [_, cr_rec_FTM2_3, _] = BaseNet([in_BBX5, ZeroPadding()(in_FTM2), ZeroPadding()(in_IMU_XT)])
        [_, _, cr_rec_IMU_XT_4] = BaseNet([in_BBX5, ZeroPadding()(in_FTM2), ZeroPadding()(in_IMU_XT)])
        [cr_rec_BBX5_5, _, _] = BaseNet([ZeroPadding()(in_BBX5), in_FTM2, ZeroPadding()(in_IMU_XT)])
        [_, _, cr_rec_IMU_XT_6] = BaseNet([ZeroPadding()(in_BBX5), in_FTM2, ZeroPadding()(in_IMU_XT)])
        [cr_rec_BBX5_7, _, _] = BaseNet([ZeroPadding()(in_BBX5), ZeroPadding()(in_FTM2), in_IMU_XT])
        [_, cr_rec_FTM2_8, _] = BaseNet([ZeroPadding()(in_BBX5), ZeroPadding()(in_FTM2), in_IMU_XT])

        # ------------------------------------------------------------------
        #  Fused-Reconstruction
        #      BBX5 = Dec0(Env0(BBX5), Env1(FTM2), Env2(IMU_XT))
        #      FTM2 = Dec1(Env0(BBX5), Env1(FTM2), Env2(IMU_XT))
        #      IMU_XT = Dec2(Env0(BBX5), Env1(FTM2), Env2(IMU_XT))
        # ------------------------------------------------------------------
        [fs_rec_BBX5_9, _, _] = BaseNet([in_BBX5, in_FTM2, in_IMU_XT])
        [_, fs_rec_FTM2_10, _] = BaseNet([in_BBX5, in_FTM2, in_IMU_XT])
        [_, _, fs_rec_IMU_XT_11] = BaseNet([in_BBX5, in_FTM2, in_IMU_XT])

        # ------------------------------------------------------------------------------------
        #  One-to-All-Reconstruction
        #      BBX5 = Dec0(Env0(BBX5)), FTM2 = Dec1(Env0(BBX5)), IMU_XT = Dec2(Env0(BBX5))
        #      BBX5 = Dec0(Env1(FTM2)), FTM2 = Dec1(Env1(FTM2)), IMU_XT = Dec2(Env1(FTM2))
        #      BBX5 = Dec0(Env2(IMU_XT)), FTM2 = Dec1(Env2(IMU_XT)), IMU_XT = Dec2(Env2(IMU_XT))
        # ------------------------------------------------------------------------------------
        [ota_rec_BBX5_12, ota_rec_FTM2_13, ota_rec_IMU_XT_14] = BaseNet([in_BBX5, ZeroPadding()(in_FTM2), ZeroPadding()(in_IMU_XT)])
        [ota_rec_BBX5_15, ota_rec_FTM2_16, ota_rec_IMU_XT_17] = BaseNet([ZeroPadding()(in_BBX5), in_FTM2, ZeroPadding()(in_IMU_XT)])
        [ota_rec_BBX5_18, ota_rec_FTM2_19, ota_rec_IMU_XT_20] = BaseNet([ZeroPadding()(in_BBX5), ZeroPadding()(in_FTM2), in_IMU_XT])

        # ---------------------------------------------------------
        #  Cross-Domain-Reconstruction
        #      FTM2 = Dec1(Env0(BBX5)), IMU_XT = Dec2(Env0(BBX5))
        #      BBX5 = Dec0(Env1(FTM2)), BBX5 = Dec0(Env2(IMU_XT))
        # ---------------------------------------------------------
        [_, crd_rec_FTM2_21, crd_rec_IMU_XT_22] = BaseNet([in_BBX5, ZeroPadding()(in_FTM2), ZeroPadding()(in_IMU_XT)])
        [crd_rec_BBX5_23, _, _] = BaseNet([ZeroPadding()(in_BBX5), in_FTM2, in_IMU_XT])

        # -------------------------------------------
        #  Multi-Reconstruction
        #      BBX5 = Dec0(Env0(BBX5))
        #      FTM2 = Dec1(Env0(BBX5))
        #      IMU_XT = Dec2(Env0(BBX5))
        #      BBX5 = Dec0(Env1(FTM2))
        #      FTM2 = Dec1(Env1(FTM2))
        #      IMU_XT = Dec2(Env1(FTM2))
        #      BBX5 = Dec0(Env2(IMU_XT))
        #      FTM2 = Dec1(Env2(IMU_XT))
        #      IMU_XT = Dec2(Env2(IMU_XT))
        # -------------------------------------------
        [mu_rec_BBX5_24, _, _] = BaseNet([in_BBX5, ZeroPadding()(in_FTM2), ZeroPadding()(in_IMU_XT)])
        [_, mu_rec_FTM2_25, _] = BaseNet([in_BBX5, ZeroPadding()(in_FTM2), ZeroPadding()(in_IMU_XT)])
        [_, _, mu_rec_IMU_XT_26] = BaseNet([in_BBX5, ZeroPadding()(in_FTM2), ZeroPadding()(in_IMU_XT)])
        [mu_rec_BBX5_27, _, _] = BaseNet([ZeroPadding()(in_BBX5), in_FTM2, ZeroPadding()(in_IMU_XT)])
        [_, mu_rec_FTM2_28, _] = BaseNet([ZeroPadding()(in_BBX5), in_FTM2, ZeroPadding()(in_IMU_XT)])
        [_, _, mu_rec_IMU_XT_29] = BaseNet([ZeroPadding()(in_BBX5), in_FTM2, ZeroPadding()(in_IMU_XT)])
        [mu_rec_BBX5_30, _, _] = BaseNet([ZeroPadding()(in_BBX5), ZeroPadding()(in_FTM2), in_IMU_XT])
        [_, mu_rec_FTM2_31, _] = BaseNet([ZeroPadding()(in_BBX5), ZeroPadding()(in_FTM2), in_IMU_XT])
        [_, _, mu_rec_IMU_XT_32] = BaseNet([ZeroPadding()(in_BBX5), ZeroPadding()(in_FTM2), in_IMU_XT])

        # ------------------------------------------------------------------
        #  Full-Reconstruction
        #      BBX5 = Dec0(Env0(BBX5), Env1(FTM2), Env2(IMU_XT))
        #      FTM2 = Dec0(Env0(BBX5), Env1(FTM2), Env2(IMU_XT))
        #      IMU_XT = Dec1(Env0(BBX5), Env1(FTM2), Env2(IMU_XT))
        # ------------------------------------------------------------------
        [fl_rec_BBX5_27, fl_rec_FTM2_27, fl_rec_IMU_XT_27] = BaseNet([in_BBX5, in_FTM2, in_IMU_XT])

        def Bhatt_loss(y_true, y_pred):
            small_num = 0.000001
            # print('np.shape(y_true): ', np.shape(y_true)) # debug # e.g. (None, 10, 2)
            y_true, y_pred = tf.cast(y_true, dtype='float64'), tf.cast(y_pred, dtype='float64')
            mu_true, sig_true = tf.cast(y_true[:,:,0], dtype='float64'), tf.cast(y_true[:,:,1], dtype='float64')
            mu_pred, sig_pred = tf.cast(y_pred[:,:,0], dtype='float64'), tf.cast(y_pred[:,:,1], dtype='float64')
            # print('np.shape(mu_true): ', np.shape(mu_true), ', np.shape(sig_true): ', np.shape(sig_true))
            # print('mu_true: ', mu_true, ', sig_true: ', sig_true)
            term0 = tf.math.truediv(tf.math.log(tf.math.add(tf.math.add(tf.math.truediv(\
                            tf.math.truediv(tf.math.pow(sig_true, 2), tf.math.pow(sig_pred, 2) + small_num), 4.), \
                            tf.math.truediv(tf.math.pow(sig_pred, 2), tf.math.pow(sig_true, 2) + small_num)), 2.) + small_num), 4.)
            term1 = tf.math.truediv(tf.math.truediv(tf.math.pow((mu_true - mu_pred), 2), tf.math.add(tf.math.pow(sig_true, 2), tf.math.pow(sig_pred, 2) + small_num)), 4.)
            return tf.reduce_mean((term0 + term1)) #, axis=-1)  # Note the `axis=-1`

        # if C.args.loss == 'mse':
        if self.cfg['loss_func'] == 'MSE_xydwh':
            self.net = Model([in_BBX5, in_FTM2, in_IMU_XT], \
                [sl_rec_BBX5_0, sl_rec_FTM2_1, sl_rec_IMU_XT_2, \
                cr_rec_FTM2_3, cr_rec_IMU_XT_4, cr_rec_BBX5_5, \
                cr_rec_IMU_XT_6, cr_rec_BBX5_7, cr_rec_FTM2_8, \
                fs_rec_BBX5_9, fs_rec_FTM2_10, fs_rec_IMU_XT_11, \
                ota_rec_BBX5_12, ota_rec_FTM2_13, ota_rec_IMU_XT_14, \
                ota_rec_BBX5_15, ota_rec_FTM2_16, ota_rec_IMU_XT_17, \
                ota_rec_BBX5_18, ota_rec_FTM2_19, ota_rec_IMU_XT_20, \
                crd_rec_FTM2_21, crd_rec_IMU_XT_22, crd_rec_BBX5_23, \
                mu_rec_BBX5_24, mu_rec_FTM2_25, mu_rec_IMU_XT_26, \
                mu_rec_BBX5_27, mu_rec_FTM2_28, mu_rec_IMU_XT_29, \
                mu_rec_BBX5_30, mu_rec_FTM2_31, mu_rec_IMU_XT_32])

            self.net.compile(loss=['mse', 'mse', 'mse',  'mse', 'mse', 'mse',  'mse', 'mse', 'mse', \
                                    'mse', 'mse', 'mse',  'mse', 'mse', 'mse',  'mse', 'mse', 'mse', \
                                    'mse', 'mse', 'mse',  'mse', 'mse', 'mse',  'mse', 'mse', 'mse', \
                                    'mse', 'mse', 'mse',  'mse', 'mse', 'mse', ],
                loss_weights=[1, 1, 1,  1, 1, 1,  1, 1, 1,   1, 1, 1,  1, 1, 1,  1, 1, 1,   1, 1, 1,  1, 1, 1,  1, 1, 1, \
                                1, 1, 1,  1, 1, 1,  1, 1, 1,   1, 1, 1,  1, 1, 1,  1, 1, 1,   1, 1, 1,  1, 1, 1,  1, 1, 1,
                                1, 1, 1,  1, 1, 1,  1, 1, 1,   1, 1, 1,  1, 1, 1,  1, 1, 1,   1, 1, 1,  1, 1, 1,  1, 1, 1,
                                1, 1, 1,  1, 1, 1,  1, 1, 1,   1, 1, 1,  1, 1, 1,  1, 1, 1], optimizer=self.cfg['optimizer']) # 'adam')
        elif self.cfg['loss_func'] == 'b':
            self.net = Model([in_BBX5, in_FTM2, in_IMU_XT], \
                [sl_rec_BBX5_0, sl_rec_FTM2_1, sl_rec_IMU_XT_2, \
                cr_rec_FTM2_3, cr_rec_IMU_XT_4, cr_rec_BBX5_5, \
                cr_rec_IMU_XT_6, cr_rec_BBX5_7, cr_rec_FTM2_8, \
                fs_rec_BBX5_9, fs_rec_FTM2_10, fs_rec_IMU_XT_11, \
                ota_rec_BBX5_12, ota_rec_FTM2_13, ota_rec_IMU_XT_14, \
                ota_rec_BBX5_15, ota_rec_FTM2_16, ota_rec_IMU_XT_17, \
                ota_rec_BBX5_18, ota_rec_FTM2_19, ota_rec_IMU_XT_20, \
                crd_rec_FTM2_21, crd_rec_IMU_XT_22, crd_rec_BBX5_23, \
                mu_rec_BBX5_24, mu_rec_FTM2_25, mu_rec_IMU_XT_26, \
                mu_rec_BBX5_27, mu_rec_FTM2_28, mu_rec_IMU_XT_29, \
                mu_rec_BBX5_30, mu_rec_FTM2_31, mu_rec_IMU_XT_32, \
                sl_rec_FTM2_1, cr_rec_FTM2_3, cr_rec_FTM2_8, \
                fs_rec_FTM2_10, ota_rec_FTM2_13, ota_rec_FTM2_16, \
                ota_rec_FTM2_19, crd_rec_FTM2_21, mu_rec_FTM2_25, \
                mu_rec_FTM2_28, mu_rec_FTM2_31, fl_rec_FTM2_27])

            self.net.compile(loss=['mse', 'mse', 'mse',  'mse', 'mse', 'mse',  'mse', 'mse', 'mse', \
                                    'mse', 'mse', 'mse',  'mse', 'mse', 'mse',  'mse', 'mse', 'mse', \
                                    'mse', 'mse', 'mse',  'mse', 'mse', 'mse',  'mse', 'mse', 'mse', \
                                    'mse', 'mse', 'mse',  'mse', 'mse', 'mse', \
                                    Bhatt_loss, Bhatt_loss, Bhatt_loss,  Bhatt_loss, Bhatt_loss, Bhatt_loss, \
                                    Bhatt_loss, Bhatt_loss, Bhatt_loss,  Bhatt_loss, Bhatt_loss, Bhatt_loss],
                loss_weights=[1, 1, 1,  1, 1, 1,  1, 1, 1,  1, 1, 1,  1, 1, 1, \
                              1, 1, 1,  1, 1, 1,  1, 1, 1,  1, 1, 1,  1, 1, 1,  1, 1, 1, \
                              1, 1, 1,  1, 1, 1,  1, 1, 1,  1, 1, 1], optimizer=self.cfg['optimizer']) # 'adam')
#  Baseline Vitag <<<
# --------------------

class Width_regressor(nn.Module):
    '''
    Baseline model to regress bounding box size by image coordinates.
    Used with Pedestrian Deak Reckoning.
    '''
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        out = self.linear(x)
        return out

class Height_regressor(nn.Module):
    '''
    Baseline model to regress bounding box size by image coordinates.
    Used with Pedestrian Deak Reckoning.
    '''
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        out = self.linear(x)
        return out

class Depth_regressor(nn.Module):
    '''
    Baseline model to regress depth by image coordinates.
    Used with Pedestrian Deak Reckoning.
    '''
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.linear_1 = nn.Linear(1, 2)
        self.linear_2 = nn.Linear(2, 1)

    def forward(self, x):
        x = self.linear_1(x)
        out = self.linear_2(x)
        return out

class MyKalmanFilter():
    def __init__(self, cfg):
        # print('\n MyKalmanFilter cfg: ', cfg)
        np.random.seed(cfg['seed'])
        self.kf_ls = []
        for i in range(cfg['n_subjects']):
            self.kf_ls.append(KalmanFilter(dim_x=cfg['kf_x_dim'], dim_z=cfg['kf_z_dim']))
        if cfg['kalman_filter_modality'] == 'C':
            for i in range(cfg['n_subjects']):
                # print('\n\n np.shape(self.kf_ls[i].F): ', np.shape(self.kf_ls[i].F))
                # self.kf_ls[i].F = np.array([[1., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
                #                     [0., 1., 0., 0., 0., 0., 1., 0., 0., 0.],
                #                     [0., 0., 1., 0., 0., 0., 0., 1., 0., 0.],
                #                     [0., 0., 0., 1., 0., 0., 0., 0., 1., 0.],
                #                     [0., 0., 0., 0., 1., 0., 0., 0., 0., 1.]])

                # https://filterpy.readthedocs.io/en/latest/kalman/KalmanFilter.html
                # F : ndarray (dim_x, dim_x)
                # [x, y, z, w, h, vx, vy, vz, vw, vh]
                self.kf_ls[i].F = \
                    np.array([[1., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
                              [0., 1., 0., 0., 0., 0., 1., 0., 0., 0.],
                              [0., 0., 1., 0., 0., 0., 0., 1., 0., 0.],
                              [0., 0., 0., 1., 0., 0., 0., 0., 1., 0.],
                              [0., 0., 0., 0., 1., 0., 0., 0., 0., 1.],
                              [0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
                              [0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
                              [0., 0., 0., 0., 0., 0., 0., 1., 0., 0.],
                              [0., 0., 0., 0., 0., 0., 0., 0., 1., 0.],
                              [0., 0., 0., 0., 0., 0., 0., 0., 0., 1.]])

                # H : ndarray (dim_z, dim_x)
                # measurements: 5d BBX, 5d velocity will be appended in func_forward() in kalman_filter.py
                self.kf_ls[i].H = \
                    np.array([[1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                              [0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
                              [0., 0., 1., 0., 0., 0., 0., 0., 0., 0.],
                              [0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
                              [0., 0., 0., 0., 1., 0., 0., 0., 0., 0.]])

                # noise
                if cfg['noise']:
                    std = 1 # TODO set it by config
                    # P : ndarray (dim_x, dim_x), default eye(dim_x)
                    # covariance matrix
                    self.kf_ls[i].P = np.diag(np.random.normal(loc=0, scale=cfg['std'], size=cfg['kf_x_dim']))
                    # Q : ndarray (dim_x, dim_x), default eye(dim_x)
                    # Process uncertainty/noise
                    self.kf_ls[i].Q = np.diag(np.random.normal(loc=0, scale=cfg['std'], size=cfg['kf_x_dim']))
                    # R : ndarray (dim_z, dim_z), default eye(dim_x)
                    # measurement uncertainty/noise
                    self.kf_ls[i].R = np.diag(np.random.normal(loc=0, scale=cfg['std'], size=cfg['kf_z_dim']))
                    # H : ndarray (dim_z, dim_x)
                    # measurement function
                    H_shape = (cfg['kf_z_dim'], cfg['kf_x_dim'])
                    H = np.zeros(H_shape)
                    noise_values = np.random.normal(loc=0, scale=cfg['std'], size=min(H_shape))
                    np.fill_diagonal(H, noise_values)
                    self.kf_ls[i].H = H

                print('\n self.kf_ls[i].P: ', self.kf_ls[i].P)
                print('\n self.kf_ls[i].Q: ', self.kf_ls[i].Q)
                print('\n self.kf_ls[i].R: ', self.kf_ls[i].R)
                print('\n self.kf_ls[i].H: ', self.kf_ls[i].H)
                
            # print('\n\n self.kf.F: ', self.kf.F)
            # print('\n\n self.kf.P: ', self.kf.P)
            # print('\n\n self.kf.Q: ', self.kf.Q)

class MyKalmanFilter_v2():
    def __init__(self, cfg):
        self.kf_ls = []
        for i in range(cfg['n_subjects']):
            self.kf_ls.append(KalmanFilter(dim_x=cfg['kf_x_dim'], dim_z=cfg['kf_z_dim']))
        if cfg['kalman_filter_modality'] == 'C':
            for i in range(cfg['n_subjects']):
                # print('\n\n np.shape(self.kf_ls[i].F): ', np.shape(self.kf_ls[i].F))

                # https://filterpy.readthedocs.io/en/latest/kalman/KalmanFilter.html
                # F : ndarray (dim_x, dim_x)
                # [x, y, z, w, h, vx, vy, vz, vw, vh]
                self.kf_ls[i].F = \
                    np.array([[1., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
                              [0., 1., 0., 0., 0., 0., 1., 0., 0., 0.],
                              [0., 0., 1., 0., 0., 0., 0., 1., 0., 0.],
                              [0., 0., 0., 1., 0., 0., 0., 0., 1., 0.],
                              [0., 0., 0., 0., 1., 0., 0., 0., 0., 1.],
                              [0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
                              [0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
                              [0., 0., 0., 0., 0., 0., 0., 1., 0., 0.],
                              [0., 0., 0., 0., 0., 0., 0., 0., 1., 0.],
                              [0., 0., 0., 0., 0., 0., 0., 0., 0., 1.]])

                # H : ndarray (dim_z, dim_x)
                self.kf_ls[i].H = \
                    np.array([[1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                              [0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
                              [0., 0., 1., 0., 0., 0., 0., 0., 0., 0.],
                              [0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
                              [0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
                              [0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
                              [0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
                              [0., 0., 0., 0., 0., 0., 0., 1., 0., 0.],
                              [0., 0., 0., 0., 0., 0., 0., 0., 1., 0.],
                              [0., 0., 0., 0., 0., 0., 0., 0., 0., 1.]])

                # TODO: add noise
            # print('\n\n self.kf.F: ', self.kf.F)
            # print('\n\n self.kf.P: ', self.kf.P)
            # print('\n\n self.kf.Q: ', self.kf.Q)


# New Code
# <<<

'''
References:
https://github.com/dapowan/LIMU-BERT-Public/blob/master/models.py
https://github.com/dhlee347/pytorchic-bert/blob/master/models.py
'''
