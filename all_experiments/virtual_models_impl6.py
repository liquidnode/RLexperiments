import torch
import numpy as np
from memory import HashingMemory
from torch.nn import functional as F
from collections import OrderedDict
from radam import RAdam
from models import CUDA, variable, LearnerModel, Lambda, swish, Flatten
from torch.multiprocessing import Process, Queue
from shared_memory2 import SharedMemory
import datetime
import random
from random import shuffle
import traceback
import importlib
import threading
import gym
import time
import argparse
import copy
import prio_utils
import matplotlib.pyplot as plt
from spectral_norm import SpectralNorm
from look_ahead import Lookahead
import matplotlib.pyplot as plt
import math

RECONSTRUCTION = True
USE_LOOKAHEAD = False
INFODIM = False
AUX_CAMERA_LOSS = True

#from https://github.com/napsternxg/pytorch-practice/blob/master/Pytorch%20-%20MMD%20VAE.ipynb
def compute_kernel(x, y):
    x_size = x.size(0)
    y_size = y.size(0)
    dim = x.size(1)
    x = x.unsqueeze(1) # (x_size, 1, dim)
    y = y.unsqueeze(0) # (1, y_size, dim)
    tiled_x = x.expand(x_size, y_size, dim)
    tiled_y = y.expand(x_size, y_size, dim)
    kernel_input = (tiled_x - tiled_y).pow(2).mean(2)/float(dim)
    return torch.exp(-kernel_input) # (x_size, y_size)

def compute_mmd(x, y):
    x_kernel = compute_kernel(x, x)
    y_kernel = compute_kernel(y, y)
    xy_kernel = compute_kernel(x, y)
    mmd = x_kernel.mean() + y_kernel.mean() - 2*xy_kernel.mean()
    return mmd

class SkipHashingMemory(torch.nn.Module):
    def __init__(self, input_dim, args, compressed_dim=32):
        super(SkipHashingMemory, self).__init__()
        self.args = args
        self.input_dim = input_dim
        self.memory = HashingMemory.build(input_dim, compressed_dim, self.args)
        self.reproj = torch.nn.Linear(compressed_dim, input_dim)

    def forward(self, x):
        if len(x.shape) == 3:
            time_tmp = x.shape[1]
            x = x.view(-1, self.input_dim)
        else:
            time_tmp = None
        y = self.memory(x)
        y = self.reproj(y)
        y = x + y
        if time_tmp is None:
            return y
        else:
            return y.view(-1, time_tmp, self.input_dim)

def one_hot(ac, num, device):
    one_hot = torch.zeros(list(ac.shape)+[num], dtype=torch.float32, device=device)
    one_hot.scatter_(-1, ac.unsqueeze(-1), 1.0)
    return one_hot


class VisualNet(torch.nn.Module):
    def __init__(self, state_shape, args, cont_dim=None, disc_dims=None, variational=False):
        super(VisualNet, self).__init__()
        self.args = args
        self.state_shape = state_shape
        self.variational = variational

        self.cont_dim = cont_dim
        self.disc_dims = disc_dims
        encnn_layers = []
        self.embed_dim = self.args.embed_dim
        self.temperature = self.args.temperature
        if self.cont_dim is not None:
            assert self.embed_dim == self.cont_dim + sum(self.disc_dims)
        assert len(self.state_shape['pov']) > 1
        if self.args.cnn_type == 'atari':
            sizes = [8, 4, 3]
            strides = [4, 2, 1]
            batchnorm = [False, False, False]
            spectnorm = [False, False, False]
            pooling = [1, 1, 1]
            filters = [32, 64, 32]
            padding = [0, 0, 0]
            end_size = 4
        elif self.args.cnn_type == 'mnist':
            sizes = [3, 3, 3]
            strides = [1, 1, 1]
            batchnorm = [False, False, False]
            spectnorm = [False, False, False]
            pooling = [2, 2, 2]
            filters = [32, 32, 32]
            padding = [0, 0, 0]
            end_size = 4
        elif self.args.cnn_type == 'adv':
            sizes = [4, 3, 3, 4]
            strides = [2, 2, 2, 1]
            batchnorm = [True, True, True, True]
            spectnorm = [True, True, True, True]
            pooling = [1, 1, 1, 1]
            filters = [16, 32, 64, 32]
            padding = [0, 0, 0, 0]
            end_size = 4

        in_channels = self.state_shape['pov'][-1]

        #make encoder
        for i in range(len(sizes)):
            tencnn_layers = []
            tencnn_layers.append(torch.nn.Conv2d(
                in_channels,
                filters[i],
                sizes[i],
                stride=strides[i],
                padding=padding[i],
                bias=True
            ))
            if spectnorm[i]:
                tencnn_layers[-1] = SpectralNorm(tencnn_layers[-1], power_iterations=(2 if USE_LOOKAHEAD else 1))

            if batchnorm[i]:
                tencnn_layers.append(torch.nn.BatchNorm2d(filters[i]))
            tencnn_layers.append(Lambda(lambda x: swish(x)))

            if pooling[i] > 1:
                tencnn_layers.append(torch.nn.MaxPool2d(pooling[i]))

            in_channels = filters[i]
            encnn_layers.append(torch.nn.Sequential(*tencnn_layers))

        tencnn_layers = []
        tencnn_layers.append(Flatten())
        if self.variational:
            tencnn_layers.append(torch.nn.Linear(in_channels*(end_size**2), self.cont_dim*2+sum(self.disc_dims)))
        else:
            tencnn_layers.append(torch.nn.Linear(in_channels*(end_size**2), self.embed_dim))
            #tencnn_layers.append(torch.nn.Tanh())
        encnn_layers.append(torch.nn.Sequential(*tencnn_layers))
        self.encnn_layers = torch.nn.ModuleList(encnn_layers)
        self.log_softmax = torch.nn.LogSoftmax(-1)

    def forward(self, state_pov):
        time_tmp = None
        if len(state_pov.shape) == 5:
            time_tmp = state_pov.shape[1]
            x = state_pov.contiguous().view(-1, self.state_shape['pov'][2], self.state_shape['pov'][0], self.state_shape['pov'][1])
        else:
            x = state_pov
        for _layer in self.encnn_layers:
            x = _layer(x)
        if self.variational:
            if time_tmp is not None:
                x = x.view(-1, time_tmp, self.cont_dim*2+sum(self.disc_dims))
            cont, discs = torch.split(x, self.cont_dim*2, dim=-1)
            mu, log_var = torch.split(cont, self.cont_dim, dim=-1)
            sdiscs = list(torch.split(discs, self.disc_dims, dim=-1))
            for i in range(len(sdiscs)):
                sdiscs[i] = self.log_softmax(sdiscs[i])
            discs = torch.cat(sdiscs, dim=-1)
            return mu, log_var, discs
        else:
            if time_tmp is not None:
                x = x.view(-1, time_tmp, self.embed_dim)
            return x

    def sample(self, mu, log_var, discs, training=True, hard=False):
        #sample continuous
        if training:
            std = torch.exp(0.5 * log_var)
            eps = torch.zeros(std.size(), device=mu.device).normal_()
            cont_sample = mu + std * eps
        else:
            cont_sample = mu

        #sample discrete
        sdiscs = list(torch.split(discs, self.disc_dims, dim=-1))
        disc_samples = [F.gumbel_softmax(d, self.temperature, (not training) or hard or ST_ENABLE) for d in sdiscs]


        return torch.cat([cont_sample]+disc_samples, dim=-1)

class GenNet(torch.nn.Module):
    def __init__(self, args, state_shape, embed_dim):
        super(GenNet, self).__init__()
        self.args = args
        self.state_shape = state_shape
        layers = []
        self.embed_dim = self.args.embed_dim
        assert len(self.state_shape['pov']) > 1
        if False:
            if self.args.cnn_type == 'atari':
                sizes = [8, 4, 3]
                strides = [4, 2, 1]
                batchnorm = [True, True, True]
                spectnorm = [False, False, False]
                pooling = [1, 1, 1]
                filters = [32, 64, 32]
                padding = [0, 0, 0]
                end_size = 4
            elif self.args.cnn_type == 'mnist':
                sizes = [3, 3, 3]
                strides = [1, 1, 1]
                batchnorm = [False, False, False]
                spectnorm = [False, False, False]
                pooling = [2, 2, 2]
                filters = [32, 32, 32]
                padding = [0, 0, 0]
                end_size = 4
            elif self.args.cnn_type == 'adv':
                sizes = [4, 3, 3, 4]
                strides = [2, 2, 2, 1]
                batchnorm = [True, True, True, True]
                spectnorm = [True, True, True, True]
                pooling = [1, 1, 1, 1]
                filters = [16, 32, 64, 32]
                padding = [0, 0, 0, 0]
                end_size = 4
        else:
            sizes = [4, 3, 3, 4]
            strides = [2, 2, 2, 1]
            batchnorm = [False, False, True, True]
            spectnorm = [False, False, False, False]
            pooling = [1, 1, 1, 1]
            filters = [32, 32, 64, 32]
            padding = [0, 0, 0, 0]
            end_size = 4
            
        t_layers = []
        t_layers.append(torch.nn.Linear(self.embed_dim, (end_size**2)*filters[-1]))
        t_layers.append(Lambda(lambda x: x.view(-1, filters[-1], end_size, end_size)))
        layers.append(torch.nn.Sequential(*t_layers))

        in_channels = filters[-1]

        for i in range(len(sizes)-1, -1, -1):
            t_layers = []
            if batchnorm[i]:
                t_layers.append(torch.nn.BatchNorm2d(in_channels))

            t_layers.append(Lambda(lambda x: swish(x)))

            t_layers.append(torch.nn.ConvTranspose2d(
                in_channels,
                filters[i],
                sizes[i],
                stride=strides[i],
                padding=padding[i],
                bias=True
            ))
            
            if spectnorm[i]:
                t_layers[-1] = SpectralNorm(t_layers[-1], power_iterations=(2 if USE_LOOKAHEAD else 1))

            in_channels = filters[i]
            layers.append(torch.nn.Sequential(*t_layers))
            
        t_layers = []
        if batchnorm[-1]:
            t_layers.append(torch.nn.BatchNorm2d(in_channels))

        t_layers.append(Lambda(lambda x: swish(x*0.1)))

        t_layers.append(torch.nn.ConvTranspose2d(
            in_channels,
            self.state_shape['pov'][-1],
            1,
            stride=1,
            padding=0,
            bias=True
        ))

        t_layers.append(torch.nn.Tanh())
        layers.append(torch.nn.Sequential(*t_layers))
        self.layers = torch.nn.ModuleList(layers)
        
    def forward(self, state):
        time_tmp = None
        if len(state.shape) == 3:
            time_tmp = state.shape[1]
            x = state.contiguous().view(-1, state.shape[2])
        else:
            x = state
        for _layer in self.layers:
            x = _layer(x)
        if time_tmp is not None:
            x = x.view(-1, time_tmp, x.shape[1], x.shape[2], x.shape[3])
        return x



class MemoryModel(torch.nn.Module):
    def __init__(self, args, embed_dim, past_time_deltas):
        super(MemoryModel, self).__init__()
        self.args = args

        #generate inner embeds
        max_len = len(past_time_deltas)
        self.conv1 = torch.nn.Sequential(Lambda(lambda x: x.transpose(-2, -1)),
                                         torch.nn.Conv1d(embed_dim, embed_dim, 1, 1),
                                         Lambda(lambda x: swish(x)),
                                         Lambda(lambda x: x.transpose(-2, -1)))
        self.conv_len = ((max_len - 1) // 1) + 1
        #add positional encoding
        self.pos_enc_dim = 16
        pe = torch.zeros((self.conv_len, self.pos_enc_dim))
        for j, pos in enumerate(past_time_deltas):
            for i in range(0, self.pos_enc_dim, 2):
                pe[j, i] = \
                math.sin(pos / (10000 ** (i/self.pos_enc_dim)))
                pe[j, i + 1] = \
                math.cos(pos / (10000 ** ((i + 1)/self.pos_enc_dim)))
                
        self.concat_or_add = False
        if self.concat_or_add:
            pe = pe.unsqueeze(0).expand(self.args.er, -1, -1).transpose(0, 1)
        else:
            pe = pe.unsqueeze(0).transpose(0, 1) / math.sqrt(float(embed_dim))
        self.register_buffer('pe', pe)

        #generate query using current pov_embed
        #TODO add also state of inventory?
        self.num_heads = 2
        self.query_gen = torch.nn.Sequential(torch.nn.Linear(embed_dim, embed_dim*2),
                                             torch.nn.LayerNorm(embed_dim*2),
                                             Lambda(lambda x: swish(x)),
                                             torch.nn.Linear(embed_dim*2, embed_dim*self.num_heads))
        if self.concat_or_add:
            self.attn1 = torch.nn.MultiheadAttention(embed_dim*self.num_heads, self.num_heads, vdim=embed_dim+self.pos_enc_dim, kdim=embed_dim)
        else:
            self.attn1 = torch.nn.MultiheadAttention(embed_dim*self.num_heads, self.num_heads, vdim=embed_dim, kdim=embed_dim)
        if self.concat_or_add:
            self.fclast = torch.nn.Sequential(torch.nn.Linear((embed_dim+self.pos_enc_dim)*self.num_heads, embed_dim*self.num_heads),
                                              torch.nn.LayerNorm(self.args.hidden),
                                              Lambda(lambda x: swish(x)))
            self.memory_embed_dim = embed_dim*self.num_heads
        else:
            self.fclast = torch.nn.Sequential(torch.nn.Linear(embed_dim*self.num_heads, embed_dim*self.num_heads),
                                              torch.nn.LayerNorm(embed_dim*self.num_heads),
                                              Lambda(lambda x: swish(x)))
            self.memory_embed_dim = embed_dim*self.num_heads

    def forward(self, x, current_pov_embed):
        assert len(x.shape) == 3
        x = self.conv1(x)
        if self.concat_or_add:
            value = torch.cat([x.transpose(0, 1), self.pe], dim=-1) #length, batch, embed_dim + position_dim
        else:
            value = x.transpose(0, 1)
            value[:,:,:self.pos_enc_dim] += self.pe
        key = x.transpose(0, 1) #length, batch, embed_dim
        query = self.query_gen(current_pov_embed).unsqueeze(0) #length, batch, embed_dim

        attn1_out, _ = self.attn1(query, key, value) #length, batch, (embed_dim + position_dim)*numheads
        assert attn1_out.size(0) == 1
        memory_embed = self.fclast(attn1_out[0,:]) #batch, self.args.hidden
        return memory_embed



class VirtualPG():
    def __init__(self, state_shape, action_dict, args, time_deltas):
        assert isinstance(state_shape, dict)
        self.state_shape = state_shape
        self.action_dict = copy.deepcopy(action_dict)
        self.state_keys = ['state', 'pov', 'history_action', 'history_reward', 'orientation']
        self.args = args
        
        self.action_dim = 2
        self.action_split = [2]
        for a in action_dict:
            if a != 'camera':
                self.action_dim += action_dict[a].n
                self.action_split.append(action_dict[a].n)
        self.time_deltas = time_deltas
        self.zero_time_point = 0
        for i in range(0,len(self.time_deltas)):
            if self.time_deltas[i] == 0:
                self.zero_time_point = i
                break
        self.max_predict_range = 0
        for i in range(self.zero_time_point+1,len(self.time_deltas)):
            if self.time_deltas[i] - self.max_predict_range == 1:
                self.max_predict_range = self.time_deltas[i]
            else:
                break
        print("MAX PREDICT RANGE:",self.max_predict_range)

        self.train_iter = 0

        self.past_time = self.time_deltas[:self.zero_time_point]
        self.future_time = self.time_deltas[(self.zero_time_point+1):]

        self.lookahead_applied = False

        self._model = self._make_model()
    def state_dict(self):
        return [self._model['modules'].state_dict()]

    def load_state_dict(self, s, strict=True):
        if isinstance(s, list):
            self._model['modules'].load_state_dict(s[0], strict)
        else:
            self._model['modules'].load_state_dict(s, strict)

    def loadstore(self, filename, load=True):
        if load:
            print('Load actor')
            state_dict = torch.load(filename + '-virtualPGmultiaction')
            if False:#AUX_CAMERA_LOSS:
                for n, s in self._model['modules'].state_dict().items():
                    if s.shape != state_dict[n].shape:
                        if len(s.shape) == 2:
                            A = min(s.shape[0],state_dict[n].shape[0])
                            B = min(s.shape[1],state_dict[n].shape[1])
                            s[:A, :B] = state_dict[n][:A, :B]
                        else:
                            s[:min(s.shape[0],state_dict[n].shape[0])] = state_dict[n][:min(s.shape[0],state_dict[n].shape[0])]
            else:
                self.load_state_dict(state_dict)
            if 'Lambda' in self._model:
                self._model['Lambda'] = torch.load(filename + '-virtualPGmultiactionLam')
        else:
            torch.save(self._model['modules'].state_dict(), filename + '-virtualPGmultiaction')
            if 'Lambda' in self._model:
                torch.save(self._model['Lambda'], filename + '-virtualPGmultiactionLam')

    def generate_embed(self, state_povs):
        with torch.no_grad():
            if len(state_povs.shape) == 3:
                state_povs = state_povs[None, :]
            state_povs = variable(state_povs)
            state_povs = (state_povs.permute(0, 3, 1, 2)/255.0)*2.0 - 1.0
            pov_embed = torch.tanh(self._model['modules']['pov_embed_no_tanh'](state_povs))
            return {'pov_embed': pov_embed.data.cpu()}

    def embed_desc(self):
        desc = {}
        desc['pov_embed'] = {'compression': False, 'shape': [self.args.embed_dim], 'dtype': np.float32}
        return desc
    
    def _make_model(self):
        modules = torch.nn.ModuleDict()
        modules['pov_embed_no_tanh'] = VisualNet(self.state_shape, self.args)
        self.cnn_embed_dim = self.args.embed_dim

        if 'state' in self.state_shape:
            self.state_dim = self.state_shape['state'][0]
        else:
            self.state_dim = 0
        for o in self.state_shape:
            if 'history' in o or o == 'orientation':
                self.state_dim += self.state_shape[o][0]
        #input to memory is state_embeds
        print('TODO add state_dim to memory')
        modules['memory'] = MemoryModel(self.args, self.cnn_embed_dim, self.past_time)
        self.memory_embed = modules['memory'].memory_embed_dim

        #actions [0:self.max_predict_range] autoencoder model conditioned on current state
        self.actions_embed_dim = 64
        if AUX_CAMERA_LOSS:
            modules['all_embed'] = torch.nn.Sequential(torch.nn.Linear(self.action_dim*self.max_predict_range + 5 * self.max_predict_range + self.cnn_embed_dim + self.memory_embed + self.state_dim, self.args.hidden), 
                                                       Lambda(lambda x: swish(x)))
            modules['actions_enc_no_tanh'] = torch.nn.Linear(self.args.hidden, self.actions_embed_dim)#input is all_embed
            modules['actions_dec'] = torch.nn.Sequential(torch.nn.Linear(self.actions_embed_dim + self.cnn_embed_dim + self.memory_embed + self.state_dim, self.args.hidden),
                                                         Lambda(lambda x: swish(x)),
                                                         torch.nn.Linear(self.args.hidden, self.action_dim*self.max_predict_range + 5 * self.max_predict_range))
        else:
            modules['all_embed'] = torch.nn.Sequential(torch.nn.Linear(self.action_dim*self.max_predict_range + self.cnn_embed_dim + self.memory_embed + self.state_dim, self.args.hidden), 
                                                       Lambda(lambda x: swish(x)))
            modules['actions_enc_no_tanh'] = torch.nn.Linear(self.args.hidden, self.actions_embed_dim)#input is all_embed
            modules['actions_dec'] = torch.nn.Sequential(torch.nn.Linear(self.actions_embed_dim + self.cnn_embed_dim + self.memory_embed + self.state_dim, self.args.hidden),
                                                         Lambda(lambda x: swish(x)),
                                                         torch.nn.Linear(self.args.hidden, self.action_dim*self.max_predict_range))
        
        #actions predict
        modules['actions_predict'] = torch.nn.Sequential(torch.nn.Linear(self.cnn_embed_dim + self.memory_embed + self.state_dim, self.args.hidden),
                                                                Lambda(lambda x: swish(x)),
                                                                torch.nn.Linear(self.args.hidden, self.actions_embed_dim*2))#mean + stds

        #aux prediction
        modules['r_predict'] = torch.nn.Sequential(torch.nn.Linear(self.actions_embed_dim + self.cnn_embed_dim + self.memory_embed + self.state_dim, self.args.hidden),
                                                   Lambda(lambda x: swish(x)),
                                                   torch.nn.Linear(self.args.hidden, 1))
        modules['v_predict_short'] = torch.nn.Linear(self.cnn_embed_dim, 2)
        modules['v_predict_long'] = torch.nn.Linear(self.cnn_embed_dim, 2)

        #infoDIM loss
        if INFODIM:
            modules['disc'] = torch.nn.Sequential(torch.nn.Linear(self.cnn_embed_dim*2, self.args.hidden), 
                                                  Lambda(lambda x: swish(x)),#torch.nn.LeakyReLU(0.1), 
                                                  torch.nn.Linear(self.args.hidden, 1))#
            if self.args.ganloss == "fisher":
                lam = torch.zeros((1,), requires_grad=True)

        #reconstruction
        if RECONSTRUCTION:
            modules['generate_pov'] = GenNet(self.args, self.state_shape, self.cnn_embed_dim)

        #state prediction
        modules['state_predict'] = torch.nn.Sequential(torch.nn.Linear(self.args.hidden, self.args.hidden), #input is all_embed
                                                       Lambda(lambda x: swish(x)),
                                                       torch.nn.Linear(self.args.hidden, self.cnn_embed_dim*2))#mean and variance, gaussian mixture model

        
        self.revidx = [i for i in range(self.zero_time_point-1, -1, -1)]
        self.revidx = torch.tensor(self.revidx).long()

        if torch.cuda.is_available() and CUDA:
            modules = modules.cuda()
            self.revidx = self.revidx.cuda()
            if INFODIM and self.args.ganloss == "fisher":
                lam = lam.cuda()
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        
        ce_loss = prio_utils.WeightedCELoss()
        loss = prio_utils.WeightedMSELoss()
        l1loss = torch.nn.L1Loss()
        l2loss = torch.nn.MSELoss()
        
        model_params = []
        nec_value_params = []
        for name, p in modules.named_parameters():
            if 'values.weight' in name:
                nec_value_params.append(p)
            else:
                model_params.append(p)

        optimizer = RAdam(model_params, self.args.lr)

        model = {'modules': modules, 
                 'opt': optimizer, 
                 'ce_loss': ce_loss,
                 'l1loss': l1loss,
                 'l2loss': l2loss,
                 'loss': loss}
        if len(nec_value_params) > 0:
            nec_optimizer = torch.optim.Adam(nec_value_params, self.args.nec_lr)
            model['nec_opt'] = nec_optimizer
        if INFODIM and self.args.ganloss == "fisher":
            model['Lambda'] = lam
        return model

    def inverse_map_camera(self, x):
        return x / 180.0

    def map_camera(self, x):
        return x * 180.0

    def train(self, input):
        return {}

    def pretrain(self, input):
        if USE_LOOKAHEAD and not self.lookahead_applied:
            self.lookahead_applied = True
            self._model['opt'] = Lookahead(self._model['opt'])
        for n in self._model:
            if 'opt' in n:
                self._model[n].zero_grad()
        embed_keys = []
        for k in self.embed_desc():
            embed_keys.append(k)
        states = {}
        for k in input:
            if k in self.state_keys or k in embed_keys:
                states[k] = variable(input[k])
        actions = {'camera': self.inverse_map_camera(variable(input['camera']))}
        for a in self.action_dict:
            if a != 'camera':
                actions[a] = variable(input[a], True).long()
        reward = variable(input['reward'])
        done = variable(input['done'], True).int()
        future_done = torch.cumsum(done[:,self.zero_time_point:],1).clamp(max=1).bool()
        reward[:,(self.zero_time_point+1):] = reward[:,(self.zero_time_point+1):].masked_fill(future_done[:,:-1], 0.0)

        is_weight = None
        if 'is_weight' in input:
            is_weight = variable(input['is_weight'])
        states['pov'] = (states['pov'].float()/255.0)*2.0 - 1.0
        states['pov'] = states['pov'].permute(0, 1, 4, 2, 3)
        current_state = {}
        for k in states:
            if k != 'pov':
                current_state[k] = states[k][:,self.zero_time_point]
        current_actions = {}
        for k in actions:
            current_actions[k] = actions[k][:,self.zero_time_point:(self.zero_time_point+self.max_predict_range)]


        with torch.no_grad():
            past_embeds = states['pov_embed'][:,:self.zero_time_point]
            #mask out from other episode
            past_done = done[:,:self.zero_time_point]
            past_done = past_done[:,self.revidx]
            past_done = torch.cumsum(past_done,1).clamp(max=1).bool()
            past_done = past_done[:,self.revidx].unsqueeze(-1).expand(-1,-1,past_embeds.shape[-1])
            past_embeds.masked_fill_(past_done, 0.0)
            #TODO try masking stuff in MultiHeadAttention??
            
        #get pov embeds
        states_gen_embed = states['pov'][:,:2]
        assert states_gen_embed.shape[1] == 2
        embed_no_tanh = self._model['modules']['pov_embed_no_tanh'](states_gen_embed)
        next_embed_no_tanh = embed_no_tanh[:,1]#pov_embed after actions [0:self.max_predict_range]
        current_embed_no_tanh = embed_no_tanh[:,0]

        #get current memory_embed
        memory_embed = self.get_memory_embed(past_embeds, torch.tanh(current_embed_no_tanh))

        #get action and aux loss
        loss, all_embed, rdict = self.prediction_loss(torch.tanh(current_embed_no_tanh), memory_embed, current_state, current_actions, reward[:,self.zero_time_point:])

        #get state prediction loss
        state_pred_loss, mean, std = self.state_prediction(all_embed, next_embed_no_tanh)
        loss += state_pred_loss * 0.1
        rdict['state_pred_loss'] = state_pred_loss

        #get reconstruction loss
        if RECONSTRUCTION:
            rec_pov = self._model['modules']['generate_pov'](torch.tanh(current_embed_no_tanh))
            rec_loss = self._model['loss'](rec_pov, states_gen_embed[:,0]) * 10.0
            loss += rec_loss
            rdict['rec_loss'] = rec_loss

            if True:
                #eps = torch.zeros(std.size(), device=self.device).normal_()
                state_pred_sample = mean #+ eps * std
                rec_pov2 = self._model['modules']['generate_pov'](torch.tanh(state_pred_sample))
                preP = np.concatenate([(states_gen_embed[0,0].permute(1, 2, 0)*0.5+0.5).data.cpu().numpy(),
                                       (rec_pov[0].permute(1, 2, 0)*0.5+0.5).data.cpu().numpy()], axis=1)
                aftP = np.concatenate([(states_gen_embed[0,1].permute(1, 2, 0)*0.5+0.5).data.cpu().numpy(),
                                       (rec_pov2[0].permute(1, 2, 0)*0.5+0.5).data.cpu().numpy()], axis=1)
                plt.imshow(np.concatenate([preP, aftP], axis=0))
                plt.show()

        if INFODIM:
            assert False #not implemented
            
            if self.args.ganloss == 'fisher':
                self._model['Lambda'].retain_grad()

        loss.backward()
        torch.nn.utils.clip_grad_value_(self._model['modules'].parameters(), 1.0)
        for n in self._model:
            if 'opt' in n:
                self._model[n].step()

        if INFODIM and self.args.ganloss == 'fisher':
            self._model['Lambda'].data += self.args.rho * self._model['Lambda'].grad.data
            self._model['Lambda'].grad.data.zero_()
            rdict['Lambda'] = self._model['Lambda'][0]

        #priority buffer stuff
        if 'tidxs' in input:
            #prioritize after bc_loss
            assert False #not implemented
            rdict['prio_upd'] = [{'error': rdict['HIER PRIO LOSS EINGEBEN'].data.cpu().clone(),
                                    'replay_num': 0,
                                    'worker_num': worker_num[0]}]

        for d in rdict:
            if d != 'prio_upd' and not isinstance(rdict[d], str):
                rdict[d] = rdict[d].item()

        self.train_iter += 1
        return rdict


    def state_prediction(self, all_embed, next_embed_no_tanh):
        mean_and_std = self._model['modules']['state_predict'](all_embed)
        mean, std = torch.split(mean_and_std, self.cnn_embed_dim, dim=-1)
        std = std.exp()+1e-12
        dist = torch.distributions.Normal(mean, std)
        log_prob = dist.log_prob(next_embed_no_tanh)
        return -log_prob.mean(), mean, std
    
    def get_memory_embed(self, pov_embes, current_pov_embed):
        output = self._model['modules']['memory'](pov_embes, current_pov_embed)
        return output

    #from https://github.com/pytorch/pytorch/issues/12160
    def batch_diagonal(self, input):
        # idea from here: https://discuss.pytorch.org/t/batch-of-diagonal-matrix/13560
        # batches a stack of vectors (batch x N) -> a stack of diagonal matrices (batch x N x N) 
        # works in  2D -> 3D, should also work in higher dimensions
        # make a zero matrix, which duplicates the last dim of input
        dims = [input.size(i) for i in torch.arange(input.dim())]
        dims.append(dims[-1])
        output = torch.zeros(dims, device=self.device)
        # stride across the first dimensions, add one to get the diagonal of the last dimension
        strides = [output.stride(i) for i in torch.arange(input.dim() - 1 )]
        strides.append(output.size(-1) + 1)
        # stride and copy the imput to the diagonal 
        output.as_strided(input.size(), strides ).copy_(input)
        return output

    def get_actions_loss(self, actions_logits, current_actions, aux_camera_target=None):
        actions_logits = actions_logits.view(-1, self.max_predict_range, self.action_dim if not AUX_CAMERA_LOSS else self.action_dim + 5)
        if AUX_CAMERA_LOSS:
            actions_logits, aux_camera_logits = torch.split(actions_logits, sum(self.action_split), -1)
            aux_loss = self._model['ce_loss'](aux_camera_logits.view(-1, 5), aux_camera_target.view(-1)) * 4.0
        actions_logits = torch.split(actions_logits, self.action_split, -1)
        actions_camera = torch.tanh(actions_logits[0])
        actions_logits = actions_logits[1:]
        #make smooth camera loss
        diff = torch.abs(actions_camera - current_actions['camera'])
        soft_diff = 0.01
        loss = torch.where(diff < soft_diff, 0.5 * diff ** 2, diff + 0.5 * soft_diff ** 2 - soft_diff).sum(-1).mean()# * 10.0
        if AUX_CAMERA_LOSS:
            loss += aux_loss
        #make discrete action losses
        i = 0
        for a in self.action_dict:
            if a != 'camera':
                if a == 'attack_place_equip_craft_nearbyCraft_nearbySmelt' or a == 'jump':
                    loss += self._model['ce_loss'](actions_logits[i].view(-1, self.action_dict[a].n), current_actions[a].reshape(-1)) * 1.5
                else:
                    loss += self._model['ce_loss'](actions_logits[i].view(-1, self.action_dict[a].n), current_actions[a].reshape(-1))
                i += 1
        return loss

    def prediction_loss(self, pov_embed, memory_embed, current_state, current_actions, reward):
        tembed = [pov_embed, memory_embed]
        for o in self.state_shape:
            if o in self.state_keys and o != 'pov':
                tembed.append(current_state[o])
        tembed = torch.cat(tembed, dim=-1)

        #make all_embed
        actions_one_hot = []
        for a in self.action_dict:
            if a != 'camera':
                actions_one_hot.append(one_hot(current_actions[a], self.action_dict[a].n, self.device).view(-1, self.action_dict[a].n*self.max_predict_range))
            else:
                actions_one_hot.append(current_actions[a].reshape(-1, 2*self.max_predict_range))
        if AUX_CAMERA_LOSS:
            aux_camera_target1 = ((torch.abs(current_actions['camera'][:,:,0]) < 1e-5) & (torch.abs(current_actions['camera'][:,:,1]) < 1e-5)).long()
            aux_camera_target2 = (current_actions['camera'][:,:,1] > current_actions['camera'][:,:,0]).long()
            aux_camera_target3 = (current_actions['camera'][:,:,1] > -current_actions['camera'][:,:,0]).long()
            aux_camera_target = (1 - aux_camera_target1) * (aux_camera_target2 + aux_camera_target3 * 2 + 1)
            actions_one_hot.append(one_hot(aux_camera_target, 5, self.device).view(-1, 5*self.max_predict_range))
        else:
            aux_camera_target = None
        actions_one_hot = torch.cat(actions_one_hot, dim=1)
        tembed_wactions = torch.cat([actions_one_hot, tembed], dim=-1)
        all_embed = self._model['modules']['all_embed'](tembed_wactions)

        #make action compression loss
        actions_embed_no_tanh = self._model['modules']['actions_enc_no_tanh'](all_embed)
        actions_embed = torch.tanh(actions_embed_no_tanh)
        actions_logits = self._model['modules']['actions_dec'](torch.cat([tembed, actions_embed], dim=-1))
        action_compression_loss = self.get_actions_loss(actions_logits, current_actions, aux_camera_target)
        loss = action_compression_loss.clone()
        
        #make action predict loss
        mean_std_action_embed = self._model['modules']['actions_predict'](tembed)
        mean, std = torch.split(mean_std_action_embed, self.actions_embed_dim, dim=-1)
        std = std.exp()+1e-12
        dist = torch.distributions.Normal(mean, std)
        log_prob = dist.log_prob(actions_embed_no_tanh)
        action_predict_loss = -log_prob.mean()
        loss += action_predict_loss * 0.5

        #make auxillary losses
        creward = torch.cumsum(reward, 1)
        #predict reward after actions execution
        tembed_waction_embed = torch.cat([actions_embed, tembed], dim=-1)
        target = creward[:,self.max_predict_range-1]
        value = self._model['modules']['r_predict'](tembed_waction_embed)
        r_loss = self._model['loss'](value[:,0], target)
        loss += r_loss
        
        #get short v loss
        self.short_time = 50
        target = (creward[:,self.short_time] > 1e-5).long()
        logits = self._model['modules']['v_predict_short'](pov_embed)
        v_short_loss = self._model['ce_loss'](logits, target)
        loss += v_short_loss

        #get long v loss
        self.long_time = reward.shape[1]-1
        target = (creward[:,self.long_time] > 1e-5).long()
        logits = self._model['modules']['v_predict_long'](pov_embed)
        v_long_loss = self._model['ce_loss'](logits, target)
        loss += v_long_loss



        #action_compression_loss = action_compression_loss.item()
        #action_predict_loss = action_predict_loss.item()
        #r_loss = r_loss.item()
        #v_short_loss = v_short_loss.item()

        return loss, all_embed, {'action_comp_loss': action_compression_loss,
                                 'action_predict_loss': action_predict_loss,
                                 'r_loss': r_loss,
                                 'v_short_loss': v_short_loss,
                                 'v_long_loss': v_long_loss}

    def reset(self):
        max_past = 1-min(self.time_deltas)
        self.past_embeds = torch.zeros((max_past,self.cnn_embed_dim), device=self.device, dtype=torch.float32, requires_grad=False)

    def select_action(self, state, batch=False):
        tstate = {}
        if not batch:
            for k in state:
                tstate[k] = state[k][None, :].copy()
        tstate = variable(tstate)
        with torch.no_grad():
            #pov_embed
            tstate['pov'] = (tstate['pov'].permute(0, 3, 1, 2)/255.0)*2.0 - 1.0
            pov_embed = torch.tanh(self._model['modules']['pov_embed_no_tanh'](tstate['pov']))
            #memory_embed
            memory_inp = self.past_embeds[self.past_time].unsqueeze(0)
            memory_embed = self._model['modules']['memory'](memory_inp, pov_embed)
            self.past_embeds = self.past_embeds.roll(1, dims=0)
            self.past_embeds[0] = pov_embed[0]
            #action embed
            tembed = [pov_embed, memory_embed]
            for o in self.state_shape:
                if o in self.state_keys and o != 'pov':
                    tembed.append(tstate[o])
            tembed = torch.cat(tembed, dim=-1)
            mean_std_action_embed = self._model['modules']['actions_predict'](tembed)
            mean, std = torch.split(mean_std_action_embed, self.actions_embed_dim, dim=-1)
            std = std.exp()+1e-12
            if True:
                #sample
                eps = torch.zeros(std.size(), device=mean.device).normal_()
                sample = mean + eps * std
            else:
                #just use mean
                sample = mean
            actions_embed = torch.tanh(sample)
            #decode actions
            actions_logits = self._model['modules']['actions_dec'](torch.cat([tembed, actions_embed], dim=-1))
            actions_logits = actions_logits.view(-1, self.max_predict_range, self.action_dim if not AUX_CAMERA_LOSS else self.action_dim + 5)
            if AUX_CAMERA_LOSS:
                actions_logits = torch.split(actions_logits, self.action_split + [5], -1)
            else:
                actions_logits = torch.split(actions_logits, self.action_split, -1)
            actions_camera = torch.tanh(actions_logits[0])
            actions_logits = actions_logits[1:]
            a_dict = {}
            i = 0
            for a in self.action_dict:
                if a != 'camera':
                    a_dict[a] = torch.distributions.Categorical(logits=actions_logits[i][0,0].cpu()).sample().item()
                    i += 1
                else:
                    a_dict[a] = self.map_camera(actions_camera[0, 0].data.cpu().numpy())#+np.random.normal(0.0, 1.0/180.0, size=(2,))
                    #if np.dot(a_dict[a], a_dict[a]) < 1.5:
                    #    a_dict[a] *= 0.0
                    #a_dict[a] += np.random.normal(0.0, 1.0, size=(2,))
            if AUX_CAMERA_LOSS:
                if torch.distributions.Categorical(logits=actions_logits[-1][0,0].cpu()).sample().item() == 0:
                    a_dict['camera'] *= 0.0
            return a_dict, {'pov_embed': pov_embed.data.cpu()}

