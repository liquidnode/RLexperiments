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
from virtual_replay_buffer2 import VirtualReplayBuffer, VirtualReplayBufferPER
from virtual_traj_generator import VirtualExpertGenerator, VirtualActorTrajGenerator
from trainers import Trainer
from memory import HashingMemory
import argparse
from torch.multiprocessing import Pipe, Queue
import time
from virtual_models import COMPATIBLE
import os
import numpy as np

USE_LOOKAHEAD = False
GAMMA = 0.99
AUX_CAMERA_LOSS = False
INFODIM = False
MEMORY_MODEL = False

class CombineQueue():
    def __init__(self, queue1, queue2, ratio=0.5):
        self.queue1 = queue1
        self.queue2 = queue2
        self.ratio = ratio

    def get(self):
        if np.random.sample() < self.ratio:
            return self.queue1.get()
        else:
            return self.queue2.get()
        
    def get_nowait(self):
        if np.random.sample() < self.ratio:
            return self.queue1.get_nowait()
        else:
            return self.queue2.get_nowait()
        
def sample_gaussian(mu, log_var):
    std = torch.exp(0.5 * log_var)
    eps = torch.zeros(std.size(), device=mu.device).normal_()
    return mu + std * eps

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
        if disc_dims is None:
            self.disc_dims = []
        encnn_layers = []
        self.embed_dim = self.args.embed_dim
        self.temperature = self.args.temperature
        if self.cont_dim is not None:
            assert self.embed_dim == self.cont_dim + sum(self.disc_dims)
        assert len(self.state_shape['pov']) > 1
        if self.args.cnn_type == 'atari':
            white = False
            sizes = [8, 4, 3]
            strides = [4, 2, 1]
            batchnorm = [True, True, True]
            spectnorm = [False, False, False]
            pooling = [1, 1, 1]
            filters = [32, 64, 32]
            padding = [0, 0, 0]
            end_size = 4
        elif self.args.cnn_type == 'mnist':
            white = False
            sizes = [3, 3, 3]
            strides = [1, 1, 1]
            batchnorm = [False, False, False]
            spectnorm = [False, False, False]
            pooling = [2, 2, 2]
            filters = [32, 32, 32]
            padding = [0, 0, 0]
            end_size = 4
        elif self.args.cnn_type == 'adv':
            white = False
            sizes = [4, 3, 3, 4]
            strides = [2, 2, 2, 1]
            batchnorm = [True, True, True, True]
            spectnorm = [True, True, True, True]
            pooling = [1, 1, 1, 1]
            filters = [16, 32, 64, 32]
            padding = [0, 0, 0, 0]
            end_size = 4
        elif self.args.cnn_type == 'dcgan':
            white = True
            filters = [32, 64, 128, 256, 400]
            sizes = [4, 4, 4, 4, 4]
            strides = [2, 2, 2, 2, 2]
            padding = [1, 1, 1, 1, 0]
            batchnorm = [False, True, True, True, False]
            spectnorm = [False, False, False, False, False]
            pooling = [1, 1, 1, 1, 1]
            end_size = 1



        in_channels = self.state_shape['pov'][-1]

        #make encoder
        for i in range(len(sizes)):
            tencnn_layers = []
            if white and i == 0:
                tencnn_layers.append(torch.nn.BatchNorm2d(in_channels))

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
            if len(self.disc_dims) > 0:
                cont, discs = torch.split(x, self.cont_dim*2, dim=-1)
            else:
                cont = x
            mu, log_var = torch.split(cont, self.cont_dim, dim=-1)
            if len(self.disc_dims) > 0:
                sdiscs = list(torch.split(discs, self.disc_dims, dim=-1))
                for i in range(len(sdiscs)):
                    sdiscs[i] = self.log_softmax(sdiscs[i])
                discs = torch.cat(sdiscs, dim=-1)
                return mu, log_var, discs
            else:
                return mu, log_var
        else:
            if time_tmp is not None:
                x = x.view(-1, time_tmp, self.embed_dim)
            return x

    def sample(self, mu, log_var, discs=None, training=True, hard=False):
        #sample continuous
        if training:
            cont_sample = sample_gaussian(mu, log_var)
        else:
            cont_sample = mu

        #sample discrete
        if len(self.disc_dims) > 0:
            sdiscs = list(torch.split(discs, self.disc_dims, dim=-1))
            disc_samples = [F.gumbel_softmax(d, self.temperature, (not training) or hard or ST_ENABLE) for d in sdiscs]
        else:
            disc_samples = []

        return torch.cat([cont_sample]+disc_samples, dim=-1)

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

class MultiStep():
    def __init__(self, state_shape, action_dict, args, time_deltas, only_visnet=False):
        assert isinstance(state_shape, dict)
        self.state_shape = state_shape
        self.action_dict = copy.deepcopy(action_dict)
        self.state_keys = ['state', 'pov', 'history_action', 'history_reward', 'orientation', 'env_type']
        self.args = args

        self.action_dim = 2
        self.action_split = []
        for a in action_dict:
            if a != 'camera':
                self.action_dim += action_dict[a].n
                self.action_split.append(action_dict[a].n)
        self.time_deltas = time_deltas
        self.zero_time_point = 0
        for i in range(0,len(self.time_deltas)):
            if self.time_deltas[len(self.time_deltas)-i-1] == 0:
                self.zero_time_point = len(self.time_deltas)-i-1
                break
        self.max_predict_range = 0
        for i in range(self.zero_time_point+1,len(self.time_deltas)):
            if self.time_deltas[i] - self.max_predict_range == 1:
                self.max_predict_range = self.time_deltas[i]
            else:
                break
        self.max_predict_range //= 2
        print("MAX PREDICT RANGE:",self.max_predict_range)

        self.train_iter = 0

        self.past_time = self.time_deltas[self.args.num_past_times:self.zero_time_point]
        self.future_time = self.time_deltas[(self.zero_time_point+1):]
        
        self.disc_embed_dims = []
        self.sum_logs = float(sum([np.log(d) for d in self.disc_embed_dims]))
        self.total_disc_embed_dim = sum(self.disc_embed_dims)
        self.continuous_embed_dim = self.args.hidden-self.total_disc_embed_dim

        self.lookahead_applied = False
        self.running_Q_diff = None
        self.num_policy_samples = 16

        self.only_visnet = only_visnet
        self._model = self._make_model()

    
    def load_state_dict(self, s, strict=True):
        if isinstance(s, list):
            if self.only_visnet:
                state_dict = {n: v for n, v in s[0].items() if n.startswith('pov_embed.')}
            else:
                state_dict = s[0]
        else:
            if self.only_visnet:
                state_dict = {n: v for n, v in s.items() if n.startswith('pov_embed.')}
            else:
                state_dict = s
        self._model['modules'].load_state_dict(state_dict, strict)

    def loadstore(self, filename, load=True):
        if load:
            print('Load actor')
            self.train_iter = torch.load(filename + 'train_iter')
            state_dict = torch.load(filename + 'multistep')
            if self.only_visnet:
                state_dict = {n: v for n, v in state_dict.items() if n.startswith('pov_embed.')}
            self.load_state_dict(state_dict)
            if 'Lambda' in self._model:
                self._model['Lambda'] = torch.load(filename + 'multistepLam')
        else:
            torch.save(self._model['modules'].state_dict(), filename + 'multistep')
            torch.save(self.train_iter, filename + 'train_iter')
            if 'Lambda' in self._model:
                torch.save(self._model['Lambda'], filename + 'multistepLam')
                
    def generate_embed(self, state_povs):
        with torch.no_grad():
            if len(state_povs.shape) == 3:
                state_povs = state_povs[None, :]
            state_povs = variable(state_povs)
            state_povs = (state_povs.permute(0, 3, 1, 2)/255.0)*2.0 - 1.0
            if len(self.disc_embed_dims) == 0:
                pov_embed = self._model['modules']['pov_embed'](state_povs)
                return {'pov_embed': pov_embed.data.cpu()}
            else:
                assert False #not implemented

    def embed_desc(self):
        desc = {}
        if MEMORY_MODEL:
            desc['pov_embed'] = {'compression': False, 'shape': [self.continuous_embed_dim+sum(self.disc_embed_dims)], 'dtype': np.float32}
            if len(self.disc_embed_dims) == 0:
                return desc
            else:
                assert False #not implemented
        else:
            return desc

    def _make_model(self):
        self.device = torch.device("cuda") if torch.cuda.is_available() and CUDA else torch.device("cpu")
        modules = torch.nn.ModuleDict()
        modules['pov_embed'] = VisualNet(self.state_shape, self.args, self.continuous_embed_dim, self.disc_embed_dims, False)
        self.cnn_embed_dim = self.continuous_embed_dim+sum(self.disc_embed_dims)
        if self.only_visnet:
            if torch.cuda.is_available() and CUDA:
                modules = modules.cuda()
            return {'modules': modules}


        self.state_dim = 0
        for o in self.state_shape:
            if o != 'pov' and o in self.state_keys:
                self.state_dim += self.state_shape[o][0]
        print('Input state dim', self.state_dim)
        #input to memory is state_embeds
        print('TODO add state_dim to memory')
        self.complete_state = self.cnn_embed_dim + self.state_dim
        if MEMORY_MODEL:
            modules['memory'] = MemoryModel(self.args, self.cnn_embed_dim, self.past_time[-self.args.num_past_times:])
            self.memory_embed = modules['memory'].memory_embed_dim
            self.complete_state += self.memory_embed

        modules['actions_predict'] = torch.nn.Sequential(torch.nn.Linear(self.complete_state, self.args.hidden),
                                                         Lambda(lambda x: swish(x)),
                                                         torch.nn.Linear(self.args.hidden, self.args.hidden),
                                                         Lambda(lambda x: swish(x)),
                                                         torch.nn.Linear(self.args.hidden, (self.action_dim-2)*self.max_predict_range+4*self.max_predict_range+2*self.max_predict_range))#discrete probs + mu/var of continuous actions + zero cont. probs

        self.num_horizon = 16
        modules['Q_values'] = torch.nn.Sequential((torch.nn.Linear(self.complete_state + (self.action_dim + 2) * self.max_predict_range, self.args.hidden)),#SpectralNorm
                                                  Lambda(lambda x: swish(x)),
                                                  (torch.nn.Linear(self.args.hidden, self.args.hidden)),#SpectralNorm
                                                  Lambda(lambda x: swish(x)),
                                                  HashingMemory.build(self.args.hidden, self.num_horizon, self.args))
                                                  #torch.nn.Linear(self.args.hidden, self.num_horizon))

        self.revidx = [i for i in range(self.args.num_past_times-1, -1, -1)]
        self.revidx = torch.tensor(self.revidx).long()

        if INFODIM and self.args.ganloss == "fisher":
            lam = torch.zeros((1,), requires_grad=True)

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

        self.gammas = GAMMA ** torch.arange(0, self.max_predict_range, device=self.device).float()#.unsqueeze(0).expand(self.args.er, -1)

        optimizer = RAdam(model_params, self.args.lr, weight_decay=1e-5)

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

    def pretrain(self, input, worker_num=None):
        assert False #not implemented

    def train(self, input, expert_input, worker_num=None):
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
                states[k] = torch.cat([variable(input[k]),variable(expert_input[k])],dim=0)
        actions = {'camera': self.inverse_map_camera(torch.cat([variable(input['camera']),variable(expert_input['camera'])],dim=0))}
        for a in self.action_dict:
            if a != 'camera':
                actions[a] = torch.cat([variable(input[a], True).long(),variable(expert_input[a], True).long()],dim=0)
        batch_size = variable(input['reward']).shape[0]
        reward = torch.cat([variable(input['reward']),variable(expert_input['reward'])],dim=0)
        done = torch.cat([variable(input['done'], True).int(),variable(expert_input['done'], True).int()],dim=0)
        assert not 'r_offset' in input
        future_done = torch.cumsum(done[:,self.zero_time_point:],1).clamp(max=1).bool()
        reward[:,(self.zero_time_point+1):] = reward[:,(self.zero_time_point+1):].masked_fill(future_done[:,:-1], 0.0)

        is_weight = None
        if 'is_weight' in input:
            is_weight = torch.cat([variable(input['is_weight']),variable(expert_input['is_weight'])],dim=0)
        states['pov'] = (states['pov'].float()/255.0)*2.0 - 1.0
        states['pov'] = states['pov'].permute(0, 1, 4, 2, 3)
        current_state = {}
        for k in states:
            if k != 'pov':
                current_state[k] = states[k][:,self.zero_time_point]
        next_state = {}
        for k in states:
            if k != 'pov':
                next_state[k] = states[k][:,self.zero_time_point+self.max_predict_range]
        current_actions = {}
        for k in actions:
            current_actions[k] = actions[k][:,self.zero_time_point:(self.zero_time_point+self.max_predict_range)]
        next_actions = {}
        for k in actions:
            next_actions[k] = actions[k][:,(self.zero_time_point+self.max_predict_range):(self.zero_time_point+self.max_predict_range*2)]
        assert states['pov'].shape[1] == 2
        
        if MEMORY_MODEL:
            with torch.no_grad():
                past_embeds = states['pov_embed'][:,(self.zero_time_point-self.args.num_past_times):self.zero_time_point]
                #mask out from other episode
                past_done = done[:,(self.zero_time_point-self.args.num_past_times):self.zero_time_point]
                past_done = past_done[:,self.revidx]
                past_done = torch.cumsum(past_done,1).clamp(max=1).bool()
                past_done = past_done[:,self.revidx].unsqueeze(-1).expand(-1,-1,past_embeds.shape[-1])
                past_embeds.masked_fill_(past_done, 0.0)

            
                next_past_embeds = states['pov_embed'][:,:(self.zero_time_point-self.args.num_past_times)]
                #mask out from other episode
                past_done = done[:,:(self.zero_time_point-self.args.num_past_times)]
                past_done = past_done[:,self.revidx]
                past_done = torch.cumsum(past_done,1).clamp(max=1).bool()
                past_done = past_done[:,self.revidx].unsqueeze(-1).expand(-1,-1,past_embeds.shape[-1])
                next_past_embeds.masked_fill_(past_done, 0.0)

                
        #get pov embed
        current_pov_embed = self._model['modules']['pov_embed'](states['pov'][:,0])
        with torch.no_grad():
            next_pov_embed = self._model['modules']['pov_embed'](states['pov'][:,1])

        tembed = [current_pov_embed]
        nembed = [next_pov_embed]
            
        if MEMORY_MODEL:
            #make memory embed
            current_memory_embed = self._model['modules']['memory'](past_embeds, current_pov_embed)
            with torch.no_grad():
                next_memory_embed = self._target['memory'](next_past_embeds, next_pov_embed)
            tembed.append(current_memory_embed)
            nembed.append(next_memory_embed)

        #combine all embeds
        for o in self.state_shape:
            if o in self.state_keys and o != 'pov':
                tembed.append(current_state[o])
        tembed = torch.cat(tembed, dim=-1)
        for o in self.state_shape:
            if o in self.state_keys and o != 'pov':
                nembed.append(next_state[o])
        nembed = torch.cat(nembed, dim=-1)
        
        #encode actions
        current_actions_cat = []
        for a in self.action_dict:
            if a != 'camera':
                current_actions_cat.append(one_hot(current_actions[a], self.action_dict[a].n, self.device).view(-1, self.action_dict[a].n*self.max_predict_range))
            else:
                current_actions_cat.append(current_actions[a].reshape(-1, 2*self.max_predict_range))
                current_zero = ((torch.abs(current_actions[a].view(-1, self.max_predict_range, 2)[:,:,0]) < 1e-5) & (torch.abs(current_actions[a].view(-1, self.max_predict_range, 2)[:,:,1]) < 1e-5)).float()
                current_actions_cat.append(1.0 - current_zero)
                current_actions_cat.append(current_zero)
        current_actions_cat = torch.cat(current_actions_cat, dim=1)

        #get best (next) action
        actions_logits = self._model['modules']['actions_predict'](nembed).view(-1, self.max_predict_range, self.action_dim + 4)
        dis_logits, zero_logits, camera_mu, camera_log_var = actions_logits[:,:,:(self.action_dim-2)], actions_logits[:,:,(self.action_dim-2):self.action_dim], actions_logits[:,:,self.action_dim:(self.action_dim+2)], actions_logits[:,:,(self.action_dim+2):]
        camera_mu *= 0.1
        camera_log_var = 5.0 - torch.nn.Softplus()(5.0 - torch.nn.Softplus()(camera_log_var + 5.0) + 5.0)
        camera_log_var -= 5.0
        dis_logits = torch.split(dis_logits, self.action_split, -1)
        next_Qs = None
        mean_Q = None
        loss = None
        all_actions = []
        all_Qs = []
        for s in range(self.num_policy_samples):
            ii = 0
            actions_cat = []
            c_actions = {}
            for a in self.action_dict:
                if a != 'camera':
                    c_actions[a] = torch.distributions.Categorical(logits=dis_logits[ii]).sample() # [batch, max_range]
                    actions_cat.append(one_hot(c_actions[a], self.action_dict[a].n, self.device).view(-1, self.action_dict[a].n*self.max_predict_range))
                    ii += 1
                else:
                    camera = sample_gaussian(camera_mu, camera_log_var)
                    zero_camera = torch.distributions.Categorical(logits=zero_logits).sample()
                    camera = torch.where((zero_camera == 1).unsqueeze(-1).expand(-1,-1,2), torch.zeros_like(camera), camera)
                    c_actions['camera'] = camera.clone()
                    camera = torch.tanh(camera)
                    actions_cat.append(camera.view(-1, 2*self.max_predict_range))
                    actions_cat.append(1.0 - zero_camera.float())
                    actions_cat.append(zero_camera.float())
            actions_cat = torch.cat(actions_cat, dim=1)
            with torch.no_grad():
                c_next_Qs = self._model['modules']['Q_values'](torch.cat([nembed, actions_cat], dim=1))
            all_actions.append(c_actions)
            all_Qs.append(c_next_Qs.clone())
            if next_Qs is None:
                next_Qs = c_next_Qs
                #best_actions = c_actions
                mean_Q = c_next_Qs[:,self.num_horizon-1]
            else:
                #Q_mask = (next_Qs[:,self.num_horizon-1] < c_next_Qs[:,self.num_horizon-1]).view(-1, 1)
                #for a in best_actions:
                #    best_actions[a] = np.where(Q_mask.expand(-1, c_actions[a].shape[1]), c_actions[a], best_actions[a])
                next_Qs = torch.max(next_Qs, c_next_Qs)
                mean_Q += c_next_Qs[:,self.num_horizon-1]

        if self.train_iter % 1 == 0:
            #add bc loss
            entropy_bonus = 0.001
            bc_loss = None
            ii = 0
            for a in self.action_dict:
                if a != 'camera':
                    if bc_loss is None:
                        bc_loss = self._model['ce_loss'](dis_logits[ii][batch_size:].reshape(-1, self.action_dict[a].n), next_actions[a][batch_size:].reshape(-1))
                    else:
                        bc_loss += self._model['ce_loss'](dis_logits[ii][batch_size:].reshape(-1, self.action_dict[a].n), next_actions[a][batch_size:].reshape(-1))
                    ii += 1
                else:
                    bc_loss += -self.normal_log_prob_dens(camera_mu[batch_size:], camera_log_var[batch_size:], next_actions[a][batch_size:]).mean()
                    current_zero = ((torch.abs(next_actions[a].view(-1, self.max_predict_range, 2)[batch_size:,:,0]) < 1e-5) & (torch.abs(next_actions[a].view(-1, self.max_predict_range, 2)[batch_size:,:,1]) < 1e-5)).long()
                    bc_loss += self._model['ce_loss'](zero_logits[batch_size:].reshape(-1, 2), current_zero.reshape(-1))
            loss = bc_loss
            rdict = {'bc_loss': bc_loss}

            dis_logits = [torch.nn.LogSoftmax(-1)(d) for d in dis_logits]
            zero_logits = torch.nn.LogSoftmax(-1)(zero_logits)

            #learn policy
            policy_loss = None
            mean_Q /= self.num_policy_samples
            for j in range(self.num_policy_samples):
                advantage = (all_Qs[j][:,self.num_horizon-1] - mean_Q).unsqueeze(-1).expand(-1, self.max_predict_range).detach()
                ii = 0
                for a in self.action_dict:
                    if a != 'camera':
                        if policy_loss is None:
                            policy_loss = -(advantage * torch.gather(dis_logits[ii], -1, all_actions[j][a].unsqueeze(-1))[:,:,0]).mean() - entropy_bonus * (dis_logits[ii].exp()*dis_logits[ii]).sum(-1).mean()
                        else:
                            policy_loss += -(advantage * torch.gather(dis_logits[ii], -1, all_actions[j][a].unsqueeze(-1))[:,:,0]).mean() - entropy_bonus * (dis_logits[ii].exp()*dis_logits[ii]).sum(-1).mean()
                        ii += 1
                    else:
                        policy_loss += -(advantage * self.normal_log_prob_dens(camera_mu, camera_log_var, all_actions[j][a].detach()).sum(-1)).mean() * 0.1 #+ entropy_bonus * self.normal_pre_kl_divergence(camera_mu, camera_log_var)
                        current_zero = ((torch.abs(all_actions[j][a].view(-1, self.max_predict_range, 2)[:,:,0]) < 1e-5) & (torch.abs(all_actions[j][a].view(-1, self.max_predict_range, 2)[:,:,1]) < 1e-5)).long()
                        policy_loss += -(advantage * torch.gather(zero_logits, -1, current_zero.unsqueeze(-1))[:,:,0]).mean() - entropy_bonus * (zero_logits.exp()*zero_logits).sum(-1).mean()
                policy_loss += (all_actions[j]['camera']).pow(2).mean() * 10.0
            loss += policy_loss
            rdict['policy_loss'] = policy_loss
        else:
            rdict = {}

        current_Qs = self._model['modules']['Q_values'](torch.cat([tembed, current_actions_cat], dim=1))
        next_Qs = torch.cat([torch.zeros_like(next_Qs[:,0]).unsqueeze(-1), next_Qs[:,:-1]], dim=-1)
        target = (reward[:,self.zero_time_point:(self.zero_time_point+self.max_predict_range)] * self.gammas.unsqueeze(0).expand(reward.shape[0], -1)).sum(1).unsqueeze(-1).expand(-1,self.num_horizon) + (GAMMA ** self.max_predict_range) * next_Qs
        if is_weight is None:
            Q_loss = (current_Qs - target.detach()).pow(2).mean()
        else:
            Q_loss = ((current_Qs - target.detach()).pow(2) * is_weight.view(-1, 1).expand(-1, self.num_horizon)).mean()
        if loss is None:
            loss = Q_loss.clone()
        else:
            loss += Q_loss
        with torch.no_grad():
            rdict['Q_diff'] = torch.abs(current_Qs - target).mean()
            rdict['current_Q_mean'] = current_Qs.mean()
            rdict['next_Q_mean'] = next_Qs.mean()

        
        if 'Lambda' in self._model:
            if self.args.ganloss == 'fisher':
                self._model['Lambda'].retain_grad()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self._model['modules'].parameters(), 1000.0)
        for n in self._model:
            if 'opt' in n:
                self._model[n].step()
                
        if 'Lambda' in self._model and self.args.ganloss == 'fisher':
            self._model['Lambda'].data += self.args.rho * self._model['Lambda'].grad.data
            self._model['Lambda'].grad.data.zero_()
            rdict['Lambda'] = self._model['Lambda']

        #priority buffer stuff
        with torch.no_grad():
            if 'tidxs' in input:
                #prioritize using Q_diff
                rdict['prio_upd'] = [{'error': torch.abs(current_Qs - target)[:batch_size,-1].data.cpu().clone(),
                                      'replay_num': 0,
                                      'worker_num': worker_num[0]},
                                     {'error': torch.abs(current_Qs - target)[batch_size:,-1].data.cpu().clone(),
                                      'replay_num': 1,
                                      'worker_num': worker_num[1]}]

        for d in rdict:
            if d != 'prio_upd' and not isinstance(rdict[d], str) and not isinstance(rdict[d], int):
                rdict[d] = rdict[d].data.cpu().numpy()

        self.train_iter += 1
        rdict['train_iter'] = self.train_iter / 10000
        return rdict

    def normal_pre_kl_divergence(self, mu1, log_var1):
        log_diff = log_var1 + 5.0
        nvar2 = np.exp(5.0)
        kl_values = -0.5 * (1 + log_diff - nvar2*(mu1.pow(2)) - log_diff.exp())
        kl_values = torch.sum(kl_values, dim=-1)
        kl_loss = torch.mean(kl_values)
        return kl_loss

    def normal2_kl_divergence(self, mu1, log_var1, mu2, log_var2):
        log_diff = log_var1 - log_var2
        nvar2 = (-log_var2).exp()
        kl_values = -0.5 * (1 + log_diff - nvar2*((mu1-mu2).pow(2)) - log_diff.exp())
        kl_values = torch.sum(kl_values, dim=-1)
        kl_loss = torch.mean(kl_values)
        return kl_loss
        
    def normal1_kl_divergence(self, mu, log_var):
        kl_values = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp())
        kl_values = torch.sum(kl_values, dim=-1)
        kl_loss = torch.mean(kl_values)
        #kl_loss = kl_values
        return kl_loss

    def normal_log_prob_dens(self, mu, log_var, loc):
        #log_var = log(sigma^2)
        sigma_m2 = (-log_var).exp()
        return -0.5*((loc-mu).pow(2)*sigma_m2+np.log(2.0*np.pi)+log_var)
    
    def reset(self):
        if MEMORY_MODEL:
            max_past = 1-min(self.past_time)
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
            pov_embed = self._model['modules']['pov_embed'](tstate['pov'])
            tembed = [pov_embed]
            #memory_embed
            if MEMORY_MODEL:
                memory_inp = self.past_embeds[self.past_time].unsqueeze(0)
                memory_embed = self._model['modules']['memory'](memory_inp, pov_embed)
                self.past_embeds = self.past_embeds.roll(1, dims=0)
                self.past_embeds[0] = pov_embed[0]
                tembed.append(memory_embed)

            #sample action
            for o in self.state_shape:
                if o in self.state_keys and o != 'pov':
                    tembed.append(tstate[o])
            tembed = torch.cat(tembed, dim=-1)
            current_best_actions = self._model['modules']['actions_predict'](tembed).view(-1, self.max_predict_range, self.action_dim + 4)
            dis_logits, zero_logits, camera_mu, camera_log_var = current_best_actions[:,:,:(self.action_dim-2)], current_best_actions[:,:,(self.action_dim-2):self.action_dim], current_best_actions[:,:,self.action_dim:(self.action_dim+2)], current_best_actions[:,:,(self.action_dim+2):]
            camera_mu *= 0.1
            camera_log_var = 5.0 - torch.nn.Softplus()(5.0 - torch.nn.Softplus()(camera_log_var + 5.0) + 5.0)
            camera_log_var -= 5.0
            dis_logits = torch.split(dis_logits, self.action_split, -1)
            action = OrderedDict()
            ii = 0
            for a in self.action_dict:
                if a != 'camera':
                    action[a] = torch.distributions.Categorical(logits=dis_logits[ii][:,0,:]).sample()[0].item()
                    ii += 1
                else:
                    action[a] = np.tanh(sample_gaussian(camera_mu[:,0], camera_log_var[:,0])[0,:].data.cpu().numpy())
                    zero_act = torch.distributions.Categorical(logits=zero_logits[:,0,:]).sample()[0].item()
                    if zero_act == 1:
                        action[a] *= 0.0
                    action[a] = self.map_camera(action[a])
            if MEMORY_MODEL:
                return action, {'pov_embed': pov_embed.data.cpu()}
            else:
                return action, {}


def main():
    parser = argparse.ArgumentParser(description="Multi Step")

    PRETRAIN = False
    parser.add_argument("--pretrain", action='store_true', default=PRETRAIN, help="")
    parser.add_argument("--agent_name", type=str, default='MultiStep', help="Model name")
    parser.add_argument("--agent_from", type=str, default='multi_step_models', help="Model python module")
    parser.add_argument("--needs-next", action='store_true', default=False, help="")
    parser.add_argument("--env", default='MineRLTreechop-v0', type=str, help="Gym environment to use")#MineRLNavigateDense-v0#MineRLTreechop-v0
    parser.add_argument("--minerl", action='store_true', default=True, help="The environment is a MineRL environment")
    parser.add_argument("--name", type=str, default='treechop', help="Experiment name")
    parser.add_argument("--episodes", type=int, default=int(1e5), help="Number of episodes to run")

    parser.add_argument("--erpoolsize", type=int, default=int(4e5), help="Number of experiences stored by each option for experience replay")
    parser.add_argument("--packet-size", type=int, default=int(256), help="")
    parser.add_argument("--sample-efficiency", type=int, default=int(8), help="")
    parser.add_argument("--chunk-size", type=int, default=int(100), help="")
    parser.add_argument("--num-replay-workers", type=int, default=1, help="Number of replay buffer workers")
    parser.add_argument("--per", default=False, action="store_true", help="Use prioritized experience replay.")
    parser.add_argument("--trainstart", type=int, default=int(1e4), help="")
    parser.add_argument("--er", type=int, default=64, help="Number of experiences used to build a replay minibatch")
        
    parser.add_argument("--history-ar", type=int, default=20, help="Number of historic rewards")
    parser.add_argument("--history-a", type=int, default=0, help="Number of historic actions")
    parser.add_argument("--cnn-type", default='adv', type=str, choices=['atari', 'mnist', 'adv', 'fixup'], help="General shape of the CNN, if any. Either DQN-Like, or image-classification-like with more layers")
    parser.add_argument("--hidden", default=256, type=int, help="Hidden neurons of the policy network")
    parser.add_argument("--layers", default=1, type=int, help="Number of hidden layers in the networks")
    parser.add_argument("--state-layers", default=0, type=int, help="Number of hidden layers in the state networks")
    parser.add_argument("--lr", default=1e-4, type=float, help="Learning rate of the neural network")
    parser.add_argument("--eps", default=0.1, type=float, help="Entropy bonus")
    parser.add_argument("--load", type=str, default="treechop/model_world", help="File from which to load the neural network weights")
    parser.add_argument("--pretrained_load", type=str, help="File from which to load the neural network weights")
    parser.add_argument("--save", type=str, default="treechop/model_world", help="Basename of saved weight files. If not given, nothing is saved")

    parser.add_argument("--clr", type=float, default=0.5, help="Critic learning rate")
    parser.add_argument("--nec-lr", type=float, default=1e-3, help="NEC learning rate")
    parser.add_argument("--rho", type=float, default=1e-5, help="Lambda learning rate")
    parser.add_argument("--ganloss", type=str, default="hinge", choices=['hinge', 'wasserstein', 'fisher', 'BCE'], help="GAN loss type")
    
    parser.add_argument("--needs_skip", default=False, action="store_true", help="")
    parser.add_argument("--needs_last_action", default=True, action="store_true", help="")
    parser.add_argument("--needs_orientation", default=False, action="store_true", help="")
    parser.add_argument("--dont_send_past_pov", default=False, action="store_true", help="")
    parser.add_argument("--needs_embedding", default=False, action="store_true", help="")
    parser.add_argument("--needs_random_future", default=False, action="store_true", help="")
    parser.add_argument("--num_past_times", type=int, default=int(32), help="")
    parser.add_argument("--embed_dim", type=int, default=int(128), help="")
    parser.add_argument("--temperature", type=float, default=0.8, help="")
    
    parser.add_argument("--needs_multi_env", default=False, action="store_true", help="")
    parser.add_argument("--log_reward", default=False, action="store_true", help="")

    HashingMemory.register_args(parser)

    args = parser.parse_args()

    args.needs_orientation = True
    args.needs_embedding = MEMORY_MODEL
    args.needs_last_action = False
    args.load = "treechop/model_3"#None#
    args.save = "treechop/model_3"
    args.dont_send_past_pov = True
    args.mem_query_batchnorm = False
    args.cnn_type = 'atari'
    args.ganloss = 'fisher'
    args.rho = 1e-4
    args.mem_n_keys = 256
    args.needs_random_future = False
    args.history_a = 3
    if PRETRAIN:
        args.er = 128
    else:
        args.er = 64
    args.embed_dim = 256
    if MEMORY_MODEL:
        num_past_times = 32
        args.num_past_times = num_past_times
        past_range_seconds = 60.0
        past_range = past_range_seconds / 0.05
        delta_a = (past_range-num_past_times-1)/(num_past_times**2)
        past_times = [-int(delta_a*n**2+n+1) for n in range(num_past_times)]
        past_times.reverse()
        time_skip = 10
        time_deltas = past_times + list(range(0, (time_skip*2)+1, 1))
        next_past_times = [-int(delta_a*n**2+n+1) + time_skip for n in range(num_past_times)]
        next_past_times.reverse()
        time_deltas = next_past_times + time_deltas
    else:
        args.num_past_times = 0
        time_skip = 10
        time_deltas = list(range(0, (time_skip*2)+1, 1))
    pov_time_deltas = [0, time_skip]
    if not PRETRAIN:
        args.erpoolsize = int(5e5)
        args.per = True
        
    HashingMemory.check_params(args)

    num_workers = 3
    copy_queues = [Queue(1) for n in range(num_workers)]
    if PRETRAIN:
        args.packet_size = max(256, args.er)
        args.sample_efficiency = max(args.packet_size // args.er, 1)
        args.trainstart = args.erpoolsize // num_workers
        args.lr = 1e-4
        args.nec_lr = 1e-3
        args.num_replay_workers = num_workers
        if args.per:
            args.sample_efficiency *= 2
            prio_queues = [[Queue(2) for n in range(num_workers)]]
            traj_gens = [VirtualExpertGenerator(args, 0, copy_queue=copy_queues[n], add_args=[time_deltas]) for n in range(num_workers)]
            start = 0
            end = time_deltas[-1]
            replay = VirtualReplayBufferPER(args, traj_gens[0].data_description(), [traj_gens[n].traj_pipe[0] for n in range(num_workers)], prio_queues[0], [t-start for t in time_deltas], 50+end, pov_time_deltas=pov_time_deltas)
            trainer = Trainer(args.agent_name, args.agent_from, traj_gens[0].state_shape, traj_gens[0].num_actions, args, 
                                    [replay.batch_queues], add_args=[time_deltas], prio_queues=prio_queues, copy_queues=copy_queues)
        else:
            traj_gens = [VirtualExpertGenerator(args, 0, copy_queue=copy_queues[n], add_args=[time_deltas]) for n in range(num_workers)]
            start = 0
            end = time_deltas[-1]
            replay = VirtualReplayBuffer(args, traj_gens[0].data_description(), [traj_gens[n].traj_pipe[0] for n in range(num_workers)], [t-start for t in time_deltas], 50+end, pov_time_deltas=pov_time_deltas)
            trainer = Trainer(args.agent_name, args.agent_from, traj_gens[0].state_shape, traj_gens[0].num_actions, args, 
                                    [replay.batch_queues], add_args=[time_deltas], copy_queues=copy_queues)
    else:
        num_workers = 1
        args.packet_size = max(256, args.er)
        args.sample_efficiency = max((args.packet_size * 16) // args.er, 1)
        args.lr = 1e-4
        args.nec_lr = 1e-3
        args.num_replay_workers = num_workers
        copy_queues = [Queue(1) for n in range(num_workers+1)]
        traj_gen = VirtualActorTrajGenerator(args, copy_queue=copy_queues[0], add_queues=args.num_replay_workers-1, add_args=[time_deltas])
        traj_gens = [VirtualExpertGenerator(args, 0, copy_queue=copy_queues[n+1], add_args=[time_deltas]) for n in range(num_workers)]
        #combine_traj = [CombineQueue(traj_gen.traj_pipe[n], traj_gens[n].traj_pipe[0]) for n in range(args.num_replay_workers)]
        assert len(traj_gen.traj_pipe) == args.num_replay_workers
        start = 0
        end = time_deltas[-1]
        if args.per:
            prio_queues = [[Queue(2) for n in range(num_workers)], [Queue(2) for n in range(num_workers)]]
            replay_actor = VirtualReplayBufferPER(args, traj_gen.data_description(), traj_gen.traj_pipe, prio_queues[0], [t-start for t in time_deltas], 50+end, pov_time_deltas=pov_time_deltas, blocking=True)
            args.sample_efficiency = 1
            replay_expert = VirtualReplayBufferPER(args, traj_gen.data_description(), traj_gens[0].traj_pipe, prio_queues[1], [t-start for t in time_deltas], 50+end, pov_time_deltas=pov_time_deltas, blocking=True)
            trainer = Trainer(args.agent_name, args.agent_from, traj_gen.state_shape, traj_gen.num_actions, args, 
                                [replay_actor.batch_queues, replay_expert.batch_queues], add_args=[time_deltas], prio_queues=prio_queues, copy_queues=copy_queues, blocking=False)
        else:
            assert False # CombineQueue funktioniert nicht weil sonst episodes gemischt werden
            replay = VirtualReplayBuffer(args, traj_gen.data_description(), combine_traj, [t-start for t in time_deltas], 50+end, pov_time_deltas=pov_time_deltas)
            trainer = Trainer(args.agent_name, args.agent_from, traj_gen.state_shape, traj_gen.num_actions, args, 
                                [replay.batch_queues], add_args=[time_deltas], copy_queues=copy_queues, blocking=False)

    while True:
        time.sleep(100)

if __name__ == '__main__':
    main()