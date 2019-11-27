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
INFODIM = True
AUX_CAMERA_LOSS = True
COMPATIBLE = False
GAMMA = 0.99

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

def compute_mmd_gauss(x):
    dim = x.size(1)
    y = torch.zeros((128, dim), device=x.device).normal_()
    return compute_mmd(x, y)

def tanh_or_not(x):
    return torch.tanh(x)


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

class FullDiscretizerWLast(torch.nn.Module):
    def __init__(self, num_input, device, args, action_dict):
        super(FullDiscretizerWLast, self).__init__()
        self.device = device
        self.action_dict = copy.deepcopy(action_dict)
        #del self.action_dict['camera']
        self.n_actions = 1 #commit
        self.n_actions += 5 #camera: inc/dec X/Y, inc delta
        self.action_ranges = {'commit': [0,1],
                              'camera': [1,6]} #commit range
        range_start = 6
        self.logit_dim = 0
        for a in action_dict:
            if a != 'camera':
                c_num_actions = action_dict[a].n-1 #without noop
                self.logit_dim += action_dict[a].n
                self.n_actions += c_num_actions
                self.action_ranges[a] = [range_start, range_start+c_num_actions]
                range_start += c_num_actions
        if COMPATIBLE:
            self.fc0 = torch.nn.Sequential(torch.nn.Linear(num_input+4+self.logit_dim, args.hidden), torch.nn.Tanh(), HashingMemory.build(args.hidden, self.n_actions, args))
        else:
            #self.fc0 = torch.nn.Sequential(torch.nn.Linear(num_input+4+self.logit_dim, args.hidden), torch.nn.Tanh(), HashingMemory.build(args.hidden, self.n_actions, args))
            self.fc0 = torch.nn.Sequential(torch.nn.Linear(num_input+4+self.logit_dim, args.hidden), 
                                           Lambda(lambda x: swish(x)),
                                           torch.nn.Linear(args.hidden, self.n_actions))
            #self.fc0 = HashingMemory.build(num_input+4+self.logit_dim, self.n_actions, args)
        self.softmax = torch.nn.Softmax(1)
        self.ce_loss = torch.nn.CrossEntropyLoss(reduction='none')
        if COMPATIBLE:
            self.min_delta = 0.05
        else:
            self.min_delta = 0.5 / 2**3
        if not COMPATIBLE:
            print('')
            print('')
            print("ACHTUNG compatability ist ausgeschaltet.")
            print('')
            print('')
        self.max_step = 20
        self.max_samplestep = 12
        self.max_camstep = 6
        self.current_ac = None
        self.scurrent_ac = None

    def get_loss_BC(self, fc_input, action, last_action=None, is_weight=None, test=False):
        if len(fc_input.shape) > 2:
            #convert time dim to batch dim in all inputs
            if not is_weight is None:
                is_weight = is_weight.unsqueeze(1).expand(-1, fc_input.shape[1])
                is_weight = is_weight.view(-1)
            fc_input = fc_input.view(-1, fc_input.shape[-1])
            for k in action:
                if action[k].dtype == torch.int64:
                    action[k] = action[k].contiguous().view(-1)
                else:
                    action[k] = action[k].contiguous().view(-1, action[k].shape[-1])
            if not last_action is None:
                for k in last_action:
                    if last_action[k].dtype == torch.int64:
                        last_action[k] = last_action[k].contiguous().view(-1)
                    else:
                        last_action[k] = last_action[k].contiguous().view(-1, last_action[k].shape[-1])

        if self.current_ac is None or self.current_ac.shape[0] != fc_input.shape[0]:
            self.current_ac = torch.zeros((fc_input.shape[0],2), dtype=torch.float32, device=self.device, requires_grad=False)
            self.current_delta = torch.ones((fc_input.shape[0],), dtype=torch.float32, device=self.device, requires_grad=False) * self.min_delta
            self.current_dac = torch.zeros((fc_input.shape[0],), dtype=torch.int64, device=self.device, requires_grad=False)
            self.delta_chosen = torch.zeros((fc_input.shape[0],), dtype=torch.bool, device=self.device, requires_grad=False)
            self.current_rdac = {}
            for a in self.action_dict:
                if a != 'camera':
                    if last_action is None or a == 'left_right' or a == 'attack_place_equip_craft_nearbyCraft_nearbySmelt':
                        self.current_rdac[a] = torch.zeros((fc_input.shape[0],), dtype=torch.int64, device=self.device, requires_grad=False)
        else:
            self.current_ac.fill_(0.0)
            self.current_delta.fill_(self.min_delta)
            self.current_dac.fill_(0)
            for a in self.action_dict:
                if a != 'camera':
                    if last_action is None or a == 'left_right' or a == 'attack_place_equip_craft_nearbyCraft_nearbySmelt':
                        self.current_rdac[a].fill_(0)
            self.delta_chosen.fill_(False)
        if last_action is not None:
            for a in self.action_dict:
                if a != 'camera' and a != 'left_right' and a != 'attack_place_equip_craft_nearbyCraft_nearbySmelt':
                    self.current_rdac[a] = last_action['last_'+a]
        loss = None
        cam_steps = None
        last_commit_mask = None
        norm = None
        all_dacs = []
        #all_odacs =[]
        entropy = None
        commited = None
        all_inputs = []
        for i in range(self.max_step):
            c_one_hot = []
            for a in self.action_dict:
                if a != 'camera':
                    c_one_hot.append(one_hot(self.current_rdac[a], self.action_dict[a].n, self.device))
                else:
                    c_one_hot.append(self.current_ac.clone())
            c_one_hot = torch.cat(c_one_hot, dim=1)
            all_inputs.append(torch.cat([c_one_hot.clone(),
                                         self.current_delta.unsqueeze(-1).clone(), 
                                         self.delta_chosen.unsqueeze(-1).float().clone()], dim=-1))
            with torch.no_grad():
                #reset dac
                self.current_dac.fill_(0)
                #check commit
                if cam_steps is None:
                    commit_mask = ((torch.abs(action['camera'][:,0]) < 1e-5) & (torch.abs(action['camera'][:,1]) < 1e-5))
                else:
                    commit_mask = ((torch.abs(action['camera'][:,0]) < 1e-5) & (torch.abs(action['camera'][:,1]) < 1e-5)) | (cam_steps >= self.max_camstep)
                for a in self.action_dict:
                    if a != 'camera':
                        commit_mask &= (self.current_rdac[a] == action[a])
                self.current_dac.masked_fill_(commit_mask, 0)
                modified_dac = commit_mask.clone()
                #check discrete actions
                for a in self.action_dict:
                    if a != 'camera':
                        disc_mask = (~modified_dac) & (self.current_rdac[a] != action[a])
                        self.current_dac = torch.where(disc_mask & (action[a] == 0), 
                                                       self.current_rdac[a]-1+self.action_ranges[a][0], 
                                                       torch.where(disc_mask, action[a]-1+self.action_ranges[a][0], self.current_dac))
                        self.current_rdac[a] = torch.where(disc_mask, action[a], self.current_rdac[a])
                        modified_dac |= disc_mask
                #check inc delta
                delta_mask = (~modified_dac) & (~self.delta_chosen) & ((torch.abs(action['camera'][:,0]-self.current_ac[:,0]) > (self.current_delta * 2.0)) | (torch.abs(action['camera'][:,1]-self.current_ac[:,0]) > (self.current_delta * 2.0)))
                self.current_dac.masked_fill_(delta_mask, 5)
                self.current_delta = torch.where(delta_mask, (self.current_delta * 2.0).clamp(-0.5, 0.5), self.current_delta)
                modified_dac |= delta_mask
                #check inc/dec X/Y
                decX_mask = (action['camera'][:,0] < self.current_ac[:,0])
                incX_mask = (action['camera'][:,0] >= self.current_ac[:,0])
                decY_mask = (action['camera'][:,1] < self.current_ac[:,1])
                incY_mask = (action['camera'][:,1] >= self.current_ac[:,1])
                self.current_dac.masked_fill_((~modified_dac) & decX_mask & decY_mask, 1)
                self.current_ac[:,0] = torch.where((~modified_dac) & decX_mask & decY_mask, self.current_ac[:,0] - self.current_delta, self.current_ac[:,0])
                self.current_ac[:,1] = torch.where((~modified_dac) & decX_mask & decY_mask, self.current_ac[:,1] - self.current_delta, self.current_ac[:,1])
                self.current_dac.masked_fill_((~modified_dac) & incX_mask & decY_mask, 2)
                self.current_ac[:,0] = torch.where((~modified_dac) & incX_mask & decY_mask, self.current_ac[:,0] + self.current_delta, self.current_ac[:,0])
                self.current_ac[:,1] = torch.where((~modified_dac) & incX_mask & decY_mask, self.current_ac[:,1] - self.current_delta, self.current_ac[:,1])
                self.current_dac.masked_fill_((~modified_dac) & decX_mask & incY_mask, 3)
                self.current_ac[:,0] = torch.where((~modified_dac) & decX_mask & incY_mask, self.current_ac[:,0] - self.current_delta, self.current_ac[:,0])
                self.current_ac[:,1] = torch.where((~modified_dac) & decX_mask & incY_mask, self.current_ac[:,1] + self.current_delta, self.current_ac[:,1])
                self.current_dac.masked_fill_((~modified_dac) & incX_mask & incY_mask, 4)
                self.current_ac[:,0] = torch.where((~modified_dac) & incX_mask & incY_mask, self.current_ac[:,0] + self.current_delta, self.current_ac[:,0])
                self.current_ac[:,1] = torch.where((~modified_dac) & incX_mask & incY_mask, self.current_ac[:,1] + self.current_delta, self.current_ac[:,1])
                self.current_delta = torch.where(~modified_dac, self.current_delta * 0.5, self.current_delta)
                self.delta_chosen |= (~modified_dac)
                #update cam step
                if cam_steps is None:
                    cam_steps = (~modified_dac).long()
                else:
                    cam_steps += (~modified_dac).long()
                #add norm
                if commited is not None:
                    norm += (~commited).float()
                else:
                    norm = torch.ones_like(fc_input[:,0], requires_grad=False)
            #add loss
            dac_clone = self.current_dac.clone()
            if COMPATIBLE:
                #compatability with old system
                _camera_mask = (dac_clone < 6) & (dac_clone >= 1)
                _disc_mask = (dac_clone < self.n_actions-1) & (dac_clone >= 6)
                dac_clone = torch.where(_camera_mask, dac_clone-1+self.n_actions-6, dac_clone)
                dac_clone = torch.where(_disc_mask, dac_clone-5, dac_clone)
            all_dacs.append(dac_clone)
            #all_odacs.append(torch.distributions.Categorical(logits=c_logits).sample())
            #check if all commited
            if commited is None:
                commited = commit_mask
            else:
                commited |= commit_mask
            if (~commited).sum().data.cpu().numpy() == 0:
                break
        if test:
            for a in self.current_rdac:
                print(a,self.current_rdac[a].data.cpu().numpy())
            print('camera',self.current_ac.data.cpu().numpy())
            print('norm', norm.data.cpu().numpy())

        #make loss
        all_targets = torch.stack(all_dacs, dim=1)
        all_inputs = torch.stack(all_inputs, dim=1)
        rand_sam = (torch.rand(norm.shape,device=self.device) * norm).long()
        input = all_inputs.gather(1, rand_sam.unsqueeze(1).unsqueeze(2).expand(-1,-1,all_inputs.shape[-1])).squeeze(1)
        target = all_targets.gather(1, rand_sam.unsqueeze(1)).squeeze(1)
        input = torch.cat([fc_input, input], dim=-1)
        c_logits = self.fc0(input.contiguous())
        loss = self.ce_loss(c_logits, target)
        probs = self.softmax(c_logits)
        entropy = -((probs*torch.log(probs)).sum(-1))
        max_probs = torch.mean(torch.max(probs,dim=1)[0])
        if is_weight is None:
            return loss.mean(), loss, entropy.mean(), all_targets, max_probs
        else:
            return (loss*is_weight).mean(), loss, entropy.mean(), all_targets, max_probs



    def sample(self, fc_input, last_action=None, current_dacs=None):
        with torch.no_grad():
            if len(fc_input.shape) > 2:
                assert False
            if self.scurrent_ac is None or self.scurrent_ac.shape[0] != fc_input.shape[0]:
                self.scurrent_ac = torch.zeros((fc_input.shape[0],2), dtype=torch.float32, device=self.device, requires_grad=False)
                self.scurrent_delta = torch.ones((fc_input.shape[0],), dtype=torch.float32, device=self.device, requires_grad=False) * self.min_delta
                self.sdelta_chosen = torch.zeros((fc_input.shape[0],), dtype=torch.bool, device=self.device, requires_grad=False)
                self.scurrent_rdac = {}
                for a in self.action_dict:
                    if a != 'camera':
                        if last_action is None or a == 'left_right' or a == 'attack_place_equip_craft_nearbyCraft_nearbySmelt':
                            self.scurrent_rdac[a] = torch.zeros((fc_input.shape[0],), dtype=torch.int64, device=self.device, requires_grad=False)
                self.scurrent_rdac_changed = {}
                for a in self.action_dict:
                    if a != 'camera':
                        self.scurrent_rdac_changed[a] = torch.zeros((fc_input.shape[0],), dtype=torch.bool, device=self.device, requires_grad=False)
            else:
                self.scurrent_ac.fill_(0.0)
                self.scurrent_delta.fill_(self.min_delta)
                for a in self.action_dict:
                    if a != 'camera':
                        if last_action is None or a == 'left_right' or a == 'attack_place_equip_craft_nearbyCraft_nearbySmelt':
                            self.scurrent_rdac[a].fill_(0)
                        self.scurrent_rdac_changed[a].fill_(False)
                self.sdelta_chosen.fill_(False)
            if last_action is not None:
                for a in self.action_dict:
                    if a != 'camera' and a != 'left_right' and a != 'attack_place_equip_craft_nearbyCraft_nearbySmelt':
                        self.scurrent_rdac[a] = last_action['last_'+a]
            commited = None
            entropy = None
            norm = None
            for i in range(self.max_samplestep):
                c_one_hot = []
                for a in self.action_dict:
                    if a != 'camera':
                        c_one_hot.append(one_hot(self.scurrent_rdac[a], self.action_dict[a].n, self.device))
                    else:
                        c_one_hot.append(self.scurrent_ac.clone())
                c_one_hot = torch.cat(c_one_hot, dim=1)
                c_logits = self.fc0(torch.cat([fc_input,
                                               c_one_hot,
                                               self.scurrent_delta.unsqueeze(-1).clone(), 
                                               self.sdelta_chosen.unsqueeze(-1).float().clone()], dim=-1))
                probs = self.softmax(c_logits)
                if entropy is None:
                    entropy = -(probs*torch.log(probs)).sum(-1)
                    norm = torch.ones_like(c_logits[:,0], requires_grad=False)
                else:
                    entropy += -(probs*torch.log(probs)).sum(-1) * (~commited).float()
                    norm += (~commited).float()
                if current_dacs is None:
                    current_dac = torch.distributions.Categorical(logits=c_logits)
                    current_dac = current_dac.sample()
                else:
                    current_dac = current_dacs[:,i]
                if COMPATIBLE:
                    #compatability with old system
                    _camera_mask = (current_dac < self.n_actions-1) & (current_dac >= self.n_actions-6)
                    _disc_mask = (current_dac < self.n_actions-6) & (current_dac >= 1)
                    current_dac = torch.where(_camera_mask, current_dac-self.n_actions+7, current_dac)
                    current_dac = torch.where(_disc_mask, current_dac+5, current_dac)

                #check commit
                commit_mask = (current_dac == 0)
                if commited is None:
                    commited = commit_mask.clone()
                else:
                    commited |= commit_mask
                if (~commit_mask).sum() == 0:
                    break
                current_dac.masked_fill_(commited, 0)
                #check discrete actions
                for a in self.action_dict:
                    if a != 'camera':
                        disc_mask = (~self.scurrent_rdac_changed[a]) & (current_dac >= self.action_ranges[a][0]) & (current_dac < self.action_ranges[a][1])
                        toggle_mask = disc_mask & (self.scurrent_rdac[a] == (current_dac - self.action_ranges[a][0] + 1))
                        self.scurrent_rdac[a] = torch.where(toggle_mask, torch.zeros_like(self.scurrent_rdac[a]), torch.where(disc_mask, current_dac - self.action_ranges[a][0] + 1, self.scurrent_rdac[a]))
                        self.scurrent_rdac_changed[a] |= disc_mask
                #check delta
                delta_mask = (~self.sdelta_chosen) & (current_dac == 5)
                self.scurrent_delta = torch.where(delta_mask, self.scurrent_delta * 2.0, self.scurrent_delta)
                self.sdelta_chosen |= (self.scurrent_delta >= 0.5)
                self.scurrent_delta = torch.where(self.scurrent_delta > 0.5, torch.ones_like(self.scurrent_delta) * 0.5, self.scurrent_delta)
                #check camera
                self.scurrent_ac[:,0] = torch.where((current_dac == 1) | (current_dac == 3), self.scurrent_ac[:,0] - self.scurrent_delta, self.scurrent_ac[:,0])
                self.scurrent_ac[:,0] = torch.where((current_dac == 2) | (current_dac == 4), self.scurrent_ac[:,0] + self.scurrent_delta, self.scurrent_ac[:,0])
                self.scurrent_ac[:,1] = torch.where((current_dac == 1) | (current_dac == 2), self.scurrent_ac[:,1] - self.scurrent_delta, self.scurrent_ac[:,1])
                self.scurrent_ac[:,1] = torch.where((current_dac == 3) | (current_dac == 4), self.scurrent_ac[:,1] + self.scurrent_delta, self.scurrent_ac[:,1])
                camera_mask = (current_dac >= 1) & (current_dac < 5)
                self.scurrent_delta = torch.where(camera_mask, self.scurrent_delta * 0.5, self.scurrent_delta)
                self.sdelta_chosen |= camera_mask
            rand_deltaX = torch.distributions.Uniform(-self.scurrent_delta*2.0, self.scurrent_delta*2.0).sample()
            rand_deltaY = torch.distributions.Uniform(-self.scurrent_delta*2.0, self.scurrent_delta*2.0).sample()
            self.scurrent_ac[:,0] = torch.where(~self.sdelta_chosen, self.scurrent_ac[:,0], self.scurrent_ac[:,0] + rand_deltaX)
            self.scurrent_ac[:,1] = torch.where(~self.sdelta_chosen, self.scurrent_ac[:,1], self.scurrent_ac[:,1] + rand_deltaY)
            entropy /= norm
        return self.scurrent_rdac, self.scurrent_ac.clamp(-1.0,1.0), entropy.mean()


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
            state_dict = torch.load(filename + 'virtualPGmultiaction_mmd')
            #state_dict2 = {n: v for n, v in state_dict.items() if not n.startswith('disc')}
            #upd = {n.replace('.bias','.module.bias').replace('.weight','.module.weight_bar'): v for n, v in state_dict.items() if n.startswith('disc')}
            #state_dict2.update(upd)
            #self.load_state_dict(state_dict2, False)
            self.load_state_dict(state_dict)
            if 'Lambda' in self._model:
                self._model['Lambda'] = torch.load(filename + 'virtualPGmultiaction_mmdLam')
        else:
            torch.save(self._model['modules'].state_dict(), filename + 'virtualPGmultiaction_mmd')
            if 'Lambda' in self._model:
                torch.save(self._model['Lambda'], filename + 'virtualPGmultiaction_mmdLam')

    def generate_embed(self, state_povs):
        with torch.no_grad():
            if len(state_povs.shape) == 3:
                state_povs = state_povs[None, :]
            state_povs = variable(state_povs)
            state_povs = (state_povs.permute(0, 3, 1, 2)/255.0)*2.0 - 1.0
            pov_embed = tanh_or_not(self._model['modules']['pov_embed_no_tanh'](state_povs))
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
        modules['memory'] = MemoryModel(self.args, self.cnn_embed_dim, self.past_time[-self.args.num_past_times:])
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
        self.random_dim = 64
        modules['actions_predict'] = torch.nn.Sequential(torch.nn.Linear(self.cnn_embed_dim + self.memory_embed + self.state_dim + self.random_dim, self.args.hidden),
                                                                Lambda(lambda x: swish(x)),
                                                                torch.nn.Linear(self.args.hidden, self.actions_embed_dim))
        modules['actions_predict_real'] = FullDiscretizerWLast(self.actions_embed_dim + self.cnn_embed_dim + self.memory_embed + self.state_dim, torch.device("cuda") if torch.cuda.is_available() and CUDA else torch.device("cpu"), self.args, self.action_dict)

        #aux prediction
        modules['r_predict'] = torch.nn.Sequential(torch.nn.Linear(self.actions_embed_dim + self.cnn_embed_dim + self.memory_embed + self.state_dim, self.args.hidden),
                                                   Lambda(lambda x: swish(x)),
                                                   torch.nn.Linear(self.args.hidden, 1))
        modules['v_predict_short'] = torch.nn.Linear(self.cnn_embed_dim, 2)
        modules['v_predict_long'] = torch.nn.Linear(self.cnn_embed_dim, 2)

        #infoDIM loss
        if INFODIM:
            modules['disc'] = torch.nn.Sequential(SpectralNorm(torch.nn.Linear(self.cnn_embed_dim*3+self.actions_embed_dim+1, self.args.hidden)), 
                                                  Lambda(lambda x: swish(x)),#torch.nn.LeakyReLU(0.1), 
                                                  SpectralNorm(torch.nn.Linear(self.args.hidden, self.args.hidden)), 
                                                  Lambda(lambda x: swish(x)),
                                                  SpectralNorm(torch.nn.Linear(self.args.hidden, 1)))#
            modules['disc_actions'] = torch.nn.Sequential(SpectralNorm(torch.nn.Linear(self.cnn_embed_dim + self.memory_embed + self.state_dim + self.actions_embed_dim, self.args.hidden)),
                                                  Lambda(lambda x: swish(x)),#torch.nn.LeakyReLU(0.1), 
                                                  SpectralNorm(torch.nn.Linear(self.args.hidden, 1)))#
            if self.args.ganloss == "fisher":
                lam = torch.zeros((2,), requires_grad=True)

        #reconstruction
        if RECONSTRUCTION:
            modules['generate_pov'] = GenNet(self.args, self.state_shape, self.cnn_embed_dim)

        #state prediction
        modules['state_predict_hidden'] = torch.nn.Sequential(torch.nn.Linear(self.actions_embed_dim + self.cnn_embed_dim + self.memory_embed + self.state_dim +self.random_dim+1, self.args.hidden),
                                                       Lambda(lambda x: swish(x)),
                                                       torch.nn.Linear(self.args.hidden, self.args.hidden),
                                                       Lambda(lambda x: swish(x)))
        modules['state_predict'] = torch.nn.Linear(self.args.hidden, self.cnn_embed_dim)
        modules['Q_value'] = torch.nn.Linear(self.args.hidden, 1) #input is state_predict_hidden

        
        self.revidx = [i for i in range(self.args.num_past_times-1, -1, -1)]
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

        self.gammas = GAMMA ** torch.arange(0, self.max_predict_range, device=self.device).float().unsqueeze(0).expand(self.args.er, -1)

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

    def train_Q(self, current_all_embed, next_past_embeds, next_state, next_pov_embed, rewards, next_actions=None):
        with torch.no_grad():
            #make next memory embed
            next_memory_embed = self.get_memory_embed(next_past_embeds, next_pov_embed)
            
            if next_actions is None:
                #predict next action
                next_tembed = [next_pov_embed, next_memory_embed]
                for o in self.state_shape:
                    if o in self.state_keys and o != 'pov':
                        next_tembed.append(next_state[o])
                next_tembed = torch.cat(next_tembed, dim=-1)
                next_actions_embed = tanh_or_not(self._model['modules']['actions_predict'](torch.cat([next_tembed, torch.zeros([next_tembed.shape[0], self.random_dim], device=next_tembed.device).normal_()], dim=-1)))
            else:
                #predict next action
                next_tembed = [next_pov_embed, next_memory_embed]
                for o in self.state_shape:
                    if o in self.state_keys and o != 'pov':
                        next_tembed.append(next_state[o])
                next_tembed = torch.cat(next_tembed, dim=-1)


                #make all_embed
                actions_one_hot = []
                for a in self.action_dict:
                    if a != 'camera':
                        actions_one_hot.append(one_hot(next_actions[a], self.action_dict[a].n, self.device).view(-1, self.action_dict[a].n*self.max_predict_range))
                    else:
                        actions_one_hot.append(next_actions[a].reshape(-1, 2*self.max_predict_range))
                if AUX_CAMERA_LOSS:
                    aux_camera_target1 = ((torch.abs(next_actions['camera'][:,:,0]) < 1e-5) & (torch.abs(next_actions['camera'][:,:,1]) < 1e-5)).long()
                    aux_camera_target2 = (next_actions['camera'][:,:,1] > next_actions['camera'][:,:,0]).long()
                    aux_camera_target3 = (next_actions['camera'][:,:,1] > -next_actions['camera'][:,:,0]).long()
                    aux_camera_target = (1 - aux_camera_target1) * (aux_camera_target2 + aux_camera_target3 * 2 + 1)
                    actions_one_hot.append(one_hot(aux_camera_target, 5, self.device).view(-1, 5*self.max_predict_range))
                else:
                    aux_camera_target = None
                actions_one_hot = torch.cat(actions_one_hot, dim=1)
                tembed_wactions = torch.cat([actions_one_hot, next_tembed], dim=-1)
                all_embed = self._model['modules']['all_embed'](tembed_wactions)

                #make action compression loss
                actions_embed_no_tanh = self._model['modules']['actions_enc_no_tanh'](all_embed)
                next_actions_embed = tanh_or_not(actions_embed_no_tanh)

            
            if False:
                #decode actions
                actions_logits = self._model['modules']['actions_dec'](torch.cat([next_tembed, next_actions_embed], dim=-1))
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
                        a_dict[a] = torch.distributions.Categorical(logits=actions_logits[i]).sample()
                        i += 1
                    else:
                        a_dict[a] = actions_camera
                if AUX_CAMERA_LOSS:
                    aux_camera = torch.distributions.Categorical(logits=actions_logits[-1]).sample()
                    a_dict['camera'] = torch.where((aux_camera == 0).unsqueeze(-1).expand(-1, -1, 2), torch.zeros_like(a_dict['camera']), a_dict['camera'])

                #make next all_embed
                actions_one_hot = []
                for a in self.action_dict:
                    if a != 'camera':
                        actions_one_hot.append(one_hot(a_dict[a], self.action_dict[a].n, self.device).view(-1, self.action_dict[a].n*self.max_predict_range))
                    else:
                        actions_one_hot.append(a_dict[a].reshape(-1, 2*self.max_predict_range))
                if AUX_CAMERA_LOSS:
                    aux_camera_target1 = ((torch.abs(a_dict['camera'][:,:,0]) < 1e-5) & (torch.abs(a_dict['camera'][:,:,1]) < 1e-5)).long()
                    aux_camera_target2 = (a_dict['camera'][:,:,1] > a_dict['camera'][:,:,0]).long()
                    aux_camera_target3 = (a_dict['camera'][:,:,1] > -a_dict['camera'][:,:,0]).long()
                    aux_camera_target = (1 - aux_camera_target1) * (aux_camera_target2 + aux_camera_target3 * 2 + 1)
                    actions_one_hot.append(one_hot(aux_camera_target, 5, self.device).view(-1, 5*self.max_predict_range))
                else:
                    aux_camera_target = None
                actions_one_hot = torch.cat(actions_one_hot, dim=1)

        if False:
            tembed_wactions = torch.cat([actions_one_hot, next_tembed.detach()], dim=-1)
            next_all_embed = self._model['modules']['all_embed'](tembed_wactions)
        else:
            next_all_embed = torch.cat([next_tembed.detach(), next_actions_embed.detach()], dim=-1)

        #make state predict hidden
        ran_sample = torch.zeros([next_all_embed.shape[0], self.random_dim], device=next_all_embed.device).normal_()
        next_hidden = self._model['modules']['state_predict_hidden'](torch.cat([next_all_embed, ran_sample, torch.ones([next_all_embed.shape[0], 1], device=next_all_embed.device, dtype=torch.float32)], dim=-1))

        #make next Q value
        next_Q = self._model['modules']['Q_value'](next_hidden)[:,0]
        target = (rewards[:,:self.max_predict_range] * self.gammas).sum(1) + (GAMMA ** self.max_predict_range) * next_Q

        #current Q value
        ran_sample = torch.zeros([current_all_embed.shape[0], self.random_dim], device=current_all_embed.device).normal_()
        hidden = self._model['modules']['state_predict_hidden'](torch.cat([current_all_embed, ran_sample, torch.ones([next_all_embed.shape[0], 1], device=next_all_embed.device, dtype=torch.float32)], dim=-1))
        Q = self._model['modules']['Q_value'](hidden)[:,0]

        #return (Q - (0.5 * target + 0.5 * Q).detach()).pow(2).mean(), Q.mean()
        return (Q - target).pow(2).mean(), Q
        


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
        r_offset = variable(input['r_offset'], True).float()
        r_offset /= self.max_predict_range
        r_offset = r_offset.unsqueeze(1)
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
        assert states['pov'].shape[1] == 3

        with torch.no_grad():
            past_embeds = states['pov_embed'][:,self.zero_time_point-self.args.num_past_times:self.zero_time_point]
            #mask out from other episode
            past_done = done[:,self.zero_time_point-self.args.num_past_times:self.zero_time_point]
            past_done = past_done[:,self.revidx]
            past_done = torch.cumsum(past_done,1).clamp(max=1).bool()
            past_done = past_done[:,self.revidx].unsqueeze(-1).expand(-1,-1,past_embeds.shape[-1])
            past_embeds.masked_fill_(past_done, 0.0)

            
            next_past_embeds = states['pov_embed'][:,:self.zero_time_point-self.args.num_past_times]
            #mask out from other episode
            past_done = done[:,:self.zero_time_point-self.args.num_past_times]
            past_done = past_done[:,self.revidx]
            past_done = torch.cumsum(past_done,1).clamp(max=1).bool()
            past_done = past_done[:,self.revidx].unsqueeze(-1).expand(-1,-1,past_embeds.shape[-1])
            next_past_embeds.masked_fill_(past_done, 0.0)

            #TODO try masking stuff in MultiHeadAttention??
            
        #get pov embeds
        states_gen_embed = states['pov']
        assert states_gen_embed.shape[1] == 3
        embed_no_tanh = self._model['modules']['pov_embed_no_tanh'](states_gen_embed)
        next_embed_no_tanh = embed_no_tanh[:,2]#pov_embed after actions [0:self.max_predict_range]
        current_embed_no_tanh = embed_no_tanh[:,0]
        r_next_embed_no_tanh = embed_no_tanh[:,1]

        #get current memory_embed
        memory_embed = self.get_memory_embed(past_embeds, tanh_or_not(current_embed_no_tanh))

        #get action and aux loss
        loss, all_embed, actions_embed, rdict = self.prediction_loss(tanh_or_not(current_embed_no_tanh), memory_embed, current_state, current_actions, reward[:,self.zero_time_point:])

        #get state prediction loss
        state_pred_loss, mean, std, dis_loss, gen_loss, current_state_predict_hidden = self.state_prediction(all_embed, next_embed_no_tanh, actions_embed, current_embed_no_tanh, r_offset)
        loss += state_pred_loss
        rdict['state_pred_loss'] = state_pred_loss
        rdict['dis_loss'] = dis_loss
        if not isinstance(gen_loss, int):
            rdict['gen_loss'] = gen_loss

        Q_loss, mean_Q = self.train_Q(all_embed, next_past_embeds, next_state, tanh_or_not(r_next_embed_no_tanh), reward[:,self.zero_time_point:self.zero_time_point+self.max_predict_range], next_actions)
        rdict['Q_loss'] = Q_loss
        rdict['mean_Q'] = mean_Q.mean()
        loss += Q_loss * 10.0

        #get reconstruction loss
        if RECONSTRUCTION:
            rec_pov = self._model['modules']['generate_pov'](tanh_or_not(current_embed_no_tanh))
            rec_loss = self._model['loss'](rec_pov, states_gen_embed[:,0]) * 10.0 + self._model['l1loss'](rec_pov, states_gen_embed[:,0])
            loss += rec_loss
            rdict['rec_loss'] = rec_loss
            
            if False:
                state_pred_sample = mean#torch.distributions.Normal(mean, std).sample()
                #for p in self._model['modules']['generate_pov'].parameters():
                #    p.requires_grad_(False)
                rec_pov2 = self._model['modules']['generate_pov'](tanh_or_not(state_pred_sample))
                #for p in self._model['modules']['generate_pov'].parameters():
                #    p.requires_grad_(True)
                rec_loss2 = (self._model['loss'](rec_pov2, states_gen_embed[:,2]) * 10.0 + self._model['l1loss'](rec_pov, states_gen_embed[:,2])) * 0.1
                loss += rec_loss2
                rdict['rec_loss'] += rec_loss2

            if False:
                state_pred_sample = mean#torch.distributions.Normal(mean, std).sample()
                print(r_offset[0,0].item())
                print(mean_Q[0].item())
                print(next_embed_no_tanh[0,0:4].data.cpu().numpy())
                print(mean[0,0:4].data.cpu().numpy(),std[0,0:4].data.cpu().numpy())
                rec_pov2 = self._model['modules']['generate_pov'](tanh_or_not(state_pred_sample))
                preP = np.concatenate([(states_gen_embed[0,0].permute(1, 2, 0)*0.5+0.5).data.cpu().numpy(),
                                       (rec_pov[0].permute(1, 2, 0)*0.5+0.5).data.cpu().numpy()], axis=1)
                aftP = np.concatenate([(states_gen_embed[0,2].permute(1, 2, 0)*0.5+0.5).data.cpu().numpy(),
                                       (rec_pov2[0].permute(1, 2, 0)*0.5+0.5).data.cpu().numpy()], axis=1)
                plt.imshow(np.concatenate([preP, aftP], axis=0))
                plt.show()
        #make pov mmd loss
        #pov_mmd_loss = compute_mmd_gauss(current_embed_no_tanh) * 1000.0
        #loss += pov_mmd_loss
        #rdict['pov_mmd_loss'] = pov_mmd_loss

        if INFODIM:
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
            rdict['Lambda'] = self._model['Lambda']

        #priority buffer stuff
        if 'tidxs' in input:
            #prioritize after bc_loss
            assert False #not implemented
            rdict['prio_upd'] = [{'error': rdict['HIER PRIO LOSS EINGEBEN'].data.cpu().clone(),
                                    'replay_num': 0,
                                    'worker_num': worker_num[0]}]

        for d in rdict:
            if d != 'prio_upd' and not isinstance(rdict[d], str) and not isinstance(rdict[d], int):
                rdict[d] = rdict[d].data.cpu().numpy()

        self.train_iter += 1
        return rdict


    def state_prediction(self, all_embed, next_embed_no_tanh, actions_embed, current_embed_no_tanh, r_offset):
        ran_sample = torch.zeros([all_embed.shape[0], self.random_dim], device=all_embed.device).normal_()
        hidden = self._model['modules']['state_predict_hidden'](torch.cat([all_embed, ran_sample, r_offset], dim=-1))
        mean = self._model['modules']['state_predict'](hidden)
        
        if INFODIM:
            real_pairs = torch.cat([actions_embed.detach(), tanh_or_not(current_embed_no_tanh).detach(), tanh_or_not(next_embed_no_tanh).detach(), tanh_or_not(mean).detach(), r_offset], dim=-1)
            fake_pairs = torch.cat([actions_embed.detach(), tanh_or_not(current_embed_no_tanh).detach(), tanh_or_not(mean).detach(), tanh_or_not(next_embed_no_tanh).detach(), r_offset], dim=-1)
            real_scores = self._model['modules']['disc'](real_pairs)
            fake_scores = self._model['modules']['disc'](fake_pairs)
            
            if self.args.ganloss == 'hinge':
                dis_loss = torch.nn.ReLU()(1.0 - real_scores).mean() + torch.nn.ReLU()(1.0 + fake_scores).mean()
            elif self.args.ganloss == 'wasserstein' or self.args.ganloss == 'fisher':
                dis_loss = -real_scores.mean() + fake_scores.mean()
                if self.args.ganloss == 'fisher':
                    constraint = (1 - (0.5*(real_scores**2).mean() + 0.5*(fake_scores**2).mean()))
                    dis_loss += -self._model['Lambda'][0] * constraint + (self.args.rho/2.0) * constraint**2
            else:
                dis_loss = torch.nn.BCEWithLogitsLoss()(real_scores, torch.ones(list(real_scores.shape[:-1]) + [1]).cuda()) + \
                        torch.nn.BCEWithLogitsLoss()(fake_scores, torch.zeros(list(fake_scores.shape[:-1]) + [1]).cuda())

            if self.train_iter % (10 if dis_loss.item() > 0.1 else 2) == 0:
                fake_pairs = torch.cat([actions_embed.detach(), tanh_or_not(current_embed_no_tanh).detach(), tanh_or_not(mean), tanh_or_not(next_embed_no_tanh).detach(), r_offset], dim=-1)
                for p in self._model['modules']['disc'].parameters():
                    p.requires_grad_(False)
                for l in self._model['modules']['disc']:
                    if isinstance(l, SpectralNorm):
                        l.make_update = False
                fake_scores = self._model['modules']['disc'](fake_pairs)
                for p in self._model['modules']['disc'].parameters():
                    p.requires_grad_(True)
                for l in self._model['modules']['disc']:
                    if isinstance(l, SpectralNorm):
                        l.make_update = True

                if self.args.ganloss == 'hinge' or self.args.ganloss == 'wasserstein' or self.args.ganloss == 'fisher':
                    gen_loss = -fake_scores.mean()
                else:
                    gen_loss = torch.nn.BCEWithLogitsLoss()(fake_scores, torch.ones(list(fake_scores.shape[:-1]) + [1]).cuda())
            else:
                gen_loss = 0

        return dis_loss + gen_loss, mean, mean, dis_loss, gen_loss, hidden
    
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
        soft_diff = 0.02
        loss = torch.where(diff < soft_diff, 0.5 * diff ** 2, diff + 0.5 * soft_diff ** 2 - soft_diff).sum(-1).mean() * 10.0
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
        reward[:,:self.max_predict_range] = reward[:,:self.max_predict_range] * self.gammas
        tembed = [pov_embed, memory_embed]
        for o in self.state_shape:
            if o in self.state_keys and o != 'pov':
                tembed.append(current_state[o])
        tembed = torch.cat(tembed, dim=-1)

        #make all_embed
        current_action = {}
        actions_one_hot = []
        for a in self.action_dict:
            current_action[a] = current_actions[a][:,0]
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
        actions_embed = tanh_or_not(actions_embed_no_tanh)
        all_embed = torch.cat([tembed, actions_embed], dim=-1)
        actions_logits = self._model['modules']['actions_dec'](torch.cat([tembed, actions_embed], dim=-1))
        action_compression_loss = self.get_actions_loss(actions_logits, current_actions, aux_camera_target)
        loss = action_compression_loss.clone()

        #first action
        comp_bc_loss, _, _, _, _ = self._model['modules']['actions_predict_real'].get_loss_BC(torch.cat([tembed, actions_embed], dim=-1), current_action)
        loss += comp_bc_loss * 20.0
        #with mmd loss
        #loss += compute_mmd_gauss(actions_embed_no_tanh) * 10.0
        
        #make action predict loss
        pred_actions_embed_no_tanh = self._model['modules']['actions_predict'](torch.cat([tembed, torch.zeros([tembed.shape[0], self.random_dim], device=tembed.device).normal_()], dim=-1))
        #train actions GAN
        real_pairs = torch.cat([tembed.detach(), tanh_or_not(actions_embed_no_tanh).detach()], dim=-1)
        fake_pairs = torch.cat([tembed.detach(), tanh_or_not(pred_actions_embed_no_tanh).detach()], dim=-1)
        real_scores = self._model['modules']['disc_actions'](real_pairs)
        fake_scores = self._model['modules']['disc_actions'](fake_pairs)
        if self.args.ganloss == 'hinge':
            dis_loss = torch.nn.ReLU()(1.0 - real_scores).mean() + torch.nn.ReLU()(1.0 + fake_scores).mean()
        elif self.args.ganloss == 'wasserstein' or self.args.ganloss == 'fisher':
            dis_loss = -real_scores.mean() + fake_scores.mean()
            if self.args.ganloss == 'fisher':
                constraint = (1 - (0.5*(real_scores**2).mean() + 0.5*(fake_scores**2).mean()))
                dis_loss += -self._model['Lambda'][1] * constraint + (self.args.rho/2.0) * constraint**2
        else:
            dis_loss = torch.nn.BCEWithLogitsLoss()(real_scores, torch.ones(list(real_scores.shape[:-1]) + [1]).cuda()) + \
                    torch.nn.BCEWithLogitsLoss()(fake_scores, torch.zeros(list(fake_scores.shape[:-1]) + [1]).cuda())
        loss += dis_loss

        if self.train_iter % (10 if dis_loss.item() > 0.1 else 2) == 0:
            #action_predict_loss = self._model['loss'](pred_actions_embed_no_tanh, actions_embed_no_tanh.detach()) * 0.1
            fake_pairs = torch.cat([tembed.detach(), tanh_or_not(pred_actions_embed_no_tanh)], dim=-1)
            for p in self._model['modules']['disc_actions'].parameters():
                p.requires_grad_(False)
            for l in self._model['modules']['disc_actions']:
                if isinstance(l, SpectralNorm):
                    l.make_update = False
            fake_scores = self._model['modules']['disc_actions'](fake_pairs)
            for p in self._model['modules']['disc_actions'].parameters():
                p.requires_grad_(True)
            for l in self._model['modules']['disc_actions']:
                if isinstance(l, SpectralNorm):
                    l.make_update = True

            if self.args.ganloss == 'hinge' or self.args.ganloss == 'wasserstein' or self.args.ganloss == 'fisher':
                gen_loss = -fake_scores.mean()
            else:
                gen_loss = torch.nn.BCEWithLogitsLoss()(fake_scores, torch.ones(list(fake_scores.shape[:-1]) + [1]).cuda())
            action_predict_loss = gen_loss
            loss += action_predict_loss
        else:
            action_predict_loss = 0

        #direct loss
        bc_loss, _, _, _, _ = self._model['modules']['actions_predict_real'].get_loss_BC(torch.cat([tembed, tanh_or_not(pred_actions_embed_no_tanh)], dim=-1), current_action)
        loss += bc_loss * 20.0

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

        return loss, all_embed, actions_embed, {'bc_loss': bc_loss,
                                                'action_comp_loss': action_compression_loss,
                                                'action_predict_loss': action_predict_loss,
                                                'r_loss': r_loss,
                                                'v_short_loss': v_short_loss,
                                                'v_long_loss': v_long_loss,
                                                'actions_dis_loss': dis_loss}

    def reset(self):
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
            pov_embed = tanh_or_not(self._model['modules']['pov_embed_no_tanh'](tstate['pov']))
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
            actions_embed_no_tanh = self._model['modules']['actions_predict'](torch.cat([tembed, torch.zeros([tembed.shape[0], self.random_dim], device=tembed.device).normal_()], dim=-1))
            actions_embed = tanh_or_not(actions_embed_no_tanh)
            if False:
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
            else:
                r_action = {}
                o_action = {}
                r_action, conaction, entropy = self._model['modules']['actions_predict_real'].sample(torch.cat([tembed, actions_embed], dim=-1))
                for a in r_action:
                    o_action[a] = r_action[a].data.cpu().numpy()[0]
                o_action['camera'] = conaction.data.cpu().numpy()[0,:]
                o_action['camera'] = self.map_camera(o_action['camera'])

                action = OrderedDict()
                for a in self.action_dict:
                    action[a] = o_action[a]
                return action, {'pov_embed': pov_embed.data.cpu(),
                                'entropy': entropy.data.cpu().numpy()}




