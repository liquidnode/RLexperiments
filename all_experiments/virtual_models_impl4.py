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

GAMMA = 0.99
LAMBDA = 0.95
COMPATIBLE = False
USE_LOOKAHEAD = False
DISABLE_FUTURE_POLICY_TRAINING = True

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

class VisualNetVariational(torch.nn.Module):
    def __init__(self, state_shape, args, cont_dim, disc_dims):
        super(VisualNetVariational, self).__init__()
        self.args = args
        self.state_shape = state_shape

        self.cont_dim = cont_dim
        self.disc_dims = disc_dims
        encnn_layers = []
        self.embed_dim = self.args.embed_dim
        self.temperature = self.args.temperature
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
        tencnn_layers.append(torch.nn.Linear(in_channels*(end_size**2), self.cont_dim*2+sum(self.disc_dims)))
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
        if time_tmp is not None:
            x = x.view(-1, time_tmp, self.cont_dim*2+sum(self.disc_dims))
        cont, discs = torch.split(x, self.cont_dim*2, dim=-1)
        mu, log_var = torch.split(cont, self.cont_dim, dim=-1)
        sdiscs = list(torch.split(discs, self.disc_dims, dim=-1))
        for i in range(len(sdiscs)):
            sdiscs[i] = self.log_softmax(sdiscs[i])
        discs = torch.cat(sdiscs, dim=-1)
        return mu, log_var, discs

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
        disc_samples = [F.gumbel_softmax(d, self.temperature, (not training) or hard) for d in sdiscs]


        return torch.cat([cont_sample]+disc_samples, dim=-1)



class VirtualPGVAE():
    def __init__(self, state_shape, action_dict, args, time_deltas):
        assert isinstance(state_shape, dict)
        self.state_shape = state_shape
        self.action_dict = copy.deepcopy(action_dict)
        self.state_keys = ['state', 'pov', 'history_action', 'history_reward', 'orientation']
        self.args = args
        
        self.action_dim = 2
        for a in action_dict:
            if a != 'camera':
                self.action_dim += action_dict[a].n
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

        self.train_iter = 0
        #self.clustering_batch = 1024*4
        #self.cluster_dim = 64
        #self.d_cluster = 2
        #self.num_clusters = 16
        self.num_random_slices = 0
        self.disc_embed_dims = [8, 8, 4, 4, 4, 4, 2, 2, 2, 2]
        self.sum_logs = float(sum([np.log(d) for d in self.disc_embed_dims]))
        self.total_disc_embed_dim = sum(self.disc_embed_dims)
        self.continuous_embed_dim = self.args.embed_dim-self.total_disc_embed_dim

        self.cont_capacity = (0.0, 15.0, 1e6, 50.0)
        self.disc_capacity = (0.0, 10.0, 1e6, 50.0)
        assert self.disc_capacity[1] < sum([float(np.log(disc_dim)) for disc_dim in self.disc_embed_dims])

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
            self.load_state_dict(torch.load(filename + '-virtualPGvariational'))
            if 'Lambda' in self._model:
                self._model['Lambda'] = torch.load(filename + '-virtualPGvariationalLam')
            #self.train_iter = int(0.03 * 1e5)
            self.train_iter = torch.load(filename + 'train_iter')
        else:
            torch.save(self._model['modules'].state_dict(), filename + '-virtualPGvariational')
            if 'Lambda' in self._model:
                torch.save(self._model['Lambda'], filename + '-virtualPGvariationalLam')
            torch.save(self.train_iter, filename + 'train_iter')

    def generate_embed(self, state_povs):
        with torch.no_grad():
            if len(state_povs.shape) == 3:
                state_povs = state_povs[None, :]
            state_povs = variable(state_povs)
            state_povs = (state_povs.permute(0, 3, 1, 2)/255.0)*2.0 - 1.0
            mu, log_var, discs = self._model['modules']['pov_embed'](state_povs)
            return {'pov_embed_mu': mu.data.cpu(),
                    'pov_embed_log_var': log_var.data.cpu(),
                    'pov_embed_log_alphas': discs.data.cpu()}

    def embed_desc(self):
        desc = {}
        desc['pov_embed_mu'] = {'compression': False, 'shape': [self.continuous_embed_dim], 'dtype': np.float32}
        desc['pov_embed_log_var'] = {'compression': False, 'shape': [self.continuous_embed_dim], 'dtype': np.float32}
        desc['pov_embed_log_alphas'] = {'compression': False, 'shape': [self.total_disc_embed_dim], 'dtype': np.float32}
        return desc

    def _make_model(self):
        modules = torch.nn.ModuleDict()
        modules['pov_embed'] = VisualNetVariational(self.state_shape, self.args, self.continuous_embed_dim, self.disc_embed_dims)
        self.cnn_embed_dim = self.args.embed_dim
        #modules['cluster_predict'] = torch.nn.Sequential(torch.nn.Linear(self.pov_embed_dim, 128),
        #                                                 Lambda(lambda x: swish(x)),
        #                                                 torch.nn.Linear(128, self.d_cluster * self.num_clusters))

        if 'state' in self.state_shape:
            self.state_dim = self.state_shape['state'][0]
        else:
            self.state_dim = 0
        for o in self.state_shape:
            if 'history' in o or o == 'orientation':
                self.state_dim += self.state_shape[o][0]
        self.memory_embed = self.args.hidden
        #input to memory is state_embeds
        modules['memory'] = torch.nn.GRU(input_size=self.cnn_embed_dim, hidden_size=self.memory_embed, batch_first=True)
        modules['state_hidden'] = torch.nn.Sequential(torch.nn.Linear(self.cnn_embed_dim+self.state_dim+self.memory_embed, self.args.hidden),
                                                      Lambda(lambda x: swish(x)))
        modules['action_predict'] = FullDiscretizerWLast(self.args.hidden, None, self.args, self.action_dict)
        
        modules['r_predict'] = torch.nn.Linear(self.cnn_embed_dim, 2)
        modules['v_predict_short'] = torch.nn.Linear(self.cnn_embed_dim, 2)
        modules['v_predict_long'] = torch.nn.Linear(self.cnn_embed_dim, 2)

        #infoDIM loss
        modules['disc'] = torch.nn.Sequential(torch.nn.Linear(self.cnn_embed_dim*2, self.args.hidden), 
                                              Lambda(lambda x: swish(x)),#torch.nn.LeakyReLU(0.1), 
                                              torch.nn.Linear(self.args.hidden, 1, False))#SpectralNorm(
        if self.args.ganloss == "fisher":
            lam = torch.zeros((1,), requires_grad=True)

        #predict state from actions
        modules['state_predict_memory'] = torch.nn.GRU(input_size=self.action_dim, hidden_size=self.memory_embed, batch_first=True)
        modules['state_predict'] = torch.nn.Sequential(torch.nn.Linear(self.memory_embed, self.continuous_embed_dim*2+self.total_disc_embed_dim))

        if torch.cuda.is_available() and CUDA:
            modules = modules.cuda()
            if self.args.ganloss == "fisher":
                lam = lam.cuda()
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        modules['action_predict'].device = self.device

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
        if USE_LOOKAHEAD:
            optimizer = Lookahead(optimizer)

        model = {'modules': modules, 
                 'opt': optimizer, 
                 'ce_loss': ce_loss,
                 'l1loss': l1loss,
                 'l2loss': l2loss,
                 'loss': loss}
        if len(nec_value_params) > 0:
            nec_optimizer = torch.optim.Adam(nec_value_params, self.args.nec_lr)
            model['nec_opt'] = nec_optimizer
        if self.args.ganloss == "fisher":
            model['Lambda'] = lam
        return model
    
    def inverse_map_camera(self, x):
        return x / 180.0

    def map_camera(self, x):
        return x * 180.0

    def train(self, input):
        self.train_iter += 1
        return {}

    def pretrain(self, input):
        for n in self._model:
            if 'opt' in n:
                self._model[n].zero_grad()
        embed_keys = []
        for k in self.embed_desc():
            embed_keys.append(k)
        states = {}
        lastaction = {}
        for k in input:
            if k in self.state_keys or k in embed_keys:
                states[k] = variable(input[k])
            if 'last_' in k:
                lastaction[k] = variable(input[k], k != 'last_camera')
                if k != 'last_camera':
                    lastaction[k] = lastaction[k].long()
        actions = {'camera': self.inverse_map_camera(variable(input['camera']))}
        for a in self.action_dict:
            if a != 'camera':
                actions[a] = variable(input[a], True).long()
        reward = variable(input['reward'])
        done = variable(input['done'])
        done = done.float()
        future_done = torch.cumsum(done[:,self.zero_time_point:],1).clamp(max=1.0)
        reward[:,(self.zero_time_point+1):] = reward[:,(self.zero_time_point+1):] * (1.0 - future_done[:,:-1]) #mask out future reward from different episode
        #TODO mask out past different episode
        past_done = done[:self.zero_time_point]
        assert past_done.sum().data.cpu().numpy() < 1e-5

        is_weight = None
        if 'is_weight' in input:
            is_weight = variable(input['is_weight'])
        #normalize and transpose pov state
        states['pov'] = (states['pov'].float()/255.0)*2.0 - 1.0
        states['pov'] = states['pov'].permute(0, 1, 4, 2, 3)
        current_state = {}
        for k in states:
            current_state[k] = states[k][:,self.zero_time_point].unsqueeze(1)
        current_action = {}
        for k in actions:
            current_action[k] = actions[k][:,self.zero_time_point].unsqueeze(1)
        current_lastaction = {}
        for k in lastaction:
            current_lastaction[k] = lastaction[k][:,self.zero_time_point].unsqueeze(1)
        future_state = {}
        for k in states:
            future_state[k] = states[k][:,(self.zero_time_point+1):]
        future_action = {}
        for k in actions:
            future_action[k] = actions[k][:,(self.zero_time_point+1):]
        future_lastaction = {}
        for k in lastaction:
            future_lastaction[k] = lastaction[k][:,(self.zero_time_point+1):]


        #sample embeds
        with torch.no_grad():
            embeds = self._model['modules']['pov_embed'].sample(states['pov_embed_mu'][:,:self.zero_time_point], states['pov_embed_log_var'][:,:self.zero_time_point], states['pov_embed_log_alphas'][:,:self.zero_time_point], False)

        #generate current and next embed
        states_gen_embed = states['pov'][:,self.zero_time_point:(self.zero_time_point+2)]
        assert states_gen_embed.shape[1] == 2
        mu, log_var, discs = self._model['modules']['pov_embed'](states_gen_embed)
        current_embed = self._model['modules']['pov_embed'].sample(mu, log_var, discs)
        embeds = torch.cat([embeds, current_embed[:,0].unsqueeze(1)], dim=1)
        next_embed = current_embed[:,1]
        current_embed = current_embed[:,0].unsqueeze(1)

        #get current memory_embed
        memory_embed, _, memory_hidden = self.get_memory_embed(embeds)
        #get current final state embed
        current_hidden = self.get_state_hidden(current_state, current_embed, memory_embed.unsqueeze(1))
        #make current prediction loss
        loss, rdict = self.prediction_loss(current_embed, current_hidden, current_action, reward[:,self.zero_time_point:], current_lastaction, is_weight)


        #test kl loss
        if True:
            kl_disc, probs = self.discrete1_kl_divergence_element(discs)
            for k in range(len(kl_disc)):
                print('kl',kl_disc[k].data.cpu().numpy(),'probas',probs[k][0,0].data.cpu().numpy())
            kl_cont = self.normal1_kl_divergence_element(mu, log_var)
            top_kl, inds = torch.topk(kl_cont, 10, dim=0)
            print('kl_cont',top_kl.data.cpu().numpy())
            print('mu',mu[0,0][inds].data.cpu().numpy())
            plt.imshow((states_gen_embed[0,0].permute(1, 2, 0)*0.5+0.5).data.cpu().numpy())
            plt.show()

        #make compression loss
        kl_cont = self.normal1_kl_divergence(mu, log_var)
        cont_min, cont_max, cont_num_iters, cont_gamma = \
                self.cont_capacity
        cont_cap_current = (cont_max - cont_min) * self.train_iter / float(cont_num_iters) + cont_min
        cont_cap_current = min(cont_cap_current, cont_max)
        cont_capacity_loss = cont_gamma * torch.abs(cont_cap_current - kl_cont)

        kl_disc = self.discrete1_kl_divergence(discs)
        disc_min, disc_max, disc_num_iters, disc_gamma = \
                self.disc_capacity
        disc_cap_current = (disc_max - disc_min) * self.train_iter / float(disc_num_iters) + disc_min
        disc_cap_current = min(disc_cap_current, disc_max)
        disc_capacity_loss = disc_gamma * torch.abs(disc_cap_current - kl_disc)
        capacity_loss = cont_capacity_loss + disc_capacity_loss
        loss += capacity_loss

        #make state prediction loss
        #generate rnn hidden states from actions
        future_actions = []
        for k in actions:
            future_actions.append(actions[k][:,self.zero_time_point:])
            if k != 'camera':
                future_actions[-1] = one_hot(future_actions[-1], self.action_dict[k].n, device=self.device)
        future_actions = torch.cat(future_actions, dim=-1)

        rnn_output, _ = self._model['modules']['state_predict_memory'](future_actions[:,:-1], memory_hidden)
        state_prediction = self._model['modules']['state_predict'](rnn_output)

        cont_prediction, log_alphas_prediction = torch.split(state_prediction, self.continuous_embed_dim*2, dim=-1)
        #normalize log_alphas
        slog_alphas_prediction = list(torch.split(log_alphas_prediction, self.disc_embed_dims, dim=-1))
        for i in range(len(slog_alphas_prediction)):
            slog_alphas_prediction[i] = torch.nn.LogSoftmax(-1)(slog_alphas_prediction[i])
        log_alphas_prediction = torch.cat(slog_alphas_prediction, dim=-1)
        log_alphas_target = states['pov_embed_log_alphas'][:,(self.zero_time_point+1):]
        disc_kl_loss = self.discrete2_kl_divergence(log_alphas_target, log_alphas_prediction)

        mu_prediction, log_var_prediction = torch.split(cont_prediction, self.continuous_embed_dim, dim=-1)
        mu_target, log_var_target = states['pov_embed_mu'][:,(self.zero_time_point+1):], states['pov_embed_log_var'][:,(self.zero_time_point+1):]
        cont_kl_loss = self.normal2_kl_divergence(mu_target, log_var_target, mu_prediction, log_var_prediction)
        
        prediction_kl_loss = disc_kl_loss + cont_kl_loss
        loss += prediction_kl_loss

        #make future action prediction loss
        future_embeds = self._model['modules']['pov_embed'].sample(mu_prediction, log_var_prediction, log_alphas_prediction)
        if not DISABLE_FUTURE_POLICY_TRAINING:
            _, future_memory_embeds, _ = self.get_memory_embed(future_embeds, memory_hidden)
            future_hidden = self.get_state_hidden(future_state, future_embeds, future_memory_embeds)
        else:
            future_hidden = None
        if DISABLE_FUTURE_POLICY_TRAINING:
            policy_params = list(self._model['modules']['action_predict'].parameters())
            for p in policy_params:
                p.requires_grad_(False)
        future_loss, future_retdict = self.prediction_loss(future_embeds, future_hidden, future_action, reward[:,(self.zero_time_point+1):], future_lastaction, is_weight, not DISABLE_FUTURE_POLICY_TRAINING)
        loss += 2.0 * future_loss
        if DISABLE_FUTURE_POLICY_TRAINING:
            for p in policy_params:
                p.requires_grad_(True)

        #make infoDIM loss
        dis_loss = self.get_infoDIM_loss(current_embed[:,0], next_embed)
        loss += 10.0 * dis_loss
        
        if self.args.ganloss == 'fisher':
            self._model['Lambda'].retain_grad()
        loss.backward()
        for n in self._model:
            if 'opt' in n:
                self._model[n].step()
        if self.args.ganloss == 'fisher':
            self._model['Lambda'].data += self.args.rho * self._model['Lambda'].grad.data
            self._model['Lambda'].grad.data.zero_()
            rdict['Lambda'] = self._model['Lambda'][0]

        #add diagnostics
        rdict['capacity_loss'] = capacity_loss
        rdict['prediction_loss'] = prediction_kl_loss
        rdict['disc_kl_loss'] = disc_kl_loss
        rdict['cont_kl_loss'] = cont_kl_loss
        rdict['future_v_short_loss'] = future_retdict['v_short_loss']
        rdict['infoDIM_loss'] = dis_loss
        rdict['trained'] = str((self.train_iter * 100.0) / self.cont_capacity[-2]) + " %\t" +\
            str((self.train_iter * ((self.cont_capacity[1] + self.disc_capacity[1]) / np.log(2))) / self.cont_capacity[-2]) + " bit"

        #priority buffer stuff
        if 'tidxs' in input:
            #prioritize after bc_loss
            rdict['prio_upd'] = [{'error': rdict['bc_loss'].data.cpu().clone(),
                                    'replay_num': 0,
                                    'worker_num': worker_num[0]}]

        if 'bc_loss' in rdict:
            rdict['bc_loss'] = rdict['bc_loss'].mean()
        for d in rdict:
            if d != 'prio_upd' and not isinstance(rdict[d], str):
                rdict[d] = rdict[d].data.cpu().numpy()

        self.train_iter += 1
        return rdict

    def get_infoDIM_loss(self, embed_now, embed_future):
        real_pairs = torch.cat([embed_now, embed_future], dim=-1)
        random_roll = np.random.randint(1, self.args.er-1)
        fake_pairs = torch.cat([embed_now, embed_future.roll(random_roll,0)], dim=-1)
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
            dis_loss = torch.nn.BCEWithLogitsLoss()(real_scores, torch.ones(list(embed_now.shape[:-1]) + [1]).cuda()) + \
                    torch.nn.BCEWithLogitsLoss()(fake_scores, torch.zeros(list(embed_now.shape[:-1]) + [1]).cuda())
        return dis_loss


    def discrete2_kl_divergence(self, log_alpha1, log_alpha2):
        probs = log_alpha1.exp()
        kl_values = probs * (log_alpha1 - log_alpha2)
        kl_values = torch.sum(kl_values, dim=-1)
        kl_loss = torch.mean(kl_values)
        return kl_loss

    def discrete1_kl_divergence(self, log_alpha):
        probs = log_alpha.exp()
        kl_values = probs * log_alpha
        kl_values = torch.sum(kl_values, dim=-1)
        kl_loss = torch.mean(kl_values) + self.sum_logs
        return kl_loss

    def discrete1_kl_divergence_element(self, log_alpha):
        probs = log_alpha.exp()
        log_alphas = torch.split(log_alpha, self.disc_embed_dims, dim=-1)
        probs = torch.split(probs, self.disc_embed_dims, dim=-1)
        ret = []
        for i in range(len(self.disc_embed_dims)):
            ret.append(probs[i] * log_alphas[i])
            ret[-1] = torch.sum(ret[-1], dim=-1)
            ret[-1] = torch.mean(ret[-1]) + float(np.log(self.disc_embed_dims[i]))
        return ret, probs

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
        return kl_loss

    def normal1_kl_divergence_element(self, mu, log_var):
        kl_values = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp())
        kl_loss = torch.mean(kl_values.view(-1, self.continuous_embed_dim), dim=0)
        return kl_loss


    def prediction_loss(self, cnn_embed, hidden, action, reward, lastaction, is_weight=None, train_policy=True):
        assert len(reward.shape) == 2
        if not is_weight is None:
            is_weight = is_weight.unsqueeze(1).expand(-1, hidden.shape[1])
            is_weight = is_weight.view(-1)
        if train_policy:
            loss, bc_loss, entropy, _, max_probas = self._model['modules']['action_predict'].get_loss_BC(hidden, action, is_weight=is_weight, last_action=lastaction)

        #get r loss
        target = (reward[:,:cnn_embed.shape[1]] > 1e-5).long().view(-1)
        value = self._model['modules']['r_predict'](cnn_embed).view(-1, 2)
        r_loss = self._model['ce_loss'](value, target, is_weight)
        if train_policy:
            loss += r_loss
        else:
            loss = r_loss.clone()

        
        creward = torch.cumsum(reward, 1)
        #get short v loss
        self.short_time = 10
        target = (creward[:,self.short_time:(self.short_time+cnn_embed.shape[1])] > 1e-5).long().view(-1)
        logits = self._model['modules']['v_predict_short'](cnn_embed).view(-1, 2)
        v_short_loss = self._model['ce_loss'](logits, target, is_weight)
        loss += 1.5 * v_short_loss

        #get long v loss
        self.long_time = reward.shape[1]-cnn_embed.shape[1]
        target = (creward[:,self.long_time:] > 1e-5).long().view(-1)
        logits = self._model['modules']['v_predict_long'](cnn_embed).view(-1, 2)
        v_long_loss = self._model['ce_loss'](logits, target, is_weight)
        loss += 0.2 * v_long_loss

        
        if train_policy:
            return loss, {'entropy': entropy,
                          'max_probas': max_probas,
                          'bc_loss': bc_loss, 
                          'r_loss': r_loss,
                          'v_short_loss': v_short_loss,
                          'v_long_loss': v_long_loss}
        else:
            return loss, {'r_loss': r_loss,
                          'v_short_loss': v_short_loss,
                          'v_long_loss': v_long_loss}


    def get_state_hidden(self, state, pov_embed, memory_embed):
        tembed = [pov_embed, memory_embed]
        for o in self.state_shape:
            if o in self.state_keys and o != 'pov':
                tembed.append(state[o])
        tembed = torch.cat(tembed, dim=-1)
        hidden = self._model['modules']['state_hidden'](tembed)
        return hidden

    def get_memory_embed(self, pov_embes, hidden=None):
        output, hidden = self._model['modules']['memory'](pov_embes, hidden)
        return output[:,-1], output, hidden

    
    def reset(self):
        self.hidden = torch.zeros((1,1,self.memory_embed), device=self.device, dtype=torch.float32, requires_grad=False)

    def select_action(self, state, batch=False):
        lastaction = {}
        tstate = {}
        if not batch:
            for k in state:
                if 'last_' in k:
                    if k != 'last_camera':
                        lastaction[k] = variable(np.array([state[k]]), True).long()
                    else:
                        lastaction[k] = variable(state[k][None,:])
                else:
                    tstate[k] = state[k][None, :].copy()
        tstate = variable(tstate)
        with torch.no_grad():
            #pov_embed
            tstate['pov'] = (tstate['pov'].permute(0, 3, 1, 2)/255.0)*2.0 - 1.0
            mu, log_var, discs = self._model['modules']['pov_embed'](tstate['pov'])
            pov_embed = self._model['modules']['pov_embed'].sample(mu, log_var, discs, training=False)
            #memory_embed
            memory_embed, self.hidden = self._model['modules']['memory'](pov_embed[:,None], self.hidden)
            memory_embed = memory_embed[:,0]
            #all embed
            tembed = [pov_embed, memory_embed]
            for o in self.state_shape:
                if o in self.state_keys and o != 'pov':
                    tembed.append(tstate[o])
            tembed = torch.cat(tembed, dim=-1)
            hidden = self._model['modules']['state_hidden'](tembed)
            r_action = {}
            o_action = {}
            r_action, conaction, entropy = self._model['modules']['action_predict'].sample(hidden, lastaction)
            for a in r_action:
                o_action[a] = r_action[a].data.cpu().numpy()[0]
            o_action['camera'] = conaction.data.cpu().numpy()[0,:]
            o_action['camera'] = self.map_camera(o_action['camera'])

            action = OrderedDict()
            for a in self.action_dict:
                action[a] = o_action[a]
        return action, {'entropy': entropy.data.cpu().numpy(),
                        'pov_embed_mu': mu.data.cpu().numpy(),
                        'pov_embed_log_var': log_var.data.cpu().numpy(),
                        'pov_embed_log_alphas': discs.data.cpu().numpy()}
