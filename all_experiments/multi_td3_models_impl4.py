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

STAGE1 = False


class CombineQueue():
    def __init__(self, queue1, queue2, ratio=0.8):
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
            self.fc0 = torch.nn.Sequential(torch.nn.Linear(num_input+4+self.logit_dim, args.hidden), torch.nn.Tanh(), HashingMemory.build(args.hidden, self.n_actions, args))
            #self.fc0 = torch.nn.Sequential(torch.nn.Linear(num_input+4+self.logit_dim, args.hidden), 
            #                               Lambda(lambda x: swish(x)),
            #                               torch.nn.Linear(args.hidden, self.n_actions))
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

class MultiTD4():
    def __init__(self, state_shape, action_dict, args, time_deltas, only_visnet=False):
        assert isinstance(state_shape, dict)
        self.state_shape = state_shape
        self.action_dict = copy.deepcopy(action_dict)
        self.state_keys = ['state', 'pov', 'history_action', 'history_reward', 'orientation', 'env_type']
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
        if self.args.pretrain:
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
            state_dict = torch.load(filename + 'multiaction_TD4')
            if self.only_visnet:
                state_dict = {n: v for n, v in state_dict.items() if n.startswith('pov_embed.')}
            self.load_state_dict(state_dict)
            if 'Lambda' in self._model:
                self._model['Lambda'] = torch.load(filename + 'multiaction_TD4Lam')
        else:
            torch.save(self._model['modules'].state_dict(), filename + 'multiaction_TD4')
            torch.save(self.train_iter, filename + 'train_iter')
            if 'Lambda' in self._model:
                torch.save(self._model['Lambda'], filename + 'multiaction_TD4Lam')
                
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
        desc['pov_embed'] = {'compression': False, 'shape': [self.continuous_embed_dim+sum(self.disc_embed_dims)], 'dtype': np.float32}
        if len(self.disc_embed_dims) == 0:
            return desc
        else:
            assert False #not implemented

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
        modules['memory'] = MemoryModel(self.args, self.cnn_embed_dim, self.past_time[-self.args.num_past_times:])
        self.memory_embed = modules['memory'].memory_embed_dim
        
        self.actions_embed_dim = 128
        modules['actions_enc'] = torch.nn.Sequential(torch.nn.Linear(self.action_dim*self.max_predict_range, self.args.hidden),
                                                     Lambda(lambda x: swish(x)),
                                                     torch.nn.Linear(self.args.hidden, self.actions_embed_dim))
        modules['actions_dec_all'] = torch.nn.Sequential(torch.nn.Linear(self.actions_embed_dim, self.args.hidden),
                                                         Lambda(lambda x: swish(x)),
                                                         torch.nn.Linear(self.args.hidden, self.action_dim*self.max_predict_range))
        modules['actions_dec_single'] = FullDiscretizerWLast(self.actions_embed_dim, self.device, self.args, self.action_dict)


        modules['actions_embed_predict'] = torch.nn.Sequential(torch.nn.Linear(self.cnn_embed_dim + self.memory_embed + self.state_dim, self.args.hidden),
                                                               Lambda(lambda x: swish(x)),
                                                               torch.nn.Linear(self.args.hidden, self.actions_embed_dim*2))#mean and Q value parameter
        modules['value'] = torch.nn.Sequential(torch.nn.Linear(self.cnn_embed_dim + self.memory_embed + self.state_dim, self.args.hidden),
                                               Lambda(lambda x: swish(x)),
                                               HashingMemory.build(self.args.hidden, 1, self.args))

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

    def train(self, input, worker_num=None):
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
        lastactions = {}
        for k in input:
            if k in self.state_keys or k in embed_keys:
                states[k] = variable(input[k])
            if 'last_' in k:
                lastactions[k] = variable(input[k], k != 'last_camera')
                if k != 'last_camera':
                    lastactions[k] = lastactions[k].long()
        actions = {'camera': self.inverse_map_camera(variable(input['camera']))}
        for a in self.action_dict:
            if a != 'camera':
                actions[a] = variable(input[a], True).long()
        reward = variable(input['reward'])
        done = variable(input['done'], True).int()
        assert not 'r_offset' in input
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
        last_action = {}
        for k in actions:
            current_actions[k] = actions[k][:,self.zero_time_point:(self.zero_time_point+self.max_predict_range)]
            last_action['last_'+k] = lastactions['last_'+k][:,self.zero_time_point]
        if self.args.pretrain:
            next_actions = {}
            for k in actions:
                next_actions[k] = actions[k][:,(self.zero_time_point+self.max_predict_range):(self.zero_time_point+self.max_predict_range*2)]
        assert states['pov'].shape[1] == 2

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

        #get pov embed
        pov_embeds = self._model['modules']['pov_embed'](states['pov'])
        current_pov_embed, next_pov_embed = pov_embeds[:,0], pov_embeds[:,1]

        #make memory embed
        current_memory_embed = self._model['modules']['memory'](past_embeds, current_pov_embed)
        next_memory_embed = self._model['modules']['memory'](next_past_embeds, next_pov_embed)

        #combine all embeds
        tembed = [current_pov_embed, current_memory_embed]
        for o in self.state_shape:
            if o in self.state_keys and o != 'pov':
                tembed.append(current_state[o])
        tembed = torch.cat(tembed, dim=-1)
        nembed = [next_pov_embed, next_memory_embed]
        for o in self.state_shape:
            if o in self.state_keys and o != 'pov':
                nembed.append(next_state[o])
        nembed = torch.cat(nembed, dim=-1)

        #encode actions
        current_action = {}
        current_actions_cat = []
        for a in self.action_dict:
            current_action[a] = current_actions[a][:,0]
            if a != 'camera':
                current_actions_cat.append(one_hot(current_actions[a], self.action_dict[a].n, self.device).view(-1, self.action_dict[a].n*self.max_predict_range))
            else:
                current_actions_cat.append(current_actions[a].reshape(-1, 2*self.max_predict_range))
        current_actions_cat = torch.cat(current_actions_cat, dim=1)

        current_actions_embed = self._model['modules']['actions_enc'](current_actions_cat)

        #decode loss
        dec_loss = self.get_actions_loss(self._model['modules']['actions_dec_all'](current_actions_embed), current_actions)
        loss = dec_loss.clone()
        rdict = {}
        rdict['dec_loss'] = dec_loss
        
        #up to here clone of pretrain
        if STAGE1:
            tembed = tembed.detach()
            nembed = nembed.detach()
        else:
            #direct decode loss
            single_dec_loss, _, _, _, _ = self._model['modules']['actions_dec_single'].get_loss_BC(current_actions_embed, current_action, last_action, is_weight)
            loss += single_dec_loss
            rdict['single_dec_loss'] = single_dec_loss
            
        #Q training
        current_V = self._model['modules']['value'](tembed)[:,0]
        next_V = self._model['modules']['value'](nembed)[:,0]
        if STAGE1:
            for n, p in self._model['modules']['actions_embed_predict'].named_parameters():
                if '0.' in n:
                    p.requires_grad_(False)
        tmp = self._model['modules']['actions_embed_predict'](tembed)
        current_best_action, current_Q_param = tmp[:,:self.actions_embed_dim], tmp[:,self.actions_embed_dim:]
        if STAGE1:
            current_best_action = current_best_action.detach()
        current_Q_param = current_Q_param.exp()
        current_Q = current_V - (current_Q_param * (current_actions_embed - current_best_action).pow(2)).sum(1)
        target = (reward[:,self.zero_time_point:(self.zero_time_point+self.max_predict_range)] * self.gammas).sum(1) + (GAMMA ** self.max_predict_range) * next_V
        if is_weight is None:
            Q_loss = (current_Q - target).pow(2).mean()
        else:
            Q_loss = ((current_Q - target).pow(2) * is_weight).mean()
        loss += Q_loss
        with torch.no_grad():
            rdict['Q_diff'] = torch.abs(current_Q - target).mean()
            rdict['current_Q_mean'] = current_Q.mean()
            rdict['max_Q_mean'] = current_V.mean()

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
                #prioritize after bc_loss
                rdict['prio_upd'] = [{'error': torch.abs(current_Q - target).data.cpu().clone(),
                                      'replay_num': 0,
                                      'worker_num': worker_num[0]}]

        for d in rdict:
            if d != 'prio_upd' and not isinstance(rdict[d], str) and not isinstance(rdict[d], int):
                rdict[d] = rdict[d].data.cpu().numpy()

        self.train_iter += 1
        rdict['train_iter'] = self.train_iter / 10000
        return rdict

    def pretrain(self, input, worker_num=None):
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
        lastactions = {}
        for k in input:
            if k in self.state_keys or k in embed_keys:
                states[k] = variable(input[k])
            if 'last_' in k:
                lastactions[k] = variable(input[k], k != 'last_camera')
                if k != 'last_camera':
                    lastactions[k] = lastactions[k].long()
        actions = {'camera': self.inverse_map_camera(variable(input['camera']))}
        for a in self.action_dict:
            if a != 'camera':
                actions[a] = variable(input[a], True).long()
        reward = variable(input['reward'])
        done = variable(input['done'], True).int()
        assert not 'r_offset' in input
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
        last_action = {}
        for k in actions:
            current_actions[k] = actions[k][:,self.zero_time_point:(self.zero_time_point+self.max_predict_range)]
            last_action['last_'+k] = lastactions['last_'+k][:,self.zero_time_point]
        if self.args.pretrain:
            next_actions = {}
            for k in actions:
                next_actions[k] = actions[k][:,(self.zero_time_point+self.max_predict_range):(self.zero_time_point+self.max_predict_range*2)]
        assert states['pov'].shape[1] == 2

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

        #get pov embed
        pov_embeds = self._model['modules']['pov_embed'](states['pov'])
        current_pov_embed, next_pov_embed = pov_embeds[:,0], pov_embeds[:,1]

        #make memory embed
        current_memory_embed = self._model['modules']['memory'](past_embeds, current_pov_embed)
        next_memory_embed = self._model['modules']['memory'](next_past_embeds, next_pov_embed)

        #combine all embeds
        tembed = [current_pov_embed, current_memory_embed]
        for o in self.state_shape:
            if o in self.state_keys and o != 'pov':
                tembed.append(current_state[o])
        tembed = torch.cat(tembed, dim=-1)
        nembed = [next_pov_embed, next_memory_embed]
        for o in self.state_shape:
            if o in self.state_keys and o != 'pov':
                nembed.append(next_state[o])
        nembed = torch.cat(nembed, dim=-1)

        #encode actions
        current_action = {}
        current_actions_cat = []
        next_actions_cat = []
        for a in self.action_dict:
            current_action[a] = current_actions[a][:,0]
            if a != 'camera':
                current_actions_cat.append(one_hot(current_actions[a], self.action_dict[a].n, self.device).view(-1, self.action_dict[a].n*self.max_predict_range))
                next_actions_cat.append(one_hot(next_actions[a], self.action_dict[a].n, self.device).view(-1, self.action_dict[a].n*self.max_predict_range))
            else:
                current_actions_cat.append(current_actions[a].reshape(-1, 2*self.max_predict_range))
                next_actions_cat.append(next_actions[a].reshape(-1, 2*self.max_predict_range))
        current_actions_cat = torch.cat(current_actions_cat, dim=1)
        next_actions_cat = torch.cat(next_actions_cat, dim=1)

        current_actions_embed = self._model['modules']['actions_enc'](current_actions_cat)
        next_actions_embed = self._model['modules']['actions_enc'](next_actions_cat)

        #decode loss
        dec_loss = self.get_actions_loss(self._model['modules']['actions_dec_all'](current_actions_embed), current_actions) +\
            self.get_actions_loss(self._model['modules']['actions_dec_all'](next_actions_embed), next_actions)
        loss = dec_loss.clone()
        rdict = {}
        rdict['dec_loss'] = dec_loss

        #direct decode loss
        single_dec_loss, _, _, _, _ = self._model['modules']['actions_dec_single'].get_loss_BC(current_actions_embed, current_action, last_action, is_weight)
        loss += single_dec_loss
        rdict['single_dec_loss'] = single_dec_loss

        #Q training
        current_V = self._model['modules']['value'](tembed)[:,0]
        next_V = self._model['modules']['value'](nembed)[:,0]
        tmp = self._model['modules']['actions_embed_predict'](tembed)
        current_best_action, current_Q_param = tmp[:,:self.actions_embed_dim], tmp[:,self.actions_embed_dim:]
        current_Q_param = current_Q_param.exp()
        current_Q = current_V - (current_Q_param * (current_actions_embed - current_best_action).pow(2)).sum(1)
        target = (reward[:,self.zero_time_point:(self.zero_time_point+self.max_predict_range)] * self.gammas).sum(1) + (GAMMA ** self.max_predict_range) * next_V
        if is_weight is None:
            Q_loss = (current_Q - target).pow(2).mean()
        else:
            Q_loss = ((current_Q - target).pow(2) * is_weight).mean()
        loss += Q_loss
        with torch.no_grad():
            rdict['Q_diff'] = torch.abs(current_Q - target).mean()
            rdict['current_Q_mean'] = current_Q.mean()
            rdict['max_Q_mean'] = current_V.mean()

        #behaviour cloning
        bc_loss, _, _, _, _ = self._model['modules']['actions_dec_single'].get_loss_BC(current_best_action, current_action, last_action, is_weight)
        loss += bc_loss
        rdict['bc_loss'] = bc_loss
        proc_loss = (current_Q_param.detach() * (current_actions_embed.detach() - current_best_action).pow(2)).sum(1).mean()
        loss += proc_loss
        rdict['proc_loss'] = proc_loss


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
                #prioritize after bc_loss
                rdict['prio_upd'] = [{'error': torch.abs(current_Q - target).data.cpu().clone(),
                                      'replay_num': 0,
                                      'worker_num': worker_num[0]}]

        for d in rdict:
            if d != 'prio_upd' and not isinstance(rdict[d], str) and not isinstance(rdict[d], int):
                rdict[d] = rdict[d].data.cpu().numpy()

        self.train_iter += 1
        rdict['train_iter'] = self.train_iter / 10000
        return rdict


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
                    loss += self._model['ce_loss'](actions_logits[i].view(-1, self.action_dict[a].n), current_actions[a].reshape(-1)) * 2.0
                else:
                    loss += self._model['ce_loss'](actions_logits[i].view(-1, self.action_dict[a].n), current_actions[a].reshape(-1))
                i += 1
        return loss

    def reset(self):
        max_past = 1-min(self.past_time)
        self.past_embeds = torch.zeros((max_past,self.cnn_embed_dim), device=self.device, dtype=torch.float32, requires_grad=False)

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
            pov_embed = self._model['modules']['pov_embed'](tstate['pov'])
            #memory_embed
            memory_inp = self.past_embeds[self.past_time].unsqueeze(0)
            memory_embed = self._model['modules']['memory'](memory_inp, pov_embed)
            self.past_embeds = self.past_embeds.roll(1, dims=0)
            self.past_embeds[0] = pov_embed[0]

            #sample action
            tembed = [pov_embed, memory_embed]
            for o in self.state_shape:
                if o in self.state_keys and o != 'pov':
                    tembed.append(tstate[o])
            tembed = torch.cat(tembed, dim=-1)
            value = self._model['modules']['value'](tembed)[0,0]
            current_best_action = self._model['modules']['actions_embed_predict'](tembed)[:,:self.actions_embed_dim]
            r_action = {}
            o_action = {}
            r_action, conaction, entropy = self._model['modules']['actions_dec_single'].sample(current_best_action, lastaction)
            for a in r_action:
                o_action[a] = r_action[a].data.cpu().numpy()[0]
            o_action['camera'] = conaction.data.cpu().numpy()[0,:]
            o_action['camera'] = self.map_camera(o_action['camera'])

            action = OrderedDict()
            for a in self.action_dict:
                action[a] = o_action[a]
            return action, {'pov_embed': pov_embed.data.cpu(),
                            'entropy': entropy.data.cpu().numpy(),
                            'value': value.data.cpu().numpy()}


def main():
    parser = argparse.ArgumentParser(description="Behavior Cloning")

    PRETRAIN = False
    parser.add_argument("--pretrain", action='store_true', default=PRETRAIN, help="")
    parser.add_argument("--agent_name", type=str, default='MultiTD4', help="Model name")
    parser.add_argument("--agent_from", type=str, default='multi_td3_models_impl4', help="Model python module")
    parser.add_argument("--needs-next", action='store_true', default=False, help="")
    parser.add_argument("--env", default='MineRLTreechop-v0', type=str, help="Gym environment to use")#MineRLNavigateDense-v0#MineRLTreechop-v0
    parser.add_argument("--minerl", action='store_true', default=True, help="The environment is a MineRL environment")
    parser.add_argument("--name", type=str, default='treechop', help="Experiment name")
    parser.add_argument("--episodes", type=int, default=int(1e5), help="Number of episodes to run")

    parser.add_argument("--erpoolsize", type=int, default=int(4e5), help="Number of experiences stored by each option for experience replay")
    parser.add_argument("--packet-size", type=int, default=int(256), help="")
    parser.add_argument("--sample-efficiency", type=int, default=int(16), help="")
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
    args.needs_embedding = True
    args.needs_last_action = True
    args.load = None#"treechop/model_stage2_"
    args.save = "treechop/model_stage2_2"
    args.dont_send_past_pov = True
    args.mem_query_batchnorm = False
    args.cnn_type = 'atari'
    args.ganloss = 'hinge'
    args.rho = 1e-4
    args.mem_n_keys = 512
    args.needs_random_future = False
    args.history_a = 0
    if PRETRAIN:
        args.er = 128
    else:
        args.er = 64
    args.embed_dim = 256
    num_past_times = 32
    args.num_past_times = num_past_times
    past_range_seconds = 60.0
    past_range = past_range_seconds / 0.05
    delta_a = (past_range-num_past_times-1)/(num_past_times**2)
    past_times = [-int(delta_a*n**2+n+1) for n in range(num_past_times)]
    past_times.reverse()
    time_skip = 10
    if not PRETRAIN:
        time_deltas = past_times + list(range(0, time_skip+1, 1))
    else:
        time_deltas = past_times + list(range(0, (time_skip*2)+1, 1))
    time_deltas = [-int(delta_a*n**2+n+1) + time_skip for n in range(num_past_times)] + time_deltas
    pov_time_deltas = [0, time_skip]
    if not PRETRAIN:
        args.erpoolsize = int(1e6)
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
        num_workers = 2
        args.packet_size = max(256, args.er)
        args.sample_efficiency = max((args.packet_size * 32) // args.er, 1)
        args.lr = 1e-4
        args.nec_lr = 1e-3
        args.num_replay_workers = num_workers
        copy_queues = [Queue(1) for n in range(num_workers+1)]
        traj_gen = VirtualActorTrajGenerator(args, copy_queue=copy_queues[0], add_queues=args.num_replay_workers-1, add_args=[time_deltas])
        traj_gens = [VirtualExpertGenerator(args, 0, copy_queue=copy_queues[n+1], add_args=[time_deltas]) for n in range(num_workers)]
        combine_traj = [CombineQueue(traj_gen.traj_pipe[n], traj_gens[n].traj_pipe[0]) for n in range(args.num_replay_workers)]
        assert len(traj_gen.traj_pipe) == args.num_replay_workers
        start = 0
        end = time_deltas[-1]
        if args.per:
            prio_queues = [[Queue(2) for n in range(num_workers)]]
            replay = VirtualReplayBufferPER(args, traj_gen.data_description(), combine_traj, prio_queues[0], [t-start for t in time_deltas], 50+end, pov_time_deltas=pov_time_deltas, blocking=False)
            trainer = Trainer(args.agent_name, args.agent_from, traj_gen.state_shape, traj_gen.num_actions, args, 
                                [replay.batch_queues], add_args=[time_deltas], prio_queues=prio_queues, copy_queues=copy_queues, blocking=False)
        else:
            replay = VirtualReplayBuffer(args, traj_gen.data_description(), combine_traj, [t-start for t in time_deltas], 50+end, pov_time_deltas=pov_time_deltas)
            trainer = Trainer(args.agent_name, args.agent_from, traj_gen.state_shape, traj_gen.num_actions, args, 
                                [replay.batch_queues], add_args=[time_deltas], copy_queues=copy_queues, blocking=False)

    while True:
        time.sleep(100)



if __name__ == '__main__':
    main()