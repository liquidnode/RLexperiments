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

GAMMA = 0.99
LAMBDA = 0.95
COMPATIBLE = False
USE_UNIFORM = True
USE_LOOKAHEAD = False

class SkipHashingMemory(torch.nn.Module):
    def __init__(self, input_dim, args, compressed_dim=32):
        super(SkipHashingMemory, self).__init__()
        self.args = args
        self.memory = HashingMemory.build(input_dim, compressed_dim, self.args)
        self.reproj = torch.nn.Linear(compressed_dim, input_dim)

    def forward(self, x):
        y = self.memory(x)
        y = self.reproj(y)
        return x + y

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
            assert False
        if self.current_ac is None or self.current_ac.shape[0] != fc_input.shape[0]:
            self.current_ac = torch.zeros((fc_input.shape[0],2), dtype=torch.float32, device=self.device, requires_grad=False)
            self.current_delta = torch.ones((fc_input.shape[0],), dtype=torch.float32, device=self.device, requires_grad=False) * self.min_delta
            self.current_dac = torch.zeros((fc_input.shape[0],), dtype=torch.int64, device=self.device, requires_grad=False)
            self.delta_chosen = torch.zeros((fc_input.shape[0],), dtype=torch.uint8, device=self.device, requires_grad=False)
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
            self.delta_chosen.fill_(0)
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
        c_logits = self.fc0(input)
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
                self.sdelta_chosen = torch.zeros((fc_input.shape[0],), dtype=torch.uint8, device=self.device, requires_grad=False)
                self.scurrent_rdac = {}
                for a in self.action_dict:
                    if a != 'camera':
                        if last_action is None or a == 'left_right' or a == 'attack_place_equip_craft_nearbyCraft_nearbySmelt':
                            self.scurrent_rdac[a] = torch.zeros((fc_input.shape[0],), dtype=torch.int64, device=self.device, requires_grad=False)
                self.scurrent_rdac_changed = {}
                for a in self.action_dict:
                    if a != 'camera':
                        self.scurrent_rdac_changed[a] = torch.zeros((fc_input.shape[0],), dtype=torch.uint8, device=self.device, requires_grad=False)
            else:
                self.scurrent_ac.fill_(0.0)
                self.scurrent_delta.fill_(self.min_delta)
                for a in self.action_dict:
                    if a != 'camera':
                        if last_action is None or a == 'left_right' or a == 'attack_place_equip_craft_nearbyCraft_nearbySmelt':
                            self.scurrent_rdac[a].fill_(0)
                        self.scurrent_rdac_changed[a].fill_(0)
                self.sdelta_chosen.fill_(0)
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
                #print(probs[0,self.action_ranges['attack_place_equip_craft_nearbyCraft_nearbySmelt'][0]].data.cpu().numpy())
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

    
class netG(torch.nn.Module):
    def __init__(self, action_dim, args, state_shape, lookahead = 1):
        super(netG, self).__init__()
        self.args = args
        self.state_shape = state_shape

        encnn_layers = []
        self.embed_dim = args.hidden
        self.noise_dim = 128
        self.cnn_noise_dim = 8
        assert len(self.state_shape['pov']) > 1
        if self.args.cnn_type == 'atari':
            sizes = [8, 4, 3]
            strides = [4, 2, 1]
            batchnorm = [False, False, False]
            pooling = [1, 1, 1]
            filters = [32, 64, 32]
            padding = [0, 0, 0]
            end_size = 4
        elif self.args.cnn_type == 'mnist':
            sizes = [3, 3, 3]
            strides = [1, 1, 1]
            batchnorm = [False, False, False]
            pooling = [2, 2, 2]
            filters = [32, 32, 32]
            padding = [0, 0, 0]
            end_size = 4
        elif self.args.cnn_type == 'adv':
            sizes = [4, 3, 3, 4]
            strides = [2, 2, 2, 1]
            batchnorm = [True, True, True, True]
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

            if batchnorm[i]:
                tencnn_layers.append(torch.nn.BatchNorm2d(filters[i]))
            tencnn_layers.append(Lambda(lambda x: swish(x)))

            if pooling[i] > 1:
                tencnn_layers.append(torch.nn.MaxPool2d(pooling[i]))

            in_channels = filters[i]
            encnn_layers.append(torch.nn.Sequential(*tencnn_layers))

        tencnn_layers = []
        tencnn_layers.append(Flatten())
        tencnn_layers.append(torch.nn.Linear(in_channels*(end_size**2), self.embed_dim))
        tencnn_layers.append(SkipHashingMemory(self.embed_dim, args))
        tencnn_layers.append(Lambda(lambda x: swish(x)))
        encnn_layers.append(torch.nn.Sequential(*tencnn_layers))
        self.encnn_layers = torch.nn.ModuleList(encnn_layers)


        #make decoder
        decnn_layers = []
        tdecnn_layers = []
        tdecnn_layers.append(torch.nn.Linear(self.embed_dim+self.noise_dim+action_dim*lookahead, in_channels*(end_size**2)))
        tdecnn_layers.append(Lambda(lambda x: swish(x)))
        tdecnn_layers.append(Lambda(lambda x: x.view(-1, filters[-1], end_size, end_size)))
        decnn_layers.append(torch.nn.Sequential(*tdecnn_layers))

        for i in range(len(sizes)-1, -1, -1):
            tdecnn_layers = []
            tinp = in_channels+filters[i]# if i != 0 else in_channels
            tfil = filters[i-1] if i != 0 else 32#lookahead*self.state_shape['pov'][-1]
            tdecnn_layers.append(torch.nn.ConvTranspose2d(
                tinp,
                tfil,
                sizes[i],
                stride=strides[i],
                padding=padding[i],
                bias=True
            ))
            
            if batchnorm[i]:
                tdecnn_layers.append(torch.nn.BatchNorm2d(tfil))
            if True:#i != 0:
                tdecnn_layers.append(Lambda(lambda x: swish(x)))

                if pooling[i] > 1:
                    assert False #MaxPool inverse not implemented
                    tdecnn_layers.append(torch.nn.MaxPool2d(pooling[i]))
            else:
                tdecnn_layers.append(torch.nn.Tanh())

            in_channels = tfil
            decnn_layers.append(torch.nn.Sequential(*tdecnn_layers))
            if i != 0:
                tdecnn_layers = []
                tinp = in_channels+action_dim*lookahead+self.cnn_noise_dim
                tdecnn_layers.append(torch.nn.ConvTranspose2d(
                    tinp,
                    tfil,
                    1,
                    stride=1,
                    bias=True
                ))
                if batchnorm[i]:
                    tdecnn_layers.append(torch.nn.BatchNorm2d(tfil))
                tdecnn_layers.append(Lambda(lambda x: swish(x)))
                decnn_layers.append(torch.nn.Sequential(*tdecnn_layers))
            else:
                tdecnn_layers = []
                tdecnn_layers.append(torch.nn.ConvTranspose2d(
                    in_channels,
                    lookahead*self.state_shape['pov'][-1],
                    1,
                    stride=1,
                    bias=True
                ))
                tdecnn_layers.append(torch.nn.Tanh())
                decnn_layers.append(torch.nn.Sequential(*tdecnn_layers))
        self.decnn_layers = torch.nn.ModuleList(decnn_layers)

    def forward(self, state_pov, action_vec):
        x = state_pov
        skip_connections = []

        for layer in self.encnn_layers:
            x = layer(x)
            skip_connections.append(x)
        skip_connections = list(reversed(skip_connections))
        embed = x

        def rand_gen(shape):
            if USE_UNIFORM:
                _noise = torch.rand(*shape).cuda() * 2.0 - 1.0
            else:
                _noise = torch.randn(*shape).cuda()
            return _noise
        noise = rand_gen([self.args.er, self.noise_dim])
            
        x = torch.cat([x, action_vec, noise], dim=1)
        for i, layer in enumerate(self.decnn_layers):
            if False:
                if i != 0 and i != len(self.decnn_layers)-1:
                    a_tile = action_vec.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, x.shape[-2], x.shape[-1])
                    noise_shp = list(a_tile.shape)
                    noise_shp[1] = self.cnn_noise_dim
                    noise = rand_gen(noise_shp)
                    x = torch.cat([x, a_tile, noise, skip_connections[i]], dim=1)
            else:
                if i != 0 and i != len(self.decnn_layers)-1:
                    if i % 2 == 1:
                        x = torch.cat([x, skip_connections[(i+1)//2]], dim=1)
                    else:
                        a_tile = action_vec.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, x.shape[-2], x.shape[-1])
                        noise_shp = list(a_tile.shape)
                        noise_shp[1] = self.cnn_noise_dim
                        noise = rand_gen(noise_shp)
                        x = torch.cat([x, a_tile, noise], dim=1)
            x = layer(x)
        return x, embed

    def forward_embed(self, state_pov):
        x = state_pov
        for layer in self.encnn_layers:
            x = layer(x)
        return x

class netD(torch.nn.Module):
    def __init__(self, action_dim, args, state_shape, lookahead = 1):
        super(netD, self).__init__()

        self.args = args
        self.state_shape = state_shape

        self.tlayers = []
        assert len(self.state_shape['pov']) > 1
        if self.args.cnn_type == 'atari':
            sizes = [8, 4, 3]
            strides = [4, 2, 1]
            batchnorm = [False, False, False]
            spectnorm = [True, True, True]
            pooling = [1, 1, 1]
            filters = [32, 64, 32]
            padding = [0, 0, 0]
            end_size = 4
        elif self.args.cnn_type == 'mnist':
            sizes = [3, 3, 3]
            strides = [1, 1, 1]
            batchnorm = [False, False, False]
            spectnorm = [True, True, True]
            pooling = [2, 2, 2]
            filters = [32, 32, 32]
            padding = [0, 0, 0]
            end_size = 4
        elif self.args.cnn_type == 'adv':
            sizes = [3, 3, 3, 4]
            strides = [2, 2, 2, 1]
            batchnorm = [False, False, False, False]
            spectnorm = [True, True, True, True]
            pooling = [1, 1, 1, 1]
            filters = [32, 64, 64, 32]
            padding = [0, 0, 0, 0]
            end_size = 4
            
        in_channels = self.state_shape['pov'][-1]*(1+lookahead)

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
            #tencnn_layers.append(Lambda(lambda x: swish(x)/1.09985))
            tencnn_layers.append(torch.nn.LeakyReLU(negative_slope = 0.2, inplace=True))

            if pooling[i] > 1:
                tencnn_layers.append(torch.nn.MaxPool2d(pooling[i]))

            if i != len(sizes)-1:
                in_channels = filters[i] + action_dim*lookahead
            else:
                in_channels = filters[i]
            self.tlayers.append(torch.nn.Sequential(*tencnn_layers))

        self.tlayers.append(Flatten())
        tlayer = []
        tlayer.append(torch.nn.Linear(filters[-1]*(end_size**2)+action_dim*lookahead, 18))
        tlayer.append(torch.nn.LeakyReLU(negative_slope = 0.2, inplace=True))
        tlayer.append(torch.nn.Linear(18, 1))
        self.tlayers.append(torch.nn.Sequential(*tlayer))
        self.tlayers = torch.nn.ModuleList(self.tlayers)

    def forward(self, state_pov, action_vec):
        x = state_pov.contiguous().view(self.args.er, -1, self.state_shape['pov'][0], self.state_shape['pov'][1])
        for i, layer in enumerate(self.tlayers):
            if i != 0 and i < len(self.tlayers)-2:
                a_tile = action_vec.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, x.shape[-2], x.shape[-1])
                x = torch.cat([x, a_tile], dim=1)
            if i == len(self.tlayers)-1:
                x = torch.cat([x, action_vec], dim=1)
            x = layer(x)
        return x.squeeze(-1)

class VirtualPGModel():
    def __init__(self, state_shape, action_dict, args, time_deltas):
        assert isinstance(state_shape, dict)
        self.state_shape = state_shape
        self.action_dict = copy.deepcopy(action_dict)
        self.args = args
        
        self.action_dim = 2
        for a in action_dict:
            if a != 'camera':
                self.action_dim += action_dict[a].n
        self.time_deltas = time_deltas
        self.max_predict_range = 0
        for i in range(1,len(self.time_deltas)):
            if self.time_deltas[i] - self.max_predict_range == 1:
                self.max_predict_range = self.time_deltas[i]
            else:
                break

        self.train_iter = 0
        self.gen_freq = 4
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
            self.load_state_dict(torch.load(filename + '-virtualPGmodel'))
            if 'Lambda' in self._model:
                self._model['Lambda'] = torch.load(filename + '-virtualPGmodelLam')
        else:
            torch.save(self._model['modules'].state_dict(), filename + '-virtualPGmodel')
            if 'Lambda' in self._model:
                torch.save(self._model['Lambda'], filename + '-virtualPGmodelLam')

    def _make_model(self):
        modules = torch.nn.ModuleDict()
        modules['cnn_gen'] = netG(self.action_dim, self.args, self.state_shape, lookahead = self.max_predict_range)
        self.cnn_embed_dim = modules['cnn_gen'].embed_dim
        modules['dis'] = netD(self.action_dim, self.args, self.state_shape, lookahead = self.max_predict_range)
        if 'state' in self.state_shape:
            self.state_dim = self.state_shape['state'][0]
        else:
            self.state_dim = 0
        for o in self.state_shape:
            if 'history' in o:
                self.state_dim += self.state_shape[o][0]
        modules['state_hidden'] = torch.nn.Sequential(torch.nn.Linear(self.cnn_embed_dim+self.state_dim, self.args.hidden),
                                                      Lambda(lambda x: swish(x)))
        modules['action_predict'] = FullDiscretizerWLast(self.args.hidden, None, self.args, self.action_dict)
        
        modules['r_predict'] = torch.nn.Linear(self.cnn_embed_dim, 1, False)
        modules['v_predict_short'] = torch.nn.Linear(self.cnn_embed_dim, 2)
        modules['v_predict_long'] = torch.nn.Linear(self.cnn_embed_dim, 2)

        if self.args.ganloss == "fisher":
            lam = torch.zeros((1,), requires_grad=True)

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
        
        self.dis_params = []
        model_params = []
        nec_value_params = []
        for name, p in modules.named_parameters():
            if name.startswith('dis'):
                if p.requires_grad:
                    self.dis_params.append(p)
            else:
                if 'values.weight' in name:
                    nec_value_params.append(p)
                else:
                    model_params.append(p)

        optimizer_disc = RAdam(self.dis_params, self.args.lr)
        optimizer = RAdam(model_params, self.args.lr)
        if USE_LOOKAHEAD:
            optimizer = Lookahead(optimizer)
            optimizer_disc = Lookahead(optimizer_disc)

        model = {'modules': modules, 
                 'opt': optimizer, 
                 'opt_dis': optimizer_disc,
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

    def pretrain(self, input, worker_num=None):
        state_keys = ['state', 'pov', 'history_action', 'history_reward']
        states = {}
        lastaction = {}
        for k in input:
            if k in state_keys:
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
        done = torch.cumsum(done,1).clamp(max=1.0)
        reward[:,1:] = reward[:,1:] * (1.0 - done[:,:-1]) #mask out reward from different episode
        is_weight = None
        if 'is_weight' in input:
            is_weight = variable(input['is_weight'])
        #normalize and transpose pov state
        states['pov'] = ((states['pov'].float()+torch.rand(states['pov'].shape).cuda())/256.0)*2.0 - 1.0
        states['pov'] = states['pov'].permute(0, 1, 4, 2, 3)

        #make action vector
        action_vec = []
        for a in self.action_dict:
            if a != 'camera':
                action_vec.append(one_hot(actions[a][:,:-1], self.action_dict[a].n, self.device))
            else:
                action_vec.append(actions[a][:,:-1])
        action_vec = torch.cat(action_vec, dim=-1)
        action_vec = action_vec.view(self.args.er, -1)

        retdict = self.train_disc(states, action_vec)

        if self.train_iter % self.gen_freq == 0:
            gretdict = self.train_gen(states, action_vec, actions, lastaction, reward, is_weight)
            retdict.update(gretdict)

        #priority experience stuff
        if 'tidxs' in input:
            retdict['prio_upd'] = [{'error': retdict['bc_loss'].data.cpu().clone(),
                                    'replay_num': 0,
                                    'worker_num': worker_num[0]}]
            
        if 'bc_loss' in retdict:
            retdict['bc_loss'] = retdict['bc_loss'].mean()
        for d in retdict:
            if d != 'prio_upd':
                retdict[d] = retdict[d].data.cpu().numpy()
        self.train_iter += 1
        return retdict

    def train_disc(self, state, action_vector):
        self._model['opt_dis'].zero_grad()

        with torch.no_grad():
            prediction, _ = self._model['modules']['cnn_gen'](state['pov'][:,0], action_vector)
        
        real_scores = self._model['modules']['dis'](state['pov'], action_vector)
        fake_scores = self._model['modules']['dis'](torch.cat([state['pov'][:,0],prediction], dim=1), action_vector)
        if True:
            pict = torch.cat([state['pov'][:,0],prediction], dim=1)[0,:].permute(1, 2, 0).data.cpu().numpy()
            plt.imshow(np.concatenate([np.concatenate([pict[:,:,:3], pict[:,:,3:6], pict[:,:,6:9]], axis=1),
                                       np.concatenate([state['pov'][0,0].permute(1, 2, 0).data.cpu().numpy(), 
                                                       state['pov'][0,1].permute(1, 2, 0).data.cpu().numpy(), 
                                                       state['pov'][0,2].permute(1, 2, 0).data.cpu().numpy()], axis=1)],axis=0)*0.5 + 0.5)
            plt.show()

        if self.args.ganloss == 'hinge':
            disc_loss = torch.nn.ReLU()(1.0 - real_scores).mean() + torch.nn.ReLU()(1.0 + fake_scores).mean()
        elif self.args.ganloss == 'wasserstein' or self.args.ganloss == 'fisher':
            disc_loss = -real_scores.mean() + fake_scores.mean()
            if self.args.ganloss == 'fisher':
                constraint = (1 - (0.5*(real_scores**2).mean() + 0.5*(fake_scores**2).mean()))
                disc_loss += -self._model['Lambda'][0] * constraint + (self.args.rho/2.0) * constraint**2
        else:
            disc_loss = torch.nn.BCEWithLogitsLoss()(real_scores, torch.ones(self.args.er, 1).cuda()) + \
                    torch.nn.BCEWithLogitsLoss()(fake_scores, torch.zeros(self.args.er, 1).cuda())
            
        o_loss = 10.0 * self._model['l1loss'](prediction.view(state['pov'][:,1:].shape), state['pov'][:,1:]) + 90.0 * self._model['l2loss'](prediction.view(state['pov'][:,1:].shape), state['pov'][:,1:])
        loss = disc_loss + 0.02 * o_loss

        if self.args.ganloss == 'fisher':
            self._model['Lambda'].retain_grad()
        loss.backward()
        self._model['opt_dis'].step()
        if self.args.ganloss == 'fisher':
            self._model['Lambda'].data += self.args.rho * self._model['Lambda'].grad.data
            self._model['Lambda'].grad.data.zero_()
            return {'disc_loss': disc_loss,
                    'o_loss': o_loss,
                    'Lambda': self._model['Lambda']}
        else:
            
            return {'disc_loss': disc_loss,
                    'o_loss': o_loss}

    def train_gen(self, state, action_vector, action, lastaction, reward, is_weight):
        self._model['opt'].zero_grad()
        if 'nec_opt' in self._model:
            self._model['nec_opt'].zero_grad()

        prediction, embed = self._model['modules']['cnn_gen'](state['pov'][:,0], action_vector)
        for p in self.dis_params:
            p.requires_grad_(False)
        fake_scores = self._model['modules']['dis'](torch.cat([state['pov'][:,0],prediction], dim=1), action_vector)
        for p in self.dis_params:
            p.requires_grad_(True)
        if self.args.ganloss == 'hinge' or self.args.ganloss == 'wasserstein':
            gen_loss = -fake_scores.mean()
        else:
            gen_loss = torch.nn.BCEWithLogitsLoss()(fake_scores, torch.ones(args.batch_size, 1).cuda())


        #predict action loss
        tembed = [embed]
        for o in self.state_shape:
            if 'history' in o:
                tembed.append(state[o][:,0])
        if 'state' in state:
            tembed.append(state['state'][:,0])
        tembed = torch.cat(tembed, dim=-1)
        hidden = self._model['modules']['state_hidden'](tembed)
        faction = {}
        for a in action:
            faction[a] = action[a][:,0]
        flaction = {}
        for a in lastaction:
            flaction[a] = lastaction[a][:,0]
        loss, bc_loss, entropy, _, max_probas = self._model['modules']['action_predict'].get_loss_BC(hidden, faction, is_weight=is_weight, last_action=flaction)
        loss += gen_loss
        
        #get r loss
        target = reward[:,0]
        value = self._model['modules']['r_predict'](embed).squeeze(1)
        r_loss = self._model['loss'](value, target, is_weight)
        loss += r_loss
        
        creward = torch.cumsum(reward, 1)
        #get short v loss
        self.short_time = 10
        target = (creward[:,self.short_time] > 1e-5).long()
        logits = self._model['modules']['v_predict_short'](embed)
        v_short_loss = self._model['ce_loss'](logits, target, is_weight)
        loss += 0.1 * v_short_loss
        
        #get long v loss
        self.long_time = reward.shape[1]
        target = (creward[:,(self.long_time-1)] > 1e-5).long()
        logits = self._model['modules']['v_predict_long'](embed)
        v_long_loss = self._model['ce_loss'](logits, target, is_weight)
        loss += 0.1 * v_long_loss

        loss.backward()
        self._model['opt'].step()
        if 'nec_opt' in self._model:
            self._model['nec_opt'].step()

        return {'gen_loss': gen_loss,
                'entropy': entropy,
                'max_probas': max_probas,
                'bc_loss': bc_loss, 
                'r_loss': r_loss,
                'v_short_loss': v_short_loss,
                'v_long_loss': v_long_loss}

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
                    tstate[k] = state[k][None, None,:].copy()
        tstate = variable(tstate)
        with torch.no_grad():
            tstate['pov'] = ((tstate['pov'].permute(0, 1, 4, 2, 3)+0.5)/256.0)*2.0 - 1.0
            embed = self._model['modules']['cnn_gen'].forward_embed(tstate['pov'][:,0])
            tembed = [embed]
            for o in self.state_shape:
                if 'history' in o:
                    tembed.append(tstate[o][:,0])
            if 'state' in state:
                tembed.append(tstate['state'][:,0])
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
        return action, {'entropy': entropy.data.cpu().numpy()}