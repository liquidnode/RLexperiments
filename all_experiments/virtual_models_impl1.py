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
from mixture import MixtureSameFamily
import argparse
import copy
import prio_utils
import matplotlib.pyplot as plt

GAMMA = 0.99
LAMBDA = 0.95
COMPATIBLE = False

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

class Discretizer(torch.nn.Module):
    def __init__(self, num_input, device, args):
        super(Discretizer, self).__init__()
        self.n_actions = 6 #inc/dec X/Y, inc delta, commit
        self.fc0 = HashingMemory.build(num_input+3, self.n_actions, args)
                   #torch.nn.Sequential(torch.nn.Linear(num_input+3, 128),
                   #                       Lambda(lambda x: swish(x)),
                   #                       torch.nn.Linear(128, self.n_actions))
        #self.nonlin = Lambda(lambda x: swish(x))
        self.ce_loss = torch.nn.CrossEntropyLoss(reduction='none')
        self.softmax = torch.nn.Softmax(1)
        self.device = device
        self.min_delta = 0.5 / 2**3
        self.max_step = 6
        self.current_ac = None
        self.current_delta = None

    def get_loss_BC(self, input, real_actions):
        if len(input.shape) > 2:
            input = input.view(-1, num_input)
            real_actions = real_actions.view(-1, 2)
        #input = self.nonlin(input)
        all_dacs = []
        if self.current_ac is None or self.current_ac.shape[0] != input.shape[0]:
            self.current_ac = torch.zeros((input.shape[0],2), dtype=torch.float32, device=self.device, requires_grad=False)
            self.current_delta = torch.ones((input.shape[0],), dtype=torch.float32, device=self.device, requires_grad=False) * self.min_delta
            self.current_dac = torch.zeros((input.shape[0],), dtype=torch.int64, device=self.device, requires_grad=False)
        else:
            self.current_ac.fill_(0.0)
            self.current_delta.fill_(self.min_delta)
            self.current_dac.fill_(0)
        loss = None
        last_commit_mask = None
        delta_mask = None
        norm = None
        entropy = None
        for i in range(self.max_step):
            #get logits
            c_logits = self.fc0(torch.cat([input, self.current_ac.clone(), self.current_delta.unsqueeze(-1).clone()], dim=-1))
            with torch.no_grad():
                probs = self.softmax(c_logits)
                if entropy is None:
                    entropy = -(probs*torch.log(probs)).sum(-1).mean().data.cpu().numpy()
                else:
                    entropy += -(probs*torch.log(probs)).sum(-1).mean().data.cpu().numpy()
                #reset dac
                self.current_dac.fill_(0)
                #check commit
                commit_mask = (torch.abs(real_actions[:,0]) < 1e-5) & (torch.abs(real_actions[:,1]) < 1e-5)
                self.current_dac.masked_fill_(commit_mask, 5)
                #check inc delta
                if delta_mask is None:
                    delta_mask = (~commit_mask) & ((torch.abs(real_actions[:,0]) > (self.current_delta * 2.0)) | (torch.abs(real_actions[:,1]) > (self.current_delta * 2.0)))
                else:
                    delta_mask = (~commit_mask) & ((torch.abs(real_actions[:,0]) > (self.current_delta * 2.0)) | (torch.abs(real_actions[:,1]) > (self.current_delta * 2.0))) & delta_mask
                self.current_dac.masked_fill_(delta_mask, 4)
                self.current_delta = torch.where(delta_mask, self.current_delta * 2.0, self.current_delta)
                #check inc/dec X
                incX_mask = (~commit_mask) & (~delta_mask) & (real_actions[:,0] > self.current_ac[:,0])
                self.current_dac.masked_fill_(incX_mask, 1)
                self.current_ac[:,0] = torch.where(commit_mask | delta_mask, self.current_ac[:,0], torch.where(incX_mask, self.current_ac[:,0] + self.current_delta, self.current_ac[:,0] - self.current_delta))
                #check inc/dec Y
                incY_mask = (~commit_mask) & (~delta_mask) & (real_actions[:,1] > self.current_ac[:,1])
                self.current_dac = torch.where(incY_mask, self.current_dac + 2, self.current_dac)
                self.current_ac[:,1] = torch.where(commit_mask | delta_mask, self.current_ac[:,1], torch.where(incY_mask, self.current_ac[:,1] + self.current_delta, self.current_ac[:,1] - self.current_delta))
                #update delta
                self.current_delta = torch.where((~commit_mask) & (~delta_mask), self.current_delta * 0.5, self.current_delta)
                if last_commit_mask is not None:
                    norm += (~last_commit_mask).float()
                else:
                    norm = torch.ones_like(c_logits[:,0], requires_grad=False)
                #print(self.current_dac.data.cpu().numpy())
            dac_clone = self.current_dac.clone()
            if loss is None:
                loss = self.ce_loss(c_logits, dac_clone)
            else:
                loss += self.ce_loss(c_logits, dac_clone) * (~last_commit_mask).float().clone()
            all_dacs.append(dac_clone)
            if last_commit_mask is None:
                last_commit_mask = commit_mask
            else:
                last_commit_mask = commit_mask | last_commit_mask
        #print(self.current_ac.data.cpu().numpy())
        entropy /= self.max_step
        loss /= norm
        return loss.mean(), entropy, torch.stack(all_dacs, dim=1)

    def sample(self, input, current_dacs=None):
        with torch.no_grad():
            if len(input.shape) > 2:
                input = input.view(-1, num_input)
                real_actions = real_actions.view(-1, 2)
            #input = self.nonlin(input)
            if self.current_ac is None or self.current_ac.shape[0] != input.shape[0]:
                self.current_ac = torch.zeros((input.shape[0],2), dtype=torch.float32, device=self.device, requires_grad=False)
                self.current_delta = torch.ones((input.shape[0],), dtype=torch.float32, device=self.device, requires_grad=False) * self.min_delta
            else:
                self.current_ac.fill_(0.0)
                self.current_delta.fill_(self.min_delta)
            entropy = None
            commit_mask = None
            delta_mask = None
            for i in range(self.max_step):
                if current_dacs is None:
                    c_logits = self.fc0(torch.cat([input, self.current_ac, self.current_delta.unsqueeze(-1)], dim=-1))
                    #print(self.softmax(c_logits).data.cpu().numpy())
                    #time.sleep(1.0)
                    current_dac = torch.distributions.Categorical(logits=c_logits)
                    current_dac = current_dac.sample()
                else:
                    current_dac = current_dacs[:,i]
                #print(current_dac.data.cpu().numpy())
                #check commit
                if commit_mask is None:
                    commit_mask = (current_dac == 5)
                else:
                    commit_mask = (current_dac == 5) | commit_mask
                if (~commit_mask).sum() == 0:
                    break
                #check delta
                if delta_mask is None:
                    delta_mask = (current_dac == 4) & (self.current_delta < 0.5)
                else:
                    delta_mask = (current_dac == 4) & (self.current_delta < 0.5) & delta_mask
                self.current_delta = torch.where(delta_mask, self.current_delta * 2.0, self.current_delta)
                #check inc/dec X
                incX_mask = (~commit_mask) & (~delta_mask) & (current_dac < 4) & (current_dac.byte() & 1)
                self.current_ac[:,0] = torch.where(commit_mask | delta_mask, self.current_ac[:,0], torch.where(incX_mask, self.current_ac[:,0] + self.current_delta, self.current_ac[:,0] - self.current_delta))
                #check inc/dec Y
                incY_mask = (~commit_mask) & (~delta_mask) & (current_dac < 4) & ((current_dac.byte() & 2) >> 1)
                self.current_ac[:,1] = torch.where(commit_mask | delta_mask, self.current_ac[:,1], torch.where(incY_mask, self.current_ac[:,1] + self.current_delta, self.current_ac[:,1] - self.current_delta))
                self.current_delta = torch.where((~commit_mask) & (~delta_mask), self.current_delta * 0.5, self.current_delta)
            rand_deltaX = torch.distributions.Uniform(-self.current_delta*2.0,self.current_delta*2.0).sample()
            rand_deltaY = torch.distributions.Uniform(-self.current_delta*2.0,self.current_delta*2.0).sample()
            self.current_ac[:,0] = torch.where(~commit_mask, self.current_ac[:,0], self.current_ac[:,0] + rand_deltaX)
            self.current_ac[:,1] = torch.where(~commit_mask, self.current_ac[:,1], self.current_ac[:,1] + rand_deltaY)
            return self.current_ac


class FullDiscretizer(torch.nn.Module):
    def __init__(self, num_input, device, args, action_dict):
        super(FullDiscretizer, self).__init__()
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
            c_num_actions = action_dict[a].n-1 #without noop
            self.logit_dim += action_dict[a].n
            self.n_actions += c_num_actions
            self.action_ranges[a] = [range_start, range_start+c_num_actions]
            range_start += c_num_actions
        self.fc0 = HashingMemory.build(num_input+4+self.logit_dim, self.n_actions, args)
        self.softmax = torch.nn.Softmax(1)
        self.ce_loss = torch.nn.CrossEntropyLoss(reduction='none')
        self.min_delta = 0.5 / 2**3
        self.max_step = 20
        self.max_samplestep = 12
        self.max_camstep = 6
        self.current_ac = None
        self.scurrent_ac = None

    def get_loss_BC(self, fc_input, action, test=False):
        if len(fc_input.shape) > 2:
            assert False
        if self.current_ac is None or self.current_ac.shape[0] != fc_input.shape[0]:
            self.current_ac = torch.zeros((fc_input.shape[0],2), dtype=torch.float32, device=self.device, requires_grad=False)
            self.current_delta = torch.ones((fc_input.shape[0],), dtype=torch.float32, device=self.device, requires_grad=False) * self.min_delta
            self.current_dac = torch.zeros((fc_input.shape[0],), dtype=torch.int64, device=self.device, requires_grad=False)
            self.delta_chosen = torch.zeros((fc_input.shape[0],), dtype=torch.uint8, device=self.device, requires_grad=False)
            self.current_rdac = {}
            for a in self.action_dict:
                self.current_rdac[a] = torch.zeros((fc_input.shape[0],), dtype=torch.int64, device=self.device, requires_grad=False)
        else:
            self.current_ac.fill_(0.0)
            self.current_delta.fill_(self.min_delta)
            self.current_dac.fill_(0)
            for a in self.action_dict:
                self.current_rdac[a].fill_(0)
            self.delta_chosen.fill_(0)
        loss = None
        cam_steps = None
        last_commit_mask = None
        norm = None
        all_dacs = []
        entropy = None
        commited = None
        for i in range(self.max_step):
            c_one_hot = []
            for a in self.action_dict:
                c_one_hot.append(one_hot(self.current_rdac[a], self.action_dict[a].n, self.device))
            c_one_hot = torch.cat(c_one_hot, dim=1)
            c_logits = self.fc0(torch.cat([fc_input, 
                                           self.current_ac.clone(), 
                                           self.current_delta.unsqueeze(-1).clone(), 
                                           self.delta_chosen.unsqueeze(-1).float().clone(), 
                                           c_one_hot], dim=-1))
            with torch.no_grad():
                probs = self.softmax(c_logits)
                if entropy is None:
                    entropy = -((probs*torch.log(probs)).sum(-1))
                else:
                    entropy += -((probs*torch.log(probs)).sum(-1)*(~commited).float())
                #reset dac
                self.current_dac.fill_(0)
                #check commit
                if cam_steps is None:
                    commit_mask = ((torch.abs(action['camera'][:,0]) < 1e-5) & (torch.abs(action['camera'][:,1]) < 1e-5))
                else:
                    commit_mask = ((torch.abs(action['camera'][:,0]) < 1e-5) & (torch.abs(action['camera'][:,1]) < 1e-5)) | (cam_steps >= self.max_camstep)
                for a in self.action_dict:
                    commit_mask &= (self.current_rdac[a] == action[a])
                self.current_dac.masked_fill_(commit_mask, 0)
                modified_dac = commit_mask.clone()
                #check discrete actions
                for a in self.action_dict:
                    disc_mask = (~modified_dac) & (self.current_rdac[a] != action[a])
                    self.current_dac = torch.where(disc_mask, action[a]-1+self.action_ranges[a][0], self.current_dac)
                    self.current_rdac[a] = torch.where(disc_mask, action[a], self.current_rdac[a])
                    modified_dac |= disc_mask
                #check inc delta
                delta_mask = (~modified_dac) & (~self.delta_chosen) & ((torch.abs(action['camera'][:,0]-self.current_ac[:,0]) > (self.current_delta * 2.0)) | (torch.abs(action['camera'][:,1]-self.current_ac[:,0]) > (self.current_delta * 2.0)))
                self.current_dac.masked_fill_(delta_mask, 5)
                self.current_delta = torch.where(delta_mask, self.current_delta * 2.0, self.current_delta)
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
                self.delta_chosen |= ~modified_dac
                #update cam step
                if cam_steps is None:
                    cam_steps = (~modified_dac).long()
                else:
                    cam_steps += (~modified_dac).long()
                #add norm
                if commited is not None:
                    norm += (~commited).float()
                else:
                    norm = torch.ones_like(c_logits[:,0], requires_grad=False)
                #add loss
            dac_clone = self.current_dac.clone()
            if loss is None:
                loss = self.ce_loss(c_logits, dac_clone)
            else:
                loss += self.ce_loss(c_logits, dac_clone) * ((~commited).float().clone())
            all_dacs.append(dac_clone)
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
        entropy /= norm
        loss /= norm
        return loss.mean(), entropy.mean(), torch.stack(all_dacs, dim=1)

    def sample(self, fc_input, current_dacs=None):
        with torch.no_grad():
            if len(fc_input.shape) > 2:
                assert False
            if self.scurrent_ac is None or self.scurrent_ac.shape[0] != fc_input.shape[0]:
                self.scurrent_ac = torch.zeros((fc_input.shape[0],2), dtype=torch.float32, device=self.device, requires_grad=False)
                self.scurrent_delta = torch.ones((fc_input.shape[0],), dtype=torch.float32, device=self.device, requires_grad=False) * self.min_delta
                self.sdelta_chosen = torch.zeros((fc_input.shape[0],), dtype=torch.uint8, device=self.device, requires_grad=False)
                self.scurrent_rdac = {}
                for a in self.action_dict:
                    self.scurrent_rdac[a] = torch.zeros((fc_input.shape[0],), dtype=torch.int64, device=self.device, requires_grad=False)
                self.scurrent_rdac_changed = {}
                for a in self.action_dict:
                    self.scurrent_rdac_changed[a] = torch.zeros((fc_input.shape[0],), dtype=torch.uint8, device=self.device, requires_grad=False)
            else:
                self.scurrent_ac.fill_(0.0)
                self.scurrent_delta.fill_(self.min_delta)
                for a in self.action_dict:
                    self.scurrent_rdac[a].fill_(0)
                    self.scurrent_rdac_changed[a].fill_(0)
                self.sdelta_chosen.fill_(0)

            commited = None
            entropy = None
            norm = None
            for i in range(self.max_samplestep):
                c_one_hot = []
                for a in self.action_dict:
                    c_one_hot.append(one_hot(self.scurrent_rdac[a], self.action_dict[a].n, self.device))
                c_one_hot = torch.cat(c_one_hot, dim=1)
                c_logits = self.fc0(torch.cat([fc_input, 
                                               self.scurrent_ac.clone(), 
                                               self.scurrent_delta.unsqueeze(-1).clone(), 
                                               self.sdelta_chosen.unsqueeze(-1).float().clone(), 
                                               c_one_hot], dim=-1))
                if entropy is None:
                    probs = self.softmax(c_logits)
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
                    disc_mask = (~self.scurrent_rdac_changed[a]) & (current_dac >= self.action_ranges[a][0]) & (current_dac < self.action_ranges[a][1])
                    self.scurrent_rdac[a] = torch.where(disc_mask, current_dac - self.action_ranges[a][0] + 1, self.scurrent_rdac[a])
                    self.scurrent_rdac_changed[a] |= disc_mask
                #check delta
                delta_mask = (~self.sdelta_chosen) & (current_dac == 5)
                self.scurrent_delta = torch.where(delta_mask, self.scurrent_delta * 2.0, self.scurrent_delta)
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
        return self.scurrent_rdac, self.scurrent_ac, entropy.mean()


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
        self.softmax = torch.nn.Softmax(1)
        self.ce_loss = torch.nn.CrossEntropyLoss(reduction='none')
        if COMPATIBLE:
            self.min_delta = 0.05#0.5 / 2**3
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

    def get_loss_BC_old(self, fc_input, action, last_action=None, is_weight=None, test=False):
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
        for i in range(self.max_step):
            c_one_hot = []
            for a in self.action_dict:
                if a != 'camera':
                    c_one_hot.append(one_hot(self.current_rdac[a], self.action_dict[a].n, self.device))
                else:
                    c_one_hot.append(self.current_ac.clone())
            c_one_hot = torch.cat(c_one_hot, dim=1)
            c_logits = self.fc0(torch.cat([fc_input,
                                            c_one_hot,
                                            self.current_delta.unsqueeze(-1).clone(), 
                                            self.delta_chosen.unsqueeze(-1).float().clone()], dim=-1))
            with torch.no_grad():
                probs = self.softmax(c_logits)
                if entropy is None:
                    entropy = -((probs*torch.log(probs)).sum(-1))
                else:
                    entropy += -((probs*torch.log(probs)).sum(-1)*(~commited).float())
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
                    norm = torch.ones_like(c_logits[:,0], requires_grad=False)
            #add loss
            dac_clone = self.current_dac.clone()
            if COMPATIBLE:
                #compatability with old system
                _camera_mask = (dac_clone < 6) & (dac_clone >= 1)
                _disc_mask = (dac_clone < self.n_actions-1) & (dac_clone >= 6)
                dac_clone = torch.where(_camera_mask, dac_clone-1+self.n_actions-6, dac_clone)
                dac_clone = torch.where(_disc_mask, dac_clone-5, dac_clone)
            if loss is None:
                loss = self.ce_loss(c_logits, dac_clone)
            else:
                loss += self.ce_loss(c_logits, dac_clone) * ((~commited).float().clone())
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
        entropy /= norm
        #loss /= norm
        #print(loss.shape)
        #print(loss.data.cpu().numpy())
        #print(norm.data.cpu().numpy())
        #time.sleep(100.0)
        #print(torch.stack(all_dacs, dim=1)[0].data.cpu().numpy())
        #print(torch.stack(all_odacs, dim=1)[0].data.cpu().numpy())
        #time.sleep(10.0)
        return loss.mean(), entropy.mean(), torch.stack(all_dacs, dim=1)


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
        #for a in self.action_dict:
        #    print(a)
        #    break
        #print(all_inputs[0,:,0].data.cpu().numpy())
        #print(all_targets[0,:].data.cpu().numpy())
        if False:
            rand_cam = (torch.rand(norm.shape,device=self.device) * cam_steps.float())
            rand_sam = (torch.rand(norm.shape,device=self.device) * (norm - rand_cam)).long()
        else:
            rand_sam = (torch.rand(norm.shape,device=self.device) * norm).long()
        #print(rand_sam[0].data.cpu().numpy())
        input = all_inputs.gather(1, rand_sam.unsqueeze(1).unsqueeze(2).expand(-1,-1,all_inputs.shape[-1])).squeeze(1)
        target = all_targets.gather(1, rand_sam.unsqueeze(1)).squeeze(1)
        #print(input[0,0].data.cpu().numpy())
        input = torch.cat([fc_input, input], dim=-1)
        c_logits = self.fc0(input)
        loss = self.ce_loss(c_logits, target)
        probs = self.softmax(c_logits)
        #print('p',probs[0,self.action_ranges['attack_place_equip_craft_nearbyCraft_nearbySmelt'][0]].data.cpu().numpy())
        #print(target[0].data.cpu().numpy())
        entropy = -((probs*torch.log(probs)).sum(-1))
        max_probs = torch.mean(torch.max(probs,dim=1)[0])
        #time.sleep(5)
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


class VirtualPGFeature():
    def __init__(self, state_shape, action_dict, args, time_deltas):
        assert isinstance(state_shape, dict)
        self.state_shape = state_shape
        self.action_dict = copy.deepcopy(action_dict)
        self.args = args
        
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
            if COMPATIBLE:
                print('Load fisherBC')
                sdict = torch.load('treechop/nec_model_fisherbc2_atari-fisherBC')
                rdict = {}
                for n in sdict:
                    if n.startswith('cnn_model'):
                        rdict[n] = sdict[n]
                    if n.startswith('macro_feature'):
                        rdict[n.replace('macro_feature','state_hidden')] = sdict[n]
                    if n.startswith('action_head'):
                        rdict[n.replace('action_head','action_predict.fc0')] = sdict[n]
                    if n.startswith('deep_infomax.'):
                        rdict[n.replace('deep_infomax','cnn_disc')] = sdict[n]
                self.load_state_dict(rdict, False)
                self._model['Lambda'] = torch.load('treechop/nec_model_fisherbc2_atari-fisherLam')
            else:
                print('Load actor')
                self.load_state_dict(torch.load(filename + '-virtualfeaturePG'))
                self._model['Lambda'] = torch.load(filename + '-virtualfeaturePGLam')
        else:
            torch.save(self._model['modules'].state_dict(), filename + '-virtualfeaturePG')
            torch.save(self._model['Lambda'], filename + '-virtualfeaturePGLam')

    def _make_model(self):
        #make CNN model
        modules = torch.nn.ModuleDict()
        cnn_layers = []
        assert len(self.state_shape['pov']) > 1
        if self.args.cnn_type == 'atari':
            sizes = [8, 4, 3]
            strides = [4, 2, 1]
            pooling = [1, 1, 1]
            filters = [32, 64, 32]
        elif self.args.cnn_type == 'mnist':
            sizes = [3, 3, 3]
            strides = [1, 1, 1]
            pooling = [2, 2, 2]
            filters = [32, 32, 32]

        in_channels = self.state_shape['pov'][-1] #HWC input

        for i in range(len(sizes)):
            cnn_layers.append(torch.nn.Conv2d(
                in_channels,
                filters[i],
                sizes[i],
                stride=strides[i],
                bias=True
            ))

            cnn_layers.append(Lambda(lambda x: swish(x)))

            if pooling[i] > 1:
                cnn_layers.append(torch.nn.MaxPool2d(pooling[i]))

            in_channels = filters[i]

        cnn_layers.append(Flatten())
        #modules['flatten'] = Flatten()
        modules['cnn_model'] = torch.nn.Sequential(*cnn_layers)
        
        #self.cnn_embed_shape = list(modules['cnn_model'](torch.zeros((1,self.state_shape['pov'][2], self.state_shape['pov'][0], self.state_shape['pov'][1]))).shape[1:])
        #self.cnn_embed_dim = self.cnn_embed_shape[0]*self.cnn_embed_shape[1]*self.cnn_embed_shape[2]
        self.comressed_pov_dim = self.args.hidden
        self.cnn_embed_dim = modules['cnn_model'](torch.zeros((1,self.state_shape['pov'][2], self.state_shape['pov'][0], self.state_shape['pov'][1]))).shape[1]
        #cnn_fea = []
        #cnn_layers.append(torch.nn.Linear(self.cnn_embed_dim, self.comressed_pov_dim))
        #cnn_fea.append(torch.nn.BatchNorm1d(self.comressed_pov_dim))
        #cnn_layers.append(Lambda(lambda x: swish(x)))#torch.nn.Tanh())#
        #cnn_layers.append(SkipHashingMemory(self.comressed_pov_dim, self.args))
        #modules['cnn_model'] = torch.nn.Sequential(*cnn_layers)
        #self.cnn_embed_dim = self.comressed_pov_dim

        if False:
            #self.infomax_dim = 64
            #modules['cnn_info'] = torch.nn.Sequential(torch.nn.Linear(self.cnn_embed_dim, self.infomax_dim),
            #                                          Lambda(lambda x: swish(x)))
            self.infomax_dim = self.cnn_embed_dim
            modules['cnn_disc'] = torch.nn.Linear(self.infomax_dim, self.infomax_dim, False)
        else:
            if COMPATIBLE:
                modules['cnn_disc'] = torch.nn.Sequential(torch.nn.Linear(self.args.hidden*2, self.args.hidden), 
                                                          Lambda(lambda x: swish(x)), 
                                                          torch.nn.Linear(self.args.hidden, 1))
            else:
                modules['cnn_disc'] = torch.nn.Sequential(torch.nn.Linear(self.args.hidden*2, self.args.hidden), 
                                                          Lambda(lambda x: swish(x)), 
                                                          torch.nn.Linear(self.args.hidden, 1, False))
        lam = torch.zeros((1,), requires_grad=True)
        
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
        #make single predict
        self.SINGLE_PREDICT = False
        if self.SINGLE_PREDICT:
            for a in self.action_dict:
                if a != 'camera':
                    modules['s_predict_'+a] = torch.nn.Linear(self.args.hidden, self.action_dict[a].n)


        if torch.cuda.is_available() and CUDA:
            modules = modules.cuda()
            lam = lam.cuda()
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        modules['action_predict'].device = self.device

        ce_loss = prio_utils.WeightedCELoss()
        loss = prio_utils.WeightedMSELoss()

        model_params = []
        nec_value_params = []
        for name, p in modules.named_parameters():
            if 'values.weight' in name:
                nec_value_params.append(p)
            else:
                model_params.append(p)
        #optimizer = torch.optim.Adam([{'params': model_params}, 
        #                              {'params': nec_value_params, 'lr': self.args.nec_lr}], lr=self.args.lr)
        optimizer = RAdam(model_params, self.args.lr)
        if len(nec_value_params) > 0:
            nec_optimizer = torch.optim.Adam(nec_value_params, self.args.nec_lr)

            return {'modules': modules, 
                    'opt': optimizer, 
                    'nec_opt': nec_optimizer,
                    'ce_loss': ce_loss,
                    'loss': loss,
                    'Lambda': lam}
        else:
            return {'modules': modules, 
                    'opt': optimizer, 
                    'ce_loss': ce_loss,
                    'loss': loss,
                    'Lambda': lam}
    
    def get_cnn_embed(self, state, _range=None):
        if _range is None:
            x = (state['pov'].permute(0, 1, 4, 2, 3)/255.0)*2.0 - 1.0
        else:
            x = (state['pov'][:,:_range].permute(0, 1, 4, 2, 3)/255.0)*2.0 - 1.0
        batch_size = state['pov'].shape[0]
        n_view = [-1] + [self.state_shape['pov'][2]] + self.state_shape['pov'][:2]
        cnn_fea = self._model['modules']['cnn_model'](x.view(*n_view))
        #cnn_fea = self._model['modules']['cnn_fea'](cnn_fea)
        n_view = [batch_size,-1,self.cnn_embed_dim]
        return cnn_fea.view(*n_view)

    def inverse_map_camera(self, x):
        return x / 180.0

    def map_camera(self, x):
        return x * 180.0

    def get_BC_value_loss(self, state, action, reward, done, cnn_embed, lastaction, is_weight=None):
        tembed = [cnn_embed[:,0]]
        for o in self.state_shape:
            if 'history' in o:
                tembed.append(state[o][:,0])
        if 'state' in state:
            tembed.append(state['state'][:,0])
        tembed = torch.cat(tembed, dim=-1)
        hidden = self._model['modules']['state_hidden'](tembed)

        #BC loss
        faction = {}
        for a in action:
            faction[a] = action[a][:,0]
            #print(a,faction[a][100].data.cpu().numpy())
        flaction = {}
        for a in lastaction:
            flaction[a] = lastaction[a][:,0]

        loss, bc_loss, entropy, _, max_probas = self._model['modules']['action_predict'].get_loss_BC(hidden, faction, is_weight=is_weight, last_action=flaction)

        #get r loss
        target = reward[:,0]
        value = self._model['modules']['r_predict'](cnn_embed[:,0]).squeeze(1)
        r_loss = self._model['loss'](value, target, is_weight)
        loss += r_loss

        creward = torch.cumsum(reward, 1)
        #get short v loss
        self.short_time = 10
        target = (creward[:,self.short_time] > 1e-5).long()
        #print(target[100].data.cpu().numpy())
        #print((state['pov'][100,0]-state['pov'][100,1]).abs().mean().data.cpu().numpy())
        #plt.imshow(np.concatenate([state['pov'][100,0].data.cpu().numpy(),state['pov'][100,1].data.cpu().numpy()],1)/255.0)
        #plt.show()
        logits = self._model['modules']['v_predict_short'](cnn_embed[:,0])
        v_short_loss = self._model['ce_loss'](logits, target, is_weight)
        loss += 0.1 * v_short_loss

        #get long v loss
        self.long_time = reward.shape[1]
        target = (creward[:,(self.long_time-1)] > 1e-5).long()
        logits = self._model['modules']['v_predict_long'](cnn_embed[:,0])
        v_long_loss = self._model['ce_loss'](logits, target, is_weight)
        loss += 0.1 * v_long_loss
        
        if self.SINGLE_PREDICT:
            s_loss = None
            for a in self.action_dict:
                if a != 'camera':
                    target = faction[a]
                    logits = self._model['modules']['s_predict_'+a](hidden)
                    if s_loss is None:
                        s_loss = self._model['ce_loss'](logits, target, is_weight)
                    else:
                        s_loss += self._model['ce_loss'](logits, target, is_weight)
            loss += 0.5 * s_loss

        return loss, {'entropy': entropy,
                      'max_probas': max_probas,
                      #'s_loss': s_loss,
                      'bc_loss': bc_loss, 
                      'r_loss': r_loss,
                      'v_short_loss': v_short_loss,
                      'v_long_loss': v_long_loss}

    def get_DIM_loss(self, cnn_embed, state):
        if False:
            cnn_info = cnn_embed#self._model['modules']['cnn_info'](cnn_embed)
            finfo = self._model['modules']['cnn_disc'](cnn_info[:,0]).unsqueeze(2)
            real_score = torch.bmm(cnn_info[:,1].unsqueeze(1), finfo)
            random_roll = np.random.randint(1, self.args.er-1)
            fake_score = torch.bmm(cnn_info[:,1].roll(random_roll,0).unsqueeze(1), finfo)
        else:
            if True:
                tembed = [cnn_embed]
                for o in self.state_shape:
                    if 'history' in o:
                        tembed.append(state[o])
                if 'state' in state:
                    tembed.append(state['state'])
                tembed = torch.cat(tembed, dim=-1)
                hidden = self._model['modules']['state_hidden'](tembed)
            else:
                hidden = cnn_embed
            real_pairs = torch.cat([hidden[:,0], hidden[:,1]], dim=1)
            real_score = self._model['modules']['cnn_disc'](real_pairs)
            random_roll = np.random.randint(1, self.args.er-1)
            fake_pairs = torch.cat([hidden[:,0], hidden[:,1].roll(random_roll, 0)], dim=1)
            fake_score = self._model['modules']['cnn_disc'](fake_pairs)


        real_f, fake_f = real_score.mean(), fake_score.mean()
        real_f2, fake_f2 = (real_score**2).mean(), (fake_score**2).mean()
        self.constraint = (1 - (0.5*real_f2 + 0.5*fake_f2))

        fisher_infomax_loss = real_f - fake_f + self._model['Lambda'][0] * self.constraint - (self.args.rho/2.0) * self.constraint**2
        loss = -0.1 * fisher_infomax_loss
        return loss, {'fisher_infomax_loss': fisher_infomax_loss,
                      'Lambda': self._model['Lambda'][0]}

    def train(self, input):
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

        self._model['opt'].zero_grad()
        if 'nec_opt' in self._model:
            self._model['nec_opt'].zero_grad()
        cnn_embed = self.get_cnn_embed(states)
        loss, retdict = self.get_BC_value_loss(states, actions, reward, done, cnn_embed, lastaction, is_weight)
        NODIM = False
        if not NODIM:
            DIMloss, DIMretdict = self.get_DIM_loss(cnn_embed, states)
            loss += DIMloss
            retdict.update(DIMretdict)

            self._model['Lambda'].retain_grad()
        loss.backward()
        
        self._model['opt'].step()
        if 'nec_opt' in self._model:
            self._model['nec_opt'].step()
        if not NODIM:
            self._model['Lambda'].data += self.args.rho * self._model['Lambda'].grad.data
            self._model['Lambda'].grad.data.zero_()

        if 'tidxs' in input:
            retdict['prio_upd'] = [{'error': retdict['bc_loss'].data.cpu().clone(),
                                    'replay_num': 0,
                                    'worker_num': worker_num[0]}]

        retdict['bc_loss'] = retdict['bc_loss'].mean()
        for d in retdict:
            if d != 'prio_upd':
                retdict[d] = retdict[d].data.cpu().numpy()
        return retdict

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
            cnn_embed = self.get_cnn_embed(tstate)
            tembed = [cnn_embed[:,0]]
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




class VirtualPGOld():
    def __init__(self, state_shape, action_dict, args, time_deltas):
        assert isinstance(state_shape, dict)
        self.state_shape = state_shape
        self.action_dict = copy.deepcopy(action_dict)
        self.args = args
        
        self.time_deltas = time_deltas
        #get GVE parameters
        assert self.time_deltas[0] == 0
        self.GVE_weigths = []
        self.GVE_norm = 0.0
        for i in range(1,len(self.time_deltas)):
            start = LAMBDA ** self.time_deltas[i]
            if i != len(self.time_deltas) - 1:
                sum_w = 1.0 - LAMBDA**(self.time_deltas[i+1]-self.time_deltas[i])
                sum_w /= 1.0 - LAMBDA
            else:
                sum_w = 1.0 / (1.0 - LAMBDA)
            self.GVE_weigths.append(start*sum_w)
            self.GVE_norm += start*sum_w
            
        self.max_predict_range = 0
        for i in range(len(self.time_deltas)):
            if self.time_deltas[i] - self.max_predict_range == 1:
                self.max_predict_range = self.time_deltas[i]
            else:
                break
        self.gen_ph = 0
        self.gen_train_freq = 5

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
            if COMPATIBLE:
                print('Load fisherBC')
                sdict = torch.load('treechop/nec_model_fisherbc2_atari-fisherBC')
                rdict = {}
                for n in sdict:
                    if n.startswith('cnn_model'):
                        rdict[n] = sdict[n]
                    if n.startswith('macro_feature'):
                        rdict[n.replace('macro_feature','action_v_hidden')] = sdict[n]
                    if n.startswith('action_head'):
                        rdict[n.replace('action_head','action_predict.fc0')] = sdict[n]
                self.load_state_dict(rdict)
            else:
                print('Load actor')
                self.load_state_dict(torch.load(filename + '-virtualPG'), False)
                self._model['Lambda'] = torch.load(filename + '-virtualPGLam')
        else:
            torch.save(self._model['modules'].state_dict(), filename + '-virtualPG')
            torch.save(self._model['Lambda'], filename + '-virtualPGLam')

    def _make_model(self):
        #make CNN
        modules = torch.nn.ModuleDict()
        cnn_layers = []
        assert len(self.state_shape['pov']) > 1
        if self.args.cnn_type == 'atari':
            sizes = [8, 4, 3]
            strides = [4, 2, 1]
            pooling = [1, 1, 1]
            filters = [32, 64, 32]
        elif self.args.cnn_type == 'mnist':
            sizes = [3, 3, 3]
            strides = [1, 1, 1]
            pooling = [2, 2, 2]
            filters = [32, 32, 32]

        in_channels = self.state_shape['pov'][-1] #HWC input

        for i in range(len(sizes)):
            cnn_layers.append(torch.nn.Conv2d(
                in_channels,
                filters[i],
                sizes[i],
                stride=strides[i],
                bias=True
            ))

            cnn_layers.append(Lambda(lambda x: swish(x)))

            if pooling[i] > 1:
                cnn_layers.append(torch.nn.MaxPool2d(pooling[i]))

            in_channels = filters[i]

        #cnn_layers.append(Flatten())
        modules['flatten'] = Flatten()
        modules['cnn_model'] = torch.nn.Sequential(*cnn_layers)
        
        self.cnn_embed_shape = list(modules['cnn_model'](torch.zeros((1,self.state_shape['pov'][2], self.state_shape['pov'][0], self.state_shape['pov'][1]))).shape[1:])
        self.cnn_embed_dim = self.cnn_embed_shape[0]*self.cnn_embed_shape[1]*self.cnn_embed_shape[2]

        self.action_logits_dim = 0
        self.action_contin_dim = 0
        for a in self.action_dict:
            if isinstance(self.action_dict[a], gym.spaces.Box):
                self.action_contin_dim += self.action_dict[a].shape[0]
            else:
                self.action_logits_dim += self.action_dict[a].n
        oaction_dict = copy.deepcopy(self.action_dict)
        #del oaction_dict['camera']

        if 'state' in self.state_shape:
            self.state_dim = self.state_shape['state'][0]
        else:
            self.state_dim = 0
        for o in self.state_shape:
            if 'history' in o:
                self.state_dim += self.state_shape[o][0]
        if COMPATIBLE:
            modules['action_v_hidden'] = torch.nn.Sequential(torch.nn.Linear(self.cnn_embed_dim+self.state_dim, self.args.hidden),
                                                             Lambda(lambda x: swish(x)))
        else:
            modules['action_v_hidden'] = torch.nn.Sequential(torch.nn.Linear(self.cnn_embed_dim+self.state_dim, self.args.hidden),
                                                             torch.nn.Tanh())
        modules['action_predict'] = FullDiscretizerWLast(self.args.hidden, None, self.args, oaction_dict)
        #modules['v_predict'] = torch.nn.Linear(self.args.hidden, 1)

        if False:
            #future predict using conditional GAN
            self.noise_dim = 32
            embed_dim = self.cnn_embed_dim+self.state_shape['history_reward'][0]
            if 'state' in self.state_shape:
                embed_dim += self.state_shape['state'][0]
            modules['embed_gen'] = torch.nn.Sequential(torch.nn.Linear(embed_dim+self.action_logits_dim+self.action_contin_dim+self.noise_dim, self.args.hidden*2),
                                                       Lambda(lambda x: swish(x)),
                                                       #SkipHashingMemory(self.args.hidden*2, self.args),
                                                       torch.nn.Linear(self.args.hidden*2, embed_dim))
            modules['embed_dis'] = torch.nn.Sequential(torch.nn.Linear(embed_dim*2+self.action_logits_dim+self.action_contin_dim, self.args.hidden*2),
                                                       Lambda(lambda x: swish(x)),
                                                       #SkipHashingMemory(self.args.hidden*2, self.args),
                                                       torch.nn.Linear(self.args.hidden*2, 1, bias=False))
        lam = torch.zeros((1,), requires_grad=True)
        modules['softmax'] = torch.nn.Softmax(1)

        if torch.cuda.is_available() and CUDA:
            modules = modules.cuda()
            lam = lam.cuda()
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        modules['action_predict'].device = self.device

        ce_loss = torch.nn.CrossEntropyLoss()
        loss = torch.nn.MSELoss()

        model_params = []
        nec_value_params = []
        for name, p in modules.named_parameters():
            if not 'embed_gen' in name and not 'embed_dis' in name:
                if 'values.weight' in name:
                    nec_value_params.append(p)
                else:
                    model_params.append(p)
        optimizer = torch.optim.Adam([{'params': model_params}, 
                                      {'params': nec_value_params, 'lr': self.args.nec_lr}], lr=self.args.lr)

        if False:
            model_params = []
            nec_value_params = []
            for name, p in modules['embed_dis'].named_parameters():
                if 'values.weight' in name:
                    nec_value_params.append(p)
                else:
                    model_params.append(p)
            optimizer_D = torch.optim.Adam([{'params': model_params}, 
                                            {'params': nec_value_params, 'lr': self.args.nec_lr}], 
                                           lr=self.args.lr,
                                           betas=(0.5, 0.999))
            optimizer_D.zero_grad()

        
            model_params = []
            nec_value_params = []
            for name, p in modules['embed_gen'].named_parameters():
                if 'values.weight' in name:
                    nec_value_params.append(p)
                else:
                    model_params.append(p)
            optimizer_G = torch.optim.Adam([{'params': model_params}, 
                                            {'params': nec_value_params, 'lr': self.args.nec_lr}], 
                                           lr=self.args.lr,
                                           betas=(0.5, 0.999))
            optimizer_G.zero_grad()

            return {'modules': modules, 
                    'opt': optimizer, 
                    'ce_loss': ce_loss,
                    'loss': loss,
                    'Lambda': lam,
                    'opt_G': optimizer_G,
                    'opt_D': optimizer_D}
        else:
            return {'modules': modules, 
                    'opt': optimizer, 
                    'ce_loss': ce_loss,
                    'loss': loss,
                    'Lambda': lam}


    def get_normal(self, shape):
        with torch.no_grad():
            return torch.normal(torch.zeros(shape, dtype=torch.float32, device=self.device), torch.ones(shape, dtype=torch.float32, device=self.device))

    def get_cnn_embed(self, state):
        x = (state['pov'].permute(0, 1, 4, 2, 3)/255.0)*2.0 - 1.0
        if False:
            batch_size = x.shape[0]
            cnn_fea = self._model['modules']['cnn_model'](x[:,0])
            cnn_fea = self._model['modules']['flatten'](cnn_fea)
            with torch.no_grad():
                n_view = [-1] + [self.state_shape['pov'][2]] + self.state_shape['pov'][:2]
                cnn_fea_ng = self._model['modules']['cnn_model'](x[:,1:].contiguous().view(*n_view))
                cnn_fea_ng = self._model['modules']['flatten'](cnn_fea_ng)
            n_view = [batch_size,-1,self.cnn_embed_dim]
            return torch.cat([cnn_fea.unsqueeze(1),cnn_fea_ng.view(*n_view)],dim=1)
        else:
            batch_size = x.shape[0]
            n_view = [-1] + [self.state_shape['pov'][2]] + self.state_shape['pov'][:2]
            cnn_fea = self._model['modules']['cnn_model'](x.view(*n_view))
            cnn_fea = self._model['modules']['flatten'](cnn_fea)
            n_view = [batch_size,-1,self.cnn_embed_dim]
            return cnn_fea.view(*n_view)


    def get_action(self, state, cnn_embed, hidden=None):
        if hidden is None:
            tembed = [cnn_embed[:,0]]
            for o in self.state_shape:
                if 'history' in o:
                    tembed.append(state[o][:,0])
            if 'state' in state:
                tembed.append(state['state'][:,0])
            tembed = torch.cat(tembed, dim=-1)
            hidden = self._model['modules']['action_v_hidden'](tembed)
        return hidden

    def get_value(self, state, cnn_embed, hidden=None):
        if hidden is None:
            tembed = [cnn_embed]
            for o in self.state_shape:
                if 'history' in o:
                    tembed.append(state[o])
            if 'state' in state:
                tembed.append(state['state'])
            tembed = torch.cat(tembed, dim=-1)
            hidden = self._model['modules']['action_v_hidden'](tembed)
        if False:
            value0 = self._model['modules']['v_predict'](hidden[:,0]).squeeze(-1)
            with torch.no_grad():
                valuer = self._model['modules']['v_predict'](hidden[:,1:]).squeeze(-1)
            return value0, valuer
        else:
            value = self._model['modules']['v_predict'](hidden).squeeze(-1)
            return value[:,0], value[:,1:], hidden

    def get_next_embed(self, embed, last_embed):
        noise = self.get_normal(list(embed.shape[:-1])+[self.noise_dim])
        inp = torch.cat([embed, noise], dim=-1)
        next_embed = self._model['modules']['embed_gen'](inp)+last_embed
        return next_embed

    def discriminate(self, embed, next_embed):
        inp = torch.cat([embed, next_embed], dim=1)
        return self._model['modules']['embed_dis'](inp)

    def inverse_map_camera(self, x):
        return x / 180.0

    def map_camera(self, x):
        return x * 180.0

    def get_BC_value_loss(self, state, action, reward, done, cnn_embed, lastaction):
        if False:
            value0, valuer, hidden = self.get_value(state, cnn_embed)

            #make truncated GVE
            with torch.no_grad():
                done = done.float()
                done = torch.cumsum(done,1).clamp(max=1.0)
                reward[:,1:] = reward[:,1:] * (1.0 - done[:,:-1]) #mask out reward from different episode
                gve_value = None
                gammas = GAMMA ** torch.arange(0, self.time_deltas[1], device=self.device).float()
                reward_slice = reward[:,:self.time_deltas[1]]
                disc_reward = (gammas.unsqueeze(0).expand(reward_slice.shape[0], -1) * reward_slice).sum(1)
                for i in range(1,len(self.time_deltas)):
                    tvalue = valuer[:,i-1]
                    tdone = done[:,self.time_deltas[i]-1]
                    if gve_value is None:
                        gve_value = self.GVE_weigths[i-1] * (disc_reward + (1.0 - tdone) * GAMMA ** self.time_deltas[i] * tvalue)
                    else:
                        gve_value += self.GVE_weigths[i-1] * (disc_reward + (1.0 - tdone) * GAMMA ** self.time_deltas[i] * tvalue)
                    if i != len(self.time_deltas) - 1:
                        reward_slice = reward[:,self.time_deltas[i]:self.time_deltas[i+1]]
                        gammas = GAMMA ** torch.arange(self.time_deltas[i], self.time_deltas[i+1], device=self.device).float()
                        disc_reward += (gammas.unsqueeze(0).expand(reward_slice.shape[0], -1) * reward_slice).sum(1)
                gve_value /= self.GVE_norm
            value_loss = self._model['loss'](value0, gve_value.detach())
            value_loss_cpu = value_loss.data.cpu().numpy()
            loss = value_loss

            faction = {}
            for a in action:
                faction[a] = action[a][:,0]
            flaction = {}
            for a in lastaction:
                flaction[a] = lastaction[a][:,0]
            aloss, entropy, _ = self._model['modules']['action_predict'].get_loss_BC(hidden[:,0], faction, flaction)
            loss += aloss
            with torch.no_grad():
                delta = torch.abs(gve_value.detach()-value0).mean()
            return loss, aloss, value0.mean(), delta, value_loss_cpu, entropy.data.cpu().numpy()
        else:
            hidden = self.get_action(state, cnn_embed)
            faction = {}
            for a in action:
                faction[a] = action[a][:,0]
            flaction = {}
            for a in lastaction:
                flaction[a] = lastaction[a][:,0]
            aloss, entropy, _ = self._model['modules']['action_predict'].get_loss_BC(hidden, faction, flaction)
            loss = aloss
            return loss, entropy.data.cpu().numpy()


    def train_predict(self, state, action, cnn_embed):
        assert self.max_predict_range != 0
        tembed = [cnn_embed[:,:self.max_predict_range].detach()]
        tembed.append(state['history_reward'][:,:self.max_predict_range])
        if 'state' in state:
            tembed.append(state['state'][:,:self.max_predict_range])
        tembed_wo_a = torch.cat(tembed, dim=-1)
        tactions = []
        for j, a in enumerate(self.action_dict):
            if a != 'camera':
                tactions.append(one_hot(action[a][:,:self.max_predict_range],self.action_dict[a].n,self.device))
        tactions.append(action['camera'][:,:self.max_predict_range])
        tactions = torch.cat(tactions, dim=-1)
        tembed.append(tactions)
        tembed = torch.cat(tembed, dim=-1)

        lossD = None
        lossG = None

        next_embed = tembed_wo_a[:,0]

        for i in range(self.max_predict_range-1):
            #train D
            next_embed = self.get_next_embed(torch.cat([next_embed, tactions[:,i]],dim=-1), next_embed)
            real_embed = tembed_wo_a[:,i+1]
            score_fake = self.discriminate(tembed[:,i], next_embed.detach())
            score_real = self.discriminate(tembed[:,i], real_embed.detach())

            E_P_f,  E_Q_f  = score_real.mean(), score_fake.mean()
            E_P_f2, E_Q_f2 = (score_real**2).mean(), (score_fake**2).mean()
            constraint = (1 - (0.5*E_P_f2 + 0.5*E_Q_f2))

            if lossD is None:
                lossD = -(E_P_f - E_Q_f + self._model['Lambda'][0] * constraint - (self.args.rho/2) * constraint**2)
            else:
                lossD += -(E_P_f - E_Q_f + self._model['Lambda'][0] * constraint - (self.args.rho/2) * constraint**2)

            if self.gen_ph % self.gen_train_freq == 0:
                #train G
                for p in self._model['modules']['embed_dis'].parameters():
                    p.requires_grad = False
                score_fake = self.discriminate(tembed[:,i], next_embed)
                for p in self._model['modules']['embed_dis'].parameters():
                    p.requires_grad = True
                if lossG is None:
                    lossG = -score_fake.mean() * (LAMBDA ** i)
                else:
                    lossG += -score_fake.mean() * (LAMBDA ** i)
                #lossG += self._model['loss'](next_embed, tembed_wo_a[:,i+1].detach()) * 0.1


        self._model['Lambda'].retain_grad()
        lossD.backward()
        self._model['opt_D'].step()
        self._model['Lambda'].data += self.args.rho * self._model['Lambda'].grad.data
        self._model['Lambda'].grad.data.zero_()
        
        if self.gen_ph % self.gen_train_freq == 0:
            lossG.backward()
            self._model['opt_G'].step()
            self._model['opt_G'].zero_grad()
        self._model['opt_D'].zero_grad()

        self.gen_ph += 1

        return lossD, lossG

    def pretrain(self, input):
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

        #train only state embedding
        self._model['opt'].zero_grad()
        cnn_embed = self.get_cnn_embed(states)
        loss, entropy = self.get_BC_value_loss(states, actions, reward, done, cnn_embed, lastaction)
        #lossD, lossG = self.train_predict(states, actions, cnn_embed)
        loss.backward()

        self._model['opt'].step()

        return {'aloss': loss.data.cpu().numpy(),
                #'lossD': lossD.data.cpu().numpy(),
                #'lossG': -1.0 if lossG is None else lossG.data.cpu().numpy(),
                #'Lambda': self._model['Lambda'][0].data.data.cpu().numpy(),
                #'value_loss': value_loss,
                #'val': value.data.cpu().numpy(),
                #'Dval': delta.data.cpu().numpy(),
                'entropy': entropy}


    def train(self, input):
        return {}
    

    def get_cnn_embed_single(self, state):
        x = (state['pov'][:,0].permute(0, 3, 1, 2)/255.0)*2.0 - 1.0
        cnn_fea = self._model['modules']['cnn_model'](x)
        cnn_fea = self._model['modules']['flatten'](cnn_fea)
        return cnn_fea

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
            cnn_embed = self.get_cnn_embed_single(tstate)
            hidden = self.get_action(tstate, cnn_embed.unsqueeze(1))
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



































class VirtualPGHalfDiscrete():
    def __init__(self, state_shape, action_dict, args, time_deltas):
        assert isinstance(state_shape, dict)
        self.state_shape = state_shape
        self.action_dict = action_dict
        self.action_bundles = [['forward_back','left_right','sneak_sprint'],
                               ['jump'],
                               ['attack_place_equip_craft_nearbyCraft_nearbySmelt']]
        self.bundle_dims = [0, 0, 0]
        self.args = args
        
        self.time_deltas = time_deltas
        #get GVE parameters
        assert self.time_deltas[0] == 1
        self.GVE_weigths = []
        self.GVE_norm = 0.0
        for i in range(len(self.time_deltas)):
            start = LAMBDA ** self.time_deltas[i]
            if i != len(self.time_deltas) - 1:
                sum_w = 1.0 - LAMBDA**(self.time_deltas[i+1]-self.time_deltas[i])
                sum_w /= 1.0 - LAMBDA
            else:
                sum_w = 1.0 / (1.0 - LAMBDA)
            self.GVE_weigths.append(start*sum_w)
            self.GVE_norm += start*sum_w
            
        self.max_predict_range = 0
        for i in range(len(self.time_deltas)):
            if self.time_deltas[i] - self.max_predict_range == 1:
                self.max_predict_range = self.time_deltas[i]
            else:
                break
        self.gen_ph = 0
        self.gen_train_freq = 5

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
            if False:
                sdict = torch.load(filename + '-virtualPG')
                rdict = {}
                for n in sdict:
                    if not 'embed_gen' in n and not 'embed_dis' in n:
                        rdict[n] = sdict[n]
                    else:
                        print(n,'discarted')
                self.load_state_dict(rdict, False)
            else:
                print('Load actor')
                self.load_state_dict(torch.load(filename + '-virtualPG'))
                self._model['Lambda'] = torch.load(filename + '-virtualPGLam')
        else:
            torch.save(self._model['modules'].state_dict(), filename + '-virtualPG')
            torch.save(self._model['Lambda'], filename + '-virtualPGLam')

    def _make_model(self):
        #make CNN
        modules = torch.nn.ModuleDict()
        cnn_layers = []
        assert len(self.state_shape['pov']) > 1
        if self.args.cnn_type == 'atari':
            sizes = [8, 4, 3]
            strides = [4, 2, 1]
            pooling = [1, 1, 1]
            filters = [32, 64, 32]
        elif self.args.cnn_type == 'mnist':
            sizes = [3, 3, 3]
            strides = [1, 1, 1]
            pooling = [2, 2, 2]
            filters = [32, 32, 32]

        in_channels = self.state_shape['pov'][-1] #HWC input

        for i in range(len(sizes)):
            cnn_layers.append(torch.nn.Conv2d(
                in_channels,
                filters[i],
                sizes[i],
                stride=strides[i],
                bias=True
            ))

            cnn_layers.append(Lambda(lambda x: swish(x)))

            if pooling[i] > 1:
                cnn_layers.append(torch.nn.MaxPool2d(pooling[i]))

            in_channels = filters[i]

        #cnn_layers.append(Flatten())
        modules['flatten'] = Flatten()
        modules['cnn_model'] = torch.nn.Sequential(*cnn_layers)
        
        self.cnn_embed_shape = list(modules['cnn_model'](torch.zeros((1,self.state_shape['pov'][2], self.state_shape['pov'][0], self.state_shape['pov'][1]))).shape[1:])
        self.cnn_embed_dim = self.cnn_embed_shape[0]*self.cnn_embed_shape[1]*self.cnn_embed_shape[2]
        modules['cnn_embed'] = torch.nn.Sequential(torch.nn.Linear(self.cnn_embed_dim, self.args.hidden),
                                                   torch.nn.Tanh())

        self.action_logits_dim = 0
        self.action_contin_dim = 0
        for a in self.action_dict:
            if isinstance(self.action_dict[a], gym.spaces.Box):
                self.action_contin_dim += self.action_dict[a].shape[0]
        for i in range(len(self.action_bundles)):
            bundle_dim = 1
            for a in self.action_bundles[i]:
                bundle_dim *= self.action_dict[a].n
            self.bundle_dims[i] = bundle_dim
            self.action_logits_dim += bundle_dim
            
        if 'state' in self.state_shape:
            self.state_dim = self.state_shape['state'][0]
        else:
            self.state_dim = 0
        for o in self.state_shape:
            if 'history' in o:
                self.state_dim += self.state_shape[o][0]
        #self.discretizer_dim = 128
        modules['action_v_hidden'] = torch.nn.Sequential(torch.nn.Linear(self.args.hidden+self.state_dim, self.args.hidden),
                                                         Lambda(lambda x: swish(x)))
        modules['action_predict'] = HashingMemory.build(self.args.hidden, self.action_logits_dim, self.args)#torch.nn.Linear(self.args.hidden, self.action_logits_dim)
        modules['action_continuous'] = Discretizer(self.args.hidden, None, self.args)
        #HashingMemory.build(self.args.hidden+self.state_dim, self.action_logits_dim+self.action_contin_dim_all, self.args)
        modules['v_predict'] = torch.nn.Linear(self.args.hidden, 1)
        #HashingMemory.build(self.args.hidden+self.state_dim, 1, self.args)
                                      #torch.nn.Sequential(torch.nn.Linear(self.args.hidden+self.state_dim, self.args.hidden),
                                      #                    Lambda(lambda x: swish(x)),
                                      #                    torch.nn.Linear(self.args.hidden, action_logits_dim+2*action_contin_dim+1))
        #TODO convert continuous action range from (-180,180) to (-inf,inf)
        #TODO convert continuous action range from (-180,180) to (-inf,inf)
        #TODO convert continuous action range from (-180,180) to (-inf,inf)
        #TODO convert continuous action range from (-180,180) to (-inf,inf)
        #TODO convert continuous action range from (-180,180) to (-inf,inf)
        #TODO convert continuous action range from (-180,180) to (-inf,inf)
        #TODO convert continuous action range from (-180,180) to (-inf,inf)
        #TODO convert continuous action range from (-180,180) to (-inf,inf)
        #TODO convert continuous action range from (-180,180) to (-inf,inf)
        #TODO convert continuous action range from (-180,180) to (-inf,inf)

        #future predict using conditional GAN
        #future preidicts DELTA of input hidden!!!!!
        self.noise_dim = 32
        embed_dim = self.args.hidden+self.state_shape['history_reward'][0]
        if 'state' in self.state_shape:
            embed_dim += self.state_shape['state'][0]
        modules['embed_gen'] = torch.nn.Sequential(torch.nn.Linear(embed_dim+self.action_logits_dim+self.action_contin_dim+self.noise_dim, self.args.hidden*2),
                                                   Lambda(lambda x: swish(x)),
                                                   SkipHashingMemory(self.args.hidden*2, self.args),
                                                   torch.nn.Linear(self.args.hidden*2, embed_dim))
        modules['embed_dis'] = torch.nn.Sequential(torch.nn.Linear(embed_dim*2+self.action_logits_dim+self.action_contin_dim, self.args.hidden*2),
                                                   Lambda(lambda x: swish(x)),
                                                   SkipHashingMemory(self.args.hidden*2, self.args),
                                                   torch.nn.Linear(self.args.hidden*2, 1, bias=False))
        lam = torch.zeros((1,), requires_grad=True)
        modules['softmax'] = torch.nn.Softmax(1)

        if torch.cuda.is_available() and CUDA:
            modules = modules.cuda()
            lam = lam.cuda()
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        modules['action_continuous'].device = self.device

        ce_loss = torch.nn.CrossEntropyLoss()
        loss = torch.nn.MSELoss()

        model_params = []
        nec_value_params = []
        for name, p in modules.named_parameters():
            if not 'embed_gen' in name and not 'embed_dis' in name:
                if 'values.weight' in name:
                    nec_value_params.append(p)
                else:
                    model_params.append(p)
        optimizer = torch.optim.Adam([{'params': model_params}, 
                                      {'params': nec_value_params, 'lr': self.args.nec_lr}], lr=self.args.lr)

        
        model_params = []
        nec_value_params = []
        for name, p in modules['embed_dis'].named_parameters():
            if 'values.weight' in name:
                nec_value_params.append(p)
            else:
                model_params.append(p)
        optimizer_D = torch.optim.Adam([{'params': model_params}, 
                                        {'params': nec_value_params, 'lr': self.args.nec_lr}], 
                                       lr=self.args.lr,
                                       betas=(0.5, 0.999))
        optimizer_D.zero_grad()

        
        model_params = []
        nec_value_params = []
        for name, p in modules['embed_gen'].named_parameters():
            if 'values.weight' in name:
                nec_value_params.append(p)
            else:
                model_params.append(p)
        optimizer_G = torch.optim.Adam([{'params': model_params}, 
                                        {'params': nec_value_params, 'lr': self.args.nec_lr}], 
                                       lr=self.args.lr,
                                       betas=(0.5, 0.999))
        optimizer_G.zero_grad()

        return {'modules': modules, 
                'opt': optimizer, 
                'ce_loss': ce_loss,
                'loss': loss,
                'Lambda': lam,
                'opt_G': optimizer_G,
                'opt_D': optimizer_D}

    def get_normal(self, shape):
        with torch.no_grad():
            return torch.normal(torch.zeros(shape, dtype=torch.float32, device=self.device), torch.ones(shape, dtype=torch.float32, device=self.device))

    def get_cnn_embed(self, state):
        x = (state['pov'].permute(0, 1, 4, 2, 3)/255.0)*2.0 - 1.0
        if False:
            batch_size = x.shape[0]
            cnn_fea = self._model['modules']['cnn_model'](x[:,0])
            cnn_fea = self._model['modules']['flatten'](cnn_fea)
            cnn_fea = self._model['modules']['cnn_embed'](cnn_fea)
            with torch.no_grad():
                n_view = [-1] + [self.state_shape['pov'][2]] + self.state_shape['pov'][:2]
                cnn_fea_ng = self._model['modules']['cnn_model'](x[:,1:].contiguous().view(*n_view))
                cnn_fea_ng = self._model['modules']['flatten'](cnn_fea_ng)
                cnn_fea_ng = self._model['modules']['cnn_embed'](cnn_fea_ng)
            n_view = [batch_size,-1,self.args.hidden]
            return torch.cat([cnn_fea.unsqueeze(1),cnn_fea_ng.view(*n_view)],dim=1)
        else:
            batch_size = x.shape[0]
            n_view = [-1] + [self.state_shape['pov'][2]] + self.state_shape['pov'][:2]
            cnn_fea = self._model['modules']['cnn_model'](x.view(*n_view))
            cnn_fea = self._model['modules']['flatten'](cnn_fea)
            cnn_fea = self._model['modules']['cnn_embed'](cnn_fea)
            n_view = [batch_size,-1,self.args.hidden]
            return cnn_fea.view(*n_view)


    def get_action(self, state, cnn_embed, hidden=None):
        if hidden is None:
            tembed = [cnn_embed[:,0]]
            for o in self.state_shape:
                if 'history' in o:
                    tembed.append(state[o][:,0])
            if 'state' in state:
                tembed.append(state['state'][:,0])
            tembed = torch.cat(tembed, dim=-1)
            hidden = self._model['modules']['action_v_hidden'](tembed)
        action = self._model['modules']['action_predict'](hidden)
        action_logits = torch.split(action[:,:self.action_logits_dim], self.bundle_dims, dim=-1)
        return action_logits, hidden

    def get_value(self, state, cnn_embed, hidden=None):
        if hidden is None:
            tembed = [cnn_embed]
            for o in self.state_shape:
                if 'history' in o:
                    tembed.append(state[o])
            if 'state' in state:
                tembed.append(state['state'])
            tembed = torch.cat(tembed, dim=-1)
            hidden = self._model['modules']['action_v_hidden'](tembed)
        if False:
            value0 = self._model['modules']['v_predict'](hidden[:,0]).squeeze(-1)
            with torch.no_grad():
                valuer = self._model['modules']['v_predict'](hidden[:,1:]).squeeze(-1)
            return value0, valuer
        else:
            value = self._model['modules']['v_predict'](hidden).squeeze(-1)
            return value[:,0], value[:,1:]

    def get_next_embed(self, embed, last_embed):
        noise = self.get_normal(list(embed.shape[:-1])+[self.noise_dim])
        inp = torch.cat([embed, noise], dim=-1)
        next_embed = self._model['modules']['embed_gen'](inp)+last_embed
        return next_embed

    def discriminate(self, embed, next_embed):
        inp = torch.cat([embed, next_embed], dim=1)
        return self._model['modules']['embed_dis'](inp)

    def inverse_map_camera(self, x):
        return x / 180.0

    def map_camera(self, x):
        return x * 180.0

    def get_BC_value_loss(self, state, action, reward, done, cnn_embed):
        action_logits, hidden = self.get_action(state, cnn_embed)
        value0, valuer = self.get_value(state, cnn_embed)

        #make truncated GVE
        with torch.no_grad():
            done = done.float()
            done = torch.cumsum(done,1).clamp(max=1.0)
            reward[:,1:] = reward[:,1:] * (1.0 - done[:,:-1]) #mask out reward from different episode
            gve_value = None
            gammas = GAMMA ** torch.arange(0, self.time_deltas[0], device=self.device).float()
            reward_slice = reward[:,:self.time_deltas[0]]
            disc_reward = (gammas.unsqueeze(0).expand(reward_slice.shape[0], -1) * reward_slice).sum(1)
            for i in range(len(self.time_deltas)):
                tvalue = valuer[:,i-1]
                tdone = done[:,self.time_deltas[i]-1]
                if gve_value is None:
                    gve_value = self.GVE_weigths[i] * (disc_reward + (1.0 - tdone) * GAMMA ** self.time_deltas[i] * tvalue)
                else:
                    gve_value += self.GVE_weigths[i] * (disc_reward + (1.0 - tdone) * GAMMA ** self.time_deltas[i] * tvalue)
                if i != len(self.time_deltas) - 1:
                    reward_slice = reward[:,self.time_deltas[i]:self.time_deltas[i+1]]
                    gammas = GAMMA ** torch.arange(self.time_deltas[i], self.time_deltas[i+1], device=self.device).float()
                    disc_reward += (gammas.unsqueeze(0).expand(reward_slice.shape[0], -1) * reward_slice).sum(1)
            gve_value /= self.GVE_norm
        value_loss = self._model['loss'](value0, self.args.clr * gve_value.detach() + (1.0 - self.args.clr) * value0.detach()) * 20.0
        value_loss_cpu = value_loss.data.cpu().numpy()
        loss = value_loss

        accuracy = 0.0
        entropy = []
        for j, bundle in enumerate(self.action_bundles):
            action_bundle = None
            action_m = 1
            for i, a in enumerate(bundle):
                if i == 0:
                    action_bundle = action[a][:,0]
                else:
                    action_bundle = action[a][:,0] * action_m
                action_m *= self.action_dict[a].n
            loss += self._model['ce_loss'](action_logits[j], action_bundle)
            with torch.no_grad():
                probs = self._model['modules']['softmax'](action_logits[j])
                entropy.append(-(probs*torch.log(probs)).sum(-1).mean().data.cpu().numpy())
                accuracy += probs.gather(1, action_bundle.unsqueeze(-1)).mean()
        with torch.no_grad():
            accuracy /= len(self.action_bundles)

        con_loss, con_entropy, _ =  self._model['modules']['action_continuous'].get_loss_BC(hidden, action['camera'][:,0])
        entropy.append(con_entropy)
        loss += con_loss
        with torch.no_grad():
            delta = torch.abs(gve_value.detach()-value0).mean()
        return loss, value0.mean(), delta, accuracy, value_loss_cpu, entropy

    def train_predict(self, state, action, cnn_embed):
        assert self.max_predict_range != 0
        tembed = [cnn_embed[:,:self.max_predict_range].detach()]
        tembed.append(state['history_reward'][:,:self.max_predict_range])
        if 'state' in state:
            tembed.append(state['state'][:,:self.max_predict_range])
        tembed_wo_a = torch.cat(tembed, dim=-1)
        tactions = []
        for j, bundle in enumerate(self.action_bundles):
            action_bundle = None
            action_m = 1
            for i, a in enumerate(bundle):
                if i == 0:
                    action_bundle = action[a][:,:self.max_predict_range]
                else:
                    action_bundle = action[a][:,:self.max_predict_range] * action_m
                action_m *= self.action_dict[a].n
            tactions.append(one_hot(action_bundle,action_m,self.device))
        tactions.append(action['camera'][:,:self.max_predict_range])
        tactions = torch.cat(tactions, dim=-1)
        tembed.append(tactions)
        tembed = torch.cat(tembed, dim=-1)

        lossD = None
        lossG = None

        next_embed = tembed_wo_a[:,0]

        for i in range(self.max_predict_range-1):
            #train D
            next_embed = self.get_next_embed(torch.cat([next_embed, tactions[:,i]],dim=-1), next_embed)
            real_embed = tembed_wo_a[:,i+1]
            score_fake = self.discriminate(tembed[:,i], next_embed.detach())
            score_real = self.discriminate(tembed[:,i], real_embed.detach())

            E_P_f,  E_Q_f  = score_real.mean(), score_fake.mean()
            E_P_f2, E_Q_f2 = (score_real**2).mean(), (score_fake**2).mean()
            constraint = (1 - (0.5*E_P_f2 + 0.5*E_Q_f2))

            if lossD is None:
                lossD = -(E_P_f - E_Q_f + self._model['Lambda'][0] * constraint - (self.args.rho/2) * constraint**2)
            else:
                lossD += -(E_P_f - E_Q_f + self._model['Lambda'][0] * constraint - (self.args.rho/2) * constraint**2)

            if self.gen_ph % self.gen_train_freq == 0:
                #train G
                for p in self._model['modules']['embed_dis'].parameters():
                    p.requires_grad = False
                score_fake = self.discriminate(tembed[:,i], next_embed)
                for p in self._model['modules']['embed_dis'].parameters():
                    p.requires_grad = True
                if lossG is None:
                    lossG = -score_fake.mean() * (LAMBDA ** i)
                else:
                    lossG += -score_fake.mean() * (LAMBDA ** i)
                #lossG += self._model['loss'](next_embed, tembed_wo_a[:,i+1].detach()) * 0.1


        self._model['Lambda'].retain_grad()
        lossD.backward()
        self._model['opt_D'].step()
        self._model['Lambda'].data += self.args.rho * self._model['Lambda'].grad.data
        self._model['Lambda'].grad.data.zero_()
        
        if self.gen_ph % self.gen_train_freq == 0:
            lossG.backward()
            self._model['opt_G'].step()
            self._model['opt_G'].zero_grad()
        self._model['opt_D'].zero_grad()

        self.gen_ph += 1

        return lossD, lossG

    def pretrain(self, input):
        state_keys = ['state', 'pov', 'history_action', 'history_reward']
        states = {}
        for k in state_keys:
            if k in input:
                states[k] = variable(input[k])
        actions = {'camera': self.inverse_map_camera(variable(input['camera']))}
        for b in self.action_bundles:
            for a in b:
                actions[a] = variable(input[a], True).long()
        reward = variable(input['reward'])
        done = variable(input['done'])

        #train state embedding
        self._model['opt'].zero_grad()
        cnn_embed = self.get_cnn_embed(states)
        loss, value, delta, accuracy, value_loss, entropy = self.get_BC_value_loss(states, actions, reward, done, cnn_embed)
        lossD, lossG = self.train_predict(states, actions, cnn_embed)
        loss.backward()
        #for n, p in self._model['modules']['cnn_model'].named_parameters():
        #    with torch.no_grad():
        #        print(n, p.grad.mean().data.cpu().numpy())
        self._model['opt'].step()

        return {'loss': loss.data.cpu().numpy(),
                'lossD': lossD.data.cpu().numpy(),
                'lossG': -1.0 if lossG is None else lossG.data.cpu().numpy(),
                'Lambda': self._model['Lambda'][0].data.data.cpu().numpy(),
                'value_loss': value_loss,
                'val': value.data.cpu().numpy(),
                'Dval': delta.data.cpu().numpy(),
                'accuracy': accuracy.data.cpu().numpy(),
                'entropy': entropy}


    def train(self, input):
        return {}
    

    def get_cnn_embed_single(self, state):
        x = (state['pov'][:,0].permute(0, 3, 1, 2)/255.0)*2.0 - 1.0
        cnn_fea = self._model['modules']['cnn_model'](x)
        cnn_fea = self._model['modules']['flatten'](cnn_fea)
        cnn_fea = self._model['modules']['cnn_embed'](cnn_fea)
        return cnn_fea

    def select_action(self, state, batch=False):
        if not batch:
            for k in state:
                state[k] = state[k][None, None,:].copy()
        state = variable(state)
        with torch.no_grad():
            cnn_embed = self.get_cnn_embed_single(state)
            action_logits, action_hidden = self.get_action(state, cnn_embed.unsqueeze(1))
            entropy = 0.0
            r_action = {}
            allprobs = []
            for i, b in enumerate(self.action_bundles):
                probs = self._model['modules']['softmax'](action_logits[i]).data.cpu().numpy()[0]
                allprobs.append(probs)
                entropy += -np.sum(probs*np.log(probs))
                action_b = int(np.random.choice(range(len(probs)), p=probs))
                for ra in self.action_bundles[i]:
                    r_action[ra] = action_b%self.action_dict[ra].n
                    action_b //= self.action_dict[ra].n
            
            r_action['camera'] = self._model['modules']['action_continuous'].sample(action_hidden)[0,:].data.cpu().numpy()
            r_action['camera'] = self.map_camera(r_action['camera'])
            #print(r_action)
            #time.sleep(1.0)
            #r_action['camera'] = np.clip(r_action['camera'], -5.0, 5.0)
            action = OrderedDict()
            for a in self.action_dict:
                action[a] = r_action[a]
        return action, {'entropy': entropy,
                        'allprobs': allprobs}




























class VirtualPGContinuous():
    def __init__(self, state_shape, action_dict, args, time_deltas):
        assert isinstance(state_shape, dict)
        self.state_shape = state_shape
        self.action_dict = action_dict
        self.action_bundles = [['forward_back','left_right','sneak_sprint'],
                               ['jump'],
                               ['attack_place_equip_craft_nearbyCraft_nearbySmelt']]
        self.bundle_dims = [0, 0, 0]
        self.args = args
        
        self.time_deltas = time_deltas
        #get GVE parameters
        assert self.time_deltas[0] == 1
        self.GVE_weigths = []
        self.GVE_norm = 0.0
        for i in range(len(self.time_deltas)):
            start = LAMBDA ** self.time_deltas[i]
            if i != len(self.time_deltas) - 1:
                sum_w = 1.0 - LAMBDA**(self.time_deltas[i+1]-self.time_deltas[i])
                sum_w /= 1.0 - LAMBDA
            else:
                sum_w = 1.0 / (1.0 - LAMBDA)
            self.GVE_weigths.append(start*sum_w)
            self.GVE_norm += start*sum_w
            
        self.max_predict_range = 0
        for i in range(len(self.time_deltas)):
            if self.time_deltas[i] - self.max_predict_range == 1:
                self.max_predict_range = self.time_deltas[i]
            else:
                break
        self.gen_ph = 0
        self.gen_train_freq = 5

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
            if False:
                sdict = torch.load(filename + '-virtualPG')
                rdict = {}
                for n in sdict:
                    if not 'embed_gen' in n and not 'embed_dis' in n:
                        rdict[n] = sdict[n]
                    else:
                        print(n,'discarted')
                self.load_state_dict(rdict, False)
            else:
                print('Load actor')
                self.load_state_dict(torch.load(filename + '-virtualPG'))
                self._model['Lambda'] = torch.load(filename + '-virtualPGLam')
        else:
            torch.save(self._model['modules'].state_dict(), filename + '-virtualPG')
            torch.save(self._model['Lambda'], filename + '-virtualPGLam')

    def _make_model(self):
        #make CNN
        modules = torch.nn.ModuleDict()
        cnn_layers = []
        assert len(self.state_shape['pov']) > 1
        if self.args.cnn_type == 'atari':
            sizes = [8, 4, 3]
            strides = [4, 2, 1]
            pooling = [1, 1, 1]
            filters = [32, 64, 32]
        elif self.args.cnn_type == 'mnist':
            sizes = [3, 3, 3]
            strides = [1, 1, 1]
            pooling = [2, 2, 2]
            filters = [32, 32, 32]

        in_channels = self.state_shape['pov'][-1] #HWC input

        for i in range(len(sizes)):
            cnn_layers.append(torch.nn.Conv2d(
                in_channels,
                filters[i],
                sizes[i],
                stride=strides[i],
                bias=True
            ))

            cnn_layers.append(Lambda(lambda x: swish(x)))

            if pooling[i] > 1:
                cnn_layers.append(torch.nn.MaxPool2d(pooling[i]))

            in_channels = filters[i]

        #cnn_layers.append(Flatten())
        modules['flatten'] = Flatten()
        modules['cnn_model'] = torch.nn.Sequential(*cnn_layers)
        
        self.cnn_embed_shape = list(modules['cnn_model'](torch.zeros((1,self.state_shape['pov'][2], self.state_shape['pov'][0], self.state_shape['pov'][1]))).shape[1:])
        self.cnn_embed_dim = self.cnn_embed_shape[0]*self.cnn_embed_shape[1]*self.cnn_embed_shape[2]
        modules['cnn_embed'] = torch.nn.Sequential(torch.nn.Linear(self.cnn_embed_dim, self.args.hidden),
                                                   Lambda(lambda x: swish(x)))

        self.action_logits_dim = 0
        self.action_contin_dim = 0
        for a in self.action_dict:
            if isinstance(self.action_dict[a], gym.spaces.Box):
                self.action_contin_dim += self.action_dict[a].shape[0]
        for i in range(len(self.action_bundles)):
            bundle_dim = 1
            for a in self.action_bundles[i]:
                bundle_dim *= self.action_dict[a].n
            self.bundle_dims[i] = bundle_dim
            self.action_logits_dim += bundle_dim
            
        if 'state' in self.state_shape:
            self.state_dim = self.state_shape['state'][0]
        else:
            self.state_dim = 0
        for o in self.state_shape:
            if 'history' in o:
                self.state_dim += self.state_shape[o][0]
        self.num_gaussian_contin = 4
        self.action_contin_dim_all = self.num_gaussian_contin * self.action_contin_dim #means
        self.action_contin_dim_all += self.num_gaussian_contin * self.action_contin_dim * (self.action_contin_dim + 1) // 2 #covariance matrix (triangular)
        self.action_contin_dim_all += self.num_gaussian_contin #gaussian logits
        modules['action_predict'] = torch.nn.Sequential(torch.nn.Linear(self.args.hidden+self.state_dim, self.args.hidden),
                                                        #Lambda(lambda x: x.permute(0,2,1) if len(x.shape) == 3 else x),
                                                        #torch.nn.BatchNorm1d(self.args.hidden),
                                                        #Lambda(lambda x: swish(x.permute(0,2,1) if len(x.shape) == 3 else x)),
                                                        Lambda(lambda x: swish(x)),
                                                        torch.nn.Linear(self.args.hidden, self.action_logits_dim+self.action_contin_dim_all))
        #HashingMemory.build(self.args.hidden+self.state_dim, self.action_logits_dim+self.action_contin_dim_all, self.args)
        modules['v_predict'] = torch.nn.Sequential(torch.nn.Linear(self.args.hidden+self.state_dim, self.args.hidden),
                                                   #Lambda(lambda x: x.permute(0,2,1) if len(x.shape) == 3 else x),
                                                   #torch.nn.BatchNorm1d(self.args.hidden),
                                                   #Lambda(lambda x: swish(x.permute(0,2,1) if len(x.shape) == 3 else x)),
                                                   Lambda(lambda x: swish(x)),
                                                   torch.nn.Linear(self.args.hidden, 1))
        #HashingMemory.build(self.args.hidden+self.state_dim, 1, self.args)
                                      #torch.nn.Sequential(torch.nn.Linear(self.args.hidden+self.state_dim, self.args.hidden),
                                      #                    Lambda(lambda x: swish(x)),
                                      #                    torch.nn.Linear(self.args.hidden, action_logits_dim+2*action_contin_dim+1))

        #future predict using conditional GAN
        #future preidicts DELTA of input hidden!!!!!
        self.noise_dim = 32
        embed_dim = self.args.hidden+self.state_shape['history_reward'][0]
        if 'state' in self.state_shape:
            embed_dim += self.state_shape['state'][0]
        modules['embed_gen'] = torch.nn.Sequential(torch.nn.Linear(embed_dim+self.action_logits_dim+self.action_contin_dim+self.noise_dim, self.args.hidden*2),
                                                   Lambda(lambda x: swish(x)),
                                                   SkipHashingMemory(self.args.hidden*2, self.args),
                                                   torch.nn.Linear(self.args.hidden*2, embed_dim))
        modules['embed_dis'] = torch.nn.Sequential(torch.nn.Linear(embed_dim*2+self.action_logits_dim+self.action_contin_dim, self.args.hidden*2),
                                                   Lambda(lambda x: swish(x)),
                                                   SkipHashingMemory(self.args.hidden*2, self.args),
                                                   torch.nn.Linear(self.args.hidden*2, 1))
        lam = torch.zeros((1,), requires_grad=True)
        modules['softmax'] = torch.nn.Softmax(1)

        if torch.cuda.is_available() and CUDA:
            modules = modules.cuda()
            lam = lam.cuda()
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        ce_loss = torch.nn.CrossEntropyLoss()
        loss = torch.nn.MSELoss()

        model_params = []
        nec_value_params = []
        for name, p in modules.named_parameters():
            if not 'embed_gen' in name and not 'embed_dis' in name:
                if 'values.weight' in name:
                    nec_value_params.append(p)
                else:
                    model_params.append(p)
        optimizer = torch.optim.Adam([{'params': model_params}, 
                                      {'params': nec_value_params, 'lr': self.args.nec_lr}], lr=self.args.lr)

        
        model_params = []
        nec_value_params = []
        for name, p in modules['embed_dis'].named_parameters():
            if 'values.weight' in name:
                nec_value_params.append(p)
            else:
                model_params.append(p)
        optimizer_D = torch.optim.Adam([{'params': model_params}, 
                                        {'params': nec_value_params, 'lr': self.args.nec_lr}], 
                                       lr=self.args.lr,
                                       betas=(0.5, 0.999))
        optimizer_D.zero_grad()

        
        model_params = []
        nec_value_params = []
        for name, p in modules['embed_gen'].named_parameters():
            if 'values.weight' in name:
                nec_value_params.append(p)
            else:
                model_params.append(p)
        optimizer_G = torch.optim.Adam([{'params': model_params}, 
                                        {'params': nec_value_params, 'lr': self.args.nec_lr}], 
                                       lr=self.args.lr,
                                       betas=(0.5, 0.999))
        optimizer_G.zero_grad()

        return {'modules': modules, 
                'opt': optimizer, 
                'ce_loss': ce_loss,
                'loss': loss,
                'Lambda': lam,
                'opt_G': optimizer_G,
                'opt_D': optimizer_D}

    def get_normal(self, shape):
        with torch.no_grad():
            return torch.normal(torch.zeros(shape, dtype=torch.float32, device=self.device), torch.ones(shape, dtype=torch.float32, device=self.device))

    def get_cnn_embed(self, state):
        x = (state['pov'].permute(0, 1, 4, 2, 3)/255.0)*2.0 - 1.0
        if False:
            batch_size = x.shape[0]
            cnn_fea = self._model['modules']['cnn_model'](x[:,0])
            cnn_fea = self._model['modules']['flatten'](cnn_fea)
            cnn_fea = self._model['modules']['cnn_embed'](cnn_fea)
            with torch.no_grad():
                n_view = [-1] + [self.state_shape['pov'][2]] + self.state_shape['pov'][:2]
                cnn_fea_ng = self._model['modules']['cnn_model'](x[:,1:].contiguous().view(*n_view))
                cnn_fea_ng = self._model['modules']['flatten'](cnn_fea_ng)
                cnn_fea_ng = self._model['modules']['cnn_embed'](cnn_fea_ng)
            n_view = [batch_size,-1,self.args.hidden]
            return torch.cat([cnn_fea.unsqueeze(1),cnn_fea_ng.view(*n_view)],dim=1)
        else:
            batch_size = x.shape[0]
            n_view = [-1] + [self.state_shape['pov'][2]] + self.state_shape['pov'][:2]
            cnn_fea = self._model['modules']['cnn_model'](x.view(*n_view))
            cnn_fea = self._model['modules']['flatten'](cnn_fea)
            cnn_fea = self._model['modules']['cnn_embed'](cnn_fea)
            n_view = [batch_size,-1,self.args.hidden]
            return cnn_fea.view(*n_view)


    def get_action(self, state, cnn_embed):
        tembed = [cnn_embed[:,0]]
        for o in self.state_shape:
            if 'history' in o:
                tembed.append(state[o][:,0])
        if 'state' in state:
            tembed.append(state['state'][:,0])
        tembed = torch.cat(tembed, dim=-1)
        action = self._model['modules']['action_predict'](tembed)
        action_logits = torch.split(action[:,:self.action_logits_dim], self.bundle_dims, dim=-1)
        action_con_mean = action[:,self.action_logits_dim:(self.action_logits_dim+self.num_gaussian_contin*self.action_contin_dim)]
        action_con_mean = action_con_mean.view(-1, self.num_gaussian_contin, self.action_contin_dim)
        action_con_std = action[:,(self.action_logits_dim+self.num_gaussian_contin*self.action_contin_dim):(self.action_logits_dim+self.num_gaussian_contin*(self.action_contin_dim+self.action_contin_dim*(self.action_contin_dim+1)//2))]
        #action_con_std = F.softplus(action_con_std-0.5)
        action_con_std = action_con_std.view(-1, self.num_gaussian_contin, self.action_contin_dim*(self.action_contin_dim+1)//2)
        action_con_cov = torch.zeros(list(action_con_std.shape[:-1])+[self.action_contin_dim, self.action_contin_dim], dtype=torch.float32, device=self.device)
        if self.action_contin_dim == 2:
            action_con_cov[:,:,0,0] = F.softplus(action_con_std[:,:,0]-3.0)
            action_con_cov[:,:,0,1] = action_con_std[:,:,1]*1e-5
            action_con_cov[:,:,1,0] = action_con_std[:,:,1]*1e-5
            action_con_cov[:,:,1,1] = F.softplus(action_con_std[:,:,1]-3.0)
        else:
            assert False #not implemented
        action_gau_logits = action[:,(self.action_logits_dim+self.num_gaussian_contin*(self.action_contin_dim+self.action_contin_dim*(self.action_contin_dim+1)//2)):]
        #make MixtureModel
        categ = torch.distributions.Categorical(logits=action_gau_logits)
        gaussians = torch.distributions.MultivariateNormal(action_con_mean*1e-1, scale_tril=action_con_cov)
        #print(gaussians.batch_shape)
        #gaussians = torch.distributions.Independent(gaussians, 1)
        #print(gaussians.batch_shape)
        #print(categ.batch_shape)
        mix = MixtureSameFamily(categ, gaussians)
        return action_logits, mix

    def get_value(self, state, cnn_embed):
        tembed = [cnn_embed]
        for o in self.state_shape:
            if 'history' in o:
                tembed.append(state[o])
        if 'state' in state:
            tembed.append(state['state'])
        tembed = torch.cat(tembed, dim=-1)
        if False:
            value0 = self._model['modules']['v_predict'](tembed[:,0]).squeeze(-1)
            with torch.no_grad():
                valuer = self._model['modules']['v_predict'](tembed[:,1:]).squeeze(-1)
            return value0, valuer
        else:
            value = self._model['modules']['v_predict'](tembed).squeeze(-1)
            return value[:,0], value[:,1:]

    def get_next_embed(self, embed, last_embed):
        noise = self.get_normal(list(embed.shape[:-1])+[self.noise_dim])
        inp = torch.cat([embed, noise], dim=-1)
        next_embed = self._model['modules']['embed_gen'](inp)+last_embed
        return next_embed

    def discriminate(self, embed, next_embed):
        inp = torch.cat([embed, next_embed], dim=1)
        return self._model['modules']['embed_dis'](inp)

    def inverse_map_camera(self, x):
        y = x / 180.0
        return 0.5*torch.log((1+y)/(1-y))

    def map_camera(self, x):
        return np.tanh(x) * 180.0

    def get_BC_value_loss(self, state, action, reward, done, cnn_embed):
        action_logits, distribution = self.get_action(state, cnn_embed)
        value0, valuer = self.get_value(state, cnn_embed)

        #make truncated GVE
        with torch.no_grad():
            done = done.float()
            done = torch.cumsum(done,1).clamp(max=1.0)
            reward[:,1:] = reward[:,1:] * (1.0 - done[:,:-1]) #mask out reward from different episode
            gve_value = None
            gammas = GAMMA ** torch.arange(0, self.time_deltas[0], device=self.device).float()
            reward_slice = reward[:,:self.time_deltas[0]]
            disc_reward = (gammas.unsqueeze(0).expand(reward_slice.shape[0], -1) * reward_slice).sum(1)
            for i in range(len(self.time_deltas)):
                tvalue = valuer[:,i-1]
                tdone = done[:,self.time_deltas[i]-1]
                if gve_value is None:
                    gve_value = self.GVE_weigths[i] * (disc_reward + (1.0 - tdone) * GAMMA ** self.time_deltas[i] * tvalue)
                else:
                    gve_value += self.GVE_weigths[i] * (disc_reward + (1.0 - tdone) * GAMMA ** self.time_deltas[i] * tvalue)
                if i != len(self.time_deltas) - 1:
                    reward_slice = reward[:,self.time_deltas[i]:self.time_deltas[i+1]]
                    gammas = GAMMA ** torch.arange(self.time_deltas[i], self.time_deltas[i+1], device=self.device).float()
                    disc_reward += (gammas.unsqueeze(0).expand(reward_slice.shape[0], -1) * reward_slice).sum(1)
            gve_value /= self.GVE_norm
        value_loss = self._model['loss'](value0, gve_value.detach()) * 10.0
        value_loss_cpu = value_loss.data.cpu().numpy()
        loss = value_loss

        accuracy = 0.0
        entropy = []
        for j, bundle in enumerate(self.action_bundles):
            action_bundle = None
            action_m = 1
            for i, a in enumerate(bundle):
                if i == 0:
                    action_bundle = action[a][:,0]
                else:
                    action_bundle = action[a][:,0] * action_m
                action_m *= self.action_dict[a].n
            loss += self._model['ce_loss'](action_logits[j], action_bundle)
            with torch.no_grad():
                probs = self._model['modules']['softmax'](action_logits[j])
                entropy.append(-(probs*torch.log(probs)).sum(-1).mean().data.cpu().numpy())
                accuracy += probs.gather(1, action_bundle.unsqueeze(-1)).mean()
        with torch.no_grad():
            accuracy /= len(self.action_bundles)

        log_p = distribution.log_prob(action['camera'][:,0])
        loss += -log_p.mean()
        with torch.no_grad():
            delta = torch.abs(gve_value.detach()-value0).mean()
            variance = distribution.variance.mean().data.cpu().numpy()
            real_variance = action['camera'][:,0].var(0).mean().data.cpu().numpy()
        return loss, value0.mean(), delta, accuracy, value_loss_cpu, variance, real_variance, entropy

    def train_predict(self, state, action, cnn_embed):
        assert self.max_predict_range != 0
        tembed = [cnn_embed[:,:self.max_predict_range].detach()]
        tembed.append(state['history_reward'][:,:self.max_predict_range])
        if 'state' in state:
            tembed.append(state['state'][:,:self.max_predict_range])
        tembed_wo_a = torch.cat(tembed, dim=-1)
        tactions = []
        for j, bundle in enumerate(self.action_bundles):
            action_bundle = None
            action_m = 1
            for i, a in enumerate(bundle):
                if i == 0:
                    action_bundle = action[a][:,:self.max_predict_range]
                else:
                    action_bundle = action[a][:,:self.max_predict_range] * action_m
                action_m *= self.action_dict[a].n
            def one_hot(ac, num):
                one_hot = torch.zeros(list(ac.shape)+[num], dtype=torch.float32, device=self.device)
                one_hot.scatter_(-1, ac.unsqueeze(-1), 1.0)
                return one_hot
            tactions.append(one_hot(action_bundle,action_m))
        tactions.append(action['camera'][:,:self.max_predict_range])
        tactions = torch.cat(tactions, dim=-1)
        tembed.append(tactions)
        tembed = torch.cat(tembed, dim=-1)

        lossD = None
        lossG = None

        next_embed = tembed_wo_a[:,0]

        for i in range(self.max_predict_range-1):
            #train D
            next_embed = self.get_next_embed(torch.cat([next_embed, tactions[:,i]],dim=-1), next_embed)
            real_embed = tembed_wo_a[:,i+1]
            score_fake = self.discriminate(tembed[:,i], next_embed.detach())
            score_real = self.discriminate(tembed[:,i], real_embed.detach())

            E_P_f,  E_Q_f  = score_real.mean(), score_fake.mean()
            E_P_f2, E_Q_f2 = (score_real**2).mean(), (score_fake**2).mean()
            constraint = (1 - (0.5*E_P_f2 + 0.5*E_Q_f2))

            if lossD is None:
                lossD = -(E_P_f - E_Q_f + self._model['Lambda'][0] * constraint - (self.args.rho/2) * constraint**2)
            else:
                lossD += -(E_P_f - E_Q_f + self._model['Lambda'][0] * constraint - (self.args.rho/2) * constraint**2)

            if self.gen_ph % self.gen_train_freq == 0:
                #train G
                for p in self._model['modules']['embed_dis'].parameters():
                    p.requires_grad = False
                score_fake = self.discriminate(tembed[:,i], next_embed)
                for p in self._model['modules']['embed_dis'].parameters():
                    p.requires_grad = True
                if lossG is None:
                    lossG = -score_fake.mean() * (LAMBDA ** i)
                else:
                    lossG += -score_fake.mean() * (LAMBDA ** i)
                lossG += self._model['loss'](next_embed, tembed_wo_a[:,i+1].detach()) * 0.1


        self._model['Lambda'].retain_grad()
        lossD.backward()
        self._model['opt_D'].step()
        self._model['Lambda'].data += self.args.rho * self._model['Lambda'].grad.data
        self._model['Lambda'].grad.data.zero_()
        
        if self.gen_ph % self.gen_train_freq == 0:
            lossG.backward()
            self._model['opt_G'].step()
            self._model['opt_G'].zero_grad()
        self._model['opt_D'].zero_grad()

        self.gen_ph += 1

        return lossD, lossG

    def pretrain(self, input):
        state_keys = ['state', 'pov', 'history_action', 'history_reward']
        states = {}
        for k in state_keys:
            if k in input:
                states[k] = variable(input[k])
        actions = {'camera': self.inverse_map_camera(variable(input['camera']))}
        for b in self.action_bundles:
            for a in b:
                actions[a] = variable(input[a], True).long()
        reward = variable(input['reward'])
        done = variable(input['done'])

        #train state embedding
        self._model['opt'].zero_grad()
        cnn_embed = self.get_cnn_embed(states)
        loss, value, delta, accuracy, value_loss, variance, real_variance, entropy = self.get_BC_value_loss(states, actions, reward, done, cnn_embed)
        lossD, lossG = self.train_predict(states, actions, cnn_embed)
        loss.backward()
        #for n, p in self._model['modules']['cnn_model'].named_parameters():
        #    with torch.no_grad():
        #        print(n, p.grad.mean().data.cpu().numpy())
        self._model['opt'].step()

        return {'loss': loss.data.cpu().numpy(),
                'lossD': lossD.data.cpu().numpy(),
                'lossG': -1.0 if lossG is None else lossG.data.cpu().numpy(),
                'Lambda': self._model['Lambda'][0].data.data.cpu().numpy(),
                'value_loss': value_loss,
                'val': value.data.cpu().numpy(),
                'Dval': delta.data.cpu().numpy(),
                'accuracy': accuracy.data.cpu().numpy(),
                'entropy': entropy,
                'variance': variance,
                'real_variance': real_variance}


    def train(self, input):
        return {}
    

    def get_cnn_embed_single(self, state):
        x = (state['pov'][:,0].permute(0, 3, 1, 2)/255.0)*2.0 - 1.0
        cnn_fea = self._model['modules']['cnn_model'](x)
        cnn_fea = self._model['modules']['flatten'](cnn_fea)
        cnn_fea = self._model['modules']['cnn_embed'](cnn_fea)
        return cnn_fea

    def select_action(self, state, batch=False):
        if not batch:
            for k in state:
                state[k] = state[k][None, None,:].copy()
        state = variable(state)
        with torch.no_grad():
            cnn_embed = self.get_cnn_embed_single(state)
            action_logits, distribution = self.get_action(state, cnn_embed.unsqueeze(1))
            entropy = 0.0
            r_action = {}
            allprobs = []
            for i, b in enumerate(self.action_bundles):
                probs = self._model['modules']['softmax'](action_logits[i]).data.cpu().numpy()[0]
                allprobs.append(probs)
                entropy += -np.sum(probs*np.log(probs))
                action_b = int(np.random.choice(range(len(probs)), p=probs))
                for ra in self.action_bundles[i]:
                    r_action[ra] = action_b%self.action_dict[ra].n
                    action_b //= self.action_dict[ra].n
            
            r_action['camera'] = distribution.sample((1,)).data.cpu().numpy()[0,0]
            variance = distribution.variance.data.cpu().numpy()
            r_action['camera'] = self.map_camera(r_action['camera'])
            #r_action['camera'] = np.clip(r_action['camera'], -5.0, 5.0)
            action = OrderedDict()
            for a in self.action_dict:
                action[a] = r_action[a]
        return action, {'entropy': entropy,
                        'variance': variance,
                        'allprobs': allprobs}



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Discretizer Test")
    parser.add_argument("--hidden", default=256, type=int, help="Hidden neurons of the policy network")
    HashingMemory.register_args(parser)
    args = parser.parse_args()
    HashingMemory.check_params(args)

    #test discretizer
    if True:
        if False:
            test_disc = FullDiscretizer(1, torch.device("cpu"), args, {'forward_back': gym.spaces.Discrete(3), 
                                                                       'left_right': gym.spaces.Discrete(3),
                                                                       'jump': gym.spaces.Discrete(2),
                                                                       'sneak_sprint': gym.spaces.Discrete(3),
                                                                       'attack': gym.spaces.Discrete(2)})
        else:
            if False:
                test_disc = FullDiscretizer(1, torch.device("cpu"), args, {'A': gym.spaces.Discrete(3), 'B': gym.spaces.Discrete(2)})
                test_disc.max_camstep = 6

                action = {}
                action['A'] = torch.from_numpy(np.asarray([0, 2])).long()
                action['B'] = torch.from_numpy(np.asarray([1, 0])).long()
                action['camera'] = torch.from_numpy(np.asarray([[-0.13, 0.05], [0.0, 0.0]])).float()
                dummy_input = torch.from_numpy(np.asarray([[0.0], [-0.2]])).float()

                loss, entropy, alldacs = test_disc.get_loss_BC(dummy_input, action, True)

                print(loss.data.cpu().numpy(), entropy.data.cpu().numpy())
                print(alldacs.data.cpu().numpy())
                alldacs[1,3] = 4

                saction, sconaction = test_disc.sample(dummy_input, alldacs)
                print(saction)
                print(sconaction)
            else:
                test_disc = FullDiscretizerWLast(1, torch.device("cpu"), args, {'A': gym.spaces.Discrete(3), 'B': gym.spaces.Discrete(2), 'camera': gym.spaces.Box(-1.0, 1.0, shape=(2,))})
                test_disc.max_camstep = 6

                action = {}
                action['A'] = torch.from_numpy(np.asarray([0, 2])).long()
                action['B'] = torch.from_numpy(np.asarray([1, 0])).long()
                action['camera'] = torch.from_numpy(np.asarray([[-0.13, 0.05], [0.0, 0.01]])).float()
                laction = {}
                laction['last_A'] = torch.from_numpy(np.asarray([1, 2])).long()
                laction['last_B'] = torch.from_numpy(np.asarray([0, 1])).long()
                laction['last_camera'] = torch.from_numpy(np.asarray([[0.0, 0.0], [0.5, 0.0]])).float()
                dummy_input = torch.from_numpy(np.asarray([[0.0], [-0.2]])).float()

                loss, entropy, alldacs = test_disc.get_loss_BC(dummy_input, action, laction, True)

                print(loss.data.cpu().numpy(), entropy.data.cpu().numpy())
                print(alldacs.data.cpu().numpy())
                #alldacs[1,3] = 5

                saction, sconaction, _ = test_disc.sample(dummy_input, laction, alldacs)
                print(saction)
                print(sconaction)
    else:
        test_disc = Discretizer(1, torch.device("cpu"), args)
        test_disc.max_step = 10

        real_action = torch.from_numpy(np.asarray([[-0.13, 0.05], [0.2, -0.2]])).float()
        dummy_input = torch.from_numpy(np.asarray([[0.0], [-0.2]])).float()

        loss, _, alldacs = test_disc.get_loss_BC(dummy_input, real_action)

        smpl = test_disc.sample(dummy_input, alldacs)

        print(smpl.data.cpu().numpy())
        print(smpl.shape)