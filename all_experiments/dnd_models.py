import torch
import numpy as np
from memory import HashingMemory
from torch.nn import functional as F
from collections import OrderedDict
from radam import RAdam
from models import CUDA, variable, LearnerModel, Lambda, swish, Flatten
from dnd import DND
from torch.multiprocessing import Process, Queue
from shared_memory2 import SharedMemory
import datetime
import random
from random import shuffle
import traceback
import importlib
import threading

GAMMA = 0.995

class NECAgent():
    def __init__(self, state_shape, num_actions, args):
        assert isinstance(state_shape, dict)
        self.state_shape = state_shape
        self.num_actions = num_actions
        self.args = args
        self.last_state = None
        self.no_update = True
        self.cnn_fea_t = None
        
        self._setup()
        self._a = self._models[0]

    def _setup(self):
        self._models = [self._make_model()]

    def state_dict(self):
        return [self._models[0]['modules'].state_dict()]

    def load_state_dict(self, s, strict=True):
        self.no_update = False
        param_found = False
        if isinstance(s, list):
            del_list = []
            for k in s[0]:
                if 'dnd_list' in k and 'keys' in k:
                    param_found = True
                    h = int(k.replace('dnd_list.','').replace('.keys',''))
                    self._models[0]['modules']['dnd_list'][h].keys = torch.nn.Parameter(s[0][k].data)
                    del_list.append(k)
                if 'dnd_list' in k and 'values' in k:
                    h = int(k.replace('dnd_list.','').replace('.values',''))
                    self._models[0]['modules']['dnd_list'][h].values = torch.nn.Parameter(s[0][k].data)
                    del_list.append(k)
            for k in del_list:
                del s[0][k]
            self._models[0]['modules'].load_state_dict(s[0], False)
        else:
            del_list = []
            for k in s:
                if 'dnd_list' in k and 'keys' in k:
                    param_found = True
                    h = int(k.replace('dnd_list.','').replace('.keys',''))
                    self._models[0]['modules']['dnd_list'][h].keys = torch.nn.Parameter(s[k].data)
                    del_list.append(k)
                if 'dnd_list' in k and 'values' in k:
                    h = int(k.replace('dnd_list.','').replace('.values',''))
                    self._models[0]['modules']['dnd_list'][h].values = torch.nn.Parameter(s[k].data)
                    del_list.append(k)
            for k in del_list:
                del s[k]
            self._models[0]['modules'].load_state_dict(s, False)
        assert param_found
        for dnd in self._models[0]['modules']['dnd_list']:
            dnd.kdtree.build_index(dnd.keys.data.cpu().numpy())
            dnd.stale_index = False
        self._a = self._models[0]

    def loadstore(self, filename, load=True):
        if load:
            self.load_state_dict(torch.load(filename + '-dnd_nec'))
        else:
            torch.save(self._models[0]['modules'].state_dict(), filename + '-dnd_nec')


    def _make_model(self):
        #make CNN
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
        self.cnn_model = torch.nn.Sequential(*cnn_layers)

        inp_size = self.cnn_model(torch.zeros((1,self.state_shape['pov'][2], self.state_shape['pov'][0], self.state_shape['pov'][1]))).shape[-1]
        self.inp_size = inp_size+self.state_shape['state'][0]

        self.feature_dim = 64
        self.feature = torch.nn.Sequential(torch.nn.Linear(self.inp_size, self.args.hidden), Lambda(lambda x: swish(x)), torch.nn.Linear(self.args.hidden, self.feature_dim), torch.nn.Tanh())
        self.action_predict = torch.nn.Linear(self.feature_dim, self.num_actions)
        self.V_predict = torch.nn.Linear(self.feature_dim, 1)

        if torch.cuda.is_available() and CUDA:
            self.cnn_model = self.cnn_model.cuda()
            self.feature = self.feature.cuda()
            self.action_predict = self.action_predict.cuda()
            self.V_predict = self.V_predict.cuda()

        ce_loss = torch.nn.CrossEntropyLoss()
        loss = torch.nn.MSELoss()
        modules = torch.nn.ModuleDict()
        modules['cnn_model'] = self.cnn_model
        modules['feature'] = self.feature
        modules['action_predict'] = self.action_predict
        modules['V_predict'] = self.V_predict

        model_params = []
        nec_value_params = []
        for name, p in modules.named_parameters():
            if 'values.weight' in name:
                nec_value_params.append(p)
            else:
                model_params.append(p)


        optimizer = RAdam([{'params': model_params}, 
                           {'params': nec_value_params, 'lr': self.args.nec_lr}], lr=self.args.lr)

        def inverse_distance(h, h_i, epsilon=1e-3):
            return 1 / (((h - h_i)**2).sum(-1)**0.5 + epsilon)
        self.dnd_list = [DND(inverse_distance, 32, int(5e5), self.args.lr)
                     for _ in range(self.num_actions)]
        self.dnd_list = torch.nn.ModuleList(self.dnd_list).cuda()
        modules['dnd_list'] = self.dnd_list

        return {'modules': modules, 
                'opt': optimizer, 
                'ce_loss': ce_loss,
                'loss': loss}


    def forward(self, state, cnn_fea=None):
        if cnn_fea is None:
            x = (state['pov'].permute(0, 3, 1, 2)/255.0)*2.0 - 1.0
            cnn_fea = self._a['modules']['cnn_model'](x)
        fea = torch.cat([cnn_fea, state['state']], dim=1)

        fea = self._a['modules']['feature'](fea)
        return fea, cnn_fea


    def read_state(self, input, expert_input=None):
        state_keys = ['state', 'pov']

        states = {}
        for k in state_keys:
            if k in input:
                states[k] = variable(input[k])
        next_states = {}
        for k in state_keys:
            next_k = 'next_'+k
            if next_k in input:
                next_states[k] = variable(input[next_k])
        actions = variable(input['action'], True).long()
        rewards = variable(input['reward'])
        done = input['done'].data.cpu().numpy()

        if expert_input is not None:
            expert_states = {}
            for k in state_keys:
                if k in expert_input:
                    expert_states[k] = variable(expert_input[k])
            expert_next_states = {}
            for k in state_keys:
                next_k = 'next_'+k
                if next_k in expert_input:
                    expert_next_states[k] = variable(expert_input[next_k])


            expert_actions = variable(expert_input['action'], True).long()
            expert_rewards = variable(expert_input['reward'])
            expert_done = expert_input['done'].data.cpu().numpy()

            #concat everything
            for k in states:
                states[k] = torch.cat([states[k],expert_states[k]], dim=0)
            for k in next_states:
                next_states[k] = torch.cat([next_states[k],expert_states[k]], dim=0)
            actions = torch.cat([actions, expert_actions], dim=0)
            rewards = torch.cat([rewards, expert_rewards], dim=0)

            done = np.concatenate([done, expert_done], axis=0)


        return states, next_states, actions, rewards, done

    def train_dnd(self, input, expert_input=None):
        #print("start train dnd")
        states, next_states, actions, rewards, done = self.read_state(input, expert_input)
        next_indexes = np.nonzero(np.logical_not(done))[0]
        batch_size = done.shape[0]
        cpu_actions = actions.data.cpu().numpy()

        with torch.no_grad():
            feas, _ = self.forward(states)
            next_feas, _ = self.forward(next_states)
            next_feas_cpu = next_feas.cpu()
            allactions = set()
            for i in range(batch_size):
                #print(i)
                #rewards[i] = rewards[i] + GAMMA ** (self.args.skip + 1) * torch.cat([dnd.lookup(next_feas[i][None, :], lookup_key_cpu=next_feas_cpu) for dnd in self._a['modules']['dnd_list']]).max()
                #fea = feas[i]
                action = cpu_actions[i]
                allactions.add(action)
                #dnd = self._a['modules']['dnd_list'][action]

                #embedding_index = dnd.get_index(fea[None, :])
                #if embedding_index is None:
                #    dnd.insert(fea[None, :].detach(), reward[None].detach())
                #else:
                #    Q = self.Q_update(dnd.values[embedding_index], reward)
                #    dnd.update(Q[None].detach(), embedding_index)
            value = []
            for action in allactions:
                dnd = self._a['modules']['dnd_list'][action]
                value.append(dnd.lookup_batch(next_feas, lookup_key_cpu=next_feas_cpu)[:,None])
            rewards[next_indexes] = rewards[next_indexes] + GAMMA ** (self.args.skip + 1) * torch.cat(value, dim=1).max(1)[0][next_indexes]

            input['reward'] = rewards[:input['reward'].shape[0]]
            if expert_input is not None:
                expert_input['reward'] = rewards[input['reward'].shape[0]:]
            
            for action in allactions:
                dnd = self._a['modules']['dnd_list'][action]
                indices = ((actions==action).nonzero()).squeeze(1)
                dnd.insert_batch(feas[indices], rewards[indices])
                dnd.commit_insert()
            #print("end train dnd")
            for k in input:
                input[k] = input[k].detach().cpu().share_memory_()
            return input, expert_input

    def Q_update(self, q_initial, q_n):
        return q_initial + self.args.clr * (q_n - q_initial)

    def train(self, input, expert_input=None):
        states, next_states, actions, rewards, done = self.read_state(input, expert_input)
        print('A')
        next_indexes = np.nonzero(np.logical_not(done))[0]
        batch_size = done.shape[0]
        fea, _ = self.forward(states)
        print('B')
        if False:
            cpu_actions = actions.data.cpu().numpy()
            allactions = set()
            for i in range(batch_size):
                allactions.add(cpu_actions[i])

            predicted = torch.zeros_like(rewards)
            for action in allactions:
                dnd = self._a['modules']['dnd_list'][action]
                indices = ((actions==action).nonzero()).squeeze(1)
                afea = fea[indices]
                avals = dnd.lookup_batch(afea, update_flag=True)
                predicted.scatter_(0, indices, avals)

            #predicted = torch.cat([self._a['modules']['dnd_list'][actions[i]].lookup(fea[i][None, :], update_flag=True) for i in range(batch_size)])
            loss = self._a['loss'](predicted, rewards)

        logits_action = self._a['modules']['action_predict'](fea)
        ce_loss = self._a['ce_loss'](logits_action, actions)
        loss = ce_loss

        V_pred = self._a['modules']['V_predict'](fea).squeeze(-1)
        loss += self._a['loss'](V_pred, rewards)
        
        print('C')
        self._a['opt'].zero_grad()
        print('D')
        loss.backward()
        print('E')
        self._a['opt'].step()
        print('F')
        if False:
            for action in allactions:
                dnd = self._a['modules']['dnd_list'][action]
                with_rebuild = True if np.random.sample() < 1.0/100.0 else False
                dnd.update_params(with_rebuild)

        return {'loss': loss.data.cpu().numpy(),
                'vals': V_pred.mean().data.cpu().numpy(),
                'ce_loss': ce_loss.data.cpu().numpy()}

    def select_action(self, state):
        make_latent = True
        if self.last_state is not None:
            d = 0.0
            d += np.sum(np.abs(state['pov'] - self.last_state))
            if d < 1e-8:
                make_latent = False
        self.last_state = np.copy(state['pov'])
        for k in state:
            state[k] = state[k][None,:]
        state = variable(state)
        if self.no_update:
            action_index = np.random.randint(0, self.num_actions)
            return action_index, {}
        else:
            with torch.no_grad():
                if make_latent or self.cnn_fea_t is None:
                    fea, self.cnn_fea_t = self.forward(state)
                else:
                    fea, self.cnn_fea_t = self.forward(state, self.cnn_fea_t)
                q_estimates = [dnd.lookup(fea)[None] for dnd in self._a['modules']['dnd_list']]
                action_index = torch.cat(q_estimates).max(0)[1].data[0].cpu().numpy()
            if np.random.sample() < 0.01:
                action_index = np.random.randint(0, self.num_actions)
        return action_index, {'value': torch.cat(q_estimates).max(0)[0].data[0].cpu().numpy()}



class NECTrainer():
    def __init__(self, agent_name, agent_from, state_shape, num_actions, args, input_queues, from_traj_generator, to_replay_buffer, copy_queues=None):
        self.copy_queues = copy_queues
        self.to_replay_buffer = to_replay_buffer
        self.p = Process(target=NECTrainer.train_worker, args=(agent_name, agent_from, state_shape, num_actions, args, input_queues, self.copy_queues, from_traj_generator, self.to_replay_buffer))
        self.p.start()


    @staticmethod
    def train_worker(agent_name, agent_from, state_shape, num_actions, args, input_queues, copy_queues, from_traj_generator, to_replay_buffer):
        try:
            AgentClass = getattr(importlib.import_module(agent_from), agent_name)
            agent = AgentClass(state_shape, num_actions, args)
            if args.load is not None:
                agent.loadstore(args.load)

            old_dt = datetime.datetime.now()
            old_dt2 = datetime.datetime.now()

            lock = threading.Lock()
            def dnd_trainer():
                jj = 0
                #train dnd
                while True:
                    if to_replay_buffer[jj].qsize() < 3:
                        gen = from_traj_generator[jj]
                        data_shs = None
                        #try:
                        data_shs = gen.get()
                        #except:
                        #    pass
                        if data_shs is not None:
                            with lock:
                                new_shs,_ = agent.train_dnd(data_shs)
                            to_replay_buffer[jj].put(new_shs)
                    jj = (jj + 1) % len(from_traj_generator)


            dnd_th = threading.Thread(target=dnd_trainer)
            dnd_th.start()

            i = 0
            kk = 0
            queues_active = []
            for q in input_queues:
                queues_active.append([False for i in range(len(q))])
            while True:
                input = []
                for queues in input_queues:
                    queue = random.sample(queues, 1)[0]
                    input.append(queue.get())
                with lock:
                    print("start train")
                    output = agent.train(*input)
                    print("end train")
                for _input in input:
                    for k in _input:
                        del _input[k]
                        _input[k] = None
                del input

                if i % 100 == 0:
                    for k in output:
                        print(k,output[k])

                if copy_queues is not None and (datetime.datetime.now() - old_dt2).total_seconds() > 2*60.0:
                    with lock:
                        print("copy start")
                        state_dict = agent._models[0]['modules'].state_dict()
                        for param_name in state_dict:
                            state_dict[param_name] = state_dict[param_name].cpu().clone().share_memory_()
                        print("copy end")

                    for copy_queue in copy_queues:
                        if copy_queue.full():
                            try:
                                copy_queue.get_nowait()
                            except:
                                pass
                        copy_queue.put(state_dict)

                    old_dt2 = datetime.datetime.now()

                if (datetime.datetime.now() - old_dt).total_seconds() > 15*60.0:
                    if args.save is not None:
                        print("")
                        print("")
                        print("Save agent")
                        print("")
                        print("")
                        with lock:
                            agent.loadstore(args.save, load=False)

                    old_dt = datetime.datetime.now()
                i += 1
            dnd_th.join()
        except:
            print("FATAL error in Trainer")
            traceback.print_exc()
            dnd_th.join()