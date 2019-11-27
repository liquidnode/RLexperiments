from shared_memory2 import SharedMemory
from models import Learner, SoftQ, IQN
import numpy as np
from torch.multiprocessing import Process, Queue
import datetime
import torch
import random
from random import shuffle
import traceback
import importlib
import time

GAMMA = 0.999
KAPPA = 1.0

class Actor(Learner):
    def __init__(self, state_shape, num_actions, args):
        super(Actor, self).__init__(state_shape, num_actions, args, False)

    def predict(self, state, batch=False):
        """ Return a probability distribution over actions
        """
        if not batch:
            for k in state:
                state[k] = state[k][None,:]
        return self._predict_model(self._models[0], state)[0]

    def select_action(self, state):
        probas = self.predict(state)

        action_index = int(np.random.choice(range(self.num_actions), p=probas))
        entropy = -np.sum(probas*np.log2(probas))
        return action_index, entropy

    def train(self, input):
        state_keys = ['state', 'pov']
        states = {}
        for k in state_keys:
            if k in input:
                states[k] = input[k]

        actions = input['action']
        critic_qvalues = input['critic_qvalues'].cpu().numpy()

        CN = np.arange(critic_qvalues.shape[0])

        # Pursuit: Value of the current state
        max_indexes = critic_qvalues.argmax(1)

        # Pursuit: Update actor
        train_probas = np.zeros_like(critic_qvalues)
        train_probas[CN, max_indexes] = 1.0

        # Normalize the direction to be pursued
        actor_probas = self._predict_model(self._models[0], states)
        entropy = np.mean(np.sum(-actor_probas*np.log2(actor_probas), axis=1))
        entropy_std = np.mean(np.std(actor_probas, axis=0))
        
        # Discuss gradient vs target, and say that https://www.sciencedirect.com/science/article/pii/S0016003205000645 uses
        # a gradient-based approach with continuous actions (which sure works, it's policy gradient)
        train_probas /= 1e-6 + train_probas.sum(1)[:, None]
        train_probas = (1. - self.args.alr) * actor_probas + self.args.alr * train_probas

        self._train_model(
            self._models[0],
            states,
            train_probas,
            self.args.aepochs)

        return entropy, entropy_std

    def pretrain(self, input):
        state_keys = ['state', 'pov']
        states = {}
        for k in state_keys:
            if k in input:
                states[k] = input[k]

        actions = input['action']

        return self._pretrain_model(
            self._models[0],
            states,
            actions)
    
    def loadstore(self, filename, load=True):
        if load:
            self.load_state_dict(torch.load(filename + '-actor'))
        else:
            torch.save(self._models[0]['model'].state_dict(), filename + '-actor')


class ActorTrainer():
    def __init__(self, state_shape, num_actions, args, input_queues, copy_queues=None):
        self.p = Process(target=ActorTrainer.train_worker, args=(state_shape, num_actions, args, input_queues, copy_queues))
        self.p.start()


    @staticmethod
    def train_worker(state_shape, num_actions, args, input_queues, copy_queues):
        try:
            actor = Actor(state_shape, num_actions, args)
            if args.load is not None:
                actor.loadstore(args.load)

            old_dt = datetime.datetime.now()
            old_dt2 = datetime.datetime.now()

            loss = None
            i = 0
            while True:
                queue = random.sample(input_queues, 1)[0]
                input = queue.get()
                if args.pretrain:
                    loss = actor.pretrain(input)
                else:
                    entropy, entropy_std = actor.train(input)
                #free input
                for k in input:
                    del input[k]
                    input[k] = None
                del input

                if i % 100 == 0:
                    if args.pretrain:
                        print("actor loss\t"+str(loss))
                    else:
                        print("actor entropy",entropy,"/",entropy_std)
                    
                if copy_queues is not None and (datetime.datetime.now() - old_dt2).total_seconds() > 60.0:
                    state_dict = actor._models[0]['model'].state_dict()
                    for param_name in state_dict:
                        state_dict[param_name] = state_dict[param_name].clone().cpu().share_memory_()

                    for copy_queue in copy_queues:
                        if copy_queue.full():
                            try:
                                copy_queue.get_nowait()
                            except:
                                pass
                        copy_queue.put(state_dict)

                    old_dt2 = datetime.datetime.now()

                if (datetime.datetime.now() - old_dt).total_seconds() > 5*60.0:
                    if args.save is not None:
                        print("")
                        print("Saving")
                        print("")
                        actor.loadstore(args.save, load=False)

                    old_dt = datetime.datetime.now()
                i += 1
        except:
            print("FATAL error in ActorTrainer")
            traceback.print_exc()



class Critic(Learner):
    def __init__(self, state_shape, num_actions, args):
        super(Critic, self).__init__(state_shape, num_actions, args, True)
        self._a, self._b = self._models

    def loadstore(self, filename, id, load=True, pretrain=False):
        if load:
            if not pretrain:
                self.load_state_dict([torch.load(filename + '-criticA' + str(id)),
                                      torch.load(filename + '-criticB' + str(id))])
            else:
                self.load_state_dict([torch.load(filename + '-actor'),
                                      torch.load(filename + '-actor')])
                
                if self.args.neural_episodic_control:
                    self._models[0]['model'].state_dict()['1.values.weight'].data.zero_()
                    self._models[1]['model'].state_dict()['1.values.weight'].data.zero_()
                else:
                    assert False #Not implemented
        else:
            torch.save(self._models[0]['model'].state_dict(), filename + '-criticA' + str(id))
            torch.save(self._models[1]['model'].state_dict(), filename + '-criticB' + str(id))

    def train(self, input):
        if self.args.per:
            assert False #not implemented yet
        else:
            is_weights = None
        state_keys = ['state', 'pov']
        states = {}
        for k in state_keys:
            if k in input:
                states[k] = input[k]
        next_states = {}
        for k in state_keys:
            next_k = 'next_'+k
            if next_k in input:
                next_states[k] = input[next_k]

        actions = input['action'].data.cpu().numpy()
        rewards = input['reward'].data.cpu().numpy()
        done = input['done'].data.cpu().numpy()
        next_indexes = np.nonzero(np.logical_not(done))[0]

        error = np.zeros_like(rewards)

        for i in range(self.args.q_loops):
            # Q-Learning
            critic_qvalues = self._predict_model(self._a, states)

            error = np.maximum(
                self._train_loop(
                    states,
                    actions,
                    rewards,
                    critic_qvalues,
                    next_states,
                    next_indexes,
                    is_weights
                ), error)

            # Clipped DQN, as Double DQN does, swaps the models after every training iteration
            self._a, self._b = self._b, self._a

        if self.args.per:
            #update per
            assert False #not implemented yet

        return (states, actions, critic_qvalues, error)

    def _train_loop(self, states, actions, rewards, critic_qvalues, next_states, next_indexes, is_weights):
        QN = np.arange(actions.shape[0])
        next_values = np.copy(rewards)
        next_values[next_indexes] += GAMMA * self._get_values(next_states)[next_indexes]

        #advantage learning
        if KAPPA < 1.0:
            max_a = np.max(critic_qvalues, axis=-1)
            next_values -= max_a
            next_values = max_a + (next_values / KAPPA)

        error = np.abs(next_values - critic_qvalues[QN, actions])
        critic_qvalues[QN, actions] += self.args.clr * (next_values - critic_qvalues[QN, actions])

        self._train_model(
            self._a,
            states,
            critic_qvalues,
            self.args.cepochs,
            is_weights)
        return error

    def _get_values(self, states):
        """ Return a list of values, one for each state.
        """
        qvalues_a = self._predict_model(self._a, states)
        qvalues_b = self._predict_model(self._b, states)
        QN = np.arange(qvalues_a.shape[0])

        qvalues = np.minimum(qvalues_a, qvalues_b)  # Clipped DQN target

        return qvalues[QN, qvalues_a.argmax(1)]

class CriticTrainer():
    def __init__(self, state_shape, num_actions, args, input_queues, id, output_queue=None, num_critics=1):
        self.output_queue = output_queue
        self.p = Process(target=CriticTrainer.train_worker, args=(state_shape, num_actions, args, input_queues, id, self.output_queue, num_critics))
        self.p.start()


    @staticmethod
    def train_worker(state_shape, num_actions, args, input_queues, id, output_queue, num_critics):
        try:
            citic_n = 0
            critics = []
            for n in range(num_critics):
                critic = Critic(state_shape, num_actions, args)
                if args.load is not None:
                    critic.loadstore(args.load, id*num_critics+n)
                elif args.pretrain_load is not None:
                    critic.loadstore(args.pretrain_load, id*num_critics+n, pretrain=True)
                critics.append(critic)

            old_dt = datetime.datetime.now()

            r_accuracy = None
            i = 0
            while True:
                queue = random.sample(input_queues, 1)[0]
                input = queue.get()
                critic = critics[citic_n]
                citic_n += 1
                if citic_n == len(critics):
                    citic_n = 0
                    shuffle(critics)
                output = critic.train(input)

                if output_queue is not None:
                    input['critic_qvalues'] = SharedMemory(output[2])
                    input['critic_qvalues'] = input['critic_qvalues'].shared_memory()
                    #input['critic_qvalues'] = output[2]
                    del input['reward']
                    del input['done']
                    output_queue.put(input)
                else:
                    #free input
                    for k in input:
                        del input[k]
                        input[k] = None
                    del input

                actions = output[1]
                critic_qvalues = output[2]
                critic_actions = np.argmax(critic_qvalues, axis=1)
                batch_size = actions.shape[0]
                if args.minerl:
                    s = np.where(actions==critic_actions, 1, 0)*np.where(actions!=0, 1, 0)
                    s2 = np.where(actions==0, 1, 0)
                    if batch_size-np.sum(s2) != 0:
                        accuracy = np.sum(s) / (batch_size-np.sum(s2))
                        if r_accuracy is None:
                            r_accuracy = accuracy
                        else:
                            r_accuracy = 0.99 * r_accuracy + 0.01 * accuracy
                else:
                    s = np.where(actions==critic_actions, 1, 0)
                    accuracy = np.sum(s) / batch_size
                    if r_accuracy is None:
                        r_accuracy = accuracy
                    else:
                        r_accuracy = 0.99 * r_accuracy + 0.01 * accuracy

                if i % 100 == 0:
                    print("value",np.mean(np.max(critic_qvalues, axis=1)))
                    print("accuracy",r_accuracy)
                    print("error",np.mean(output[3]))

                if (datetime.datetime.now() - old_dt).total_seconds() > 5*60.0:
                    if args.save is not None:
                        print("Save critics"+str(id))
                        for n, critic in enumerate(critics):
                            critic.loadstore(args.save, id*num_critics+n, load=False)

                    old_dt = datetime.datetime.now()
                i += 1
        except:
            print("FATAL error in CriticTrainer")
            traceback.print_exc()



class SoftQTrainer():
    def __init__(self, state_shape, num_actions, args, input_queues, expert_queues=None, copy_queues=None):
        self.copy_queues = copy_queues
        self.p = Process(target=SoftQTrainer.train_worker, args=(state_shape, num_actions, args, input_queues, expert_queues, self.copy_queues))
        self.p.start()


    @staticmethod
    def train_worker(state_shape, num_actions, args, input_queues, expert_queues, copy_queues):
        try:
            softq = SoftQ(state_shape, num_actions, args)
            if args.load is not None:
                softq.loadstore(args.load)

            old_dt = datetime.datetime.now()
            old_dt2 = datetime.datetime.now()

            i = 0
            while True:
                queue = random.sample(input_queues, 1)[0]
                input = queue.get()
                if expert_queues is not None:
                    queue = random.sample(expert_queues, 1)[0]
                    expert_input = queue.get()
                else:
                    expert_input = None
                if args.pretrain:
                    output = softq.pretrain(input)
                else:
                    output = softq.train(input, expert_input)
                for k in input:
                    del input[k]
                    input[k] = None
                del input
                if expert_input is not None:
                    for k in expert_input:
                        del expert_input[k]
                        expert_input[k] = None
                    del expert_input

                if i % 100 == 0:
                    if args.pretrain:
                        loss = output[0]
                        print("loss",loss)
                        print("entropy",output[1])
                        print("max_probs",output[2])
                    else:
                        critic_qvalues = output[0]
                        entropy = output[1]
                        loss = output[2]
                        print("value",np.mean(np.max(critic_qvalues, axis=1)))
                        print("entropy",entropy)
                        print("loss",loss,"eps",output[3])

                if copy_queues is not None and (datetime.datetime.now() - old_dt2).total_seconds() > 60.0:
                    state_dict = softq._models[0]['Q_values'].state_dict()
                    for param_name in state_dict:
                        state_dict[param_name] = state_dict[param_name].clone().cpu().share_memory_()

                    for copy_queue in copy_queues:
                        if copy_queue.full():
                            try:
                                copy_queue.get_nowait()
                            except:
                                pass
                        copy_queue.put(state_dict)

                    old_dt2 = datetime.datetime.now()

                if (datetime.datetime.now() - old_dt).total_seconds() > 5*60.0:
                    if args.save is not None:
                        print("Save softq")
                        softq.loadstore(args.save, load=False)

                    old_dt = datetime.datetime.now()
                i += 1
        except:
            print("FATAL error in SoftQTrainer")
            traceback.print_exc()



class IQNTrainer():
    def __init__(self, state_shape, num_actions, args, input_queues, expert_queues=None, copy_queues=None):
        self.copy_queues = copy_queues
        self.p = Process(target=IQNTrainer.train_worker, args=(state_shape, num_actions, args, input_queues, expert_queues, self.copy_queues))
        self.p.start()


    @staticmethod
    def train_worker(state_shape, num_actions, args, input_queues, expert_queues, copy_queues):
        try:
            iqn = IQN(state_shape, num_actions, args)
            if args.load is not None:
                iqn.loadstore(args.load)

            old_dt = datetime.datetime.now()
            old_dt2 = datetime.datetime.now()

            i = 0
            while True:
                queue = random.sample(input_queues, 1)[0]
                input = queue.get()
                if expert_queues is not None:
                    queue = random.sample(expert_queues, 1)[0]
                    expert_input = queue.get()
                else:
                    expert_input = None
                output = iqn.train(input, expert_input)
                for k in input:
                    del input[k]
                    input[k] = None
                del input
                if expert_input is not None:
                    for k in expert_input:
                        del expert_input[k]
                        expert_input[k] = None
                    del expert_input

                if i % 100 == 0:
                    loss = output[0]
                    print("loss",loss)
                    print("q_value",output[1])
                    print("loss_actor",output[2])
                    print("entropy",output[3])
                    print("max_probs",output[4])
                    print("accuracy",output[5])

                if copy_queues is not None and (datetime.datetime.now() - old_dt2).total_seconds() > 60.0:
                    state_dict = iqn._models[0]['modules'].state_dict()
                    for param_name in state_dict:
                        state_dict[param_name] = state_dict[param_name].clone().cpu().share_memory_()

                    for copy_queue in copy_queues:
                        if copy_queue.full():
                            try:
                                copy_queue.get_nowait()
                            except:
                                pass
                        copy_queue.put(state_dict)

                    old_dt2 = datetime.datetime.now()

                if (datetime.datetime.now() - old_dt).total_seconds() > 5*60.0:
                    if args.save is not None:
                        print("")
                        print("")
                        print("Save iqn")
                        print("")
                        print("")
                        iqn.loadstore(args.save, load=False)

                    old_dt = datetime.datetime.now()
                i += 1
        except:
            print("FATAL error in IQNTrainer")
            traceback.print_exc()












class Trainer():
    def __init__(self, agent_name, agent_from, state_shape, num_actions, args, input_queues, copy_queues=None, add_args=None, blocking=True, prio_queues=None):
        self.copy_queues = copy_queues
        self.p = Process(target=Trainer.train_worker, args=(agent_name, agent_from, state_shape, num_actions, args, input_queues, self.copy_queues, add_args, blocking, prio_queues))
        self.p.start()


    @staticmethod
    def train_worker(agent_name, agent_from, state_shape, num_actions, args, input_queues, copy_queues, add_args, blocking, prio_queues):
        try:
            AgentClass = getattr(importlib.import_module(agent_from), agent_name)
            if add_args is None:
                agent = AgentClass(state_shape, num_actions, args)
            else:
                agent = AgentClass(state_shape, num_actions, args, *add_args)
            if args.load is not None:
                agent.loadstore(args.load)
            try:
                agent.build_target()
            except:
                pass

            old_dt = datetime.datetime.now()
            old_dt2 = datetime.datetime.now()

            i = 0
            while True:
                input = []
                if prio_queues is None:
                    if blocking:
                        for queues in input_queues:
                            queue = random.sample(queues, 1)[0]
                            input.append(queue.get())
                    else:
                        for queues in input_queues:
                            queue = random.sample(queues, 1)[0]
                            while queue.qsize() == 0:
                                queue = random.sample(queues, 1)[0]
                            input.append(queue.get())
                else:
                    queue_inds = []
                    if blocking:
                        for queues in input_queues:
                            queue_num = random.randint(0, len(queues)-1)
                            input.append(queues[queue_num].get())
                            queue_inds.append(queue_num)
                    else:
                        for queues in input_queues:
                            queue_num = random.randint(0, len(queues)-1)
                            while queues[queue_num].qsize() == 0:
                                queue_num = random.randint(0, len(queues)-1)
                            input.append(queues[queue_num].get())
                            queue_inds.append(queue_num)
                    input.append(queue_inds)


                if args.pretrain:
                    output = agent.pretrain(*input)
                else:
                    output = agent.train(*input)
                #update prio_queue
                if 'prio_upd' in output and prio_queues is not None:
                    prio_upd = output['prio_upd']
                    for upd in prio_upd:
                        replay_num = upd['replay_num']
                        worker_num = upd['worker_num']
                        package = {'tidxs': input[replay_num]['tidxs'],
                                   'error': upd['error'].share_memory_()}
                        prio_queues[replay_num][worker_num].put(package)
                for _input in input:
                    if isinstance(_input, dict):
                        for k in _input:
                            if k != 'tidxs':
                                del _input[k]
                                _input[k] = None
                #del input

                if i % 100 == 0:
                    for k in output:
                        if k != 'prio_upd':
                            print(k,output[k])

                if copy_queues is not None and (datetime.datetime.now() - old_dt2).total_seconds() > 60.0:
                    try:
                        state_dict = agent._models[0]['modules'].state_dict()
                    except:
                        state_dict = agent._model['modules'].state_dict()
                    for param_name in state_dict:
                        state_dict[param_name] = state_dict[param_name].clone().cpu().share_memory_()

                    for copy_queue in copy_queues:
                        if copy_queue.full():
                            try:
                                copy_queue.get_nowait()
                            except:
                                pass
                        copy_queue.put(state_dict)

                    old_dt2 = datetime.datetime.now()

                if (datetime.datetime.now() - old_dt).total_seconds() > 5*60.0:
                    if args.save is not None:
                        print("")
                        print("")
                        print("Save agent")
                        print("")
                        print("")
                        agent.loadstore(args.save, load=False)

                    old_dt = datetime.datetime.now()
                i += 1
        except:
            print("FATAL error in Trainer")
            traceback.print_exc()