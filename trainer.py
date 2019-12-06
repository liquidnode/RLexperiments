from shared_memory import SharedMemory
import numpy as np
from torch.multiprocessing import Process, Queue
import datetime
import torch
import random
import traceback
import importlib
import time

#the trainer process reads a stream of batched data and trains an agent
#it also produces update data for proesses that depend on an up-to-date agent (like ActorTrajGenerator) 
class Trainer():
    def __init__(self, agent_name, agent_from, state_shape, num_actions, args, input_queues, copy_queues=None, add_args=None, blocking=True, prio_queues=None, no_process=False):
        self.copy_queues = copy_queues
        if not no_process:
            self.p = Process(target=Trainer.train_worker, args=(agent_name, agent_from, state_shape, num_actions, args, input_queues, self.copy_queues, add_args, blocking, prio_queues))
            self.p.start()
        else:
            self.p = lambda: Trainer.train_worker(agent_name, agent_from, state_shape, num_actions, args, input_queues, self.copy_queues, add_args, blocking, prio_queues)


    @staticmethod
    def train_worker(agent_name, agent_from, state_shape, num_actions, args, input_queues, copy_queues, add_args, blocking, prio_queues):
        try:
            AgentClass = getattr(importlib.import_module(agent_from), agent_name)
            if add_args is None:
                agent = AgentClass(state_shape, num_actions, args)
            else:
                agent = AgentClass(state_shape, num_actions, args, *add_args)
            try:
                agent.build_target()
            except:
                pass
            if args.load is not None:
                agent.loadstore(args.load)

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