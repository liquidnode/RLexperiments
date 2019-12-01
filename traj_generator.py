import numpy as np
import gym
import traceback
from random import shuffle
import torch 
import importlib
import threading

import torch.multiprocessing as mp
from torch.multiprocessing import Pipe, Process, Queue
from shared_memory import SharedMemory

def make_shared_buffer(data):
    data_sh = SharedMemory(data)
    return data_sh.shared_memory()

#template class for trajectory generators
class TrajectoryGenerator():
    def __init__(self, args, copy_queue=None):
        self.args = args

        self.traj_pipe = [Queue(maxsize=10)]
        self.comm_pipe = Queue(maxsize=5)
        if args.needs_skip:
            self.traj_pipe.append(Queue(maxsize=10))
        packet_size = args.packet_size
        p = Process(target=self.generator_process, args=(args, self.traj_pipe, packet_size, copy_queue, self.comm_pipe))
        p.start()

        #wait for state and action shape
        self.state_shape = TrajectoryGenerator.read_pipe(self.traj_pipe[0])
        self.num_actions = TrajectoryGenerator.read_pipe(self.traj_pipe[0])
        print('Number of primitive actions:', self.num_actions)
        print('State shape:', self.state_shape)
        
    @staticmethod
    def read_pipe(traj_pipe):
        return traj_pipe.get()
    
    @staticmethod
    def write_pipe(traj_pipe, data):
        return traj_pipe.put(data)

    def data_description(self):
        raise Exception("Not implemented!")

    @staticmethod
    def generator_process(args, traj_pipe, copy_queue, comm_pipe):
        try:
            raise Exception("Not implemented!")
        except:
            print("FATAL error in TrajectoryGenerator")
            traceback.print_exc()