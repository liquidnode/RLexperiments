from torch.multiprocessing import Process, Queue
import torch.multiprocessing
import zarr
import numpy as np
from shared_memory import SharedMemory
import copy
from traj_generator import make_shared_buffer

class FutureRewardTimeLeftModifier():
    def __init__(self, data_description, input_queue, just_density=False):
        self.data_description = copy.deepcopy(data_description)
        if not just_density:
            self.data_description['future_reward'] = {'compression': False, 'shape': [], 'dtype': np.float32}
            self.data_description['time_left'] = {'compression': False, 'shape': [], 'dtype': np.int32}
        else:
            self.data_description['reward_density'] = {'compression': False, 'shape': [], 'dtype': np.float32}
        self.traj_pipe = [Queue(maxsize=10)]

        params = (input_queue, self.traj_pipe[0], just_density)

        #start process
        self.proc_pool = Process(target=FutureRewardTimeLeftModifier.worker, args=params)
        self.proc_pool.start()
        
    @staticmethod
    def worker(input_queue, output_queue, just_density):
        current_traj = []
        buffered_trajs = []
        data_shs = None
        while True:
            if len(buffered_trajs) < 2:
                try:
                    data_shs = input_queue.get_nowait()
                except:
                    pass

            if data_shs is not None:
                current_traj.append(data_shs)
                #check done
                done_sh = SharedMemory(data_shs['done'])
                done_np = done_sh.numpy()
                if np.any(done_np):
                    #calculate future rewards and time left for every state in current trajectory
                    rewards = []
                    buffer_sizes = []
                    for buffer in current_traj:
                        buffer_reward = SharedMemory(data_shs['reward']).numpy()
                        rewards.append(buffer_reward)
                        buffer_sizes.append(buffer_reward.shape[0])
                    rewards = np.concatenate(rewards, axis=0)
                    if not just_density:
                        future_rewards = np.cumsum(rewards, axis=0)
                        time_left = np.arange(rewards.shape[0]-1, -1, -1, dtype = np.int32)
                    else:
                        reward_density = np.ones_like(rewards)*np.sum(rewards)/max(1.0, rewards.shape[0])
                    #redistribute to buffers
                    i = 0
                    for j, buffer in enumerate(current_traj):
                        if not just_density:
                            buffer['future_reward'] = make_shared_buffer(future_rewards[i:(i+buffer_sizes[j])])
                            buffer['time_left'] = make_shared_buffer(time_left[i:(i+buffer_sizes[j])])
                        else:
                            buffer['reward_density'] = make_shared_buffer(reward_density[i:(i+buffer_sizes[j])])
                        i += buffer_sizes[j]
                    #push to buffered_trajs
                    buffered_trajs.append(current_traj)
                    current_traj = []
                data_shs = None

            if len(buffered_trajs) > 0:
                #send buffered trajectories
                if len(buffered_trajs[0]) > 0:
                    output_queue.put(buffered_trajs[0][0])
                    buffered_trajs[0].pop(0)
                else:
                    buffered_trajs.pop(0)