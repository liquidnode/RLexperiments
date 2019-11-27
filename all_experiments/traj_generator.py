import numpy as np
import gym
import traceback
from random import shuffle
import env_wrappers
import torch 
import importlib
import threading

import torch.multiprocessing as mp
from torch.multiprocessing import Pipe, Process, Queue
from shared_memory2 import SharedMemory
from trainers import Actor
import models
import importlib
import copy

# Import additional environments if available
import gym_envs
import gym_envs.contwrapper

try:
    import gym_miniworld
except:
    pass

try:
    import minerl
    import expert_traj
except:
    pass

def make_shared_buffer(data):
    #data_packet = data.tobytes()
    #data_sh = SharedMemory(create=True, size=len(data_packet))
    #data_sh.buf = data_packet
    #return data_sh.name
    data_sh = SharedMemory(data)
    return data_sh.shared_memory()

class TrajectoryGenerator():
    def __init__(self, args, copy_queue=None):
        self.args = args

        #self.traj_pipe = Pipe(False)
        self.traj_pipe = [Queue(maxsize=10)]
        self.comm_pipe = Queue(maxsize=5)
        if args.needs_skip:
            self.traj_pipe.append(Queue(maxsize=10))
        packet_size = args.packet_size
        p = Process(target=self.generator_process, args=(args, self.traj_pipe, packet_size, copy_queue, self.comm_pipe))
        p.start()

        #wait for state and action shape
        self.state_shape = TrajectoryGenerator.read_pipe(self.traj_pipe[0])#self.traj_pipe_read.recv()
        self.num_actions = TrajectoryGenerator.read_pipe(self.traj_pipe[0])#self.traj_pipe_read.recv()
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


class ExpertGenerator(TrajectoryGenerator):
    def __init__(self, args):
        assert args.minerl #expert trajectories only for minerl enviroment
        if args.needs_skip:
            assert args.skip > 0
        super().__init__(args)
        
    def data_description(self):
        if self.args.needs_next:
            return {'pov': {'compression': True, 'shape': list(self.state_shape['pov']), 'dtype': np.uint8, 'has_next': True},
                    'state': {'compression': False, 'shape': list(self.state_shape['state']), 'dtype': np.float32, 'has_next': True},
                    'action': {'compression': False, 'shape': [], 'dtype': np.int32},
                    'reward': {'compression': False, 'shape': [], 'dtype': np.float32},
                    'done': {'compression': False, 'shape': [], 'dtype': np.bool}}
        else:
            return {'pov': {'compression': True, 'shape': list(self.state_shape['pov']), 'dtype': np.uint8},
                    'state': {'compression': False, 'shape': list(self.state_shape['state']), 'dtype': np.float32},
                    'action': {'compression': False, 'shape': [], 'dtype': np.int32},
                    'reward': {'compression': False, 'shape': [], 'dtype': np.float32},
                    'done': {'compression': False, 'shape': [], 'dtype': np.bool}}

    @staticmethod
    def generator_process(args, traj_pipe, packet_size, copy_queue, comm_pipe):
        try:
            data = minerl.data.make(args.env)
            trajs = data.get_trajectory_names()
            shuffle(trajs)

            #wrappers
            combine = env_wrappers.ICombineActionWrapper(data)
            if args.history_ar > 0:
                history_ar = env_wrappers.IHistoryActionReward(combine, args.history_ar, args)
                multistep = env_wrappers.IMultiStepWrapper(history_ar)
            else:
                multistep = env_wrappers.IMultiStepWrapper(combine)
            minerlobs = env_wrappers.IMineRLObsWrapper(multistep)

            obs = minerlobs.observation_space
            assert isinstance(obs, dict)

            #convert to state_shape
            state_shape = {}
            for k in obs:
                state_shape[k] = obs[k].shape
            #traj_pipe.send(state_shape)
            TrajectoryGenerator.write_pipe(traj_pipe[0], state_shape)

            #get num actions
            aspace = minerlobs.action_space

            if isinstance(aspace, gym.spaces.Tuple):
                aspace = aspace.spaces
            else:
                aspace = [aspace]               # Ensure that the action space is a list for all the environments

            if isinstance(aspace[0], gym.spaces.Discrete):
                # Discrete actions
                num_actions = int(np.prod([a.n for a in aspace]))
            elif isinstance(aspace[0], gym.spaces.MultiBinary):
                # Retro actions are binary vectors of pressed buttons. Quick HACK,
                # only press one button at a time
                num_actions = int(np.prod([a.n for a in aspace]))

            #traj_pipe.send(num_actions)
            TrajectoryGenerator.write_pipe(traj_pipe[0], num_actions)

            traj_index = 0
            #start generator
            while True:
                current_traj = trajs[traj_index]
                
                data_frames = list(data.load_data(current_traj, include_metadata=False))

                #convert actions and add current action to obs
                #combine actions
                for i in range(len(data_frames)):
                    data_frames[i][1] = combine.reverse_action(data_frames[i][1])
                    if args.history_ar > 0:
                        if i == 0:
                            data_frames[0][0] = history_ar._reset(data_frames[0][0])
                        else:
                            data_frames[i][0] = history_ar._step(data_frames[i][0], data_frames[i-1][2], data_frames[i-1][1])
                    

                if args.needs_skip:
                    skip = args.skip
                    pov_l = np.zeros((len(data_frames), 64, 64, 3), dtype=np.uint8)
                    obs_l = []
                    reward_l = np.zeros((len(data_frames),), dtype=np.float32)
                    rdone_l = np.zeros((len(data_frames),), dtype=np.bool)
                    #convert obs
                    for i in range(len(data_frames)):
                        obs = minerlobs.observation(multistep._add_current_to_obs(copy.deepcopy(data_frames[i][0]), multistep.rnoop))
                        pov_l[i] = obs['pov']
                        obs_l.append(obs['state'])
                        reward_l[i] = data_frames[i][2]
                        rdone_l[i] = data_frames[i][4]

                    #convert to numpy arrays
                    if args.needs_next:
                        obs_l.append(obs_l[-1])
                        pov_l = np.concatenate([pov_l, np.expand_dims(pov_l[-1], 0)], axis=0)
                    obs_l = np.array(np.stack(obs_l, axis=0), dtype=np.float32)

                    #make skip and skip next lists
                    rreward = np.copy(reward_l[:(-skip+1)])
                    for i in range(1,skip-1):
                        rreward += (models.tGAMMA**i)*reward_l[i:(i-skip+1)]
                    rreward = np.array(rreward, dtype=np.float32)
                    robs = obs_l[:-skip]
                    robs_next = obs_l[skip:]
                    rpov = pov_l[:-skip]
                    rpov_next = pov_l[skip:]
                    rdone_l = rdone_l[(skip-1):]

                    #slice and pack into shared memory buffers
                    #def packet_writer():
                    i = 0
                    while i < len(rreward):
                        current_packet_size = min(packet_size, len(rreward) - i)
                        pov_sh = make_shared_buffer(rpov[i:(i+current_packet_size)])
                        obs_sh = make_shared_buffer(robs[i:(i+current_packet_size)])
                        rew_sh = make_shared_buffer(rreward[i:(i+current_packet_size)])
                        don_sh = make_shared_buffer(rdone_l[i:(i+current_packet_size)])
                        if args.needs_next:
                            next_pov_sh = make_shared_buffer(rpov_next[i:(i+current_packet_size)])
                            next_obs_sh = make_shared_buffer(robs_next[i:(i+current_packet_size)])
                    
                        #send packet
                        packet = {'pov': pov_sh,
                                    'state': obs_sh,
                                    'reward': rew_sh,
                                    'done': don_sh}
                        if args.needs_next:
                            packet['next_pov'] = next_pov_sh
                            packet['next_state'] = next_obs_sh
                        TrajectoryGenerator.write_pipe(traj_pipe[1], packet)
                        i += packet_size

                    #th = threading.Thread(target=packet_writer)
                    #th.start()


                #convert to multistep actions
                obs_l = []
                action_l = []
                reward_l = []
                done_l = []
                obs_l.append(multistep.reverse_reset(data_frames[0][0]))
                for i in range(len(data_frames)):
                    next_obs = None
                    if i != len(data_frames) - 1:
                        next_obs = data_frames[i+1][0]
                    else:
                        if not data_frames[i][4]:
                            #print('The end of an episode should have set done = True')
                            pass
                    robs_l, raction_l, rreward_l, rdone_l, _ = multistep.reverse_step(next_obs, data_frames[i][1], data_frames[i][2], data_frames[i][4])
                    obs_l.extend(robs_l)
                    action_l.extend(raction_l)
                    reward_l.extend(rreward_l)
                    done_l.extend(rdone_l)

                pov_l = np.zeros((len(obs_l), 64, 64, 3), dtype=np.uint8)
                #convert obs
                for i in range(len(obs_l)):
                    obs_l[i] = minerlobs.observation(obs_l[i])
                    pov_l[i] = obs_l[i]['pov']
                    obs_l[i] = obs_l[i]['state']

                #convert to numpy arrays
                if args.needs_next:
                    obs_l.append(obs_l[-1])
                    pov_l = np.concatenate([pov_l, np.expand_dims(pov_l[-1], 0)], axis=0)
                obs_l = np.array(np.stack(obs_l, axis=0), dtype=np.float32)
                action_l = np.array(action_l, dtype=np.int32)
                reward_l = np.array(reward_l, dtype=np.float32)
                done_l = np.array(done_l, dtype=np.bool)
                done_l[-1] = True

                #slice and pack into shared memory buffers
                i = 0
                while i < action_l.shape[0]:
                    current_packet_size = min(packet_size, action_l.shape[0] - i)
                    pov_sh = make_shared_buffer(pov_l[i:(i+current_packet_size)])
                    obs_sh = make_shared_buffer(obs_l[i:(i+current_packet_size)])
                    if args.needs_next:
                        next_pov_sh = make_shared_buffer(pov_l[(i+1):(i+current_packet_size+1)])
                        next_obs_sh = make_shared_buffer(obs_l[(i+1):(i+current_packet_size+1)])
                    action_sh = make_shared_buffer(action_l[i:(i+current_packet_size)])
                    reward_sh = make_shared_buffer(reward_l[i:(i+current_packet_size)])
                    done_sh = make_shared_buffer(done_l[i:(i+current_packet_size)])
                    
                    #send packet
                    packet = {'pov': pov_sh,
                              'state': obs_sh,
                              'action': action_sh,
                              'reward': reward_sh,
                              'done': done_sh}
                    if args.needs_next:
                        packet['next_pov'] = next_pov_sh
                        packet['next_state'] = next_obs_sh
                    TrajectoryGenerator.write_pipe(traj_pipe[0], packet)
                    i += packet_size
                
                #if args.needs_skip:
                #    th.join()

                traj_index += 1
                traj_index = traj_index % len(trajs)

                
            TrajectoryGenerator.write_pipe(traj_pipe, "finish")
        except:
            print("FATAL error in ExpertGenerator")
            traceback.print_exc()


class ActorTrajGenerator(TrajectoryGenerator):
    def __init__(self, args, copy_queue=None):
        super().__init__(args, copy_queue)

    def data_description(self):
        desc = {'action': {'compression': False, 'shape': [], 'dtype': np.int32},
                'reward': {'compression': False, 'shape': [], 'dtype': np.float32},
                'done': {'compression': False, 'shape': [], 'dtype': np.bool}}
        for k in self.state_shape:
            if k == 'pov':
                desc[k] = {'compression': True, 'shape': list(self.state_shape[k]), 'dtype': np.uint8, 'has_next': True}
            if k == 'state':
                desc[k] = {'compression': False, 'shape': list(self.state_shape[k]), 'dtype': np.float32, 'has_next': True}
        return desc

    @staticmethod
    def generator_process(args, traj_pipe, packet_size, copy_queue, comm_pipe):
        try:
            env = gym.make(args.env)

            if args.minerl:
                env = env_wrappers.ResetTrimInfoWrapper(env)
                combine = env_wrappers.CombineActionWrapper(env)
                if args.history_ar > 0:
                    history_ar = env_wrappers.HistoryActionReward(combine, args.history_ar, args)
                    multistep = env_wrappers.MultiStepWrapper(history_ar)
                else:
                    multistep = env_wrappers.MultiStepWrapper(combine)
                minerlobs = env_wrappers.MineRLObsWrapper(multistep)
                env = minerlobs
            else:
                env = env_wrappers.DictObsWrapper(env)

            obs = env.observation_space
            assert isinstance(obs, dict)

            #convert to state_shape
            state_shape = {}
            for k in obs:
                state_shape[k] = obs[k].shape

            TrajectoryGenerator.write_pipe(traj_pipe[0], state_shape)

            #get num actions
            aspace = env.action_space

            if isinstance(aspace, gym.spaces.Tuple):
                aspace = aspace.spaces
            else:
                aspace = [aspace]               # Ensure that the action space is a list for all the environments

            if isinstance(aspace[0], gym.spaces.Discrete):
                # Discrete actions
                num_actions = int(np.prod([a.n for a in aspace]))
            elif isinstance(aspace[0], gym.spaces.MultiBinary):
                # Retro actions are binary vectors of pressed buttons. Quick HACK,
                # only press one button at a time
                num_actions = int(np.prod([a.n for a in aspace]))

            TrajectoryGenerator.write_pipe(traj_pipe[0], num_actions)


            #load actor
            AgentClass = getattr(importlib.import_module(args.agent_from), args.agent_name)
            actor = AgentClass(state_shape, num_actions, args)
            #if args.iqn:
            #    actor = models.IQN(state_shape, num_actions, args)
            #else:
            #    if args.softq:
            #        actor = models.SoftQ(state_shape, num_actions, args)
            #    else:
            #        actor = Actor(state_shape, num_actions, args)
            if args.load is not None:
                actor.loadstore(args.load)

            batch_index = 0
            batch = {}
            for k in state_shape:
                if k == 'pov':
                    batch[k] = np.zeros([packet_size]+list(state_shape[k]), dtype=np.uint8)
                    batch['next_'+k] = np.zeros([packet_size]+list(state_shape[k]), dtype=np.uint8)
                if k == 'state':
                    batch[k] = np.zeros([packet_size]+list(state_shape[k]), dtype=np.float32)
                    batch['next_'+k] = np.zeros([packet_size]+list(state_shape[k]), dtype=np.float32)
            batch['action'] = np.zeros([packet_size], dtype=np.int32)
            batch['reward'] = np.zeros([packet_size], dtype=np.float32)
            batch['done'] = np.zeros([packet_size], dtype=np.bool)

            
            if args.needs_skip:
                skip = args.skip
                skip_batch_index = 0
                skip_batch = {}
                skip_delay_line = {}
                skip_delay_index = 0
                for k in state_shape:
                    if k == 'pov':
                        skip_batch[k] = np.zeros([packet_size]+list(state_shape[k]), dtype=np.uint8)
                        skip_batch['next_'+k] = np.zeros([packet_size]+list(state_shape[k]), dtype=np.uint8)
                        skip_delay_line[k] = np.zeros([skip]+list(state_shape[k]), dtype=np.uint8)
                    if k == 'state':
                        skip_batch[k] = np.zeros([packet_size]+list(state_shape[k]), dtype=np.float32)
                        skip_batch['next_'+k] = np.zeros([packet_size]+list(state_shape[k]), dtype=np.float32)
                        skip_delay_line[k] = np.zeros([skip]+list(state_shape[k]), dtype=np.float32)
                    skip_batch['reward'] = np.zeros([packet_size], dtype=np.float32)
                    skip_batch['action'] = np.zeros([packet_size], dtype=np.int32)
                    skip_delay_line['action'] = np.zeros([skip], dtype=np.int32)
                    skip_delay_line['reward'] = np.zeros([skip], dtype=np.float32)
                    skip_batch['done'] = np.zeros([packet_size], dtype=np.bool)
                    skip_delay_line['valid'] = np.zeros([skip], dtype=np.bool)

                


            r_cumulative_reward = None

            num_episode = 0
            if args.name is not None or args.name != '':
                f = open('out-' + args.name, 'w')
            for ie in range(args.episodes):
                #start new episode
                done = False
                cumulative_reward = 0.0
                env_state = env.reset()
                
                if args.needs_skip:
                    #reset current delay line
                    skip_delay_index = 0
                    for k in state_shape:
                        if k == 'pov':
                            skip_delay_line[k] = np.zeros([skip]+list(state_shape[k]), dtype=np.uint8)
                        if k == 'state':
                            skip_delay_line[k] = np.zeros([skip]+list(state_shape[k]), dtype=np.float32)
                    skip_delay_line['reward'] = np.zeros([skip], dtype=np.float32)
                    skip_delay_line['action'] = np.zeros([skip], dtype=np.int32)
                    skip_delay_line['valid'] = np.zeros([skip], dtype=np.bool)
                    last_not_virtual_step = True

                ii = 0
                while (not done):
                    if copy_queue is not None and ii % 10 == 0:
                        if not copy_queue.empty():
                            try:
                                new_state_dict = copy_queue.get_nowait()
                            except:
                                pass
                            else:
                                if torch.cuda.is_available() and models.CUDA:
                                    for p in new_state_dict:
                                        new_state_dict[p] = new_state_dict[p].cuda()
                                actor.load_state_dict(new_state_dict)
                                print("copied actor")

                    for k in env_state:
                        batch[k][batch_index] = env_state[k]
                    action, entropy = actor.select_action(env_state)
                    batch['action'][batch_index] = action
                                
                    if args.needs_skip and last_not_virtual_step:
                        #add state to delay line
                        for k in env_state:
                            skip_delay_line[k][skip_delay_index] = env_state[k]
                        skip_delay_line['action'][skip_delay_index] = action
                        skip_delay_line['valid'][skip_delay_index] = True
                        skip_delay_index = (skip_delay_index + 1) % skip

                    env_state, reward, done, info = env.step(action)
                    batch['reward'][batch_index] = reward# + 0.01 * entropy
                    batch['done'][batch_index] = done
                    for k in env_state:
                        batch['next_'+k][batch_index] = env_state[k]

                    last_not_virtual_step = False
                    if isinstance(info, dict) and 'virtual_step' in info:
                        last_not_virtual_step = not info['virtual_step']
                    else:
                        last_not_virtual_step = True
                    if args.needs_skip and last_not_virtual_step:
                        #add to batch
                        index = (skip_delay_index - 1) % skip
                        if skip_delay_line['valid'][index]:
                            skip_delay_line['reward'][index] = reward
                        if skip_delay_line['valid'][skip_delay_index]:
                            #write new entry in batch
                            for k in env_state:
                                skip_batch[k][skip_batch_index] = np.copy(skip_delay_line[k][skip_delay_index])
                                skip_batch['next_'+k][skip_batch_index] = env_state[k]
                            #calc discounted reward
                            disc_reward = 0.0
                            for i in range(0, skip):
                                if skip_delay_line['valid'][(skip_delay_index+i)%skip]:
                                    disc_reward += (models.tGAMMA**i)*skip_delay_line['reward'][(skip_delay_index+i)%skip]
                            skip_batch['reward'][skip_batch_index] = disc_reward
                            skip_batch['action'][skip_batch_index] = skip_delay_line['action'][skip_delay_index]
                            skip_batch['done'][skip_batch_index] = done
                            if skip_batch_index == packet_size - 1:
                                #send packet to replay buffer
                                r_batch = {}
                                for b in skip_batch:
                                    r_batch[b] = SharedMemory(skip_batch[b])
                                    r_batch[b] = r_batch[b].shared_memory()
                                TrajectoryGenerator.write_pipe(traj_pipe[1], r_batch)
                            skip_batch_index = (skip_batch_index + 1) % packet_size

                    if batch_index == packet_size - 1:
                        #send packet to replay buffer
                        r_batch = {}
                        for b in batch:
                            r_batch[b] = SharedMemory(batch[b])
                            r_batch[b] = r_batch[b].shared_memory()
                        TrajectoryGenerator.write_pipe(traj_pipe[0], r_batch)

                    batch_index = (batch_index + 1) % packet_size

                    cumulative_reward += reward
                    ii += 1

                    if ii % 400 == 0:
                        if isinstance(entropy, dict):
                            for k in entropy:
                                print(k,entropy[k])
                        else:
                            print("actor entropy", entropy)

                if r_cumulative_reward is None:
                    r_cumulative_reward = cumulative_reward
                else:
                    r_cumulative_reward = 0.95* r_cumulative_reward + 0.05 * cumulative_reward
                
                print("cumulative_reward",cumulative_reward,"/", r_cumulative_reward,num_episode)
                num_episode += 1
                if args.name is not None or args.name != '':
                    print(cumulative_reward, r_cumulative_reward,num_episode,file=f)

            if args.name is not None or args.name != '':
                f.close()
        except:
            print("FATAL error in ActorTrajGenerator")
            traceback.print_exc()



class RealExpertGenerator(TrajectoryGenerator):
    def __init__(self, args):
        assert args.minerl #expert trajectories only for minerl enviroment
        super().__init__(args)
        
    def data_description(self):
        if self.args.needs_next:
            return {'real_pov': {'compression': True, 'shape': list(self.state_shape['pov']), 'dtype': np.uint8, 'has_next': True},
                    'real_state': {'compression': False, 'shape': list(self.state_shape['state']), 'dtype': np.float32, 'has_next': True},
                    #'action': {'compression': False, 'shape': [], 'dtype': np.int32},
                    'real_reward': {'compression': False, 'shape': [], 'dtype': np.float32},
                    'real_done': {'compression': False, 'shape': [], 'dtype': np.bool}}
        else:
            return {'real_pov': {'compression': True, 'shape': list(self.state_shape['pov']), 'dtype': np.uint8},
                    'real_state': {'compression': False, 'shape': list(self.state_shape['state']), 'dtype': np.float32},
                    #'action': {'compression': False, 'shape': [], 'dtype': np.int32},
                    'real_reward': {'compression': False, 'shape': [], 'dtype': np.float32},
                    'real_done': {'compression': False, 'shape': [], 'dtype': np.bool}}

    @staticmethod
    def generator_process(args, traj_pipe, packet_size, copy_queue):
        try:
            data = minerl.data.make(args.env)
            trajs = data.get_trajectory_names()
            shuffle(trajs)
            skip = args.skip

            #wrappers
            combine = env_wrappers.ICombineActionWrapper(data)
            if args.history_ar > 0:
                history_ar = env_wrappers.IHistoryActionReward(combine, args.history_ar)
                minerlobs = env_wrappers.IMineRLObsWrapper(history_ar)
            else:
                minerlobs = env_wrappers.IMineRLObsWrapper(combine)

            obs = minerlobs.observation_space
            assert isinstance(obs, dict)

            #convert to state_shape
            state_shape = {}
            for k in obs:
                state_shape[k] = obs[k].shape
            #traj_pipe.send(state_shape)
            TrajectoryGenerator.write_pipe(traj_pipe[0], state_shape)

            #get num actions
            aspace = minerlobs.action_space

            if isinstance(aspace, gym.spaces.Tuple):
                aspace = aspace.spaces
            else:
                aspace = [aspace]               # Ensure that the action space is a list for all the environments

            if isinstance(aspace[0], gym.spaces.Discrete):
                # Discrete actions
                num_actions = int(np.prod([a.n for a in aspace]))
            elif isinstance(aspace[0], gym.spaces.MultiBinary):
                # Retro actions are binary vectors of pressed buttons. Quick HACK,
                # only press one button at a time
                num_actions = int(np.prod([a.n for a in aspace]))

            #traj_pipe.send(num_actions)
            TrajectoryGenerator.write_pipe(traj_pipe[0], 0)

            traj_index = 0
            #start generator
            while True:
                current_traj = trajs[traj_index]
                
                data_frames = list(data.load_data(current_traj, include_metadata=False))

                #convert actions and add current action to obs
                #combine actions
                for i in range(len(data_frames)):
                    data_frames[i][1] = combine.reverse_action(data_frames[i][1])
                    if args.history_ar > 0:
                        if i == 0:
                            data_frames[0][0] = history_ar._reset(data_frames[0][0])
                        else:
                            data_frames[i][0] = history_ar._step(data_frames[i][0], data_frames[i-1][2], data_frames[i-1][1])

                pov_l = np.zeros((len(data_frames), 64, 64, 3), dtype=np.uint8)
                obs_l = []
                reward_l = np.zeros((len(data_frames),), dtype=np.float32)
                done_l = np.zeros((len(data_frames),), dtype=np.bool)
                #convert obs
                for i in range(len(data_frames)):
                    obs = minerlobs.observation(data_frames[i][0])
                    pov_l[i] = obs['pov']
                    obs_l.append(obs['state'])
                    reward_l[i] = data_frames[i][2]
                    done_l[i] = data_frames[i][4]

                #convert to numpy arrays
                #if args.needs_next:
                #    obs_l.append(obs_l[-1])
                #    pov_l = np.concatenate([pov_l, np.expand_dims(pov_l[-1], 0)], axis=0)
                #obs_l = np.array(np.stack(obs_l, axis=0), dtype=np.float32)
                obs_l = np.array(np.stack(obs_l, axis=0), dtype=np.float32)

                #make skip and skip next lists
                rreward = np.copy(reward_l[:-skip])
                for i in range(1,skip):
                    rreward += (models.tGAMMA**i)*reward_l[i:(i-skip)]
                rreward = np.array(rreward, dtype=np.float32)
                robs = obs_l[:-skip]
                robs_next = obs_l[skip:]
                rpov = pov_l[:-skip]
                rpov_next = pov_l[skip:]
                done_l = done_l[skip:]

                #slice and pack into shared memory buffers
                i = 0
                while i < len(rreward):
                    current_packet_size = min(packet_size, len(rreward) - i)
                    pov_sh = make_shared_buffer(rpov[i:(i+current_packet_size)])
                    obs_sh = make_shared_buffer(robs[i:(i+current_packet_size)])
                    rew_sh = make_shared_buffer(rreward[i:(i+current_packet_size)])
                    don_sh = make_shared_buffer(done_l[i:(i+current_packet_size)])
                    if args.needs_next:
                        next_pov_sh = make_shared_buffer(rpov_next[i:(i+current_packet_size)])
                        next_obs_sh = make_shared_buffer(robs_next[i:(i+current_packet_size)])
                    
                    #send packet
                    packet = {'real_pov': pov_sh,
                              'real_state': obs_sh,
                              'real_reward': rew_sh,
                              'real_done': don_sh}
                    if args.needs_next:
                        packet['next_real_pov'] = next_pov_sh
                        packet['next_real_state'] = next_obs_sh
                    TrajectoryGenerator.write_pipe(traj_pipe[0], packet)
                    i += packet_size
                
                traj_index += 1
                traj_index = traj_index % len(trajs)

                
            TrajectoryGenerator.write_pipe(traj_pipe[0], "finish")
        except:
            print("FATAL error in ExpertGenerator")
            traceback.print_exc()



class EntropyPlot():
    def __init__(self, args, agent_name):
        self.args = args

        self.env = gym.make(args.env)

        if args.minerl:
            self.env = env_wrappers.ResetTrimInfoWrapper(self.env)
            combine = env_wrappers.CombineActionWrapper(self.env)
            if args.history_ar > 0:
                history_ar = env_wrappers.HistoryActionReward(combine, args.history_ar, args)
                multistep = env_wrappers.MultiStepWrapper(history_ar)
            else:
                multistep = env_wrappers.MultiStepWrapper(combine)
            minerlobs = env_wrappers.MineRLObsWrapper(multistep)
            self.env = minerlobs
        else:
            self.env = env_wrappers.DictObsWrapper(self.env)

        obs = self.env.observation_space
        assert isinstance(obs, dict)

        #convert to state_shape
        state_shape = {}
        for k in obs:
            state_shape[k] = obs[k].shape

        #get num actions
        aspace = self.env.action_space

        if isinstance(aspace, gym.spaces.Tuple):
            aspace = aspace.spaces
        else:
            aspace = [aspace]               # Ensure that the action space is a list for all the environments

        if isinstance(aspace[0], gym.spaces.Discrete):
            # Discrete actions
            num_actions = int(np.prod([a.n for a in aspace]))
        elif isinstance(aspace[0], gym.spaces.MultiBinary):
            # Retro actions are binary vectors of pressed buttons. Quick HACK,
            # only press one button at a time
            num_actions = int(np.prod([a.n for a in aspace]))

        
        AgentClass = getattr(importlib.import_module("models"), agent_name)
        self.actor = AgentClass(state_shape, num_actions, args)
        if args.load is not None:
            self.actor.loadstore(args.load)

    def run_episode(self):
        #start new episode
        done = False
        cumulative_reward = 0.0
        env_state = self.env.reset()
        i = 0
        entropyLst = []
        rewardLst = []
        while (not done):
            action, entropy = self.actor.select_action(env_state)

            env_state, reward, done, info = self.env.step(action)

            entropyLst.append(entropy)
            rewardLst.append(reward)
            cumulative_reward += reward
            i += 1

        print("c reward", cumulative_reward)
        return rewardLst, entropyLst




class ClusteringPlot():
    def __init__(self, args, agent_name):
        self.trajectory = ExpertGenerator(args)
        self.args = args

        AgentClass = getattr(importlib.import_module("models"), agent_name)
        self.actor = AgentClass(self.trajectory.state_shape, self.trajectory.num_actions, args)
        if args.load is not None:
            self.actor.loadstore(args.load)

    def get_features(self, ran = False):
        run = True
        features = []
        actions = []
        povs = []
        states = []
        num_samples = 0
        while run:
            input = self.trajectory.traj_pipe.get()
            if isinstance(input, str):
                if input == "finish":
                    run = False
                    break
                
            if not ran:
                features.append(self.actor.get_features(input))
                actions.append(input['action'])
                povs.append(input['pov'])
                states.append(input['state'])
                if len(features) > 30000 // self.args.packet_size:
                    break
            else:
                random_ind = torch.from_numpy(np.random.randint(0,input['action'].shape[0],size=8)).long()
                features.append(self.actor.get_features(input)[random_ind])
                actions.append(input['action'][random_ind])
                povs.append(input['pov'][random_ind])
                states.append(input['state'][random_ind])
                num_samples += 8
                if num_samples > 30000:
                    break
        return [torch.cat(features, dim=0),
                torch.cat(actions, dim=0),
                torch.cat(povs, dim=0),
                torch.cat(states, dim=0)]
                