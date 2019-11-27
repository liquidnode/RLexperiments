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
from traj_generator import TrajectoryGenerator, make_shared_buffer
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

GAMMA = 0.995


class SkipExpertGenerator(TrajectoryGenerator):
    def __init__(self, args, add_queues=0):
        assert args.minerl #expert trajectories only for minerl enviroment
        assert args.skip > 0
        super().__init__(args)
        for i in range(add_queues):
            self.traj_pipe.append(Queue(maxsize=10))
        
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
    def generator_process(args, traj_pipe, packet_size, copy_queue):
        try:
            data = minerl.data.make(args.env)
            trajs = data.get_trajectory_names()
            shuffle(trajs)

            #wrappers
            combine = env_wrappers.ICombineActionWrapper(data)
            if args.history_ar > 0:
                history_ar = env_wrappers.IHistoryActionReward(combine, args.history_ar)
                multistep = env_wrappers.IMultiStepWrapper(history_ar)
            else:
                multistep = env_wrappers.IMultiStepWrapper(combine)
            minerlobs = env_wrappers.IMineRLObsWrapper(multistep)

            obs = minerlobs.observation_space
            assert isinstance(obs, dict)
            skip = args.skip

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
                    obs_l.extend([obs_l[-1]]+[obs_l[-1] for i in range(1, skip+1)])
                    pov_l = np.concatenate([pov_l, np.expand_dims(pov_l[-1], 0)] + [np.expand_dims(pov_l[-1], 0) for i in range(1, skip+1)], axis=0)
                    action_l.extend([0 for i in range(1, skip+1)])
                    reward_l.extend([0.0 for i in range(1, skip+1)])
                    done_l[-1] = True
                    done_l.extend([True for i in range(1, skip+1)])
                obs_l = np.array(np.stack(obs_l, axis=0), dtype=np.float32)
                action_l = np.array(action_l, dtype=np.int32)
                reward_l = np.array(reward_l, dtype=np.float32)
                done_l = np.array(done_l, dtype=np.bool)
                done_l[-1] = True

                if skip > 0:
                    disc_rewards = np.copy(reward_l[:-skip])
                    for i in range(1, skip+1):
                        if i-skip < 0:
                            disc_rewards += GAMMA ** i * reward_l[i:(i-skip)]
                        else:
                            disc_rewards += GAMMA ** i * reward_l[i:]


                #slice and pack into shared memory buffers
                i = 0
                while i < action_l.shape[0] - skip:
                    current_packet_size = min(packet_size, action_l.shape[0] - skip - i)
                    pov_sh = make_shared_buffer(pov_l[i:(i+current_packet_size)])
                    obs_sh = make_shared_buffer(obs_l[i:(i+current_packet_size)])
                    if args.needs_next:
                        next_pov_sh = make_shared_buffer(pov_l[(i+1+skip):(i+current_packet_size+1+skip)])
                        next_obs_sh = make_shared_buffer(obs_l[(i+1+skip):(i+current_packet_size+1+skip)])
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
                    for pipe in traj_pipe:
                        TrajectoryGenerator.write_pipe(pipe, packet)
                    i += packet_size
                

                traj_index += 1
                traj_index = traj_index % len(trajs)

                
            TrajectoryGenerator.write_pipe(traj_pipe, "finish")
        except:
            print("FATAL error in ExpertGenerator")
            traceback.print_exc()


class SkipActorTrajGenerator(TrajectoryGenerator):
    def __init__(self, args, copy_queue=None, add_queues=0):
        super().__init__(args, copy_queue)
        for i in range(add_queues):
            self.traj_pipe.append(Queue(maxsize=10))

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
    def generator_process(args, traj_pipe, packet_size, copy_queue):
        try:
            env = gym.make(args.env)

            if args.minerl:
                env = env_wrappers.ResetTrimInfoWrapper(env)
                combine = env_wrappers.CombineActionWrapper(env)
                if args.history_ar > 0:
                    history_ar = env_wrappers.HistoryActionReward(combine, args.history_ar)
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

            skip = args.skip
            batch_index = 0
            batch = {}
            for k in state_shape:
                if k == 'pov':
                    batch[k] = np.zeros([packet_size+skip]+list(state_shape[k]), dtype=np.uint8)
                    batch['next_'+k] = np.zeros([packet_size+skip]+list(state_shape[k]), dtype=np.uint8)
                if k == 'state':
                    batch[k] = np.zeros([packet_size+skip]+list(state_shape[k]), dtype=np.float32)
                    batch['next_'+k] = np.zeros([packet_size+skip]+list(state_shape[k]), dtype=np.float32)
            batch['action'] = np.zeros([packet_size+skip], dtype=np.int32)
            batch['reward'] = np.zeros([packet_size+skip], dtype=np.float32)
            batch['done'] = np.zeros([packet_size+skip], dtype=np.bool)

                


            r_cumulative_reward = None

            num_episode = 0
            if args.name is not None or args.name != '':
                f = open('out-' + args.name, 'w')
            for ie in range(args.episodes):
                #start new episode
                done = False
                cumulative_reward = 0.0
                env_state = env.reset()

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

                    if batch_index == packet_size + skip - 1:
                        #discount reward
                        if np.sum(batch['done']) == 0:
                            disc_reward = np.copy(batch['reward'][:packet_size])
                            for i in range(1,skip+1):
                                disc_reward += GAMMA**i * batch['reward'][i:(packet_size+i)]
                            batch['reward'][:packet_size] = disc_reward
                        else:
                            disc_reward = np.copy(batch['reward'][:packet_size])
                            for j in range(packet_size):
                                for i in range(1,skip+1):
                                    if batch['done'][i+j-1]:
                                        break
                                    else:
                                        disc_reward[j] +=  GAMMA**i * batch['reward'][i+j]
                            #discount done
                            for i in range(1,skip+1):
                                batch['done'][:packet_size] = np.logical_or(batch['done'][:packet_size], batch['done'][i:(packet_size+i)])
                        #send packet to replay buffer
                        r_batch = {}
                        for b in batch:
                            if 'next' in b:
                                r_batch[b] = SharedMemory(batch[b][skip:(packet_size+skip)])
                                r_batch[b] = r_batch[b].shared_memory()
                            else:
                                r_batch[b] = SharedMemory(batch[b][:packet_size])
                                r_batch[b] = r_batch[b].shared_memory()
                        for pipe in traj_pipe:
                            TrajectoryGenerator.write_pipe(pipe, r_batch)
                        #shift batch
                        if skip > 0:
                            for b in batch:
                                batch[b][:skip] = batch[b][-skip:]
                        batch_index = skip
                    else:
                        batch_index += 1

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
