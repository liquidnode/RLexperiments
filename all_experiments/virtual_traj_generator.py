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
import time
from collections import OrderedDict
from virtual_models import COMPATIBLE
import time
import own_data
import cv2
import json
from scipy.misc import imsave
import subprocess

# Import additional environments if available
try:
    import gym_envs
    import gym_envs.contwrapper
except:
    pass

try:
    import gym_miniworld
except:
    pass

try:
    import minerl
except:
    pass


OLD_RECORDING_SYSTEM = False

class VirtualExpertGenerator(TrajectoryGenerator):
    def __init__(self, args, add_queues=0, copy_queue=None, add_args=None):
        assert args.minerl #expert trajectories only for minerl enviroment
        super().__init__(args, copy_queue)
        TrajectoryGenerator.write_pipe(self.comm_pipe, add_args)
        if args.needs_embedding:
            self.add_desc = TrajectoryGenerator.read_pipe(self.traj_pipe[0])
        else:
            self.add_desc = {}
        for i in range(add_queues):
            self.traj_pipe.append(Queue(maxsize=10))
        
    def data_description(self):
        desc = {'reward': {'compression': False, 'shape': [], 'dtype': np.float32},
                'done': {'compression': False, 'shape': [], 'dtype': np.bool}}
        for o in self.state_shape:
            if o == 'pov':
                desc['pov'] = {'compression': True, 'shape': list(self.state_shape['pov']), 'dtype': np.uint8, 'dontsendpast': self.args.dont_send_past_pov}
            else:
                if len(self.state_shape[o]) == 0:
                    desc[o] = {'compression': False, 'shape': [], 'dtype': np.int32}
                else:
                    desc[o] = {'compression': False, 'shape': list(self.state_shape[o]), 'dtype': np.float32}
        for a in self.num_actions:
            if isinstance(self.num_actions[a], gym.spaces.Discrete):
                desc[a] = {'compression': False, 'shape': [], 'dtype': np.int32}
            else:
                desc[a] = {'compression': False, 'shape': list(self.num_actions[a].shape), 'dtype': np.float32}
        if self.args.needs_embedding:
            desc.update(self.add_desc)
        return desc

    @staticmethod
    def generator_process(args, traj_pipe, packet_size, copy_queue, comm_pipe):
        try:
            multi_env = False
            try:
                multi_env = args.needs_multi_env
            except:
                pass
            if not multi_env:
                data = minerl.data.make(args.env)
                trajs = data.get_trajectory_names()
                shuffle(trajs)
            else:
                #configure data from multiple envs
                all_envs = ['MineRLTreechop-v0',
                            'MineRLNavigateDense-v0',
                            'MineRLNavigate-v0',
                            'MineRLNavigateExtremeDense-v0',
                            'MineRLNavigateExtreme-v0',
                            'MineRLObtainIronPickaxe-v0',
                            'MineRLObtainIronPickaxeDense-v0',
                            'MineRLObtainDiamond-v0',
                            'MineRLObtainDiamondDense-v0']
                probs = [1.0,
                         0.0,
                         0.0,
                         0.0,
                         0.0,
                         1.0,
                         1.0,
                         1.0,
                         1.0]
                probs = np.array(probs)
                probs /= np.sum(probs)
                datas = [own_data.make(tenv, num_workers=0) for tenv in all_envs]
                env_ind = np.random.choice(len(all_envs), p=probs)#np.random.randint(0, len(all_envs))
                all_trajs = [datas[i].get_trajectory_names() for i in range(len(datas))]
                trajs = all_trajs[env_ind]
                shuffle(trajs)


            #wrappers
            if multi_env:
                multi_data = env_wrappers.MultiDataWrapper(datas)
                data = multi_data
            combine = env_wrappers.ICombineActionWrapper(data)
            if args.history_ar > 0:
                history_ar = env_wrappers.IHistoryActionReward(combine, args.history_ar, args)
                env = history_ar
            else:
                env = combine
            if args.needs_orientation:
                orientation = env_wrappers.IOrientation(env)
                env = orientation
            if not multi_env:
                minerlobs = env_wrappers.IMineRLObsWrapper2(env)
            else:
                minerlobs = env_wrappers.IMineRLObsWrapper3(env)
            env = minerlobs
            if args.needs_last_action:
                lastactionw = env_wrappers.ILastAction(minerlobs)
                
            if args.needs_last_action:
                obs = lastactionw.observation_space
            else:
                obs = minerlobs.observation_space
            assert isinstance(obs, dict)

            #convert to state_shape
            state_shape = {}
            for k in obs:
                if isinstance(obs[k], gym.spaces.Box):
                    state_shape[k] = list(obs[k].shape)
                else:
                    state_shape[k] = []



            #traj_pipe.send(state_shape)
            TrajectoryGenerator.write_pipe(traj_pipe[0], state_shape)

            #get num actions
            if not True:#COMPATIBLE:
                aspace = env.action_space.spaces
            else:
                #in original system actions were reordered
                aspace = OrderedDict([
                    ('forward_back',
                        env.action_space.spaces['forward_back']),
                    ('left_right',
                        env.action_space.spaces['left_right']),
                    ('jump',
                        env.action_space.spaces['jump']),
                    ('sneak_sprint',
                        env.action_space.spaces['sneak_sprint']),
                    ('camera',
                        env.action_space.spaces['camera']),
                    ('attack_place_equip_craft_nearbyCraft_nearbySmelt',
                        env.action_space.spaces['attack_place_equip_craft_nearbyCraft_nearbySmelt'])])


            #traj_pipe.send(num_actions)
            TrajectoryGenerator.write_pipe(traj_pipe[0], aspace)

            add_args = TrajectoryGenerator.read_pipe(comm_pipe)
            if args.needs_embedding:
                AgentClass = getattr(importlib.import_module(args.agent_from), args.agent_name)
                if add_args is None:
                    actor = AgentClass(state_shape, aspace, args)
                else:
                    add_args.append(True)
                    actor = AgentClass(state_shape, aspace, args, *add_args)
                try:
                    add_desc = actor.embed_desc()
                    TrajectoryGenerator.write_pipe(traj_pipe[0], add_desc)
                except:
                    assert False #embed_desc not implemented
                if args.load is not None:
                    actor.loadstore(args.load)
                actor._model['modules'] = actor._model['modules'].train(False)

            traj_index = 0
            pipe_number = 0
            #start generator
            while True:
                current_traj = trajs[traj_index]
                
                if not multi_env:
                    data_frames = list(data.load_data(current_traj, include_metadata=False))
                else:
                    data_frames = list(datas[env_ind].load_data(current_traj, include_metadata=False))
                    for i in range(len(data_frames)):
                        data_frames[i][0] = multi_data._new_obs(data_frames[i][0], env_ind)
                        data_frames[i][1] = multi_data._new_act(data_frames[i][1])
                        try:
                            if args.log_reward and not ('Navigate' in all_envs[env_ind] and 'Dense' in all_envs[env_ind]):
                                if data_frames[i][2] > 1.0:
                                    data_frames[i][2] = np.log2(data_frames[i][2]) + 1.0
                        except:
                            pass

                #convert actions and add current action to obs
                #combine actions
                for i in range(len(data_frames)):
                    data_frames[i][1] = combine.reverse_action(data_frames[i][1])
                    if i == 0:
                        if args.history_ar > 0:
                            data_frames[0][0] = history_ar._reset(data_frames[0][0])
                        if args.needs_last_action:
                            data_frames[0][0] = lastactionw._reset(data_frames[0][0])
                        if args.needs_orientation:
                            data_frames[0][0] = orientation._reset(data_frames[0][0])
                    else:
                        if args.history_ar > 0:
                            data_frames[i][0] = history_ar._step(data_frames[i][0], data_frames[i-1][2], data_frames[i-1][1])
                        if args.needs_last_action:
                            data_frames[i][0] = lastactionw._step(data_frames[i][0], data_frames[i-1][1])
                        if args.needs_orientation:
                            data_frames[i][0] = orientation._step(data_frames[i][0], data_frames[i-1][1])
                    #test orientation
                    #if i == 0 or i == (len(data_frames)//4) or i == i == (len(data_frames)//2) or i == ((4*len(data_frames))//5) or i == ((7*len(data_frames))//8) or i == ((3*len(data_frames))//4) or i == len(data_frames)-1:
                    #    print('orientation', data_frames[i][0]['orientation'])
                    #    plt.imshow(data_frames[i][0]['pov']/255.0)
                    #    plt.show()


                #make action dict
                action_dict = {}
                for name in aspace:
                    if isinstance(aspace[name], gym.spaces.Discrete):
                        action_dict[name] = np.zeros((len(data_frames),), dtype=np.int32)
                    else:
                        action_dict[name] = np.zeros((len(data_frames), aspace[name].shape[0]), dtype=np.float32)
                    for i in range(len(data_frames)):
                        action_dict[name][i] = data_frames[i][1][name]

                obs_l = {}
                for o in state_shape:
                    if o == 'pov':
                        obs_l[o] = np.zeros([len(data_frames)]+list(state_shape[o]), dtype=np.uint8)
                    else:
                        obs_l[o] = np.zeros([len(data_frames)]+list(state_shape[o]), dtype=np.float32)
                reward_l = np.zeros([len(data_frames)], dtype=np.float32)
                done_l = np.zeros([len(data_frames)], dtype=np.bool)
                #convert obs, reward and done
                for i in range(len(data_frames)):
                    t_obs = minerlobs.observation(data_frames[i][0])
                    for o in t_obs:
                        obs_l[o][i] = t_obs[o]
                    reward_l[i] = data_frames[i][2]
                    done_l[i] = data_frames[i][-1]



                #slice and pack into shared memory buffers
                i = 0
                while i < reward_l.shape[0]:
                    current_packet_size = min(packet_size, reward_l.shape[0] - i)
                    reward_sh = make_shared_buffer(reward_l[i:(i+current_packet_size)])
                    done_sh = make_shared_buffer(done_l[i:(i+current_packet_size)])
                    
                    #send packet
                    packet = {'reward': reward_sh,
                              'done': done_sh}

                    if args.needs_embedding:
                        if copy_queue is not None:
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
                                    print("copied actor in expert TrajectoryGenerator")
                        packet.update(actor.generate_embed(obs_l['pov'][i:(i+current_packet_size)]))

                    for o in obs_l:
                        packet[o] = make_shared_buffer(obs_l[o][i:(i+current_packet_size)])
                    for a in action_dict:
                        packet[a] = make_shared_buffer(action_dict[a][i:(i+current_packet_size)])
                    TrajectoryGenerator.write_pipe(traj_pipe[pipe_number], packet)
                    i += packet_size
                
                pipe_number = (pipe_number + 1) % len(traj_pipe)
                if not multi_env:
                    traj_index += 1
                    traj_index = traj_index % len(trajs)
                else:
                    env_ind = np.random.choice(len(all_envs), p=probs)#np.random.randint(0, len(all_envs))
                    trajs = all_trajs[env_ind]
                    traj_index = np.random.randint(0, len(trajs))
                    #print("HERE")
                    if 'Navigate' in all_envs[env_ind]:
                        print("ALARM")

            for pipe in traj_pipe:
                TrajectoryGenerator.write_pipe(pipe, "finish")
        except:
            print("FATAL error in VirtualExpertGenerator")
            traceback.print_exc()


class VirtualActorTrajGenerator(TrajectoryGenerator):
    def __init__(self, args, copy_queue=None, add_queues=0, add_args=None):
        super().__init__(args, copy_queue)
        TrajectoryGenerator.write_pipe(self.comm_pipe, add_args)
        if args.needs_embedding:
            self.add_desc = TrajectoryGenerator.read_pipe(self.traj_pipe[0])
        else:
            self.add_desc = {}
        for i in range(add_queues):
            self.traj_pipe.append(Queue(maxsize=10))

    def data_description(self):
        desc = {'reward': {'compression': False, 'shape': [], 'dtype': np.float32},
                'done': {'compression': False, 'shape': [], 'dtype': np.bool}}
        for k in self.state_shape:
            if k == 'pov':
                desc[k] = {'compression': True, 'shape': list(self.state_shape[k]), 'dtype': np.uint8, 'dontsendpast': self.args.dont_send_past_pov}
            else:
                if len(self.state_shape[k]) == 0:
                    desc[k] = {'compression': False, 'shape': [], 'dtype': np.int32}
                else:
                    desc[k] = {'compression': False, 'shape': list(self.state_shape[k]), 'dtype': np.float32}
        for a in self.num_actions:
            if isinstance(self.num_actions[a], gym.spaces.Discrete):
                desc[a] = {'compression': False, 'shape': [], 'dtype': np.int32}
            else:
                desc[a] = {'compression': False, 'shape': list(self.num_actions[a].shape), 'dtype': np.float32}
        if self.args.needs_embedding:
            desc.update(self.add_desc)
        return desc

    @staticmethod
    def generator_process(args, traj_pipe, packet_size, copy_queue, comm_pipe):
        try:
            multi_env = False
            try:
                multi_env = args.needs_multi_env
            except:
                pass
            try:
                record_episode = args.record_first_episode
            except:
                record_episode = False
            if not multi_env:
                env = gym.make(args.env)
                orig_env = env
                if record_episode:
                    saveorigenv = env_wrappers.SaveOrigObsAct(env)
                    env = saveorigenv
            else:
                env = gym.make(args.env)
                orig_env = env
                if record_episode:
                    saveorigenv = env_wrappers.SaveOrigObsAct(env)
                    env = saveorigenv
                all_envs = ['MineRLTreechop-v0',
                            'MineRLNavigateDense-v0',
                            'MineRLNavigate-v0',
                            'MineRLNavigateExtremeDense-v0',
                            'MineRLNavigateExtreme-v0',
                            'MineRLObtainIronPickaxe-v0',
                            'MineRLObtainIronPickaxeDense-v0',
                            'MineRLObtainDiamond-v0',
                            'MineRLObtainDiamondDense-v0']
                env_type = all_envs.index(args.env)
                datas = [own_data.make(tenv, num_workers=0) for tenv in all_envs]
                env = env_wrappers.ConstantMultiDataWrapper(env, datas, env_type)
                print(env.observation_space)

            if args.minerl:
                if False:
                    env = env_wrappers.ResetTrimInfoWrapper(env)
                    combine = env_wrappers.CombineActionWrapper(env)
                    env = combine
                    if args.needs_last_action:
                        combine = env_wrappers.LastAction(combine)
                        env = combine
                    if args.history_ar > 0:
                        history_ar = env_wrappers.HistoryActionReward(env, args.history_ar, args)
                        env = history_ar
                    if args.needs_orientation:
                        orientation = env_wrappers.Orientation(env)
                        env = orientation
                    if not multi_env:
                        minerlobs = env_wrappers.MineRLObsWrapper2(env)
                    else:
                        minerlobs = env_wrappers.MineRLObsWrapper3(env)
                    env = minerlobs
                else:
                    env = env_wrappers.ResetTrimInfoWrapper(env)
                    combine = env_wrappers.CombineActionWrapper(env)
                    env = combine
                    if args.history_ar > 0:
                        history_ar = env_wrappers.HistoryActionReward(env, args.history_ar, args)
                        env = history_ar
                    if args.needs_orientation:
                        orientation = env_wrappers.Orientation(env)
                        env = orientation
                    if not multi_env:
                        minerlobs = env_wrappers.MineRLObsWrapper2(env)
                    else:
                        minerlobs = env_wrappers.MineRLObsWrapper3(env)
                    env = minerlobs
                    if args.needs_last_action:
                        env = env_wrappers.LastAction(env)
            else:
                env = env_wrappers.DictObsWrapper(env)

            obs = env.observation_space
            assert isinstance(obs, dict)

            #convert to state_shape
            state_shape = {}
            for k in obs:
                if isinstance(obs[k], gym.spaces.Box):
                    state_shape[k] = list(obs[k].shape)
                else:
                    state_shape[k] = []

            TrajectoryGenerator.write_pipe(traj_pipe[0], state_shape)

            #get num actions
            if not True:#COMPATIBLE:
                aspace = env.action_space.spaces
            else:
                #in original system actions were reordered
                aspace = OrderedDict([
                    ('forward_back',
                        env.action_space.spaces['forward_back']),
                    ('left_right',
                        env.action_space.spaces['left_right']),
                    ('jump',
                        env.action_space.spaces['jump']),
                    ('sneak_sprint',
                        env.action_space.spaces['sneak_sprint']),
                    ('camera',
                        env.action_space.spaces['camera']),
                    ('attack_place_equip_craft_nearbyCraft_nearbySmelt',
                        env.action_space.spaces['attack_place_equip_craft_nearbyCraft_nearbySmelt'])])

            TrajectoryGenerator.write_pipe(traj_pipe[0], aspace)

            add_args = TrajectoryGenerator.read_pipe(comm_pipe)

            #load actor
            AgentClass = getattr(importlib.import_module(args.agent_from), args.agent_name)
            if add_args is None:
                actor = AgentClass(state_shape, aspace, args)
            else:
                actor = AgentClass(state_shape, aspace, args, *add_args)
            if args.needs_embedding:
                try:
                    add_desc = actor.embed_desc()
                    TrajectoryGenerator.write_pipe(traj_pipe[0], add_desc)
                except:
                    assert False #embed_desc not implemented
            else:
                add_desc = {}


            if args.load is not None:
                actor.loadstore(args.load)
            actor._model['modules'] = actor._model['modules'].train(False)

            batch_index = 0
            batch = {}
            for k in state_shape:
                if k == 'pov':
                    batch[k] = np.zeros([packet_size]+list(state_shape[k]), dtype=np.uint8)
                else:
                    batch[k] = np.zeros([packet_size]+list(state_shape[k]), dtype=np.float32)
            batch['reward'] = np.zeros([packet_size], dtype=np.float32)
            batch['done'] = np.zeros([packet_size], dtype=np.bool)
            for name in aspace:
                if isinstance(aspace[name], gym.spaces.Discrete):
                    batch[name] = np.zeros([packet_size], dtype=np.int32)
                else:
                    batch[name] = np.zeros([packet_size]+list(aspace[name].shape), dtype=np.float32)
            if args.needs_embedding:
                for a in add_desc:
                    batch[a] = np.zeros([packet_size]+add_desc[a]['shape'], dtype=add_desc[a]['dtype'])
                
            if record_episode:
                if OLD_RECORDING_SYSTEM:
                    fourcc = cv2.VideoWriter_fourcc(*'avc1')
                    out = cv2.VideoWriter('recorded/recording.mp4', fourcc, 20, (64,64))
                    #out.set(cv2.VIDEOWRITER_PROP_QUALITY, 100.0)
                else:
                    frame_num = 0
                record_obs_action_reward = {}
                for obs_s in orig_env.observation_space.spaces:
                    if obs_s != 'pov':
                        record_obs_action_reward['observation_'+obs_s] = []
                for act_s in orig_env.action_space.spaces:
                    record_obs_action_reward['action_'+act_s] = []
                record_obs_action_reward['reward'] = []
                #write a bunch of black frames at the start
                for i in range(0, 10):
                    if OLD_RECORDING_SYSTEM:
                        out.write(np.zeros((64, 64, 3), np.uint8))
                    else:
                        imsave("recorded/frames/%07d.png" % frame_num, np.zeros((64, 64, 3), np.uint8))
                        frame_num += 1



            r_cumulative_reward = None

            num_episode = 0
            if args.name is not None or args.name != '':
                f = open('out-' + args.name, 'w')
            for ie in range(args.episodes):
                #start new episode
                done = False
                cumulative_reward = 0.0
                env_state = env.reset()
                try:
                    actor.reset()
                except:
                    pass
                if record_episode:
                    #record first obs
                    frame = np.copy(saveorigenv.orig_obs['pov'])
                    if OLD_RECORDING_SYSTEM:
                        cv2.cvtColor(frame, code=cv2.COLOR_BGR2RGB, dst=frame)
                        out.write(frame)
                    else:
                        imsave("recorded/frames/%07d.png" % frame_num, frame)
                        frame_num += 1
                    for obs_s in orig_env.observation_space.spaces:
                        if obs_s != 'pov':
                            record_obs_action_reward['observation_'+obs_s].append(saveorigenv.orig_obs[obs_s])

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
                    action, info = actor.select_action(env_state)
                    for k in add_desc:
                        batch[k][batch_index] = info[k]
                    for name in aspace:
                        batch[name][batch_index] = action[name]

                    env_state, reward, done, _ = env.step(action)
                    try:
                        if args.log_reward:
                            if reward > 0.0:
                                if reward < 1.0:
                                    assert False # wrong enviroment
                                reward = np.log2(reward) + 1.0
                    except:
                        pass
                    batch['reward'][batch_index] = reward
                    batch['done'][batch_index] = done
                    if record_episode:
                        #record exp.
                        frame = np.copy(saveorigenv.orig_obs['pov'])
                        if OLD_RECORDING_SYSTEM:
                            cv2.cvtColor(frame, code=cv2.COLOR_BGR2RGB, dst=frame)
                            out.write(frame)
                        else:
                            imsave("recorded/frames/%07d.png" % frame_num, frame)
                            frame_num += 1
                        for obs_s in orig_env.observation_space.spaces:
                            if obs_s != 'pov':
                                record_obs_action_reward['observation_'+obs_s].append(saveorigenv.orig_obs[obs_s])
                        for act_s in orig_env.action_space.spaces:
                            record_obs_action_reward['action_'+act_s].append(saveorigenv.orig_action[act_s])
                        record_obs_action_reward['reward'].append(reward)



                    if batch_index == packet_size - 1:
                        #send packet to replay buffer
                        r_batch = {}
                        for b in batch:
                            r_batch[b] = SharedMemory(batch[b])
                            r_batch[b] = r_batch[b].shared_memory()
                        TrajectoryGenerator.write_pipe(traj_pipe[ie%len(traj_pipe)], r_batch)
                    batch_index = (batch_index + 1)%packet_size

                    cumulative_reward += reward

                    if ii % 100 == 0:
                        if isinstance(info, dict):
                            for k in info:
                                if not k in add_desc:
                                    print(k,info[k])
                        else:
                            print("actor entropy", info)
                    ii += 1

                if r_cumulative_reward is None:
                    r_cumulative_reward = cumulative_reward
                else:
                    r_cumulative_reward = 0.95 * r_cumulative_reward + 0.05 * cumulative_reward
                
                print("cumulative_reward",cumulative_reward,"/", r_cumulative_reward,num_episode)
                num_episode += 1
                if args.name is not None or args.name != '':
                    print(cumulative_reward, r_cumulative_reward,num_episode,file=f)
                if record_episode:
                    record_episode = False
                    if OLD_RECORDING_SYSTEM:
                        out.release()
                    else:
                        subprocess.call(["ffmpeg.exe","-y","-r","20","-i", "recorded/frames/%07d.png","-vcodec","libx264","-acodec","aac", "-b:v","543000", "-r", "20", "recorded/recording.mp4"])
                    for k in record_obs_action_reward:
                        record_obs_action_reward[k] = np.array(record_obs_action_reward[k])
                    np.savez('recorded/rendered.npz', **record_obs_action_reward)
                    meta_data = {}
                    with open('recorded/metadata.json', 'w') as fl:
                        json.dump(meta_data, fl)

            if args.name is not None or args.name != '':
                f.close()
        except:
            print("FATAL error in ActorTrajGenerator")
            traceback.print_exc()