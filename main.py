import argparse
from all_experiments import HashingMemory
from torch.multiprocessing import Queue
from expert_traj_generator import ExpertGenerator
from actor_traj_generator import ActorTrajGenerator
from replaybuffers import ReplayBuffer, ReplayBufferPER
from trainer import Trainer
from traj_modifiers import FutureRewardTimeLeftModifier
import time
import copy
import importlib

DEBUG_ACTOR_REPLAY_BUFFER = False
DEBUG_EXPERT_REPLAY_BUFFER = False
DEBUG_TRAINER = True

def pretrain(args):
    AgentClass = getattr(importlib.import_module(args.agent_from), args.agent_name)
    need_dict, time_deltas, pov_time_deltas = AgentClass.needed_state_info(args)
    min_num_future_rewards = need_dict['min_num_future_rewards']
    #time_deltas and pov_time_deltas are used to mark multiple past and future states as relevant for the training
    
    HashingMemory.check_params(args)

    args.save += 'pretrain_'
    num_workers = args.num_replay_workers
    args.packet_size = max(256, args.er)
    args.sample_efficiency = max(args.packet_size // args.er, 1)
    #copy queues for actor updates from the training process
    copy_queues = [Queue(1) for n in range(num_workers)]
    #create expert data trajectory generator process
    expert_traj = [ExpertGenerator(args, 
                                   0, 
                                   copy_queue=copy_queues[n], 
                                   add_args=[time_deltas]) 
                   for n in range(num_workers)]
    data_description = expert_traj[0].data_description()
    state_shape = expert_traj[0].state_shape
    num_actions = expert_traj[0].num_actions

    #apply modifier workers
    if 'needs_future_reward_and_time_left' in need_dict and need_dict['needs_future_reward_and_time_left']:
        needs_only_reward_density=False
        if 'needs_only_reward_density' in need_dict:
            needs_only_reward_density = need_dict['needs_only_reward_density']
        expert_traj = [FutureRewardTimeLeftModifier(data_description, expert_traj[n].traj_pipe[0], just_density=needs_only_reward_density) 
                       for n in range(len(expert_traj))]
        data_description = expert_traj[0].data_description

    print("Data:")
    print(data_description.keys())
    end = time_deltas[-1]
    if args.per:
        #if per is set use the "Prioritized Experience Replay"-buffer workers
        #these queus are used to send priority updates back to the replay buffers
        prio_queues = [[Queue(2) for n in range(num_workers)]]
        args = copy.deepcopy(args)
        args.sample_efficiency = 1
        replay_expert = ReplayBufferPER(args, 
                                        data_description, 
                                        expert_traj[0].traj_pipe, 
                                        prio_queues[0], 
                                        time_deltas, 
                                        min_num_future_rewards+end, 
                                        pov_time_deltas=pov_time_deltas, 
                                        blocking=True,
                                        no_process=DEBUG_EXPERT_REPLAY_BUFFER)
    else:
        #use uniform sampling replay buffer workers
        prio_queues = None
        args = copy.deepcopy(args)
        args.sample_efficiency = 1
        replay_expert = ReplayBuffer(args, 
                                     data_description, 
                                     expert_traj[0].traj_pipe, 
                                     time_deltas, 
                                     min_num_future_rewards+end, 
                                     pov_time_deltas=pov_time_deltas, 
                                     blocking=True,
                                     no_process=DEBUG_EXPERT_REPLAY_BUFFER)
    #create the trainer process
    trainer = Trainer(args.agent_name, 
                      args.agent_from, 
                      state_shape, 
                      num_actions, 
                      args, 
                      [replay_expert.batch_queues], 
                      add_args=[time_deltas], 
                      prio_queues=prio_queues, 
                      copy_queues=copy_queues, 
                      blocking=False,
                      no_process=DEBUG_TRAINER)

    if not DEBUG_EXPERT_REPLAY_BUFFER \
        and not DEBUG_TRAINER:
        while True:
            time.sleep(1)
    elif DEBUG_EXPERT_REPLAY_BUFFER:
        replay_expert.p()
    elif DEBUG_TRAINER:
        trainer.p()

def train(args):
    AgentClass = getattr(importlib.import_module(args.agent_from), args.agent_name)
    need_dict, time_deltas, pov_time_deltas = AgentClass.needed_state_info(args)
    min_num_future_rewards = need_dict['min_num_future_rewards']
    #time_deltas and pov_time_deltas are used to mark multiple past and future states as relevant for the training
    
    HashingMemory.check_params(args)

    num_workers = args.num_replay_workers
    args.packet_size = max(256, args.er)
    args.sample_efficiency = max((args.packet_size * args.sample_efficiency) // args.er, 1)
    #copy queues for actor updates from the training process
    copy_queues = [Queue(1) for n in range(num_workers+1)]
    #create actor trajectory generator process
    actor_traj = ActorTrajGenerator(args, 
                                    copy_queue=copy_queues[0], 
                                    add_queues=args.num_replay_workers-1, 
                                    add_args=[time_deltas])
    #create expert data trajectory generator process
    expert_traj = [ExpertGenerator(args, 
                                   0, 
                                   copy_queue=copy_queues[n+1], 
                                   add_args=[time_deltas]) 
                   for n in range(num_workers)]
    data_description = actor_traj.data_description()
    state_shape = actor_traj.state_shape
    num_actions = actor_traj.num_actions

    #apply modifier workers
    if 'needs_future_reward_and_time_left' in need_dict and need_dict['needs_future_reward_and_time_left']:
        needs_only_reward_density=False
        if 'needs_only_reward_density' in need_dict:
            needs_only_reward_density = need_dict['needs_only_reward_density']
        actor_traj = FutureRewardTimeLeftModifier(data_description, actor_traj.traj_pipe[0], just_density=needs_only_reward_density)
        data_description = actor_traj.data_description
        expert_traj = [FutureRewardTimeLeftModifier(data_description, expert_traj[n].traj_pipe[0], just_density=needs_only_reward_density) 
                       for n in range(len(expert_traj))]

    print("Data:")
    print(data_description.keys())
    assert len(actor_traj.traj_pipe) == args.num_replay_workers
    end = time_deltas[-1]
    if args.per:
        #if per is set use the "Prioritized Experience Replay"-buffer workers
        #these queus are used to send priority updates back to the replay buffers
        prio_queues = [[Queue(2) for n in range(num_workers)], [Queue(2) for n in range(num_workers)]]
        replay_actor = ReplayBufferPER(args, 
                                       data_description, 
                                       actor_traj.traj_pipe, 
                                       prio_queues[0], 
                                       time_deltas, 
                                       min_num_future_rewards+end, 
                                       pov_time_deltas=pov_time_deltas, 
                                       blocking=True,
                                       no_process=DEBUG_ACTOR_REPLAY_BUFFER)
        args = copy.deepcopy(args)
        args.sample_efficiency = 1
        replay_expert = ReplayBufferPER(args, 
                                        data_description, 
                                        expert_traj[0].traj_pipe, 
                                        prio_queues[1], 
                                        time_deltas, 
                                        min_num_future_rewards+end, 
                                        pov_time_deltas=pov_time_deltas, 
                                        blocking=True,
                                        no_process=DEBUG_EXPERT_REPLAY_BUFFER)
    else:
        #use uniform sampling replay buffer workers
        prio_queues = None
        replay_actor = ReplayBuffer(args, 
                                    data_description, 
                                    actor_traj.traj_pipe, 
                                    time_deltas, 
                                    min_num_future_rewards+end, 
                                    pov_time_deltas=pov_time_deltas, 
                                    blocking=True,
                                    no_process=DEBUG_ACTOR_REPLAY_BUFFER)
        args = copy.deepcopy(args)
        args.sample_efficiency = 1
        replay_expert = ReplayBuffer(args, 
                                     data_description, 
                                     expert_traj[0].traj_pipe, 
                                     time_deltas, 
                                     min_num_future_rewards+end, 
                                     pov_time_deltas=pov_time_deltas, 
                                     blocking=True,
                                     no_process=DEBUG_EXPERT_REPLAY_BUFFER)
    #create the trainer process
    trainer = Trainer(args.agent_name, 
                      args.agent_from, 
                      state_shape, 
                      num_actions, 
                      args, 
                      [replay_actor.batch_queues, replay_expert.batch_queues], 
                      add_args=[time_deltas], 
                      prio_queues=prio_queues, 
                      copy_queues=copy_queues, 
                      blocking=False,
                      no_process=DEBUG_TRAINER)

    
    if not DEBUG_ACTOR_REPLAY_BUFFER \
        and not DEBUG_EXPERT_REPLAY_BUFFER \
        and not DEBUG_TRAINER:
        while True:
            time.sleep(1)
    elif DEBUG_ACTOR_REPLAY_BUFFER:
        replay_actor.p()
    elif DEBUG_EXPERT_REPLAY_BUFFER:
        replay_expert.p()
    elif DEBUG_TRAINER:
        trainer.p()

def main():
    parser = argparse.ArgumentParser(description="RLexperiments")

    parser.add_argument("--pretrain", action='store_true', default=False, help="")
    parser.add_argument("--agent_name", type=str, default='MuZeroModel', help="Model name")
    parser.add_argument("--agent_from", type=str, default='muzero_models', help="Model python module")
    parser.add_argument("--needs-next", action='store_true', default=False, help="")
    parser.add_argument("--env", default='MineRLTreechop-v0', type=str, help="Gym environment to use")
    parser.add_argument("--minerl", action='store_true', default=True, help="The environment is a MineRL environment")
    parser.add_argument("--name", type=str, default='treechop', help="Experiment name")
    parser.add_argument("--episodes", type=int, default=int(1e5), help="Number of episodes to run")

    parser.add_argument("--erpoolsize", type=int, default=int(5e5), help="Number of experiences stored by each option for experience replay")
    parser.add_argument("--packet-size", type=int, default=int(256), help="")
    parser.add_argument("--sample-efficiency", type=int, default=int(64), help="")
    parser.add_argument("--chunk-size", type=int, default=int(100), help="")
    parser.add_argument("--num-replay-workers", type=int, default=1, help="Number of replay buffer workers")
    parser.add_argument("--per", default=False, action="store_true", help="Use prioritized experience replay.")
    parser.add_argument("--trainstart", type=int, default=int(1e4), help="")
    parser.add_argument("--er", type=int, default=128, help="Number of experiences used to build a replay minibatch")
        
    parser.add_argument("--history-ar", type=int, default=10, help="Number of historic rewards")
    parser.add_argument("--history-a", type=int, default=5, help="Number of historic actions")
    parser.add_argument("--cnn-type", default='atari', type=str, choices=['atari', 'mnist', 'adv', 'fixup'], help="General shape of the CNN, if any. Either DQN-Like, or image-classification-like with more layers")
    parser.add_argument("--hidden", default=256, type=int, help="Hidden neurons of the network")
    parser.add_argument("--layers", default=1, type=int, help="Number of hidden layers in the networks")
    parser.add_argument("--state-layers", default=0, type=int, help="Number of hidden layers in the state networks")
    parser.add_argument("--lr", default=1e-4, type=float, help="Learning rate of the neural network")
    parser.add_argument("--eps", default=0.1, type=float, help="Entropy bonus")
    parser.add_argument("--load", type=str, default="treechop/model_", help="File from which to load the neural network weights")
    parser.add_argument("--pretrained_load", type=str, help="File from which to load the neural network weights")
    parser.add_argument("--save", type=str, default="treechop/model_", help="Basename of saved weight files. If not given, nothing is saved")

    parser.add_argument("--clr", type=float, default=0.5, help="Critic learning rate")
    parser.add_argument("--nec-lr", type=float, default=1e-3, help="NEC learning rate")
    parser.add_argument("--rho", type=float, default=1e-4, help="Lambda learning rate")
    parser.add_argument("--ganloss", type=str, default="fisher", choices=['hinge', 'wasserstein', 'fisher', 'BCE'], help="GAN loss type")
    
    parser.add_argument("--needs_skip", default=False, action="store_true", help="")
    parser.add_argument("--needs_last_action", default=False, action="store_true", help="")
    parser.add_argument("--needs_orientation", default=True, action="store_true", help="")
    parser.add_argument("--dont_send_past_pov", default=True, action="store_true", help="")
    parser.add_argument("--needs_embedding", default=False, action="store_true", help="")
    parser.add_argument("--needs_add_data", default=True, action="store_true", help="")
    parser.add_argument("--needs_random_future", default=False, action="store_true", help="")
    parser.add_argument("--num_past_times", type=int, default=int(32), help="")
    parser.add_argument("--embed_dim", type=int, default=int(256), help="")
    parser.add_argument("--temperature", type=float, default=0.8, help="")
    parser.add_argument("--act_on_cpu", default=True, action="store_true", help="")
    
    parser.add_argument("--needs_multi_env", default=False, action="store_true", help="")
    parser.add_argument("--log_reward", default=False, action="store_true", help="")

    HashingMemory.register_args(parser)

    args = parser.parse_args()


    if not args.pretrain:
        train(args)
    else:
        pretrain(args)


    
if __name__ == '__main__':
    main()