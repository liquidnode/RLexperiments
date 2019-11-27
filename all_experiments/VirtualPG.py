from virtual_replay_buffer import VirtualReplayBuffer, VirtualReplayBufferPER
from virtual_traj_generator import VirtualExpertGenerator, VirtualActorTrajGenerator
from trainers import Trainer
from memory import HashingMemory
import argparse
from torch.multiprocessing import Pipe, Queue
import time
from virtual_models import COMPATIBLE


def main():
    parser = argparse.ArgumentParser(description="Virtual Policy Gradient")

    PRETRAIN = False
    parser.add_argument("--pretrain", action='store_true', default=PRETRAIN, help="")
    parser.add_argument("--agent_name", type=str, default='VirtualPGFeature', help="Model name")
    parser.add_argument("--agent_from", type=str, default='virtual_models_impl1', help="Model python module")
    parser.add_argument("--needs-next", action='store_true', default=False, help="")
    parser.add_argument("--env", default='MineRLTreechop-v0', type=str, help="Gym environment to use")#MineRLNavigateDense-v0#MineRLTreechop-v0
    parser.add_argument("--minerl", action='store_true', default=True, help="The environment is a MineRL environment")
    parser.add_argument("--name", type=str, default='treechop', help="Experiment name")
    parser.add_argument("--episodes", type=int, default=500, help="Number of episodes to run")

    parser.add_argument("--erpoolsize", type=int, default=int(4e5), help="Number of experiences stored by each option for experience replay")
    parser.add_argument("--packet-size", type=int, default=int(256), help="")
    parser.add_argument("--sample-efficiency", type=int, default=int(16), help="")
    parser.add_argument("--chunk-size", type=int, default=int(100), help="")
    parser.add_argument("--num-replay-workers", type=int, default=1, help="Number of replay buffer workers")
    parser.add_argument("--per", default=False, action="store_true", help="Use prioritized experience replay.")
    parser.add_argument("--trainstart", type=int, default=int(1e4), help="")
    parser.add_argument("--er", type=int, default=256, help="Number of experiences used to build a replay minibatch")
        
    parser.add_argument("--history-ar", type=int, default=20, help="Number of historic rewards")
    parser.add_argument("--history-a", type=int, default=0, help="Number of historic actions")
    parser.add_argument("--cnn-type", default='atari', type=str, choices=['atari', 'mnist', 'fixup'], help="General shape of the CNN, if any. Either DQN-Like, or image-classification-like with more layers")
    parser.add_argument("--hidden", default=256, type=int, help="Hidden neurons of the policy network")
    parser.add_argument("--layers", default=1, type=int, help="Number of hidden layers in the networks")
    parser.add_argument("--state-layers", default=0, type=int, help="Number of hidden layers in the state networks")
    parser.add_argument("--lr", default=1e-4, type=float, help="Learning rate of the neural network")
    parser.add_argument("--eps", default=0.1, type=float, help="Entropy bonus")
    parser.add_argument("--load_pretain", type=str, help="File from which to load the neural network weights")#, default="treechop/nec_model_softq_atari_fine-pretrain-softq"
    parser.add_argument("--load", type=str, default="treechop/model_atari4", help="File from which to load the neural network weights")#, default="main_model"
    parser.add_argument("--save", type=str, default=None, help="Basename of saved weight files. If not given, nothing is saved")#, default="main_model"

    parser.add_argument("--clr", type=float, default=0.5, help="Critic learning rate")
    parser.add_argument("--nec-lr", type=float, default=1e-3, help="NEC learning rate")
    parser.add_argument("--rho", type=float, default=1e-5, help="Lambda learning rate")
    parser.add_argument("--needs_skip", default=False, action="store_true", help="")
    parser.add_argument("--needs_last_action", default=True, action="store_true", help="")

    #neu
    parser.add_argument("--needs_orientation", default=False, action="store_true", help="")
    parser.add_argument("--dont_send_past_pov", default=False, action="store_true", help="")
    parser.add_argument("--needs_embedding", default=False, action="store_true", help="")
    parser.add_argument("--needs_random_future", default=False, action="store_true", help="")
    parser.add_argument("--num_past_times", type=int, default=int(0), help="")
    parser.add_argument("--embed_dim", type=int, default=int(128), help="")
    parser.add_argument("--temperature", type=float, default=0.8, help="")
    
    parser.add_argument("--needs_multi_env", default=False, action="store_true", help="")
    parser.add_argument("--log_reward", default=False, action="store_true", help="")
    parser.add_argument("--record_first_episode", default=True, action="store_true", help="")

    HashingMemory.register_args(parser)

    args = parser.parse_args()
    
    if COMPATIBLE:
        args.mem_n_keys = 512
    else:
        args.mem_n_keys = 512
        args.mem_query_batchnorm = False
    HashingMemory.check_params(args)
    #args.mem_query_batchnorm = True

    if PRETRAIN:
        if False:
            args.packet_size = 256
            args.sample_efficiency = 1#args.packet_size // args.er
            args.trainstart = args.erpoolsize // 2
            args.lr = 1e-4
            args.nec_lr = 1e-3
            time_deltas = [0, 1, 2, 3, 4, 5, 10, 15, 25]
            args.num_replay_workers = 2
            traj_gen = VirtualExpertGenerator(args, 0)
            traj_gen2 = VirtualExpertGenerator(args, 0)
            replay = VirtualReplayBuffer(args, traj_gen.data_description(), [traj_gen.traj_pipe[0], traj_gen.traj_pipe[0]], time_deltas)
            if True:
                trainer = Trainer(args.agent_name, args.agent_from, traj_gen.state_shape, traj_gen.num_actions, args, 
                                    [replay.batch_queues], add_args=[time_deltas])
            else:
                import torch
                import numpy as np
                import matplotlib.pyplot as plt
                #test replay buffer
                while True:
                    input = replay.batch_queues[0].get()

                    obs_l = input['pov'].data.cpu().numpy()
                    plt.imshow(np.concatenate([obs_l[5,0]/255.0, obs_l[5,1]/255.0, obs_l[5,2]/255.0], axis=1))
                    plt.show()
        else:
            #representation learning
            #args.mem_query_batchnorm = False
            args.packet_size = 256
            args.sample_efficiency = 1#args.packet_size // args.er
            args.trainstart = args.erpoolsize // 2
            args.lr = 1e-4
            args.nec_lr = 1e-3
            time_deltas = [0, 1]
            args.num_replay_workers = 2
            if args.per:
                args.sample_efficiency *= 4
                prio_queues = [[Queue(2), Queue(2)]]
                traj_gen = VirtualExpertGenerator(args, 0)
                traj_gen2 = VirtualExpertGenerator(args, 0)
                replay = VirtualReplayBufferPER(args, traj_gen.data_description(), [traj_gen.traj_pipe[0], traj_gen2.traj_pipe[0]], prio_queues[0], time_deltas, 50)
                trainer = Trainer(args.agent_name, args.agent_from, traj_gen.state_shape, traj_gen.num_actions, args, 
                                        [replay.batch_queues], add_args=[time_deltas], prio_queues=prio_queues)
            else:
                traj_gen = VirtualExpertGenerator(args, 0)
                traj_gen2 = VirtualExpertGenerator(args, 0)
                replay = VirtualReplayBuffer(args, traj_gen.data_description(), [traj_gen.traj_pipe[0], traj_gen2.traj_pipe[0]], time_deltas, 50)
                trainer = Trainer(args.agent_name, args.agent_from, traj_gen.state_shape, traj_gen.num_actions, args, 
                                        [replay.batch_queues], add_args=[time_deltas])
    else:
        time_deltas = [0, 1]
        traj_gen = VirtualActorTrajGenerator(args, add_queues=args.num_replay_workers-1, add_args=[time_deltas])
        assert len(traj_gen.traj_pipe) == args.num_replay_workers
        replay = VirtualReplayBuffer(args, traj_gen.data_description(), traj_gen.traj_pipe, time_deltas)
        trainer = Trainer(args.agent_name, args.agent_from, traj_gen.state_shape, traj_gen.num_actions, args, 
                            [replay.batch_queues], add_args=[time_deltas])
        
    while True:
        time.sleep(1)

if __name__ == '__main__':
    main()
