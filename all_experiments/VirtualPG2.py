from virtual_replay_buffer2 import VirtualReplayBuffer, VirtualReplayBufferPER
from virtual_traj_generator import VirtualExpertGenerator, VirtualActorTrajGenerator
from trainers import Trainer
from memory import HashingMemory
import argparse
from torch.multiprocessing import Pipe, Queue
import time
from virtual_models import COMPATIBLE
import os
import numpy as np

class CombineQueue():
    def __init__(self, queue1, queue2, ratio=3.0/4.0):
        self.queue1 = queue1
        self.queue2 = queue2
        self.ratio = ratio

    def get(self):
        if np.random.sample() < self.ratio:
            return self.queue1.get()
        else:
            return self.queue2.get()
        
    def get_nowait(self):
        if np.random.sample() < self.ratio:
            return self.queue1.get_nowait()
        else:
            return self.queue2.get_nowait()

def main():
    parser = argparse.ArgumentParser(description="Virtual Policy Gradient")

    TRAIN_VAE_MODEL = False
    TRAIN_VAE_MODEL2 = False
    TRAIN_VAE_MODEL3 = False
    TRAIN_VAE_MODEL4 = False
    if False:
        TRAIN_VAE_MODEL5 = True #very good bc_loss
        TRAIN12 = True
        MULTITD3_TRAIN = False
    else:
        TRAIN_VAE_MODEL5 = False
        TRAIN12 = False
        MULTITD3_TRAIN = True

    PRETRAIN = False
    parser.add_argument("--pretrain", action='store_true', default=PRETRAIN, help="")
    parser.add_argument("--agent_name", type=str, default='VirtualPGModel', help="Model name")
    parser.add_argument("--agent_from", type=str, default='virtual_models_impl2', help="Model python module")
    parser.add_argument("--needs-next", action='store_true', default=False, help="")
    parser.add_argument("--env", default='MineRLTreechop-v0', type=str, help="Gym environment to use")#MineRLNavigateDense-v0#MineRLTreechop-v0
    parser.add_argument("--minerl", action='store_true', default=True, help="The environment is a MineRL environment")
    parser.add_argument("--name", type=str, default='treechop', help="Experiment name")
    parser.add_argument("--episodes", type=int, default=int(1e5), help="Number of episodes to run")

    parser.add_argument("--erpoolsize", type=int, default=int(4e5), help="Number of experiences stored by each option for experience replay")
    parser.add_argument("--packet-size", type=int, default=int(256), help="")
    parser.add_argument("--sample-efficiency", type=int, default=int(16), help="")
    parser.add_argument("--chunk-size", type=int, default=int(100), help="")
    parser.add_argument("--num-replay-workers", type=int, default=1, help="Number of replay buffer workers")
    parser.add_argument("--per", default=False, action="store_true", help="Use prioritized experience replay.")
    parser.add_argument("--trainstart", type=int, default=int(1e4), help="")
    parser.add_argument("--er", type=int, default=64, help="Number of experiences used to build a replay minibatch")
        
    parser.add_argument("--history-ar", type=int, default=20, help="Number of historic rewards")
    parser.add_argument("--history-a", type=int, default=0, help="Number of historic actions")
    parser.add_argument("--cnn-type", default='adv', type=str, choices=['atari', 'mnist', 'adv', 'fixup'], help="General shape of the CNN, if any. Either DQN-Like, or image-classification-like with more layers")
    parser.add_argument("--hidden", default=256, type=int, help="Hidden neurons of the policy network")
    parser.add_argument("--layers", default=1, type=int, help="Number of hidden layers in the networks")
    parser.add_argument("--state-layers", default=0, type=int, help="Number of hidden layers in the state networks")
    parser.add_argument("--lr", default=1e-4, type=float, help="Learning rate of the neural network")
    parser.add_argument("--eps", default=0.1, type=float, help="Entropy bonus")
    parser.add_argument("--load", type=str, default="treechop/model_world", help="File from which to load the neural network weights")
    parser.add_argument("--pretrained_load", type=str, help="File from which to load the neural network weights")
    parser.add_argument("--save", type=str, default="treechop/model_world", help="Basename of saved weight files. If not given, nothing is saved")

    parser.add_argument("--clr", type=float, default=0.5, help="Critic learning rate")
    parser.add_argument("--nec-lr", type=float, default=1e-3, help="NEC learning rate")
    parser.add_argument("--rho", type=float, default=1e-5, help="Lambda learning rate")
    parser.add_argument("--ganloss", type=str, default="hinge", choices=['hinge', 'wasserstein', 'fisher', 'BCE'], help="GAN loss type")
    
    parser.add_argument("--needs_skip", default=False, action="store_true", help="")
    parser.add_argument("--needs_last_action", default=True, action="store_true", help="")
    parser.add_argument("--needs_orientation", default=False, action="store_true", help="")
    parser.add_argument("--dont_send_past_pov", default=False, action="store_true", help="")
    parser.add_argument("--needs_embedding", default=False, action="store_true", help="")
    parser.add_argument("--needs_random_future", default=False, action="store_true", help="")
    parser.add_argument("--num_past_times", type=int, default=int(32), help="")
    parser.add_argument("--embed_dim", type=int, default=int(128), help="")
    parser.add_argument("--temperature", type=float, default=0.8, help="")

    HashingMemory.register_args(parser)

    args = parser.parse_args()
    
    if COMPATIBLE:
        args.mem_n_keys = 512
    else:
        args.mem_n_keys = 512
        args.mem_query_batchnorm = True

    #hier aenderungen einbauen
    pov_time_deltas = None
    if TRAIN_VAE_MODEL:
        args.agent_name = "VirtualPGVAE"
        args.agent_from = "virtual_models_impl4"
        args.needs_orientation = True
        args.needs_embedding = True
        args.ganloss = 'BCE'
        args.load = "treechop/model_"
        args.save = "treechop/model_"
        args.mem_query_batchnorm = False
        args.cnn_type = 'atari'
        args.history_a = 3
        time_deltas = range(-5, 5, 1)
    elif TRAIN_VAE_MODEL2:
        args.agent_name = "VirtualPGVAE"
        args.agent_from = "virtual_models_impl5"
        args.needs_orientation = True
        args.needs_embedding = True
        args.ganloss = 'fisher'
        args.load = "treechop/model5_"
        args.save = "treechop/model5_"
        args.dont_send_past_pov = True
        args.mem_query_batchnorm = False
        args.cnn_type = 'atari'
        args.history_a = 0
        args.er = 256
        num_past_times = 32
        past_range_seconds = 60.0
        past_range = past_range_seconds / 0.05
        delta_a = (past_range-num_past_times-1)/(num_past_times**2)
        past_times = [-int(delta_a*n**2+n+1) for n in range(num_past_times)]
        past_times.reverse()
        time_deltas = past_times + list(range(0, 5, 1))
    elif TRAIN_VAE_MODEL3 or TRAIN_VAE_MODEL4:
        args.agent_name = "VirtualPG"
        args.agent_from = "virtual_models_impl7"
        if TRAIN_VAE_MODEL4:
            args.agent_from = "virtual_models_impl9"
        args.needs_orientation = True
        args.needs_embedding = True
        args.needs_last_action = False
        args.load = "treechop/model_"
        args.save = "treechop/model_"
        args.dont_send_past_pov = True
        args.mem_query_batchnorm = False
        args.cnn_type = 'atari'
        args.ganloss = 'hinge'
        args.rho = 1e-4
        args.needs_random_future = TRAIN_VAE_MODEL4
        args.history_a = 10
        args.er = 128
        args.embed_dim = 128
        num_past_times = 32
        args.num_past_times = num_past_times
        past_range_seconds = 60.0
        past_range = past_range_seconds / 0.05
        delta_a = (past_range-num_past_times-1)/(num_past_times**2)
        past_times = [-int(delta_a*n**2+n+1) for n in range(num_past_times)]
        past_times.reverse()
        if TRAIN_VAE_MODEL4:
            time_deltas = past_times + list(range(0, 41, 1))
            time_deltas = [-int(delta_a*n**2+n+1) + 20 for n in range(num_past_times)] + time_deltas
            pov_time_deltas = [0, 20, 20]
        else:
            time_deltas = past_times + list(range(0, 21, 1))
            pov_time_deltas = [0, 20]
    elif TRAIN_VAE_MODEL5:
        args.agent_name = "VirtualPG"
        args.agent_from = "virtual_models_impl11"
        if TRAIN12:
            args.agent_from = "virtual_models_impl12"
        args.needs_orientation = True
        args.needs_embedding = True
        args.needs_last_action = False
        args.load = "treechop/model_"
        args.save = "treechop/model_"
        args.dont_send_past_pov = True
        args.mem_query_batchnorm = False
        args.cnn_type = 'dcgan'
        args.ganloss = 'hinge'
        args.rho = 1e-4
        args.mem_n_keys = 128
        args.needs_random_future = True
        args.history_a = 0
        if PRETRAIN:
            args.er = 128
        else:
            args.er = 64
        args.embed_dim = 128
        num_past_times = 32
        args.num_past_times = num_past_times
        past_range_seconds = 60.0
        past_range = past_range_seconds / 0.05
        delta_a = (past_range-num_past_times-1)/(num_past_times**2)
        past_times = [-int(delta_a*n**2+n+1) for n in range(num_past_times)]
        past_times.reverse()
        time_deltas = past_times + list(range(0, 41, 1))
        time_deltas = [-int(delta_a*n**2+n+1) + 20 for n in range(num_past_times)] + time_deltas
        pov_time_deltas = [0, 20, 20]
    elif MULTITD3_TRAIN:
        args.agent_name = "MultiTD3"
        args.agent_from = "multi_td3_models_impl3"
        args.mem_heads = 2

        args.needs_orientation = True
        args.needs_embedding = True
        args.needs_last_action = False
        args.pretrained_load = None#"treechop/model2_multiaction_TD3_pre"
        args.load = "treechop/model4_"
        args.save = "treechop/model4_"
        args.dont_send_past_pov = True
        args.mem_query_batchnorm = False
        args.cnn_type = 'atari'
        args.ganloss = 'hinge'
        args.rho = 1e-4
        args.mem_n_keys = 512
        args.needs_random_future = False
        args.history_a = 0
        if PRETRAIN:
            args.er = 128
        else:
            args.er = 64
        args.embed_dim = 256
        num_past_times = 32
        args.num_past_times = num_past_times
        past_range_seconds = 60.0
        past_range = past_range_seconds / 0.05
        delta_a = (past_range-num_past_times-1)/(num_past_times**2)
        past_times = [-int(delta_a*n**2+n+1) for n in range(num_past_times)]
        past_times.reverse()
        if not PRETRAIN:
            time_deltas = past_times + list(range(0, 21, 1))
        else:
            time_deltas = past_times + list(range(0, 41, 1))
        time_deltas = [-int(delta_a*n**2+n+1) + 20 for n in range(num_past_times)] + time_deltas
        pov_time_deltas = [0, 20]
        if not PRETRAIN:
            args.per = True
            args.erpoolsize = int(1e6)
    else:
        time_deltas = [0, 1, 2]



    HashingMemory.check_params(args)
    
    
    num_workers = 3
    copy_queues = [Queue(1) for n in range(num_workers)]
    if PRETRAIN:
        args.packet_size = max(256, args.er)
        args.sample_efficiency = max(args.packet_size // args.er, 1)
        args.trainstart = args.erpoolsize // num_workers
        args.lr = 1e-4
        args.nec_lr = 1e-3
        args.num_replay_workers = num_workers
        if args.per:
            args.sample_efficiency *= 2
            prio_queues = [[Queue(2) for n in range(num_workers)]]
            traj_gens = [VirtualExpertGenerator(args, 0, copy_queue=copy_queues[n], add_args=[time_deltas]) for n in range(num_workers)]
            if not (TRAIN_VAE_MODEL2 or TRAIN_VAE_MODEL3 or TRAIN_VAE_MODEL4 or TRAIN_VAE_MODEL5):
                start = time_deltas[0]
                end = time_deltas[-1]
            else:
                start = 0
                end = time_deltas[-1]
            replay = VirtualReplayBufferPER(args, traj_gens[0].data_description(), [traj_gens[n].traj_pipe[0] for n in range(num_workers)], prio_queues[0], [t-start for t in time_deltas], 50+end, pov_time_deltas=pov_time_deltas)
            trainer = Trainer(args.agent_name, args.agent_from, traj_gens[0].state_shape, traj_gens[0].num_actions, args, 
                                    [replay.batch_queues], add_args=[time_deltas], prio_queues=prio_queues, copy_queues=copy_queues)
        else:
            traj_gens = [VirtualExpertGenerator(args, 0, copy_queue=copy_queues[n], add_args=[time_deltas]) for n in range(num_workers)]
            #traj_gen = VirtualExpertGenerator(args, 0, copy_queue=copy_queues[0], add_args=[time_deltas])
            #traj_gen2 = VirtualExpertGenerator(args, 0, copy_queue=copy_queues[1], add_args=[time_deltas])
            if not (TRAIN_VAE_MODEL2 or TRAIN_VAE_MODEL3 or TRAIN_VAE_MODEL4 or TRAIN_VAE_MODEL5):
                start = time_deltas[0]
                end = time_deltas[-1]
            else:
                start = 0
                end = time_deltas[-1]
            replay = VirtualReplayBuffer(args, traj_gens[0].data_description(), [traj_gens[n].traj_pipe[0] for n in range(num_workers)], [t-start for t in time_deltas], 50+end, pov_time_deltas=pov_time_deltas)
            trainer = Trainer(args.agent_name, args.agent_from, traj_gens[0].state_shape, traj_gens[0].num_actions, args, 
                                    [replay.batch_queues], add_args=[time_deltas], copy_queues=copy_queues)
    else:
        num_workers = 2
        args.packet_size = max(256, args.er)
        args.sample_efficiency = max((args.packet_size * 32) // args.er, 1)
        args.lr = 1e-4
        args.nec_lr = 1e-3
        args.num_replay_workers = num_workers
        copy_queues = [Queue(1) for n in range(num_workers+1)]
        traj_gen = VirtualActorTrajGenerator(args, copy_queue=copy_queues[0], add_queues=args.num_replay_workers-1, add_args=[time_deltas])
        traj_gens = [VirtualExpertGenerator(args, 0, copy_queue=copy_queues[n+1], add_args=[time_deltas]) for n in range(num_workers)]
        combine_traj = [CombineQueue(traj_gen.traj_pipe[n], traj_gens[n].traj_pipe[0]) for n in range(args.num_replay_workers)]
        assert len(traj_gen.traj_pipe) == args.num_replay_workers
        if not (TRAIN_VAE_MODEL2 or TRAIN_VAE_MODEL3 or TRAIN_VAE_MODEL4 or TRAIN_VAE_MODEL5):
            start = time_deltas[0]
            end = time_deltas[-1]
        else:
            start = 0
            end = time_deltas[-1]
        if args.per:
            prio_queues = [[Queue(2) for n in range(num_workers)]]
            replay = VirtualReplayBufferPER(args, traj_gen.data_description(), combine_traj, prio_queues[0], [t-start for t in time_deltas], 50+end, pov_time_deltas=pov_time_deltas)
            trainer = Trainer(args.agent_name, args.agent_from, traj_gen.state_shape, traj_gen.num_actions, args, 
                                [replay.batch_queues], add_args=[time_deltas], prio_queues=prio_queues, copy_queues=copy_queues, blocking=False)
        else:
            replay = VirtualReplayBuffer(args, traj_gen.data_description(), combine_traj, [t-start for t in time_deltas], 50+end, pov_time_deltas=pov_time_deltas)
            trainer = Trainer(args.agent_name, args.agent_from, traj_gen.state_shape, traj_gen.num_actions, args, 
                                [replay.batch_queues], add_args=[time_deltas], copy_queues=copy_queues, blocking=False)

    while True:
        time.sleep(100)
        #for t in combine_traj:
        #    print("COMB",t.queue1.qsize(), t.queue2.qsize())
        #for r in replay.batch_queues:
        #    print("REPL",r.qsize())
        #for p in prio_queues[0]:
        #    print("PRIO",p.qsize())

if __name__ == '__main__':
    main()