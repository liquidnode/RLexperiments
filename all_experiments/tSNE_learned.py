from trainers import ActorTrainer, CriticTrainer, SoftQTrainer, IQNTrainer
from replay_buffer import ReplayBuffer
from traj_generator import ExpertGenerator, ActorTrajGenerator, EntropyPlot
from memory import HashingMemory
import argparse
from torch.multiprocessing import Pipe, Queue
import time
from models import IQN
import models
from sklearn.manifold import TSNE
import torch
import numpy as np
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser(description="Distributed BDPI")

    parser.add_argument("--pretrain", action='store_true', default=False, help="")
    parser.add_argument("--time", action='store_true', default=False, help="")
    parser.add_argument("--softq", action='store_true', default=False, help="")
    parser.add_argument("--iqn", action='store_true', default=True, help="")
    parser.add_argument("--needs-next", action='store_true', default=True, help="")
    parser.add_argument("--env", default='MineRLTreechop-v0', type=str, help="Gym environment to use")#MineRLNavigateDense-v0#MineRLTreechop-v0
    parser.add_argument("--minerl", action='store_true', default=True, help="The environment is a MineRL environment")
    parser.add_argument("--name", type=str, default='treechop', help="Experiment name")
    parser.add_argument("--episodes", type=int, default=500, help="Number of episodes to run")

    parser.add_argument("--erpoolsize", type=int, default=int(4e5), help="Number of experiences stored by each option for experience replay")
    parser.add_argument("--packet-size", type=int, default=int(256), help="")
    parser.add_argument("--sample-efficiency", type=int, default=int(64), help="")
    parser.add_argument("--chunk-size", type=int, default=int(100), help="")
    parser.add_argument("--num-replay-workers", type=int, default=2, help="Number of replay buffer workers")
    parser.add_argument("--per", default=False, action="store_true", help="Use prioritized experience replay.")
    parser.add_argument("--trainstart", type=int, default=int(10e4), help="")
    parser.add_argument("--er", type=int, default=16, help="Number of experiences used to build a replay minibatch")
        
    parser.add_argument("--history-ar", type=int, default=20, help="Number of historic actions and rewards")
    parser.add_argument("--cnn-type", default='atari', type=str, choices=['atari', 'mnist', 'fixup'], help="General shape of the CNN, if any. Either DQN-Like, or image-classification-like with more layers")
    parser.add_argument("--hidden", default=256, type=int, help="Hidden neurons of the policy network")
    parser.add_argument("--layers", default=1, type=int, help="Number of hidden layers in the networks")
    parser.add_argument("--state-layers", default=0, type=int, help="Number of hidden layers in the state networks")
    parser.add_argument("--lr", default=1e-5, type=float, help="Learning rate of the neural network")
    parser.add_argument("--eps", default=0.1, type=float, help="Entropy bonus")
    parser.add_argument("--load_pretain", type=str, default="treechop/nec_model_softq_atari_fine-pretrain-softq", help="File from which to load the neural network weights")#, default="treechop/nec_model_softq_atari_fine-pretrain-softq"
    parser.add_argument("--load", type=str, default="treechop/nec_model_iqn_atari", help="File from which to load the neural network weights")#, default="main_model"
    parser.add_argument("--save", type=str, help="Basename of saved weight files. If not given, nothing is saved")#, default="main_model"
        
    parser.add_argument("--alr", type=float, default=0.05, help="Actor learning rate")

    parser.add_argument("--neural-episodic-control", action='store_true', default=True, help="")
    parser.add_argument("--nec-lr", type=float, default=1e-4, help="NEC learning rate")

    #IQN stuff
    parser.add_argument("--kappa", type=float, default=1.0, help="Huber loss kappa")
    parser.add_argument('--N_tau', type=int, default=8, help='Paper N')
    parser.add_argument('--Np_tau', type=int, default=8, help='Paper Nprime')

    HashingMemory.register_args(parser)

    args = parser.parse_args()

    HashingMemory.check_params(args)


    traj_gen_expert = ExpertGenerator(args)
    iqn = IQN(traj_gen_expert.state_shape, traj_gen_expert.num_actions, args)
    iqn.loadstore(args.load)

    keys_data = iqn._a['modules']['actor_model'].state_dict()['keys'].data.cpu().numpy()

    print(keys_data.shape)
    print("Oke")

    for i in range(0, 2):
        for j in range(0, 2):
            keys_embed = TSNE(n_components=2).fit_transform(keys_data[i,j])

            #print(keys_embed.shape)
            print(i, j)
            plt.scatter(keys_embed[:,0], keys_embed[:,1])
            plt.show()




if __name__ == '__main__':
    main()