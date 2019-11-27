from trainers import ActorTrainer, CriticTrainer, SoftQTrainer, IQNTrainer, Trainer
from replay_buffer import ReplayBuffer
from traj_generator import ExpertGenerator, ActorTrajGenerator, EntropyPlot, ClusteringPlot, RealExpertGenerator
from memory import HashingMemory
import argparse
from torch.multiprocessing import Pipe, Queue
import time
from models import IQN
from hier_models import IQNValHier
import models
from sklearn.manifold import TSNE

class CombineQueue():
    def __init__(self, queue1, queue2):
        self.queue1 = queue1
        self.queue2 = queue2
        self.switch = False

    def get(self):
        self.switch = not self.switch
        if self.switch:
            return self.queue1.get()
        else:
            return self.queue2.get()


def main():
    parser = argparse.ArgumentParser(description="Distributed BDPI")

    DEBUG = False

    PRETRAIN_ACTOR = False
    PRETRAIN_CRITIC = False
    HALLWAY = False
    LARGEGRID = False
    SOFTQ_TRAIN = False
    PRETRAIN = True
    IQN_TRAIN = False
    IQN_HIER_TRAIN = False
    ADVANCED_BC = False
    SIMPLE_BC = True
    FEATURE_BC = False
    if PRETRAIN_ACTOR:
        parser.add_argument("--pretrain", action='store_true', default=True, help="")
        parser.add_argument("--softq", action='store_true', default=False, help="")
        parser.add_argument("--needs-next", action='store_true', default=False, help="")
        parser.add_argument("--env", default='MineRLTreechop-v0', type=str, help="Gym environment to use")#MineRLNavigateDense-v0#MineRLTreechop-v0
        parser.add_argument("--minerl", action='store_true', default=True, help="The environment is a MineRL environment")
        parser.add_argument("--name", type=str, default='', help="Experiment name")

        parser.add_argument("--erpoolsize", type=int, default=int(4e5), help="Number of experiences stored by each option for experience replay")
        parser.add_argument("--packet-size", type=int, default=int(256), help="")
        parser.add_argument("--sample-efficiency", type=int, default=int(1), help="")
        parser.add_argument("--chunk-size", type=int, default=int(100), help="")
        parser.add_argument("--num-replay-workers", type=int, default=4, help="Number of replay buffer workers")
        parser.add_argument("--per", default=False, action="store_true", help="Use prioritized experience replay.")
        parser.add_argument("--trainstart", type=int, default=int(2e5), help="")
        parser.add_argument("--er", type=int, default=256, help="Number of experiences used to build a replay minibatch")

        parser.add_argument("--cnn-type", default='atari', type=str, choices=['atari', 'mnist', 'fixup'], help="General shape of the CNN, if any. Either DQN-Like, or image-classification-like with more layers")
        parser.add_argument("--hidden", default=256, type=int, help="Hidden neurons of the policy network")
        parser.add_argument("--layers", default=1, type=int, help="Number of hidden layers in the networks")
        parser.add_argument("--state-layers", default=0, type=int, help="Number of hidden layers in the state networks")
        parser.add_argument("--lr", default=1e-4, type=float, help="Learning rate of the neural network")
        parser.add_argument("--load", type=str, help="File from which to load the neural network weights")#, default="main_model"
        parser.add_argument("--save", type=str, default="be_cloning/pretrained_model_nec_atari3", help="Basename of saved weight files. If not given, nothing is saved")#, default="main_model"

        parser.add_argument("--neural-episodic-control", action='store_true', default=True, help="")
        parser.add_argument("--nec-lr", type=float, default=1e-3, help="NEC learning rate")

    if PRETRAIN_CRITIC:
        parser.add_argument("--pretrain", action='store_true', default=True, help="")
        parser.add_argument("--softq", action='store_true', default=False, help="")
        parser.add_argument("--needs-next", action='store_true', default=True, help="")
        parser.add_argument("--env", default='MineRLTreechop-v0', type=str, help="Gym environment to use")#MineRLNavigateDense-v0#MineRLTreechop-v0
        parser.add_argument("--minerl", action='store_true', default=True, help="The environment is a MineRL environment")
        parser.add_argument("--name", type=str, default='', help="Experiment name")

        parser.add_argument("--erpoolsize", type=int, default=int(4e5), help="Number of experiences stored by each option for experience replay")
        parser.add_argument("--packet-size", type=int, default=int(256), help="")
        parser.add_argument("--sample-efficiency", type=int, default=int(64), help="")
        parser.add_argument("--chunk-size", type=int, default=int(100), help="")
        parser.add_argument("--num-replay-workers", type=int, default=4, help="Number of replay buffer workers")
        parser.add_argument("--per", default=False, action="store_true", help="Use prioritized experience replay.")
        parser.add_argument("--trainstart", type=int, default=int(1e4), help="")
        parser.add_argument("--er", type=int, default=256, help="Number of experiences used to build a replay minibatch")

        parser.add_argument("--cnn-type", default='atari', type=str, choices=['atari', 'mnist', 'fixup'], help="General shape of the CNN, if any. Either DQN-Like, or image-classification-like with more layers")
        parser.add_argument("--hidden", default=256, type=int, help="Hidden neurons of the policy network")
        parser.add_argument("--layers", default=1, type=int, help="Number of hidden layers in the networks")
        parser.add_argument("--state-layers", default=0, type=int, help="Number of hidden layers in the state networks")
        parser.add_argument("--lr", default=1e-5, type=float, help="Learning rate of the neural network")
        parser.add_argument("--load", type=str, help="File from which to load the neural network weights")#, default="main_model"
        parser.add_argument("--pretrain-load", type=str, default="be_cloning/pretrained_model_nec_atari2", help="File from which to load the neural network weights")
        parser.add_argument("--save", type=str, default="be_cloning/pretrained_model_nec_atari2", help="Basename of saved weight files. If not given, nothing is saved")#, default="main_model"

        parser.add_argument("--cepochs", type=int, default=20, help="Number of epochs used to fit the critic")
        parser.add_argument("--q-loops", type=int, default=4, help="Number of training iterations performed on the critic for each training epoch")
        parser.add_argument("--clr", type=float, default=0.2, help="Critic learning rate")

        parser.add_argument("--neural-episodic-control", action='store_true', default=True, help="")
        parser.add_argument("--nec-lr", type=float, default=1e-4, help="NEC learning rate")

    if HALLWAY:
        parser.add_argument("--pretrain", action='store_true', default=False, help="")
        parser.add_argument("--softq", action='store_true', default=False, help="")
        parser.add_argument("--needs-next", action='store_true', default=True, help="")
        parser.add_argument("--env", default='MiniWorld-Hallway-v0', type=str, help="Gym environment to use")
        parser.add_argument("--minerl", action='store_true', default=False, help="The environment is a MineRL environment")
        parser.add_argument("--name", type=str, default='hallway', help="Experiment name")
        parser.add_argument("--episodes", type=int, default=500, help="Number of episodes to run")

        parser.add_argument("--erpoolsize", type=int, default=int(20000), help="Number of experiences stored by each option for experience replay")
        parser.add_argument("--packet-size", type=int, default=int(32), help="")
        parser.add_argument("--sample-efficiency", type=int, default=int(32*16), help="")
        parser.add_argument("--chunk-size", type=int, default=int(200), help="")
        parser.add_argument("--num-replay-workers", type=int, default=1, help="Number of replay buffer workers")
        parser.add_argument("--num-critic-workers", type=int, default=1, help="Number of critic train workers")
        parser.add_argument("--per", default=False, action="store_true", help="Use prioritized experience replay.")
        parser.add_argument("--trainstart", type=int, default=int(256), help="")
        parser.add_argument("--er", type=int, default=256, help="Number of experiences used to build a replay minibatch")

        parser.add_argument("--cnn-type", default='atari', type=str, choices=['atari', 'mnist', 'fixup'], help="General shape of the CNN, if any. Either DQN-Like, or image-classification-like with more layers")
        parser.add_argument("--hidden", default=256, type=int, help="Hidden neurons of the policy network")
        parser.add_argument("--layers", default=1, type=int, help="Number of hidden layers in the networks")
        parser.add_argument("--state-layers", default=0, type=int, help="Number of hidden layers in the state networks")
        parser.add_argument("--lr", default=1e-5, type=float, help="Learning rate of the neural network")
        parser.add_argument("--load", type=str, help="File from which to load the neural network weights")#, default="main_model"
        parser.add_argument("--pretrain-load", type=str, help="File from which to load the neural network weights")
        parser.add_argument("--save", type=str, default="hallway/model", help="Basename of saved weight files. If not given, nothing is saved")#, default="main_model"

        parser.add_argument("--critic-count", type=int, default=16, help="Number of critics used by BDPI")
        parser.add_argument("--aepochs", type=int, default=1, help="Number of epochs used to fit the actor")
        parser.add_argument("--cepochs", type=int, default=1, help="Number of epochs used to fit the critic")
        parser.add_argument("--q-loops", type=int, default=1, help="Number of training iterations performed on the critic for each training epoch")
        parser.add_argument("--alr", type=float, default=0.05, help="Actor learning rate")
        parser.add_argument("--clr", type=float, default=0.2, help="Critic learning rate")

        parser.add_argument("--neural-episodic-control", action='store_true', default=False, help="")
        parser.add_argument("--nec-lr", type=float, default=1e-4, help="NEC learning rate")

    if LARGEGRID:
        parser.add_argument("--pretrain", action='store_true', default=False, help="")
        parser.add_argument("--softq", action='store_true', default=False, help="")
        parser.add_argument("--needs-next", action='store_true', default=True, help="")
        parser.add_argument("--env", default='LargeGrid-v0', type=str, help="Gym environment to use")
        parser.add_argument("--minerl", action='store_true', default=False, help="The environment is a MineRL environment")
        parser.add_argument("--name", type=str, default='largegrid', help="Experiment name")
        parser.add_argument("--episodes", type=int, default=100, help="Number of episodes to run")

        parser.add_argument("--erpoolsize", type=int, default=int(20000), help="Number of experiences stored by each option for experience replay")
        parser.add_argument("--packet-size", type=int, default=int(1), help="")
        parser.add_argument("--sample-efficiency", type=int, default=int(16), help="")
        parser.add_argument("--chunk-size", type=int, default=int(200), help="")
        parser.add_argument("--num-replay-workers", type=int, default=1, help="Number of replay buffer workers")
        parser.add_argument("--num-critic-workers", type=int, default=1, help="Number of critic train workers")
        parser.add_argument("--per", default=False, action="store_true", help="Use prioritized experience replay.")
        parser.add_argument("--trainstart", type=int, default=int(256), help="")
        parser.add_argument("--er", type=int, default=256, help="Number of experiences used to build a replay minibatch")

        parser.add_argument("--cnn-type", default='atari', type=str, choices=['atari', 'mnist', 'fixup'], help="General shape of the CNN, if any. Either DQN-Like, or image-classification-like with more layers")
        parser.add_argument("--hidden", default=32, type=int, help="Hidden neurons of the policy network")
        parser.add_argument("--layers", default=1, type=int, help="Number of hidden layers in the networks")
        parser.add_argument("--state-layers", default=0, type=int, help="Number of hidden layers in the state networks")
        parser.add_argument("--lr", default=1e-4, type=float, help="Learning rate of the neural network")
        parser.add_argument("--load", type=str, help="File from which to load the neural network weights")#, default="main_model"
        parser.add_argument("--pretrain-load", type=str, help="File from which to load the neural network weights")
        parser.add_argument("--save", type=str, default='largegrid/lamodel', help="Basename of saved weight files. If not given, nothing is saved")#, default="main_model"

        parser.add_argument("--critic-count", type=int, default=16, help="Number of critics used by BDPI")
        parser.add_argument("--aepochs", type=int, default=20, help="Number of epochs used to fit the actor")
        parser.add_argument("--cepochs", type=int, default=20, help="Number of epochs used to fit the critic")
        parser.add_argument("--q-loops", type=int, default=4, help="Number of training iterations performed on the critic for each training epoch")
        parser.add_argument("--alr", type=float, default=0.05, help="Actor learning rate")
        parser.add_argument("--clr", type=float, default=0.2, help="Critic learning rate")

        parser.add_argument("--neural-episodic-control", action='store_true', default=False, help="")
        parser.add_argument("--nec-lr", type=float, default=1e-4, help="NEC learning rate")
        
    if SOFTQ_TRAIN:
        parser.add_argument("--agent_name", type=str, default='SoftQ', help="Model name")
        parser.add_argument("--agent_from", type=str, default='models', help="Model python module")
        parser.add_argument("--pretrain", action='store_true', default=PRETRAIN, help="")
        parser.add_argument("--time", action='store_true', default=False, help="")
        parser.add_argument("--softq", action='store_true', default=True, help="")
        parser.add_argument("--needs-next", action='store_true', default=True, help="")
        parser.add_argument("--env", default='MineRLTreechop-v0', type=str, help="Gym environment to use")#MineRLNavigateDense-v0#MineRLTreechop-v0
        parser.add_argument("--minerl", action='store_true', default=True, help="The environment is a MineRL environment")
        parser.add_argument("--name", type=str, default='treechop', help="Experiment name")
        parser.add_argument("--episodes", type=int, default=500, help="Number of episodes to run")

        parser.add_argument("--erpoolsize", type=int, default=int(4e5), help="Number of experiences stored by each option for experience replay")
        parser.add_argument("--packet-size", type=int, default=int(256), help="")
        parser.add_argument("--sample-efficiency", type=int, default=int(64//4), help="")
        parser.add_argument("--chunk-size", type=int, default=int(100), help="")
        parser.add_argument("--num-replay-workers", type=int, default=2, help="Number of replay buffer workers")
        parser.add_argument("--per", default=False, action="store_true", help="Use prioritized experience replay.")
        parser.add_argument("--trainstart", type=int, default=int(1e4), help="")
        parser.add_argument("--er", type=int, default=32*4, help="Number of experiences used to build a replay minibatch")
        
        parser.add_argument("--history-ar", type=int, default=20, help="Number of historic actions and rewards")
        parser.add_argument("--cnn-type", default='atari', type=str, choices=['atari', 'mnist', 'fixup'], help="General shape of the CNN, if any. Either DQN-Like, or image-classification-like with more layers")
        parser.add_argument("--hidden", default=256, type=int, help="Hidden neurons of the policy network")
        parser.add_argument("--layers", default=1, type=int, help="Number of hidden layers in the networks")
        parser.add_argument("--state-layers", default=0, type=int, help="Number of hidden layers in the state networks")
        parser.add_argument("--lr", default=1e-4, type=float, help="Learning rate of the neural network")
        parser.add_argument("--eps", default=0.05, type=float, help="Entropy bonus")
        parser.add_argument("--load", type=str, default="treechop/nec_model_softq_atari_fine", help="File from which to load the neural network weights")#, default="main_model"
        parser.add_argument("--save", type=str, help="Basename of saved weight files. If not given, nothing is saved")#, default="main_model"
        
        parser.add_argument("--clr", type=float, default=0.8, help="Critic learning rate")

        parser.add_argument("--neural-episodic-control", action='store_true', default=True, help="")
        parser.add_argument("--nec-lr", type=float, default=1e-3, help="NEC learning rate")

    if IQN_TRAIN or IQN_HIER_TRAIN:
        parser.add_argument("--pretrain", action='store_true', default=PRETRAIN, help="")
        parser.add_argument("--time", action='store_true', default=False, help="")
        parser.add_argument("--softq", action='store_true', default=False, help="")
        parser.add_argument("--iqn", action='store_true', default=True, help="")
        parser.add_argument("--agent_name", type=str, default='IQNValHier', help="Model name")
        parser.add_argument("--agent_from", type=str, default='hier_models', help="Model python module")
        parser.add_argument("--needs-next", action='store_true', default=True, help="")
        parser.add_argument("--env", default='MineRLTreechop-v0', type=str, help="Gym environment to use")#MineRLNavigateDense-v0#MineRLTreechop-v0
        parser.add_argument("--minerl", action='store_true', default=True, help="The environment is a MineRL environment")
        parser.add_argument("--name", type=str, default='treechop', help="Experiment name")
        parser.add_argument("--episodes", type=int, default=500, help="Number of episodes to run")

        parser.add_argument("--erpoolsize", type=int, default=int(2e5), help="Number of experiences stored by each option for experience replay")
        parser.add_argument("--packet-size", type=int, default=int(256), help="")
        parser.add_argument("--sample-efficiency", type=int, default=int(16), help="")
        parser.add_argument("--chunk-size", type=int, default=int(100), help="")
        parser.add_argument("--num-replay-workers", type=int, default=2, help="Number of replay buffer workers")
        parser.add_argument("--per", default=False, action="store_true", help="Use prioritized experience replay.")
        parser.add_argument("--trainstart", type=int, default=int(1e4), help="")
        parser.add_argument("--er", type=int, default=64, help="Number of experiences used to build a replay minibatch")
        
        parser.add_argument("--history-ar", type=int, default=2, help="Number of historic actions and rewards")
        parser.add_argument("--cnn-type", default='atari', type=str, choices=['atari', 'mnist', 'fixup'], help="General shape of the CNN, if any. Either DQN-Like, or image-classification-like with more layers")
        parser.add_argument("--hidden", default=256, type=int, help="Hidden neurons of the policy network")
        parser.add_argument("--layers", default=1, type=int, help="Number of hidden layers in the networks")
        parser.add_argument("--state-layers", default=0, type=int, help="Number of hidden layers in the state networks")
        parser.add_argument("--lr", default=1e-4, type=float, help="Learning rate of the neural network")
        parser.add_argument("--eps", default=0.1, type=float, help="Entropy bonus")
        parser.add_argument("--load_pretain", type=str, help="File from which to load the neural network weights")#, default="treechop/nec_model_softq_atari_fine-pretrain-softq"
        parser.add_argument("--load", type=str, default="treechop/nec_model_iqn_hier_atari", help="File from which to load the neural network weights")#, default="main_model"
        parser.add_argument("--save", type=str, default="treechop/nec_model_iqn_hier_atari", help="Basename of saved weight files. If not given, nothing is saved")#, default="main_model"
        
        parser.add_argument("--alr", type=float, default=0.05, help="Actor learning rate")

        parser.add_argument("--neural-episodic-control", action='store_true', default=True, help="")
        parser.add_argument("--nec-lr", type=float, default=1e-3, help="NEC learning rate")

        #IQN stuff
        parser.add_argument("--kappa", type=float, default=1.0, help="Huber loss kappa")
        parser.add_argument('--N_tau', type=int, default=16, help='Paper N')
        parser.add_argument('--Np_tau', type=int, default=16, help='Paper Nprime')

    if ADVANCED_BC or SIMPLE_BC or FEATURE_BC:
        parser.add_argument("--pretrain", action='store_true', default=True, help="")
        parser.add_argument("--time", action='store_true', default=False, help="")
        parser.add_argument("--softq", action='store_true', default=False, help="")
        parser.add_argument("--iqn", action='store_true', default=False, help="")
        parser.add_argument("--needs-next", action='store_true', default=False, help="")
        parser.add_argument("--env", default='MineRLTreechop-v0', type=str, help="Gym environment to use")#MineRLNavigateDense-v0#MineRLTreechop-v0
        parser.add_argument("--minerl", action='store_true', default=True, help="The environment is a MineRL environment")
        parser.add_argument("--name", type=str, default='treechop', help="Experiment name")
        parser.add_argument("--episodes", type=int, default=500, help="Number of episodes to run")

        parser.add_argument("--erpoolsize", type=int, default=int(2e5), help="Number of experiences stored by each option for experience replay")
        parser.add_argument("--packet-size", type=int, default=int(256), help="")
        parser.add_argument("--sample-efficiency", type=int, default=int(1), help="")
        parser.add_argument("--chunk-size", type=int, default=int(100), help="")
        parser.add_argument("--num-replay-workers", type=int, default=4, help="Number of replay buffer workers")
        parser.add_argument("--per", default=False, action="store_true", help="Use prioritized experience replay.")
        parser.add_argument("--trainstart", type=int, default=int(1e5), help="")
        parser.add_argument("--er", type=int, default=256, help="Number of experiences used to build a replay minibatch")
        
        parser.add_argument("--history-ar", type=int, default=20, help="Number of historic actions and rewards")
        parser.add_argument("--cnn-type", default='atari', type=str, choices=['atari', 'mnist', 'fixup'], help="General shape of the CNN, if any. Either DQN-Like, or image-classification-like with more layers")
        parser.add_argument("--hidden", default=256, type=int, help="Hidden neurons of the policy network")
        parser.add_argument("--layers", default=1, type=int, help="Number of hidden layers in the networks")
        parser.add_argument("--state-layers", default=0, type=int, help="Number of hidden layers in the state networks")
        parser.add_argument("--lr", default=1e-5, type=float, help="Learning rate of the neural network")
        parser.add_argument("--eps", default=0.1, type=float, help="Entropy bonus")
        parser.add_argument("--load_pretain", type=str, help="File from which to load the neural network weights")
        parser.add_argument("--load", type=str, default="treechop/nec_model_fisherbc2_atari", help="File from which to load the neural network weights")##, default="treechop/nec_model_featurebc_atari"
        parser.add_argument("--save", type=str, help="Basename of saved weight files. If not given, nothing is saved")#, default="treechop/nec_model_featurebc_atari"

        parser.add_argument("--neural-episodic-control", action='store_true', default=True, help="")
        parser.add_argument("--nec-lr", type=float, default=1e-4, help="NEC learning rate")

    parser.add_argument("--rho", type=float, default=1e-4, help="Lambda learning rate")
    parser.add_argument("--skip", type=int, default=0, help="Skip inside real trajectories")
    parser.add_argument("--needs_skip", default=False, action="store_true", help="")

    HashingMemory.register_args(parser)

    args = parser.parse_args()

    HashingMemory.check_params(args)

    if not DEBUG:
        if PRETRAIN_ACTOR:
            traj_gen = ExpertGenerator(args)
            replay = ReplayBuffer(args, traj_gen.data_description(), traj_gen.traj_pipe[0])
            actor_trainer = ActorTrainer(traj_gen.state_shape, traj_gen.num_actions, args, replay.batch_queues)

        if PRETRAIN_CRITIC:
            traj_gen_expert = ExpertGenerator(args)
            args.load = args.save
            traj_gen_actor = ActorTrajGenerator(args)
            combined_traj = CombineQueue(traj_gen_expert.traj_pipe[0], traj_gen_actor.traj_pipe[0])
            replay = ReplayBuffer(args, traj_gen_actor.data_description(), combined_traj)
            args.load = None
            critic_trainer = CriticTrainer(traj_gen_actor.state_shape, traj_gen_actor.num_actions, args, replay.batch_queues, 0)

        if HALLWAY or LARGEGRID:
            copy_queues = [Queue(1)]
            traj_gen_actor = ActorTrajGenerator(args, copy_queues[0])
            replay = ReplayBuffer(args, traj_gen_actor.data_description(), traj_gen_actor.traj_pipe[0])
            critic_trainers = []
            critic_queues = []
            for i in range(args.num_critic_workers):
                critic_queues.append(Queue(10))
                critic_trainers.append(CriticTrainer(traj_gen_actor.state_shape, traj_gen_actor.num_actions, args, replay.batch_queues, i, critic_queues[i], args.critic_count//args.num_critic_workers))
            actor_trainer = ActorTrainer(traj_gen_actor.state_shape, traj_gen_actor.num_actions, args, critic_queues, copy_queues)
    
        if SOFTQ_TRAIN:
            if PRETRAIN:
                args.er = 128
                args.needs_next = False
                args.lr = 1e-4
                args.nec_lr = 1e-3
                args.trainstart = args.erpoolsize // 2
                args.num_replay_workers *= 2
                args.sample_efficiency = 1
                traj_gen_expert = ExpertGenerator(args)
                traj_gen_expert2 = ExpertGenerator(args)
                combined_traj = CombineQueue(traj_gen_expert.traj_pipe[0], traj_gen_expert2.traj_pipe[0])
                replay_expert = ReplayBuffer(args, traj_gen_expert.data_description(), combined_traj)
                softq_trainer = SoftQTrainer(traj_gen_expert.state_shape, traj_gen_expert.num_actions, args, replay_expert.batch_queues)
            else:
                copy_queues = [Queue(1)]
                traj_gen_actor = ActorTrajGenerator(args, copy_queues[0])
                #args.trainstart = args.erpoolsize - 1
                replay_actor = ReplayBuffer(args, traj_gen_actor.data_description(), traj_gen_actor.traj_pipe[0])
                args.sample_efficiency = 1
                traj_gen_expert = ExpertGenerator(args)
                replay_expert = ReplayBuffer(args, traj_gen_actor.data_description(), traj_gen_expert.traj_pipe[0])
                softq_trainer = SoftQTrainer(traj_gen_actor.state_shape, traj_gen_actor.num_actions, args, replay_actor.batch_queues, replay_expert.batch_queues, copy_queues)

        if IQN_TRAIN:
            copy_queues = [Queue(1)]
            traj_gen_actor = ActorTrajGenerator(args, copy_queues[0])
            replay_actor = ReplayBuffer(args, traj_gen_actor.data_description(), traj_gen_actor.traj_pipe[0])
            args.sample_efficiency = 1
            traj_gen_expert = ExpertGenerator(args)
            replay_expert = ReplayBuffer(args, traj_gen_actor.data_description(), traj_gen_expert.traj_pipe[0])
            softq_trainer = IQNTrainer(traj_gen_actor.state_shape, traj_gen_actor.num_actions, args, replay_actor.batch_queues, replay_expert.batch_queues, copy_queues)

            
        if IQN_HIER_TRAIN:
            copy_queues = [Queue(1)]
            traj_gen_actor = ActorTrajGenerator(args, copy_queues[0])
            replay_actor = ReplayBuffer(args, traj_gen_actor.data_description(), traj_gen_actor.traj_pipe[0])
            args.sample_efficiency = 1
            args.trainstart = args.erpoolsize // 2
            traj_gen_expert = ExpertGenerator(args)
            replay_expert = ReplayBuffer(args, traj_gen_actor.data_description(), traj_gen_expert.traj_pipe[0])
            args.skip = 15
            traj_skip_expert = RealExpertGenerator(args)
            args.erpoolsize = int(2e4)
            args.trainstart = args.erpoolsize // 2
            tmp = args.er
            args.er = 128
            replay_skip_expert = ReplayBuffer(args, traj_skip_expert.data_description(), traj_skip_expert.traj_pipe[0])
            args.er = tmp

            trainer = Trainer("IQNValHier", "hier_models", traj_gen_actor.state_shape, traj_gen_actor.num_actions, args, 
                              [replay_actor.batch_queues, 
                               replay_skip_expert.batch_queues, 
                               replay_expert.batch_queues], copy_queues)

        if ADVANCED_BC or SIMPLE_BC or FEATURE_BC:
            args.er = 128
            args.needs_next = SIMPLE_BC
            args.lr = 1e-4
            args.nec_lr = 1e-3
            args.trainstart = args.erpoolsize // 2
            args.sample_efficiency = 1
            if FEATURE_BC:
                args.lr = 1e-4
                args.nec_lr = 1e-3
                args.rho = 1e-5
                args.pretrain = True
                traj_gen_expert = RealExpertGenerator(args)
                traj_gen_expert2 = RealExpertGenerator(args)
            else:
                traj_gen_expert = ExpertGenerator(args)
                traj_gen_expert2 = ExpertGenerator(args)
            combined_traj = CombineQueue(traj_gen_expert.traj_pipe[0], traj_gen_expert2.traj_pipe[0])
            replay_expert = ReplayBuffer(args, traj_gen_expert.data_description(), combined_traj)
            if FEATURE_BC:
                trainer = Trainer("FeatureFisherBC", "models", traj_gen_expert.state_shape, traj_gen_expert.num_actions, args, [replay_expert.batch_queues])
            else:
                if SIMPLE_BC:
                    trainer = Trainer("FisherBC", "models", traj_gen_expert.state_shape, traj_gen_expert.num_actions, args, [replay_expert.batch_queues])
                else:
                    trainer = Trainer("AdvancedBC", "models", traj_gen_expert.state_shape, traj_gen_expert.num_actions, args, [replay_expert.batch_queues])

        while True:
            time.sleep(1)
    else:
        import torch
        import numpy as np
        import matplotlib.pyplot as plt
        if False:
            traj_gen_expert = ExpertGenerator(args)
            iqn = IQN(traj_gen_expert.state_shape, traj_gen_expert.num_actions, args)
            iqn.loadstore(args.load)

            while True:
                tau = np.arange(0.0, 1.0, 0.01)[:,np.newaxis]
                tau = models.variable(tau)

                inputs = traj_gen_expert.traj_pipe[0].get()
                state_keys = ['state', 'pov']

                states = {}
                for k in state_keys:
                    if k in inputs:
                        states[k] = models.variable(inputs[k][0,:])
                        states[k] = states[k][None, :]
                with torch.no_grad():
                    dist, _, _, _ = iqn._get_distribution(iqn._a, states, tau.size(0), tau)

                    dist = dist.data.cpu().numpy()
                    tau = tau.data.cpu().numpy()
                curraction = states['state'][0,-17:]
            
                print('fwd_back',curraction[:3])
                print('lft_rght',curraction[3:6])
                print('jump',curraction[6:8])
                print('snk_spr',curraction[8:11])
                print('camera',curraction[11:13])
                print('attack',curraction[13:15])

                labels = [
                    'commit',
                    'forward',
                    'back',
                    'left',
                    'right',
                    'jump',
                    'sneak',
                    'sprint',
                    'camera0',
                    'camera1',
                    'camera2',
                    'camera3',
                    'camera4',
                    'attack'
                    ]

                plt.subplot(1, 2, 1)
                for i in range(traj_gen_expert.num_actions):
                    plt.plot(tau[:,0], dist[0, i], label=labels[i])
                plt.legend()
                plt.subplot(1, 2, 2)
                plt.imshow(states['pov'][0].data.cpu().numpy()/255.0)
                plt.show()

                print("Oke")
        else:
            if True:
                plotter = EntropyPlot(args, "FisherBC")

                while True:
                    rewards, entropy = plotter.run_episode()
                    plt.plot(np.arange(0, len(rewards), 1), rewards, label='reward')
                    plt.plot(np.arange(0, len(rewards), 1), entropy, label='entropy')
                    plt.legend()
                    plt.show()

                    #for i in range(options.shape[1]):
                    #    plt.plot(np.arange(0, len(rewards), 1), options[:,i], label='option'+str(i))
                    #plt.legend()
                    #plt.show()
            else:
                plotter = ClusteringPlot(args, "FisherBC")
                ran = True
                features, actions, povs, states = plotter.get_features(ran)
                #subind = np.random.randint(0,features.shape[0],size=2000)
                features = features#[subind]
                actions = actions.data.cpu().numpy()#[subind]
                povs = povs.data.cpu().numpy()#[subind]
                states = states.data.cpu().numpy()#[subind]

                def KMeans(x, K=10, Niter=10, verbose=True):
                    N, D = x.shape  # Number of samples, dimension of the ambient space

                    # K-means loop:
                    # - x  is the point cloud,
                    # - cl is the vector of class labels
                    # - c  is the cloud of cluster centroids
                    start = time.time()
                    random_ind = torch.from_numpy(np.random.randint(0,N,size=K)).long()
                    if torch.cuda.is_available():
                        random_ind = random_ind.cuda()
                    c = x[random_ind, :].clone()  # Simplistic random initialization
                    x_i = x[:, None, :]  # (Npoints, 1, D)

                    for i in range(Niter):

                        c_j = c[None, :, :]  # (1, Nclusters, D)
                        D_ij = ((x_i - c_j) ** 2).sum(-1)  # (Npoints, Nclusters) symbolic matrix of squared distances
                        cl = D_ij.min(dim=1)[1].long().view(-1)  # Points -> Nearest cluster

                        Ncl = torch.bincount(cl).float()  # Class weights
                        for d in range(D):  # Compute the cluster centroids with torch.bincount:
                            c[:, d] = torch.bincount(cl, weights=x[:, d]) / Ncl

                    end = time.time()

                    if verbose:
                        print("K-means example with {:,} points in dimension {:,}, K = {:,}:".format(N, D, K))
                        print('Timing for {} iterations: {:.5f}s = {} x {:.5f}s\n'.format(
                                Niter, end - start, Niter, (end-start) / Niter))

                    return cl, c

                cl, c = KMeans(features, K=20, Niter=2000)

                cl = cl.data.cpu().numpy()

                if not ran:
                    plt.plot(np.arange(0, cl.shape[0], 1), cl)
                    plt.legend()
                    plt.show()

                if True:
                    embed = TSNE(n_components=2).fit_transform(features.data.cpu().numpy())
                    z = cl

                    plt.scatter(embed[:,0], embed[:,1], c=z)
                    plt.show()

                while True:
                    try:
                        t = int(input("Type:"))
                        ts = -1
                        while t != ts:
                            i = np.random.randint(0,cl.shape[0])
                            ts = cl[i]
                        pov = povs[i]
                        state = states[i]
                        curraction = state[-17:]
            
                        print('fwd_back',curraction[:3])
                        print('lft_rght',curraction[3:6])
                        print('jump',curraction[6:8])
                        print('snk_spr',curraction[8:11])
                        print('camera',curraction[11:13])
                        print('attack',curraction[13:15])

                        plt.imshow(pov/255.0)
                        plt.show()
                    except:
                        pass

if __name__ == '__main__':
    main()


