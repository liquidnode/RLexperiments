from trainers import Trainer
from replay_buffer import ReplayBuffer
from traj_generator import ExpertGenerator, ActorTrajGenerator
from memory import HashingMemory
import argparse
from torch.multiprocessing import Pipe, Queue
import time

def main():
    parser = argparse.ArgumentParser(description="Feudal Network")

    parser.add_argument("--pretrain", action='store_true', default=False, help="")
    parser.add_argument("--agent_name", type=str, default='FeudalNet', help="Model name")
    parser.add_argument("--agent_from", type=str, default='hier_models', help="Model python module")
    parser.add_argument("--needs-next", action='store_true', default=True, help="")
    parser.add_argument("--env", default='MineRLTreechop-v0', type=str, help="Gym environment to use")#MineRLNavigateDense-v0#MineRLTreechop-v0
    parser.add_argument("--minerl", action='store_true', default=True, help="The environment is a MineRL environment")
    parser.add_argument("--name", type=str, default='treechop', help="Experiment name")
    parser.add_argument("--episodes", type=int, default=500, help="Number of episodes to run")

    parser.add_argument("--erpoolsize", type=int, default=int(4e5), help="Number of experiences stored by each option for experience replay")
    parser.add_argument("--packet-size", type=int, default=int(256), help="")
    parser.add_argument("--sample-efficiency", type=int, default=int(8), help="")
    parser.add_argument("--chunk-size", type=int, default=int(100), help="")
    parser.add_argument("--num-replay-workers", type=int, default=1, help="Number of replay buffer workers")
    parser.add_argument("--per", default=False, action="store_true", help="Use prioritized experience replay.")
    parser.add_argument("--trainstart", type=int, default=int(4e4), help="")
    parser.add_argument("--er", type=int, default=64, help="Number of experiences used to build a replay minibatch")
        
    parser.add_argument("--history-ar", type=int, default=0, help="Number of historic actions and rewards")
    parser.add_argument("--cnn-type", default='atari', type=str, choices=['atari', 'mnist', 'fixup'], help="General shape of the CNN, if any. Either DQN-Like, or image-classification-like with more layers")
    parser.add_argument("--hidden", default=256, type=int, help="Hidden neurons of the policy network")
    parser.add_argument("--layers", default=1, type=int, help="Number of hidden layers in the networks")
    parser.add_argument("--state-layers", default=0, type=int, help="Number of hidden layers in the state networks")
    parser.add_argument("--lr", default=1e-4, type=float, help="Learning rate of the neural network")
    parser.add_argument("--eps", default=0.1, type=float, help="Entropy bonus")
    parser.add_argument("--load_pretain", type=str, help="File from which to load the neural network weights")#, default="treechop/nec_model_softq_atari_fine-pretrain-softq"
    parser.add_argument("--load", type=str, help="File from which to load the neural network weights")#, default="main_model"
    parser.add_argument("--save", type=str, default="treechop/model_atari", help="Basename of saved weight files. If not given, nothing is saved")#, default="main_model"

    parser.add_argument("--clr", type=float, default=0.2, help="Critic learning rate")
    parser.add_argument("--nec-lr", type=float, default=1e-3, help="NEC learning rate")

    parser.add_argument("--skip", type=int, default=10, help="Skip inside trajectories")
    parser.add_argument("--needs_skip", default=True, action="store_true", help="")

    #IQN stuff
    parser.add_argument("--kappa", type=float, default=1.0, help="Huber loss kappa")
    parser.add_argument('--N_tau', type=int, default=8, help='Paper N')
    parser.add_argument('--Np_tau', type=int, default=8, help='Paper Nprime')

    HashingMemory.register_args(parser)

    args = parser.parse_args()

    HashingMemory.check_params(args)

    if False:
        #Best bis jetzt 6.5 reward im durchscnitt
        copy_queues = [Queue(1)]
        traj_gen_actor = ActorTrajGenerator(args, copy_queues[0])
        replay_actor = ReplayBuffer(args, traj_gen_actor.data_description(), traj_gen_actor.traj_pipe[0])
        desc = traj_gen_actor.data_description()
        del desc['action']
        skip_replay_actor = ReplayBuffer(args, desc, traj_gen_actor.traj_pipe[1], blocking=False)
        args.sample_efficiency = 1
        traj_gen_expert = ExpertGenerator(args)
        replay_expert = ReplayBuffer(args, traj_gen_actor.data_description(), traj_gen_expert.traj_pipe[0])
        desc = traj_gen_actor.data_description()
        del desc['action']
        skip_replay_expert = ReplayBuffer(args, desc, traj_gen_expert.traj_pipe[1], blocking=False)

        trainer = Trainer("FeudalNet", "hier_models", traj_gen_actor.state_shape, traj_gen_actor.num_actions, args, 
                            [replay_actor.batch_queues, 
                            replay_expert.batch_queues, 
                            skip_replay_actor.batch_queues, 
                            skip_replay_expert.batch_queues], copy_queues)
    else:
        args.needs_skip = False
        args.agent_name = "SimpleIQN"
        copy_queues = [Queue(1)]
        traj_gen_actor = ActorTrajGenerator(args, copy_queues[0])
        replay_actor = ReplayBuffer(args, traj_gen_actor.data_description(), traj_gen_actor.traj_pipe[0])
        args.sample_efficiency = 1
        traj_gen_expert = ExpertGenerator(args)
        replay_expert = ReplayBuffer(args, traj_gen_actor.data_description(), traj_gen_expert.traj_pipe[0])
        
        trainer = Trainer("SimpleIQN", "hier_models", traj_gen_actor.state_shape, traj_gen_actor.num_actions, args, 
                            [replay_actor.batch_queues, 
                            replay_expert.batch_queues], copy_queues)

    while True:
        time.sleep(1)
        #print(replay_actor.batch_queues[0].qsize())
        #print(replay_expert.batch_queues[0].qsize())
        #print(skip_replay_actor.batch_queues[0].qsize())
        #print(skip_replay_expert.batch_queues[0].qsize())

if __name__ == '__main__':
    main()