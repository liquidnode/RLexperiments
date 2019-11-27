import torch
import numpy as np
from memory import HashingMemory
from torch.nn import functional as F
from collections import OrderedDict
from radam import RAdam
from models import tGAMMA, CUDA, variable, LearnerModel, Lambda, swish, Flatten

GAMMA = 0.99

#modified from https://github.com/dannysdeng/dqn-pytorch/blob/master/DQN_network.py
class IQNHead(torch.nn.Module):
    def __init__(self, input_dim, num_actions, args):
        super(IQNHead, self).__init__()
        self.num_actions = num_actions
        self.args = args
        self.quantile_embedding_dim = 64
        self.pi = np.pi

        self.fc2 = torch.nn.Linear(self.args.hidden, self.num_actions)#HashingMemory.build(self.args.hidden, self.num_actions, self.args)

        self.quantile_fc0 = torch.nn.Linear(self.quantile_embedding_dim, input_dim)
        self.quantile_fc1 = torch.nn.Linear(input_dim, self.args.hidden)

        self.quantile_fc_value = torch.nn.Linear(self.args.hidden, 1)#HashingMemory.build(self.args.hidden, 1, self.args)

    def forward(self, x, num_quantiles, tau=None):
        BATCH_SIZE     = x.size(0)
        state_net_size = x.size(1)

        if tau is None:
            tau = torch.FloatTensor(BATCH_SIZE * num_quantiles, 1).to(x)
            tau.uniform_(0, 1)

        quantile_net = torch.FloatTensor([i for i in range(1, 1+self.quantile_embedding_dim)]).to(x)

        tau_expand   = tau.unsqueeze(-1).expand(-1, -1, self.quantile_embedding_dim)
        quantile_net = quantile_net.view(1, 1, -1) # [1 x 1 x 64] -->         [Batch*Np x 1 x 64]
        quantile_net = quantile_net.expand(BATCH_SIZE*num_quantiles, 1, self.quantile_embedding_dim)
        cos_tau      = torch.cos(quantile_net * self.pi * tau_expand)       # [Batch*Np x 1 x 64]
        cos_tau      = cos_tau.squeeze(1)                                   # [Batch*Np x 64]

        out          = F.relu(self.quantile_fc0(cos_tau))                   # [Batch*Np x feaSize]
        fea_tile     = x.unsqueeze(1).expand(-1, num_quantiles, -1)         # [Batch x Np x feaSize]
        out          = out.view(BATCH_SIZE, num_quantiles, -1)              # [Batch x Np x feaSize]
        product      = (fea_tile * out).view(BATCH_SIZE*num_quantiles, -1)
        combined_fea = F.relu(self.quantile_fc1(product))                   # (Batch*atoms, hidden)

        values   = self.quantile_fc_value(combined_fea)
        values   = values.view(-1, num_quantiles).unsqueeze(1)
        
        x        = self.fc2(combined_fea)

        x_batch  = x.view(BATCH_SIZE, num_quantiles, self.num_actions)
        x_batch  = x_batch.transpose(1, 2).contiguous()

        action_component = x_batch - x_batch.mean(1, keepdim=True)
        y = values + action_component # [batch x actions x atoms]

        return y, tau, action_component

class IQNValHier():
    def __init__(self, state_shape, num_actions, args):
        assert isinstance(state_shape, dict)
        self.state_shape = state_shape
        self.num_actions = num_actions
        self.args = args
        self.last_action_index = 0

        self._setup()
        self._a, self._b = self._models

    def _setup(self):
        self._models = [self._make_model() for i in range(2)]

    def state_dict(self):
        return [self._models[0]['modules'].state_dict(), self._models[1]['modules'].state_dict()]

    def load_state_dict(self, s, strict=True):
        if isinstance(s, list):
            self._models[0]['modules'].load_state_dict(s[0], strict)
            self._models[1]['modules'].load_state_dict(s[1], strict)
        else:
            self._models[0]['modules'].load_state_dict(s, strict)

    def loadstore(self, filename, load=True):
        if load:
            if self.args.load_pretain is not None:
                pretrain_dict = torch.load(self.args.load_pretain)
                this_state_dict = self._models[0]['modules'].state_dict()
                pretain_load_tail = OrderedDict([])
                pretain_load_actor = OrderedDict([])
                for k, p in pretrain_dict.items():
                    if k.startswith('0.'):
                        k = k[2:]
                        pretain_load_tail[k] = p
                    if k.startswith('1.nec_layer_adv.'):
                        k = k.replace('1.nec_layer_adv.','')
                        pretain_load_actor[k] = p
                self._models[0]['modules']['tail_model'].load_state_dict(pretain_load_tail)
                self._models[1]['modules']['tail_model'].load_state_dict(pretain_load_tail)
                self._models[0]['modules']['actor_model'].load_state_dict(pretain_load_actor)
                self._models[1]['modules']['actor_model'].load_state_dict(pretain_load_actor)
            else:
                self.load_state_dict([torch.load(filename + '-iqnA'),
                                        torch.load(filename + '-iqnB')])
        else:
            torch.save(self._models[0]['modules'].state_dict(), filename + '-iqnA')
            torch.save(self._models[1]['modules'].state_dict(), filename + '-iqnB')

    def _make_model(self):
        tail_model = LearnerModel(self.args, self.state_shape)
        head_model = IQNHead(self.args.hidden, self.num_actions, self.args)
        value_model = torch.nn.Sequential(torch.nn.Linear(tail_model.embed_dim, self.args.hidden), Lambda(lambda x: swish(x)), torch.nn.Linear(self.args.hidden, 1))#HashingMemory.build(self.args.hidden, 1, self.args) #

        smax = torch.nn.Softmax(1)

        if torch.cuda.is_available() and CUDA:
            if smax is not None:
                smax = smax.cuda()
            tail_model = tail_model.cuda()
            head_model = head_model.cuda()
            value_model = value_model.cuda()
            

        scale = 1.0# if self.is_critic else 0.1
        if not self.args.neural_episodic_control:
            optimizer = torch.optim.Adam(tail_model.parameters()+head_model.parameters(), lr=self.args.lr*scale)
        else:
            model_params = []
            nec_value_params = []
            for name, p in tail_model.named_parameters():
                if 'values.weight' in name:
                    nec_value_params.append(p)
                else:
                    model_params.append(p)
            for name, p in head_model.named_parameters():
                if 'values.weight' in name:
                    nec_value_params.append(p)
                else:
                    model_params.append(p)
            nec_value_params.extend(smax.parameters())
            optimizer = RAdam([{'params': model_params}, 
                               {'params': nec_value_params, 'lr': self.args.nec_lr*scale}], lr=self.args.lr*scale, eps=0.01)#torch.optim.Adam
            
        #TODO add skynet penalty
        if self.args.per:
            loss = WeightedMSELoss()
        else:
            loss = torch.nn.MSELoss()#torch.nn.SmoothL1Loss()#

        ce_loss = torch.nn.CrossEntropyLoss()
        modules = torch.nn.ModuleDict()
        modules['tail_model'] = tail_model
        modules['head_model'] = head_model
        modules['value_model'] = value_model
        return {'modules': modules, 'opt': optimizer, 'loss': loss, 'ce_loss': ce_loss, 'softmax': smax, 'mse_loss': torch.nn.MSELoss()}

    def select_action(self, state):
        for k in state:
            state[k] = state[k][None,:]
        with torch.no_grad():
            dist,_,_,_ = self._get_distribution(self._a, variable(state), self.args.N_tau)
            vals = torch.mean(dist, dim=2)
            action_index = vals.max(1)[1][0].data.cpu().numpy()
        if action_index == 0 and self.last_action_index == 0:
            return np.random.randint(0, self.num_actions), -1.0
        self.last_action_index = action_index
        return action_index, {'val': vals.max(1)[0][0].data.cpu().numpy()}

    def _get_distribution(self, model, state, num_tau, tau=None):
        tail = model['modules']['tail_model'](state)
        dist, Ttau, action_comp = model['modules']['head_model'](tail, num_tau, tau)
        return dist, Ttau, tail, action_comp

    def _get_next_distribution(self, next_states, rewards, next_indices):
        def get_argmax(next_states, batch_size):
            next_dist, Ttau, _, _ = self._get_distribution(self._a, next_states, self.args.Np_tau)
            combined  = next_dist.mean(dim=2) 
            next_Q_sa = combined.max(1)[1]   # next_Q_sa is of size: [batch ] of action index 
            next_Q_sa = next_Q_sa.view(batch_size, 1, 1)  # Make it to be size of [32 x 1 x 1]
            next_Q_sa = next_Q_sa.expand(-1, -1, self.args.Np_tau)  # Expand to be [32 x 1 x 51], one action, expand to support
            return next_Q_sa, next_dist, Ttau

        with torch.no_grad():
            action_argmax, next_yA, Ttau = get_argmax(next_states, rewards.size(0))
            next_yB, _, _, _ = self._get_distribution(self._b, next_states, self.args.Np_tau, Ttau)
            next_y = torch.min(next_yA, next_yB)
            quantiles_next = torch.clone(rewards).unsqueeze(dim=-1).expand(-1, self.args.Np_tau)

            quantiles_next[next_indices] += (GAMMA * next_y.gather(1, action_argmax).squeeze(1))[next_indices]
            #quantiles_next = quantiles_next.clamp(0.0, 1.0)
            return quantiles_next.detach()

    def _get_value(self, model, inp):
        if 'real_pov' in inp:
            x = inp['real_pov']
        else:
            x = inp['pov']
        x = (x.permute(0, 3, 1, 2)/255.0)*2.0 - 1.0
        cnn_fea = model['modules']['tail_model'].cnn_model(x)
        return model['modules']['value_model'](cnn_fea).squeeze(1)

    def train(self, input, real_input, expert_input=None):
        if self.args.per:
            assert False #not implemented yet
        else:
            is_weights = None
        state_keys = ['state', 'pov']
        self._a['opt'].zero_grad()

        states = {}
        for k in state_keys:
            if k in input:
                states[k] = variable(input[k])
        next_states = {}
        for k in state_keys:
            next_k = 'next_'+k
            if next_k in input:
                next_states[k] = variable(input[next_k])
        actions = variable(input['action'], True).long()
        rewards = variable(input['reward'])
        done = input['done'].data.cpu().numpy()
        next_indexes = np.nonzero(np.logical_not(done))[0]
        batch_size = done.shape[0]

        if expert_input is not None:
            expert_states = {}
            for k in state_keys:
                if k in input:
                    expert_states[k] = variable(expert_input[k])
            expert_next_states = {}
            for k in state_keys:
                next_k = 'next_'+k
                if next_k in expert_input:
                    expert_next_states[k] = variable(expert_input[next_k])


            expert_actions = variable(expert_input['action'], True).long()
            expert_rewards = variable(expert_input['reward'])
            expert_done = expert_input['done'].data.cpu().numpy()

            #concat everything
            for k in states:
                states[k] = torch.cat([expert_states[k],states[k]], dim=0)
            for k in next_states:
                next_states[k] = torch.cat([expert_states[k],next_states[k]], dim=0)
            actions = torch.cat([expert_actions, actions], dim=0)
            rewards = torch.cat([expert_rewards, rewards], dim=0)

            done = np.concatenate([expert_done, done], axis=0)
            next_indexes = np.nonzero(np.logical_not(done))[0]

        #train hierarchical value net
        real_state_keys = ['real_state', 'real_pov']
        real_states = {}
        for k in real_state_keys:
            if k in real_input:
                real_states[k] = variable(real_input[k])
        next_real_states = {}
        for k in real_state_keys:
            next_k = 'next_'+k
            if next_k in real_input:
                next_real_states[k] = variable(real_input[next_k])
        real_rewards = variable(real_input['real_reward'])
        real_done = real_input['real_done'].data.cpu().numpy()
        next_real_indexes = np.nonzero(np.logical_not(real_done))[0]
        value = self._get_value(self._a, real_states)
        target = real_rewards
        with torch.no_grad():
            target[next_real_indexes] += ((tGAMMA**(self.args.skip+1))*torch.min(self._get_value(self._a, next_real_states),
                                                                                 self._get_value(self._b, next_real_states)))[next_real_indexes]
        target = target.detach()
        loss_value = self._a['mse_loss'](value, target)
        allloss = loss_value

        #modify rewards
        with torch.no_grad():
            phi_s_prime = self._get_value(self._a, next_states)
            phi_s = self._get_value(self._a, states)
        mod_rewards = phi_s_prime.detach() - phi_s.detach()#GAMMA * 
        rewards += mod_rewards

        #train IQN
        y, my_tau, tail, action_comp = self._get_distribution(self._a, states, self.args.N_tau)
        iqn_actions = actions.unsqueeze(dim=-1).unsqueeze(dim=-1).expand(-1, -1, self.args.N_tau)
        quantiles = y.gather(1, iqn_actions).squeeze(1)
        quantiles_next = self._get_next_distribution(next_states, rewards, next_indexes)

        quantiles      = quantiles.unsqueeze(-1)
        quantiles_next = quantiles_next.unsqueeze(-1)

        diff = quantiles_next.unsqueeze(2) - quantiles.unsqueeze(1)

        def huber(x, k=1.0):
            return torch.where(x.abs() < k, 0.5 * x.pow(2), k * (x.abs() - 0.5 * k))
        huber_diff = huber(diff, self.args.kappa)

        my_tau = my_tau.view(y.shape[0], self.args.N_tau, 1)   # [N_tau x Batch x 1]
        my_tau = my_tau.unsqueeze(1) # [Batch x 1 x N_tau x 1]
        my_tau = my_tau.expand(-1, self.args.Np_tau, -1, -1) # [Batch x Np_tau x N_tau x 1]

        loss = (huber_diff * (my_tau - (diff<0).float()).abs()) / self.args.kappa

        loss = loss.squeeze(3).sum(-1).mean(-1)

        #loss += ((action_comp < 0).float() * action_comp.pow(2)).mean() * 5.0

        if self.args.per:
            loss_PER = loss.detach().abs().cpu().numpy() 
            loss = loss * batch_weight_IS

        loss = loss.mean()
        ds = y.detach() * 1.0 / y.size(1)
        Q_sa = ds.sum(dim=2).gather(1, actions.unsqueeze(dim=-1))

        #actor train
        #y [batch x actions x atoms]
        with torch.no_grad():
            argmax_y = y.detach().max(1)[1].unsqueeze(1) # [batch x 1 x atoms]
            probas = torch.zeros_like(y)
            probas.scatter_(1, argmax_y, 1.0)
            probas = torch.mean(probas, dim=2) #[batch x actions]
                
        if False:
            actor_logits = self._a['modules']['actor_model'](tail)
            actor_probas = self._a['softmax'](actor_logits)
            with torch.no_grad():
                entropy = -torch.sum(actor_probas.detach()*torch.log(actor_probas.detach()), dim=1)
                alr_weights = torch.clamp(entropy, max=1.8)[:,None]
                alr_weights = alr_weights.detach()
                train_probas = actor_probas.detach() * (1.0 - self.args.alr * alr_weights) + self.args.alr * alr_weights * probas
            if self.args.per:
                entropy = -torch.sum(actor_probas*torch.log(actor_probas), dim=1)
                loss_actor = self._a['loss'](actor_probas, train_probas, is_weights)
            else:
                #entropy weigths
                loss_actor = self._a['loss'](actor_probas, train_probas)
            #if expert_input is not None:
            #    loss_actor += 0.05 * self._a['ce_loss'](actor_logits[:expert_done.size], expert_actions)
            if False:
                loss += loss_actor
            max_probs = torch.mean(torch.max(actor_probas,dim=1)[0])
            entropy = torch.mean(entropy)
        accuracy = torch.mean(probas[:,actions])
        allloss += loss

        allloss.backward()
        self._a['opt'].step()

        Qval = Q_sa.cpu().detach().numpy().squeeze()

        self._a, self._b = self._b, self._a

        return {'loss': loss.data.cpu().numpy(), 
                'loss_value': loss_value.data.cpu().numpy(), 
                'mod_rewards': mod_rewards.mean().data.cpu().numpy(),
                'val': np.mean(Qval),
                'accuracy': accuracy.data.cpu().numpy(),
                'TD_diff': (value-target).abs().mean().data.cpu().numpy()}







class FeudalNet():
    def __init__(self, state_shape, num_actions, args):
        assert isinstance(state_shape, dict)
        self.state_shape = state_shape
        self.num_actions = num_actions
        self.args = args
        self.state_action_dim = 17
        self.last_state = None
        self.last_action = -1

        self._setup()
        self._a = self._models[0]
        self.ABswitch = False

    def _setup(self):
        self._models = [self._make_model()]
    
    def state_dict(self):
        return [self._models[0]['modules'].state_dict()]

    def load_state_dict(self, s, strict=True):
        if isinstance(s, list):
            self._models[0]['modules'].load_state_dict(s[0], strict)
        else:
            self._models[0]['modules'].load_state_dict(s, strict)

    def loadstore(self, filename, load=True):
        if load:
            self.load_state_dict([torch.load(filename + '-fun')])
        else:
            torch.save(self._models[0]['modules'].state_dict(), filename + '-fun')

    def _make_model(self):
        cnn_layers = []
        assert len(self.state_shape['pov']) > 1
        if self.args.cnn_type == 'atari':
            sizes = [8, 4, 3]
            strides = [4, 2, 1]
            pooling = [1, 1, 1]
            filters = [32, 64, 32]
        elif self.args.cnn_type == 'mnist':
            sizes = [3, 3, 3]
            strides = [1, 1, 1]
            pooling = [2, 2, 2]
            filters = [32, 32, 32]

        in_channels = self.state_shape['pov'][-1] #HWC input

        for i in range(len(sizes)):
            cnn_layers.append(torch.nn.Conv2d(
                in_channels,
                filters[i],
                sizes[i],
                stride=strides[i],
                bias=True
            ))

            cnn_layers.append(Lambda(lambda x: swish(x)))

            if pooling[i] > 1:
                cnn_layers.append(torch.nn.MaxPool2d(pooling[i]))

            in_channels = filters[i]

        self.flatten = Flatten()
        self.cnn_model = torch.nn.Sequential(*cnn_layers)

        self.goal_embed_dim = 32
        self.inp_size = self.cnn_model(torch.zeros((1,self.state_shape['pov'][2], self.state_shape['pov'][0], self.state_shape['pov'][1]))).shape[1:]

        latent_input = self.inp_size[0]*self.inp_size[1]*self.inp_size[2]+self.state_shape['state'][0]-self.state_action_dim
        self.latent_state = torch.nn.Sequential(torch.nn.Linear(latent_input, self.args.hidden), Lambda(lambda x: swish(x)))

        #goal embedding
        self.goal_embed = torch.nn.Linear(self.args.hidden, self.goal_embed_dim, False)

        #Q networks
        self.Q_net_A = torch.nn.Linear(self.args.hidden+self.state_action_dim, self.num_actions*self.goal_embed_dim)
        self.Q_net_B = torch.nn.Linear(self.args.hidden+self.state_action_dim, self.num_actions*self.goal_embed_dim)

        #goal and value network
        self.goal_net = torch.nn.Sequential(torch.nn.Linear(self.args.hidden, self.args.hidden), Lambda(lambda x: swish(x)), torch.nn.Linear(self.args.hidden, self.args.hidden))
        self.value_net = HashingMemory.build(self.args.hidden, 1, self.args)
        
        if torch.cuda.is_available() and CUDA:
            self.cnn_model = self.cnn_model.cuda()
            self.flatten = self.flatten.cuda()
            self.latent_state = self.latent_state.cuda()
            self.goal_embed = self.goal_embed.cuda()
            self.Q_net_A = self.Q_net_A.cuda()
            self.Q_net_B = self.Q_net_B.cuda()
            self.goal_net = self.goal_net.cuda()
            self.value_net = self.value_net.cuda()
            
        modules = torch.nn.ModuleDict()
        modules['cnn_model'] = self.cnn_model
        modules['flatten'] = self.flatten
        modules['latent_state'] = self.latent_state
        modules['goal_embed'] = self.goal_embed
        modules['Q_net_A'] = self.Q_net_A
        modules['Q_net_B'] = self.Q_net_B
        modules['goal_net'] = self.goal_net
        modules['value_net'] = self.value_net
        
        model_params = []
        nec_value_params = []
        for name, p in modules.named_parameters():
            if 'values.weight' in name:
                nec_value_params.append(p)
            else:
                model_params.append(p)
        optimizer = RAdam([{'params': model_params}, 
                           {'params': nec_value_params, 'lr': self.args.nec_lr}], lr=self.args.lr)

        return {'modules': modules, 
                'opt': optimizer, 
                'loss': torch.nn.MSELoss()}

    def _get_cnn_fea(self, state):
        x = (state['pov'].permute(0, 3, 1, 2)/255.0)*2.0 - 1.0
        cnn_fea = self._a['modules']['cnn_model'](x) # [batch, channels, 4, 4]
        return cnn_fea

    def _get_latent(self, state, cnn_fea=None):
        if cnn_fea is None:
            cnn_fea = self._get_cnn_fea(state)
            cnn_fea = self._a['modules']['flatten'](cnn_fea)
        fea = torch.cat([cnn_fea, state['state'][:,:-self.state_action_dim]], dim=1)
        return self._a['modules']['latent_state'](fea)

    def _get_goal(self, l_state):
        return self._a['modules']['goal_net'](l_state)

    def _get_Q(self, inp_Q, QAB, goal, pre_value=None):
        if pre_value is None:
            pre_value = self._a['modules']['Q_net_'+QAB](inp_Q)
            pre_value = pre_value.view(-1, self.num_actions, self.goal_embed_dim)
        goal_embed = self._a['modules']['goal_embed'](goal).unsqueeze(1).expand(-1, self.num_actions, -1)
        return torch.sum(pre_value*goal_embed, dim=2), pre_value

    def train(self, input, expert_input, skip_input, skip_expert_input):
        if self.args.per:
            assert False #not implemented yet
        else:
            is_weights = None
        state_keys = ['state', 'pov']
        self._a['opt'].zero_grad()

        states = {}
        for k in state_keys:
            if k in input:
                states[k] = torch.cat([variable(input[k]), variable(expert_input[k])], dim=0)
        next_states = {}
        for k in state_keys:
            next_k = 'next_'+k
            if next_k in input:
                next_states[k] = torch.cat([variable(input[next_k]), variable(expert_input[next_k])], dim=0)
        actions = torch.cat([variable(input['action'], True).long(), variable(expert_input['action'], True).long()], dim=0)
        rewards = 10.0 * torch.cat([variable(input['reward']), variable(expert_input['reward'])], dim=0)
        done = np.concatenate([input['done'].data.cpu().numpy(), expert_input['done'].data.cpu().numpy()], axis=0)
        next_indexes = np.nonzero(np.logical_not(done))[0]
        batch_size = next_indexes.shape[0]
        QQN = np.arange(batch_size)

        #get latent states
        l_state = self._get_latent(states)
        with torch.no_grad():
            l_next_state = self._get_latent(next_states).detach()

        #modify reward with goal net
        with torch.no_grad():
            goal = self._get_goal(l_state)
            goal = (goal/(torch.norm(goal, p=2, dim=1, keepdim=True)+1e-8)).detach()
            next_goal = self._get_goal(l_next_state)
            next_goal = (next_goal/(torch.norm(next_goal, p=2, dim=1, keepdim=True)+1e-8)).detach()
            g_rewards = torch.clone(rewards)
            diff = l_next_state - l_state
            diff = (diff/(torch.norm(diff, p=2, dim=1, keepdim=True)+1e-8)).detach()
            g_rewards += torch.sum(diff*goal, dim=1)

        QAB = 'A' if self.ABswitch else 'B'
        QBA = 'A' if not self.ABswitch else 'B'
        inp_Q = torch.cat([l_state, states['state'][:,-self.state_action_dim:]], dim=1)
        value, pre_value = self._get_Q(inp_Q, QAB, goal)
        value = value.gather(1, actions.unsqueeze(1)).squeeze(1)
        ret_value = np.mean(value.data.cpu().numpy())
        with torch.no_grad():
            next_inp_Q = torch.cat([l_next_state, next_states['state'][:,-self.state_action_dim:]], dim=1)
            value_nextA, pre_value_nextA = self._get_Q(next_inp_Q, QAB, next_goal)
            value_nextB, pre_value_nextB = self._get_Q(next_inp_Q, QBA, next_goal)
            argmaxA = torch.max(value_nextA, dim=1)[1].unsqueeze(1)
            target = g_rewards
            target[next_indexes] += (GAMMA * torch.min(value_nextA.gather(1, argmaxA).squeeze(1), 
                                                       value_nextB.gather(1, argmaxA).squeeze(1)))[next_indexes]
            target = self.args.clr * target + (1.0 - self.args.clr) * value.detach()
        assert value.shape == target.shape
        allloss = self._a['loss'](value, target)

        if False:
            #does not work because goal is almost always 0
            #needs skip in next states

            #make positive and negative hindsight reward
            with torch.no_grad():
                goal = l_next_state - l_state
                goal = (goal/(torch.norm(goal, p=2, dim=1, keepdim=True)+1e-8)).detach()
                g_rewards = torch.clone(rewards)
                g_rewards += 1.0
            value, _ = self._get_Q(inp_Q, QAB, goal, pre_value)
            value = value.gather(1, actions.unsqueeze(1)).squeeze(1)
            with torch.no_grad():
                value_nextA, _ = self._get_Q(next_inp_Q, QAB, goal, pre_value_nextA)
                value_nextB, _ = self._get_Q(next_inp_Q, QBA, goal, pre_value_nextB)
                argmaxA = torch.max(value_nextA, dim=1)[1].unsqueeze(1)
                target = g_rewards
                target[next_indexes] += (GAMMA * torch.min(value_nextA.gather(1, argmaxA).squeeze(1), 
                                                           value_nextB.gather(1, argmaxA).squeeze(1)))[next_indexes]
            assert value.shape == target.shape
            allloss += 0.2 * self._a['loss'](value, target)

            with torch.no_grad():
                goal = l_state - l_next_state
                goal = (goal/(torch.norm(goal, p=2, dim=1, keepdim=True)+1e-8)).detach()
                g_rewards = torch.clone(rewards)
                g_rewards -= 1.0
            value, _ = self._get_Q(inp_Q, QAB, goal, pre_value)
            value = value.gather(1, actions.unsqueeze(1)).squeeze(1)
            with torch.no_grad():
                value_nextA, _ = self._get_Q(next_inp_Q, QAB, goal, pre_value_nextA)
                value_nextB, _ = self._get_Q(next_inp_Q, QBA, goal, pre_value_nextB)
                argmaxA = torch.max(value_nextA, dim=1)[1].unsqueeze(1)
                target = g_rewards
                target[next_indexes] += (GAMMA * torch.min(value_nextA.gather(1, argmaxA).squeeze(1), 
                                                           value_nextB.gather(1, argmaxA).squeeze(1)))[next_indexes]
            assert value.shape == target.shape
            allloss += 0.2 * self._a['loss'](value, target)


        self.ABswitch = not self.ABswitch


        #make goal and value net loss
        states = {}
        for k in state_keys:
            if k in skip_input:
                states[k] = torch.cat([variable(skip_input[k]), variable(skip_expert_input[k])], dim=0)
        next_states = {}
        for k in state_keys:
            next_k = 'next_'+k
            if next_k in skip_input:
                next_states[k] = torch.cat([variable(skip_input[next_k]), variable(skip_expert_input[next_k])], dim=0)
        rewards = torch.cat([variable(skip_input['reward']), variable(skip_expert_input['reward'])], dim=0)
        done = np.concatenate([skip_input['done'].data.cpu().numpy(), skip_expert_input['done'].data.cpu().numpy()], axis=0)
        next_indexes = np.nonzero(np.logical_not(done))[0]

        
        #get latent states
        l_state = self._get_latent(states)
        with torch.no_grad():
            l_next_state = self._get_latent(next_states).detach()
        value = self._a['modules']['value_net'](l_state)[:,0]
        with torch.no_grad():
            value_next = torch.clone(rewards)
            value_next[next_indexes] += (((tGAMMA**self.args.skip)*self._a['modules']['value_net'](l_next_state))[:,0])[next_indexes]
            
        assert value.shape == value_next.shape
        allloss += 2.0 * self._a['loss'](value, value_next)
        
        with torch.no_grad():
            advantage = value_next - value
            advantage = advantage.detach()
            diff = l_next_state - l_state
            diff = (diff/(torch.norm(diff, p=2, dim=1, keepdim=True)+1e-8)).detach()

        goal = self._get_goal(l_state)
        goal = (goal/(torch.norm(goal, p=2, dim=1, keepdim=True)+1e-8)).detach()
        allloss -= (advantage * torch.sum(diff*goal, dim=1)).mean()
        ret_value2 = np.mean(value.detach().cpu().numpy())
        
        allloss.backward()
        self._a['opt'].step()

        return {'loss': allloss.data.cpu().numpy(),
                'Q_val': ret_value,
                'value': ret_value2}

    def select_action(self, state):
        make_latent = True
        if self.last_state is not None:
            d = 0.0
            for k in state:
                if k != 'state':
                    d += np.sum(np.abs(state[k] - self.last_state[k]))
                else:
                    d += np.sum(np.abs(state[k][:-self.state_action_dim] - self.last_state[k][:-self.state_action_dim]))
            if d < 1e-8:
                make_latent = False
        self.last_state = {}
        for k in state:
            self.last_state[k] = np.copy(state[k])
        for k in state:
            state[k] = state[k][None,:]
        states = variable(state)
        with torch.no_grad():
            if make_latent:
                latent = self._get_latent(states)
                goal = self._get_goal(latent)
                self.last_latent = latent
                self.last_goal = goal
            else:
                latent = self.last_latent
                goal = self.last_goal
            QAB = 'A' if self.ABswitch else 'B'
            Q_inp = torch.cat([latent, states['state'][:,-self.state_action_dim:]], dim=1)
            Q_vals, _ = self._get_Q(Q_inp, QAB, goal)
            action = torch.max(Q_vals, dim=1)[1][0].data.cpu().numpy()
            value = torch.max(Q_vals, dim=1)[0][0].data.cpu().numpy()
        self.ABswitch = not self.ABswitch
        if self.last_action == 0 and action == 0:
            action = np.random.randint(0, self.num_actions)
        self.last_action = action
        if np.random.uniform() < 0.005:
            action = np.random.randint(0, self.num_actions)
        return action, {'value': value}





class LearnerModelS(torch.nn.Module):
    def __init__(self, args, state_shape, state_action_dim):
        super(LearnerModelS, self).__init__()
        self.args = args
        self.state_shape = state_shape
        self.state_action_dim = state_action_dim
        model = {}

        inp_size = 0
        self.cnn_model = None
        if 'pov' in self.state_shape:
            #make CNN
            cnn_layers = []
            assert len(self.state_shape['pov']) > 1
            if self.args.cnn_type == 'atari':
                sizes = [8, 4, 3]
                strides = [4, 2, 1]
                pooling = [1, 1, 1]
                filters = [32, 64, 32]
            elif self.args.cnn_type == 'mnist':
                sizes = [3, 3, 3]
                strides = [1, 1, 1]
                pooling = [2, 2, 2]
                filters = [32, 32, 32]

            in_channels = self.state_shape['pov'][-1] #HWC input

            for i in range(len(sizes)):
                cnn_layers.append(torch.nn.Conv2d(
                    in_channels,
                    filters[i],
                    sizes[i],
                    stride=strides[i],
                    bias=True
                ))

                cnn_layers.append(Lambda(lambda x: swish(x)))

                if pooling[i] > 1:
                    cnn_layers.append(torch.nn.MaxPool2d(pooling[i]))

                in_channels = filters[i]

            cnn_layers.append(Flatten())
            self.cnn_model = torch.nn.Sequential(*cnn_layers)

            self.embed_dim = self.cnn_model(torch.zeros((1,self.state_shape['pov'][2], self.state_shape['pov'][0], self.state_shape['pov'][1]))).shape[1]
            inp_size += self.embed_dim
            print("CNN EMBED",self.embed_dim) 

        self.state_model = None
        if 'state' in self.state_shape:
            if self.args.state_layers != 0:
                state_layers = []
                for i in range(self.args.state_layers):
                    state_layers.append(torch.nn.Linear(self.state_shape['state'][0]-self.state_action_dim if i == 0 else self.args.hidden, self.args.hidden))
                    state_layers.append(torch.nn.Tanh())
                self.state_model = torch.nn.Sequential(*state_layers)
                inp_size += self.args.hidden
            else:
                inp_size += self.state_shape['state'][0]-self.state_action_dim

        head_layers = []
        for i in range(self.args.layers):
            head_layers.append(torch.nn.Linear(inp_size if i == 0 else self.args.hidden, self.args.hidden))
            head_layers.append(torch.nn.Tanh())
        self.head_model = torch.nn.Sequential(*head_layers)


    def forward(self, inp):
        body_out = []
        if not self.cnn_model is None:
            if len(inp['pov'].shape) == 2:
                body_out.append(inp['pov'])
            else:
                x = (inp['pov'].permute(0, 3, 1, 2)/255.0)*2.0 - 1.0
                body_out.append(self.cnn_model(x))
        if not self.state_model is None:
            body_out.append(self.state_model(inp['state'][:,:-self.state_action_dim]))
        else:
            if 'state' in inp:
                body_out.append(inp['state'][:,:-self.state_action_dim])
        x = torch.cat(body_out, dim=1)
        return self.head_model(x)


class SimpleIQN():
    def __init__(self, state_shape, num_actions, args):
        assert isinstance(state_shape, dict)
        self.state_shape = state_shape
        self.num_actions = num_actions
        self.args = args
        self.last_action_index = 0
        self.ABswitch = True
        self.state_action_dim = 17
        self.last_state = None

        self._setup()
        self._a = self._models[0]

    def _setup(self):
        self._models = [self._make_model()]

    def state_dict(self):
        return [self._models[0]['modules'].state_dict()]

    def load_state_dict(self, s, strict=True):
        if isinstance(s, list):
            self._models[0]['modules'].load_state_dict(s[0], strict)
        else:
            self._models[0]['modules'].load_state_dict(s, strict)

    def loadstore(self, filename, load=True):
        if load:
            if self.args.load_pretain is not None:
                pretrain_dict = torch.load(self.args.load_pretain)
                this_state_dict = self._models[0]['modules'].state_dict()
                pretain_load_tail = OrderedDict([])
                pretain_load_actor = OrderedDict([])
                for k, p in pretrain_dict.items():
                    if k.startswith('0.'):
                        k = k[2:]
                        pretain_load_tail[k] = p
                    if k.startswith('1.nec_layer_adv.'):
                        k = k.replace('1.nec_layer_adv.','')
                        pretain_load_actor[k] = p
                self._models[0]['modules']['tail_model'].load_state_dict(pretain_load_tail)
                self._models[1]['modules']['tail_model'].load_state_dict(pretain_load_tail)
            else:
                self.load_state_dict([torch.load(filename + '-simpleiqnA')])
        else:
            torch.save(self._models[0]['modules'].state_dict(), filename + '-simpleiqnA')

    def _make_model(self):
        tail_model = LearnerModelS(self.args, self.state_shape, self.state_action_dim)
        head_modelA = IQNHead(self.args.hidden+self.state_action_dim, self.num_actions, self.args)
        head_modelB = IQNHead(self.args.hidden+self.state_action_dim, self.num_actions, self.args)

        smax = torch.nn.Softmax(1)

        if torch.cuda.is_available() and CUDA:
            if smax is not None:
                smax = smax.cuda()
            tail_model = tail_model.cuda()
            head_modelA = head_modelA.cuda()
            head_modelB = head_modelB.cuda()
            
        #TODO add skynet penalty
        if self.args.per:
            loss = WeightedMSELoss()
        else:
            loss = torch.nn.MSELoss()#torch.nn.SmoothL1Loss()#

        ce_loss = torch.nn.CrossEntropyLoss()
        modules = torch.nn.ModuleDict()
        modules['tail_model'] = tail_model
        modules['head_modelA'] = head_modelA
        modules['head_modelB'] = head_modelB

        model_params = []
        nec_value_params = []
        for name, p in modules.named_parameters():
            if 'values.weight' in name:
                nec_value_params.append(p)
            else:
                model_params.append(p)
        optimizer = RAdam([{'params': model_params}, 
                           {'params': nec_value_params, 'lr': self.args.nec_lr}], lr=self.args.lr)

        return {'modules': modules, 'opt': optimizer, 'loss': loss, 'ce_loss': ce_loss, 'softmax': smax, 'mse_loss': torch.nn.MSELoss()}

    def select_action(self, state):
        make_latent = True
        if self.last_state is not None:
            d = 0.0
            for k in state:
                if k != 'state':
                    d += np.sum(np.abs(state[k] - self.last_state[k]))
                else:
                    d += np.sum(np.abs(state[k][:-self.state_action_dim] - self.last_state[k][:-self.state_action_dim]))
            if d < 1e-8:
                make_latent = False
        self.last_state = {}
        for k in state:
            self.last_state[k] = np.copy(state[k])
        for k in state:
            state[k] = state[k][None,:]
        state = variable(state)
        with torch.no_grad():
            QAB = 'A' if self.ABswitch else 'B'
            if make_latent:
                dist, _, tail, _ = self._get_distribution(QAB, state, self.args.N_tau)
                self.last_tail = tail
            else:
                dist, _, tail, _ = self._get_distribution(QAB, state, self.args.N_tau, tail=self.last_tail)
            vals = torch.mean(dist, dim=2)
            action_index = vals.max(1)[1][0].data.cpu().numpy()
        #if action_index == 0 and self.last_action_index == 0:
        #    return np.random.randint(0, self.num_actions), -1.0
        self.last_action_index = action_index
        if np.random.sample() < 0.01:
            self.ABswitch = not self.ABswitch
        if np.random.sample() < 0.01:
            action_index = np.random.randint(0, self.num_actions)
        return action_index, {'val': vals.max(1)[0][0].data.cpu().numpy()}

    def _get_distribution(self, QAB, state, num_tau, tau=None, tail=None):
        if tail is None:
            tail = self._a['modules']['tail_model'](state)
        rtail = torch.cat([tail,state['state'][:,-self.state_action_dim:]], dim=1)
        dist, Ttau, action_comp = self._a['modules']['head_model'+QAB](rtail, num_tau, tau)
        return dist, Ttau, tail, action_comp

    def _get_next_distribution(self, next_states, rewards, next_indices):
        def get_argmax(next_states, batch_size):
            QAB = 'A' if self.ABswitch else 'B'
            next_dist, Ttau, tail, _ = self._get_distribution(QAB, next_states, self.args.Np_tau)
            combined  = next_dist.mean(dim=2) 
            next_Q_sa = combined.max(1)[1]   # next_Q_sa is of size: [batch ] of action index 
            next_Q_sa = next_Q_sa.view(batch_size, 1, 1)  # Make it to be size of [32 x 1 x 1]
            next_Q_sa = next_Q_sa.expand(-1, -1, self.args.Np_tau)  # Expand to be [32 x 1 x 51], one action, expand to support
            return next_Q_sa, next_dist, Ttau, tail

        with torch.no_grad():
            QBA = 'B' if self.ABswitch else 'A'
            action_argmax, next_yA, Ttau, tail = get_argmax(next_states, rewards.size(0))
            next_yB, _, _, _ = self._get_distribution(QBA, next_states, self.args.Np_tau, Ttau, tail)
            next_y = torch.min(next_yA, next_yB)
            quantiles_next = torch.clone(rewards).unsqueeze(dim=-1).expand(-1, self.args.Np_tau)

            quantiles_next[next_indices] += (0.999 * next_y.gather(1, action_argmax).squeeze(1))[next_indices]
            #quantiles_next = quantiles_next.clamp(0.0, 1.0)
            return quantiles_next.detach(), Ttau

    def train(self, input, expert_input=None):
        if self.args.per:
            assert False #not implemented yet
        else:
            is_weights = None
        state_keys = ['state', 'pov']
        self._a['opt'].zero_grad()

        states = {}
        for k in state_keys:
            if k in input:
                states[k] = variable(input[k])
        next_states = {}
        for k in state_keys:
            next_k = 'next_'+k
            if next_k in input:
                next_states[k] = variable(input[next_k])
        actions = variable(input['action'], True).long()
        rewards = variable(input['reward'])
        done = input['done'].data.cpu().numpy()
        next_indexes = np.nonzero(np.logical_not(done))[0]
        batch_size = done.shape[0]

        if expert_input is not None:
            expert_states = {}
            for k in state_keys:
                if k in expert_input:
                    expert_states[k] = variable(expert_input[k])
            expert_next_states = {}
            for k in state_keys:
                next_k = 'next_'+k
                if next_k in expert_input:
                    expert_next_states[k] = variable(expert_input[next_k])


            expert_actions = variable(expert_input['action'], True).long()
            expert_rewards = variable(expert_input['reward'])
            expert_done = expert_input['done'].data.cpu().numpy()

            #concat everything
            for k in states:
                states[k] = torch.cat([states[k],expert_states[k]], dim=0)
            for k in next_states:
                next_states[k] = torch.cat([next_states[k],expert_states[k]], dim=0)
            actions = torch.cat([actions, expert_actions], dim=0)
            rewards = torch.cat([rewards, expert_rewards], dim=0)

            done = np.concatenate([done, expert_done], axis=0)

        next_indexes = np.nonzero(np.logical_not(done))[0]
        next_indexes_other = np.nonzero(done)[0]
        rewards[next_indexes_other] += -0.01/(1-0.999)
        rewards = torch.where(rewards == 0.0, -0.01*torch.ones_like(rewards), rewards)


        #train IQN
        QAB = 'A' if self.ABswitch else 'B'
        y, my_tau, tail, action_comp = self._get_distribution(QAB, states, self.args.N_tau)
        iqn_actions = actions.unsqueeze(dim=-1).unsqueeze(dim=-1).expand(-1, -1, self.args.N_tau) #[batch_size,1,atoms]
        quantiles = y.gather(1, iqn_actions).squeeze(1)
        quantiles_next, Ttau = self._get_next_distribution(next_states, rewards, next_indexes)

        quantiles      = quantiles.unsqueeze(-1)
        quantiles_next = quantiles_next.unsqueeze(-1)
        #with torch.no_grad():
            #quantiles_next = self.args.clr * quantiles_next.unsqueeze(-1) + (1.0 - self.args.clr) * self._get_distribution(QAB, states, self.args.N_tau, Ttau, tail)[0].gather(1, iqn_actions).squeeze(1).detach().unsqueeze(-1)

        diff = quantiles_next.unsqueeze(2) - quantiles.unsqueeze(1)

        def huber(x, k=1.0):
            return torch.where(x.abs() < k, 0.5 * x.pow(2), k * (x.abs() - 0.5 * k))
        huber_diff = huber(diff, self.args.kappa)

        my_tau = my_tau.view(y.shape[0], self.args.N_tau, 1)   # [N_tau x Batch x 1]
        my_tau = my_tau.unsqueeze(1) # [Batch x 1 x N_tau x 1]
        my_tau = my_tau.expand(-1, self.args.Np_tau, -1, -1) # [Batch x Np_tau x N_tau x 1]

        loss = (huber_diff * (my_tau - (diff<0).float()).abs()) / self.args.kappa

        loss = loss.squeeze(3).sum(-1).mean(-1)

        #loss += ((action_comp < 0).float() * action_comp.pow(2)).mean() * 5.0

        if self.args.per:
            loss_PER = loss.detach().abs().cpu().numpy() 
            loss = loss * batch_weight_IS

        loss = loss.mean()
        ds = y.detach() * 1.0 / y.size(2)
        Q_sa = ds.sum(dim=2).max(1)[0]#gather(1, actions.unsqueeze(dim=-1))


        #actor train
        #y [batch x actions x atoms]
        with torch.no_grad():
            argmax_y = y.detach().max(1)[1].unsqueeze(1) # [batch x 1 x atoms]
            probas = torch.zeros_like(y)
            probas.scatter_(1, argmax_y, 1.0)
            probas = torch.mean(probas, dim=2) #[batch x actions]
                
        if False:
            actor_logits = self._a['modules']['actor_model'](tail)
            actor_probas = self._a['softmax'](actor_logits)
            with torch.no_grad():
                entropy = -torch.sum(actor_probas.detach()*torch.log(actor_probas.detach()), dim=1)
                alr_weights = torch.clamp(entropy, max=1.8)[:,None]
                alr_weights = alr_weights.detach()
                train_probas = actor_probas.detach() * (1.0 - self.args.alr * alr_weights) + self.args.alr * alr_weights * probas
            if self.args.per:
                entropy = -torch.sum(actor_probas*torch.log(actor_probas), dim=1)
                loss_actor = self._a['loss'](actor_probas, train_probas, is_weights)
            else:
                #entropy weigths
                loss_actor = self._a['loss'](actor_probas, train_probas)
            #if expert_input is not None:
            #    loss_actor += 0.05 * self._a['ce_loss'](actor_logits[:expert_done.size], expert_actions)
            if False:
                loss += loss_actor
            max_probs = torch.mean(torch.max(actor_probas,dim=1)[0])
            entropy = torch.mean(entropy)
        accuracy = torch.mean(probas[:,actions][-len(expert_done):])
        allloss = loss

        if False:
            #add margin loss to expert quantiles
            expert_quantiles = quantiles[-len(expert_done):].detach().squeeze(-1) #[batch, atoms]
            #argmax_quantiles = y[-expert_done.shape[0]:].max(1)[1] #[batch, atoms]
            #expert_actions = expert_actions.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, self.args.N_tau)
            #margin = torch.where(expert_actions!=argmax_quantiles,-0.02,0.0)
            margin = y[-len(expert_done):] + 0.1
            bselect = torch.arange(margin.size(0), dtype=torch.long)
            margin[bselect,expert_actions] -= 0.1
            margin = margin.max(1)[0]
            margin_loss = (margin-expert_quantiles).mean()
            allloss += 4.0 * margin_loss

        allloss.backward()
        self._a['opt'].step()

        Qval = Q_sa.cpu().detach().numpy().squeeze()

        if np.random.uniform() < 0.01:
            self.ABswitch = not self.ABswitch

        return {'allloss': allloss.data.cpu().numpy(), 
                'loss': loss.data.cpu().numpy(), 
                'val': np.mean(Qval),
                'val_exp': np.mean(Qval[-len(expert_done):]),
                'accuracy': accuracy.data.cpu().numpy()}