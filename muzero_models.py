import torch
import numpy as np
from all_experiments import HashingMemory, RAdam, prio_utils
from torch.nn import functional as F
from collections import OrderedDict
from utils import variable, Lambda, swish, Flatten, sample_gaussian, one_hot, GAMMA
import copy
import math
from modules import VisualNet, MemoryModel, CameraBias, VariationalDropoutLinear, add_loss
import kornia

INFODIM = False
LINEAR = VariationalDropoutLinear
MAX_STEPS_PREDICT = 8
ON_POLICY = True
REGULARIZE = True
PAD_SIZE = 4
NUM_REG_SAMPLE = 4

class TreeSamplerNode():
    def __init__(self, prob_double_sampling, probs):
        self.children = {}
        self.prob_double_sampling = prob_double_sampling
        if probs is not None:
            self.name = list(probs.keys())[0]
            self.prob = list(probs.values())[0]
            self.probs = probs
            #self.probs.pop(self.name)

    def remove_sample(self, sample):
        self.prob_double_sampling += np.prod([self.probs[s][sample[s]] for s in sample])
        if sample[self.name] not in self.children:
            ch_prob = None
            if len(self.probs) >= 2:
                ch_prob = self.probs.copy()
                ch_prob.pop(self.name)
            self.children[sample[self.name]] = TreeSamplerNode(0.0, ch_prob)
        if len(sample) > 0:
            if len(sample) > 1:
                ch_sample = sample.copy()
                ch_sample.pop(self.name)
                self.children[sample[self.name]].remove_sample(ch_sample)
            else:
                self.children[sample[self.name]].prob_double_sampling = 1.0

    #def prob_double_sampling(self):
    #    if len(self.children) == 0:
    #        return 0.0
    #    if self.needs_update:
    #        self._prob_double_sampling = 0.0
    #        for k in self.children:
    #            self._prob_double_sampling += self.prob[k] * self.children[k].prob_double_sampling()
    #    return self._prob_double_sampling

    def get_true_probs(self):
        if len(self.children) == 0:
            return self.prob
        prob_reject = np.zeros_like(self.prob)
        for i in self.children:
            prob_reject[i] = self.children[i].prob_double_sampling
        sum_total_reject = np.sum(self.prob * prob_reject)
        true_probs = ((1.0 - prob_reject) * self.prob) / (1.0 - sum_total_reject)
        true_probs = np.where(true_probs < 0.0, 0.0, true_probs)
        true_probs /= true_probs.sum()
        return true_probs

    def sample(self, current_sample, p=1.0):
        true_probs = self.get_true_probs()
        app_sample = np.random.choice(len(true_probs), p=true_probs)
        current_sample[self.name] = app_sample
        p *= self.probs[self.name][app_sample]
        if app_sample in self.children:
            return self.children[app_sample].sample(current_sample)
        else:
            for i in self.probs:
                if i != self.name:
                    current_sample[i] = np.random.choice(len(self.probs[i]), p=self.probs[i])
                    p *= self.probs[i][current_sample[i]]
            return current_sample, p

class TreeSampler():
    def __init__(self, probs):
        self.probs = probs
        self.root = TreeSamplerNode(0.0, probs)

    def remove_sample(self, sample):
        self.root.remove_sample(sample)

    def sample(self):
        return self.root.sample(OrderedDict())


#modified code from https://arxiv.org/src/1911.08265v1/anc/pseudocode.py
class MCTS_CUDA():
    def __init__(self, model):
        self.model = model

    def run_mcts(self, root):
        self.min_V = 0.0
        self.max_V = 1.0

        for _ in range(self.model.num_mcts_steps):
            node = root
            search_path = [node]

            while node.expanded():
                action, node = self.select_child(node)
                search_path.append(node)

            parent = search_path[-2]
            self.expand_node(node)
            for i, child in node.children.items():
                self.backpropagate(search_path+[child])

    def expand_node(self, node):
        action_stack, _, _ = self.get_action_V(node, self.model.num_policy_samples)

        #calc new hidden states for all new nodes and reward
        next_hidden_states, reward = self.get_next_states(node.hidden_state, action_stack)

        #init next nodes
        for i in range(next_hidden_states.shape[1]):
            node.children[i] = Node(1.0/self.model.num_policy_samples)
            node.children[i].hidden_state = next_hidden_states[:,i]
            node.children[i].reward = reward[0,i,0].item()
        self.calc_all_V(next_hidden_states[0], node.children)
        
    def select_child(self, node):
        _, action, child = max(
            (self.ucb_score(node, child), action, child)
            for action, child in node.children.items()
        )
        return action, child

    def backpropagate(self, search_path):
        value = search_path[-1].value
        for node in reversed(search_path):
            node.value_sum += value
            node.visit_count += 1
            self.min_V = min(value, self.min_V)
            self.max_V = max(value, self.max_V)
            value = node.reward + GAMMA * value

    def ucb_score(self, parent, child):
        p_visit = parent.visit_count
        if p_visit > 1:
            p_visit //= self.model.num_policy_samples
        c_visit = child.visit_count
        if c_visit > 1:
            c_visit //= self.model.num_policy_samples
        pb_c = math.log((p_visit + 19652.0 + 1.0)/19652.0)+1.25
        pb_c *= math.sqrt(p_visit) / (c_visit + 1)
        value_reward = child.reward + GAMMA * child.value_()
        value_score  = (value_reward - self.min_V) / (self.max_V - self.min_V)
        return pb_c * child.prior + value_score

    def get_next_states(self, hidden_state, action_stack):
        num_samples = action_stack.shape[1]
        hidden_state_exp = hidden_state.unsqueeze(1).expand([-1, num_samples, -1])
        tmp = self.model._model['modules']['dynamics'](torch.cat([hidden_state_exp,action_stack],-1))
        next_hidden_state, reward = tmp[:,:,:self.model.args.hidden], tmp[:,:,self.model.args.hidden:]
        if self.model.distributional:
            reward = F.softmax(reward-reward.logsumexp(dim=-1, keepdim=True), -1)
            reward = torch.sum((reward * self.model.reward_support),-1).unsqueeze(-1)
        return next_hidden_state, reward #[batch, num_samples, hidden_dim], [batch, num_samples, 1]

    def calc_all_V(self, hidden_states, nodes):
        actions_hidden = self.model._model['modules']['actions_predict_hidden'](hidden_states) #hidden_state [batch_size, hidden_size]
        for i in range(len(nodes)):
            nodes[i].actions_hidden = actions_hidden[i].unsqueeze(0)
        assert self.model.max_predict_range == 1 #this does not work with self.max_predict_range>1
        current_actions_V = self.model._model['modules']['V_actions_predict'](actions_hidden)#.view(-1, self.max_predict_range, self.action_dim)
        if self.model.distributional:
            values = current_actions_V[:,:self.model.num_support_value]
            values = F.softmax(values-values.logsumexp(dim=-1, keepdim=True), -1)
            values = torch.sum((values * self.model.value_support),-1)
        else:
            values = current_actions_V[:,0]
        for i in range(len(nodes)):
            nodes[i].current_actions_V = current_actions_V[i].unsqueeze(0)
            nodes[i].value = self.model.inverse_map_reward_V(values[i].item())

    def get_action_V(self, node, num_samples):
        if node.actions_hidden is None:
            # batch_size = 1
            node.actions_hidden = self.model._model['modules']['actions_predict_hidden'](node.hidden_state) #hidden_state [batch_size, hidden_size]
        assert self.model.max_predict_range == 1 #this does not work with self.max_predict_range>1
        if node.current_actions_V is None:
            node.current_actions_V = self.model._model['modules']['V_actions_predict'](node.actions_hidden)#.view(-1, self.max_predict_range, self.action_dim)
        if not self.model.distributional:
            dis_logits, zero_logits = node.current_actions_V[:,1:(self.model.action_dim-2+1)], node.current_actions_V[:,(self.model.action_dim-2+1):(self.model.action_dim+1)]
        else:
            dis_logits, zero_logits = node.current_actions_V[:,self.model.num_support_value:(self.model.action_dim-2+self.model.num_support_value)], node.current_actions_V[:,(self.model.action_dim-2+self.model.num_support_value):(self.model.action_dim+self.model.num_support_value)]
        if node.action_stack is None:
            dis_logits = torch.split(dis_logits, self.model.action_split, -1)
            cam_map = self.model._model['modules']['camera_predict'](node.actions_hidden).reshape(1,-1)
            if True:
                ii = 0
                action = OrderedDict()
                action_stack = []
                sample_rn = 0.9
                for a in self.model.action_dict:
                    if a != 'camera':
                        #action[a] = torch.distributions.Categorical(logits=dis_logits[ii]).sample()
                        action[a] = torch.cat([torch.multinomial(F.softmax((dis_logits[ii]*sample_rn)- (dis_logits[ii]*sample_rn).logsumexp(dim=-1, keepdim=True), dim=-1), num_samples-1, True),
                                               torch.multinomial(F.softmax(dis_logits[ii]- dis_logits[ii].logsumexp(dim=-1, keepdim=True), dim=-1), 1, True)], 1)
                        action_stack.append(one_hot(action[a], self.model.action_split[ii], self.model.device))
                        ii += 1
                    else:
                        #cam_choose = torch.distributions.Categorical(logits=cam_map).sample()
                        cam_choose = torch.cat([torch.multinomial(F.softmax(cam_map*sample_rn- (cam_map*sample_rn).logsumexp(dim=-1, keepdim=True), dim=-1), num_samples-1, True),
                                                torch.multinomial(F.softmax(cam_map- cam_map.logsumexp(dim=-1, keepdim=True), dim=-1), 1, True)], 1)#[batch_size, num_samples]
                        cam_choose_x = (((cam_choose % self.model.cam_map_size) + torch.rand(cam_choose.size(), device=self.model.device)) * (2.0/self.model.cam_map_size)) - 1.0
                        cam_choose_y = (((cam_choose // self.model.cam_map_size) + torch.rand(cam_choose.size(), device=self.model.device)) * (2.0/self.model.cam_map_size)) - 1.0
                        cam_choose_xy = torch.stack([cam_choose_x, cam_choose_y], dim=-1) #[batch_size, num_samples, 2]
                        #zero_act = torch.distributions.Categorical(logits=zero_logits[:,0,:]).sample()[0].item()
                        zero_act = torch.cat([torch.multinomial(F.softmax(zero_logits*sample_rn- (zero_logits*sample_rn).logsumexp(dim=-1, keepdim=True), dim=-1), num_samples-1, True),
                                              torch.multinomial(F.softmax(zero_logits- zero_logits.logsumexp(dim=-1, keepdim=True), dim=-1), 1, True)], -1)
                        action[a] = torch.where(zero_act.unsqueeze(-1) == 1, torch.zeros_like(cam_choose_xy), self.model.map_camera_torch(cam_choose_xy))
                        action_stack.append(action[a])
                action_stack = torch.cat(action_stack, dim=-1) #[batch_size, num_policy_samples, action_dim]
            else:
                assert dis_logits[0].shape[0] == 1 #batch not implemented
                dis_logits = list(dis_logits)
                action = OrderedDict()
                action_stack = []
                disc_action_stack = []
                cont_action_stack = []
                i = 0
                revert_cnt = 0
                while i < num_samples:
                    ii = 0
                    current_action = OrderedDict()
                    current_action_stack = []
                    current_disc_action_stack = []
                    current_cont_action_stack = []
                    for a in self.model.action_dict:
                        if a != 'camera':
                            #action[a] = torch.distributions.Categorical(logits=dis_logits[ii]).sample()
                            current_action[a] = torch.multinomial(F.softmax(dis_logits[ii]- dis_logits[ii].logsumexp(dim=-1, keepdim=True), dim=-1), 1, True)
                            current_action_stack.append(one_hot(current_action[a], self.model.action_split[ii], self.model.device))
                            current_disc_action_stack.append(current_action_stack[-1])
                            ii += 1
                        else:
                            #cam_choose = torch.distributions.Categorical(logits=cam_map).sample()
                            cam_choose = torch.multinomial(F.softmax(cam_map- cam_map.logsumexp(dim=-1, keepdim=True), dim=-1), 1, True) #[batch_size, 1]
                            cam_choose_x = (((cam_choose % self.model.cam_map_size) + torch.rand(cam_choose.size(), device=self.model.device)) * (2.0/self.model.cam_map_size)) - 1.0
                            cam_choose_y = (((cam_choose // self.model.cam_map_size) + torch.rand(cam_choose.size(), device=self.model.device)) * (2.0/self.model.cam_map_size)) - 1.0
                            cam_choose_xy = torch.stack([cam_choose_x, cam_choose_y], dim=-1) #[batch_size, 1, 2]
                            #zero_act = torch.distributions.Categorical(logits=zero_logits[:,0,:]).sample()[0].item()
                            zero_act = torch.multinomial(F.softmax(zero_logits- zero_logits.logsumexp(dim=-1, keepdim=True), dim=-1), 1, True)
                            current_action[a] = torch.where(zero_act.unsqueeze(-1) == 1, torch.zeros_like(cam_choose_xy), self.model.map_camera_torch(cam_choose_xy))
                            current_action_stack.append(current_action[a])
                            current_cont_action_stack.append(current_action[a])
                    current_action_stack = torch.cat(current_action_stack, dim=-1)
                    current_disc_action_stack = torch.cat(current_disc_action_stack, dim=-1)
                    current_cont_action_stack = torch.cat(current_cont_action_stack, dim=-1)
                    if len(action_stack) == 0:
                        action_stack = current_action_stack
                        disc_action_stack = current_disc_action_stack
                        cont_action_stack = current_cont_action_stack
                        action = current_action
                        revert_cnt = 0
                        i += 1
                    else:
                        dist_disc = torch.sum(torch.abs(disc_action_stack-current_disc_action_stack)).item()
                        dist_cont = torch.sum((cont_action_stack-current_cont_action_stack)**2).item()
                        if dist_disc <= 2.1 and dist_cont < 0.02:
                            if revert_cnt > 3:
                                cam_map *= 0.7
                                zero_logits *= 0.5
                                for j in range(len(dis_logits)):
                                    dis_logits[j] *= 0.7
                                revert_cnt = 0
                            revert_cnt += 1
                            continue
                        else:
                            action_stack = torch.cat([action_stack, current_action_stack], 1)
                            disc_action_stack = torch.cat([disc_action_stack, current_disc_action_stack], 1)
                            cont_action_stack = torch.cat([cont_action_stack, current_cont_action_stack], 1)
                            for a in action:
                                action[a] = torch.cat([action[a], current_action[a]], 1)
                            revert_cnt = 0
                            i += 1
        node.action_stack = action_stack
        node.action = action
        return node.action_stack, node.action, node.value
    
#https://arxiv.org/src/1911.08265v1/anc/pseudocode.py with entropy based progressive widening
class MCTS_CPU():
    def __init__(self, model):
        self.model = model
        self.max_policy_entropy = 5.0
        self.this_max_policy_entropy = 0.0

    def run_mcts(self, root):
        self.min_V = 0.0
        self.max_V = 1.0
        self.this_max_policy_entropy = 0.0

        for _ in range(self.model.num_mcts_steps):
            node = root
            search_path = [node]

            while node.expanded():
                action, node = self.select_child(node)
                search_path.append(node)

            parent = search_path[-2]
            network_output = self.recurrent_inference(parent.hidden_state,
                                                      parent.action_stack[action])
            self.expand_node(node, network_output)
            self.backpropagate(search_path, network_output['value'])
            if len(root.children) <= 1:
                break
        self.max_policy_entropy = self.max_policy_entropy * 0.999 + (self.this_max_policy_entropy * 1.2) * 0.001

    def expand_node(self, node, network_output):
        node.hidden_state = network_output['hidden_state']
        node.reward = network_output['reward']
        node.action = network_output['actions']
        node.entropy = network_output['entropy']
        node.sampler = TreeSampler(node.action)
        self.this_max_policy_entropy = max(self.this_max_policy_entropy, node.entropy)
        self.correct_num_children(node)

    def recurrent_inference(self, hidden_state, action, init=False):
        if not init:
            #get next state and reward
            tmp = self.model._model['modules']['dynamics'](torch.cat([hidden_state,action.unsqueeze(0)],-1))
            next_hidden_state, reward = tmp[:,:self.model.args.hidden], tmp[:,self.model.args.hidden:]
            if self.model.distributional:
                reward = F.softmax(reward-reward.logsumexp(dim=-1, keepdim=True), -1)
                reward = torch.sum((reward * self.model.reward_support),-1)
            reward = reward[0,0].item()
        else:
            next_hidden_state = hidden_state
            reward = 0.0

        #get policy and value
        actions_hidden = self.model._model['modules']['actions_predict_hidden'](next_hidden_state) #hidden_state [1, hidden_size]
        assert self.model.max_predict_range == 1 #this does not work with self.max_predict_range>1
        current_actions_V = self.model._model['modules']['V_actions_predict'](actions_hidden)#.view(-1, self.max_predict_range, self.action_dim)
        if not self.model.distributional:
            value = current_actions_V[:,0]
            dis_logits = current_actions_V[:,1:(self.model.action_dim-2+1)]
            zero_logits = current_actions_V[:,(self.model.action_dim-2+1):(self.model.action_dim+1)]
        else:
            value = current_actions_V[:,:self.model.num_support_value]
            dis_logits = current_actions_V[:,self.model.num_support_value:(self.model.action_dim-2+self.model.num_support_value)]
            zero_logits = current_actions_V[:,(self.model.action_dim-2+self.model.num_support_value):(self.model.action_dim+self.model.num_support_value)]
            value = F.softmax(value-value.logsumexp(dim=-1, keepdim=True), -1)
            value = torch.sum((value * self.model.value_support),-1)
        value = self.model.inverse_map_reward_V(value[0].item())
        dis_logits = torch.split(dis_logits, self.model.action_split, -1)
        cam_logits = self.model._model['modules']['camera_predict'](actions_hidden).reshape(1,-1)
        actions = OrderedDict()
        ii = 0
        entropy = []
        for a in self.model.action_dict:
            if a != 'camera':
                actions[a] = F.softmax(dis_logits[ii] - dis_logits[ii].logsumexp(dim=-1, keepdim=True))[0,:]
                entropy.append(torch.distributions.Categorical(probs=actions[a]).entropy())
                actions[a] = actions[a].data.cpu().numpy()
                ii += 1
            else:
                actions[a] = F.softmax(cam_logits - cam_logits.logsumexp(dim=-1, keepdim=True))[0,:]
                actions['zero'] = F.softmax(zero_logits - zero_logits.logsumexp(dim=-1, keepdim=True))[0,:]
                #calculate real entropy of camera prob. 
                r_cam_probs = actions[a].clone()
                r_cam_probs_delta = torch.zeros_like(r_cam_probs).reshape((self.model.cam_map_size,self.model.cam_map_size))
                r_cam_probs_delta[self.model.cam_map_size//2, self.model.cam_map_size//2] = 1.0
                r_cam_probs_delta = r_cam_probs_delta.reshape((-1))
                r_cam_probs = (r_cam_probs * actions['zero'][0]) + (r_cam_probs_delta * actions['zero'][1])
                entropy.append(torch.distributions.Categorical(probs=r_cam_probs).entropy())
                actions[a] = actions[a].data.cpu().numpy()
                actions['zero'] = actions['zero'].data.cpu().numpy()
        entropy = sum(entropy).item()

        network_output = {}
        network_output['hidden_state'] = next_hidden_state
        network_output['reward'] = reward
        network_output['value'] = value
        network_output['actions'] = actions
        network_output['entropy'] = entropy
        return network_output

    def target_num_children(self, node):
        return max(min(int((node.entropy / self.max_policy_entropy) * self.model.num_policy_samples),
                       self.model.num_policy_samples), 1)

    def correct_num_children(self, node):
        #check if new children needed
        nchl = self.target_num_children(node)
        while nchl > len(node.children):
            #sample action
            n_action, prior = node.sampler.sample()
            #for x in range(-2, 3):
            #    for y in range(-2, 3):
            #        rem_action = n_action.copy()
            #        rem_action['camera'] = (rem_action['camera'] + x + y * self.model.cam_map_size) % (self.model.cam_map_size ** 2)
            #        node.sampler.remove_sample(rem_action) #TODO filter for double sampling through 'zero'
            node.sampler.remove_sample(n_action)
            node.prior_sum += prior
            t_action = OrderedDict()
            t_action_one_hot = []
            for a in self.model.action_dict:
                if a != 'camera':
                    t_action[a] = n_action[a]
                    t_action_one_hot.append(np.zeros((self.model.action_dict[a].n)))
                    t_action_one_hot[-1][n_action[a]] = 1.0
                else:
                    cam_choose = n_action[a]
                    cam_choose_x = (((cam_choose % self.model.cam_map_size) + np.random.random_sample()) * (2.0/self.model.cam_map_size)) - 1.0
                    cam_choose_y = (((cam_choose // self.model.cam_map_size) + np.random.random_sample()) * (2.0/self.model.cam_map_size)) - 1.0
                    t_action[a] = np.array([cam_choose_x, cam_choose_y], dtype=np.float) * (1 - n_action['zero'])
                    t_action_one_hot.append(t_action[a])
            t_action_one_hot = [torch.tensor(t, dtype=torch.float32) for t in t_action_one_hot]
            t_action_one_hot = torch.cat(t_action_one_hot, 0).unsqueeze(0)
            node.action_samples.append(t_action)
            if node.action_stack is None:
                node.action_stack = t_action_one_hot
            else:
                node.action_stack = torch.cat([node.action_stack, t_action_one_hot], 0)
            node.children[len(node.children)] = Node(prior)

    def ucb_score(self, parent, child):
        p_visit = parent.visit_count
        c_visit = child.visit_count
        pb_c = math.log((p_visit + 19652.0 + 1.0)/19652.0)+1.25
        pb_c *= math.sqrt(p_visit) / (c_visit + 1)
        value_reward = child.reward + GAMMA * child.value_()
        value_score  = (value_reward - self.min_V) / (self.max_V - self.min_V)
        return pb_c * (child.prior / parent.prior_sum) + value_score

    def select_child(self, node):
        #self.correct_num_children(node)
        _, action, child = max(
            (self.ucb_score(node, child), action, child)
            for action, child in node.children.items()
        )
        return action, child
    
    def backpropagate(self, search_path, value):
        for node in reversed(search_path):
            node.value_sum += value
            node.visit_count += 1
            self.min_V = min(value, self.min_V)
            self.max_V = max(value, self.max_V)
            value = node.reward + GAMMA * value



class Node():
    def __init__(self, prior):
        self.visit_count = 0
        self.to_play = -1
        self.action_samples = []
        self.action = None
        self.actions_hidden = None
        self.value = None
        self.value_sum = 0
        self.current_actions_V = None
        self.action_stack = None
        self.children = {}
        self.hidden_state = None
        self.reward = 0
        self.prior = prior
        self.prior_sum = 0.0
        self.parent = None
        self.entropy = 0.0
        self.sampler = None

    def expanded(self):
        return len(self.children) > 0

    def value_(self):
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count

class MuZeroModel():
    def __init__(self, state_shape, action_dict, args, time_deltas, cuda_enable=True, only_visnet=False):
        assert isinstance(state_shape, dict)
        self.p_count = 0
        self.state_shape = state_shape
        self.action_dict = copy.deepcopy(action_dict)
        self.state_keys = ['state', 'pov', 'history_action', 'history_reward', 'orientation', 'env_type']
        self.args = args
        assert self.args.needs_add_data
        self.MEMORY_MODEL = self.args.needs_embedding
        self.CUDA = cuda_enable
        print("CUDA",self.CUDA)

        self.action_dim = 2
        self.action_split = []
        self.camera_start = 100
        for a in action_dict:
            if a != 'camera':
                self.action_dim += action_dict[a].n
                self.action_split.append(action_dict[a].n)
            else:
                self.camera_start = self.action_dim
        self.time_deltas = time_deltas
        self.zero_time_point = 0
        for i in range(0,len(self.time_deltas)):
            if self.time_deltas[len(self.time_deltas)-i-1] == 0:
                self.zero_time_point = len(self.time_deltas)-i-1
                break
        #self.max_predict_range = 0
        #for i in range(self.zero_time_point+1,len(self.time_deltas)):
        #    if self.time_deltas[i] - self.max_predict_range == 1:
        #        self.max_predict_range = self.time_deltas[i]
        #    else:
        #        break
        #self.max_predict_range //= 2
        #print("MAX PREDICT RANGE:",self.max_predict_range)
        self.max_predict_range = 1
        assert self.max_predict_range == 1 #check if self.max_predict_range>1 can work

        self.train_iter = 0

        self.disc_embed_dims = []
        self.sum_logs = float(sum([np.log(d) for d in self.disc_embed_dims]))
        self.total_disc_embed_dim = sum(self.disc_embed_dims)
        self.continuous_embed_dim = self.args.hidden-self.total_disc_embed_dim

        self.num_policy_samples = 8
        self.max_steps_predict = MAX_STEPS_PREDICT
        self.num_mcts_steps = 15

        self.num_support_value = 80
        self.value_start = 0.0
        self.value_end = 10.0
        self.num_support_reward = 4
        self.reward_start = 0.0
        self.reward_end = 4.0
        self.distributional = True


        self.only_visnet = only_visnet
        self._model = self._make_model()

        #camera map coeff
        x_1 = 0.7
        y_1 = 90.0/180.0
        v = np.array([1.0, y_1, y_1])
        M = np.array([[1.0, 1.0, 1.0],
                      [x_1**2, x_1, 1.0],
                      [2*x_1**2, x_1, 0.0]])
        self.cam_coeffs = np.linalg.solve(M, v)
        self.x_1 = x_1
        self.y_1 = y_1

        if self.CUDA:
            self.mcts = MCTS_CUDA(self)
        else:
            self.mcts = MCTS_CPU(self)

    def load_state_dict(self, s, strict=True):
        if isinstance(s, list):
            if self.only_visnet:
                state_dict = {n: v for n, v in s[0].items() if n.startswith('pov_embed.')}
            else:
                state_dict = s[0]
        else:
            if self.only_visnet:
                state_dict = {n: v for n, v in s.items() if n.startswith('pov_embed.')}
            else:
                state_dict = s
        if torch.cuda.is_available() and self.CUDA:
            for p in state_dict:
                state_dict[p] = state_dict[p].cuda()
        self._model['modules'].load_state_dict(state_dict, strict)

    def loadstore(self, filename, load=True):
        if load:
            print('Load actor')
            self.train_iter = torch.load(filename + 'train_iter')
            state_dict = torch.load(filename + 'muzero')
            if self.only_visnet:
                state_dict = {n: v for n, v in state_dict.items() if n.startswith('pov_embed.')}
            self.load_state_dict(state_dict)
            if 'Lambda' in self._model:
                self._model['Lambda'] = torch.load(filename + 'muzeroLam')
        else:
            torch.save(self._model['modules'].state_dict(), filename + 'muzero')
            torch.save(self.train_iter, filename + 'train_iter')
            if 'Lambda' in self._model:
                torch.save(self._model['Lambda'], filename + 'muzeroLam')

    @staticmethod
    def needed_state_info(args):
        MEMORY_MODEL = args.needs_embedding
        if MEMORY_MODEL:
            #the memory model is used
            #past states have to be send to the trainer process
            num_past_times = 32
            args.num_past_times = num_past_times
            past_range_seconds = 60.0
            past_range = past_range_seconds / 0.05
            delta_a = (past_range-num_past_times-1)/(num_past_times**2)
            past_times = [-int(delta_a*n**2+n+1) for n in range(num_past_times)]
            past_times.reverse()
            time_skip = MAX_STEPS_PREDICT
            time_deltas = past_times + list(range(0, (time_skip*2)+1, 1))
            next_past_times = [-int(delta_a*n**2+n+1) + time_skip for n in range(num_past_times)]
            next_past_times.reverse()
            time_deltas = next_past_times + time_deltas
        else:
            args.num_past_times = 0
            time_skip = MAX_STEPS_PREDICT #number of steps in mcts
            time_deltas = list(range(0, time_skip+1, 1))
        pov_time_deltas = [0]
        if ON_POLICY:
            pov_time_deltas.append(time_skip)
        return {'min_num_future_rewards': 0}, time_deltas, pov_time_deltas

    def generate_embed(self, state_povs):
        with torch.no_grad():
            if len(state_povs.shape) == 3:
                state_povs = state_povs[None, :]
            state_povs = variable(state_povs)
            state_povs = (state_povs.permute(0, 3, 1, 2)/255.0)*2.0 - 1.0
            if len(self.disc_embed_dims) == 0:
                pov_embed = self._model['modules']['pov_embed'](state_povs)
                return {'pov_embed': pov_embed.data.cpu()}
            else:
                assert False #not implemented

    def embed_desc(self):
        desc = {'value': {'compression': False, 'shape': [], 'dtype': np.float32}}
        if self.MEMORY_MODEL:
            desc['pov_embed'] = {'compression': False, 'shape': [self.continuous_embed_dim+sum(self.disc_embed_dims)], 'dtype': np.float32}
            if len(self.disc_embed_dims) == 0:
                return desc
            else:
                assert False #not implemented
        else:
            return desc

    def map_target_reward_V(self, x):
        return torch.sign(x)*(torch.sqrt(torch.abs(x)+1.0)-1.0)+0.01*x

    def inverse_map_reward_V(self, x):
        if x > 0.0:
            return ((np.sqrt(1.0+4*0.01*(np.abs(x)+1.0+0.01))-1.0)/(2*0.01))**2-1.0
        else:
            return 1.0-((np.sqrt(1.0+4*0.01*(np.abs(x)+1.0+0.01))-1.0)/(2*0.01))**2
        
    def inverse_map_reward_V_torch(self, x):
        y = ((torch.sqrt(1.0+4*0.01*(torch.abs(x)+1.0+0.01))-1.0)/(2*0.01))**2-1.0
        return torch.where(x > 0.0, y, -y)

    def _make_model(self):
        self.device = torch.device("cuda") if (torch.cuda.is_available() and self.CUDA) else torch.device("cpu")
        modules = torch.nn.ModuleDict()
        modules['pov_embed'] = VisualNet(self.state_shape, self.args, self.continuous_embed_dim, self.disc_embed_dims, False)
        self.cnn_embed_dim = self.continuous_embed_dim+sum(self.disc_embed_dims)
        if self.only_visnet:
            if torch.cuda.is_available() and self.CUDA:
                modules = modules.cuda()
            return {'modules': modules}

        self.state_dim = 0
        for o in self.state_shape:
            if o != 'pov' and o in self.state_keys:
                self.state_dim += self.state_shape[o][0]
        print('Input state dim', self.state_dim)
        #input to memory is state_embeds
        self.complete_state = self.cnn_embed_dim + self.state_dim
        if self.MEMORY_MODEL:
            print('TODO add state_dim to memory')
            modules['memory'] = MemoryModel(self.args, self.cnn_embed_dim, self.past_time[-self.args.num_past_times:])
            self.memory_embed = modules['memory'].memory_embed_dim
            self.complete_state += self.memory_embed

        def normalize(x):
            min_x = torch.min(x, -1, keepdim=True)[0]
            max_x = torch.max(x, -1, keepdim=True)[0]
            return (x - min_x) / (max_x - min_x)
        def normalize_part(x):
            hidden_stat = x[...,:-1]
            min_x = torch.min(hidden_stat, -1, keepdim=True)[0]
            max_x = torch.max(hidden_stat, -1, keepdim=True)[0]
            return torch.cat([swish((hidden_stat - min_x) / (max_x - min_x)), x[...,-1:] * 0.1], -1)
        def normalize_part_dist(x):
            hidden_stat = x[...,:self.args.hidden]
            min_x = torch.min(hidden_stat, -1, keepdim=True)[0]
            max_x = torch.max(hidden_stat, -1, keepdim=True)[0]
            return torch.cat([swish((hidden_stat - min_x) / (max_x - min_x)), x[...,self.args.hidden:]], -1)
        def post_process_V(x):
            x[..., 0] *= 0.1
            return x
        modules['state_hidden'] = torch.nn.Sequential(torch.nn.Linear(self.complete_state, self.args.hidden),
                                                         Lambda(lambda x: normalize(swish(x))))
        modules['actions_predict_hidden'] = torch.nn.Sequential(torch.nn.Linear(self.args.hidden, self.args.hidden),
                                                         Lambda(lambda x: swish(x)))

        modules_V_actions_predict = [torch.nn.Linear(self.args.hidden, (1 if not self.distributional else self.num_support_value)+(self.action_dim-2)*self.max_predict_range+2*self.max_predict_range)]
        if not self.distributional:
            modules_V_actions_predict.append(Lambda(lambda x: post_process_V(x)))
        modules['V_actions_predict'] = torch.nn.Sequential(*modules_V_actions_predict)

        modules['camera_predict'] = torch.nn.Sequential(torch.nn.Linear(self.args.hidden, 8*4*4),
                                                        Lambda(lambda x: swish(x.reshape(-1, 8, 4, 4))),
                                                        torch.nn.ConvTranspose2d(8, 8, 3, 2), #9x9
                                                        torch.nn.BatchNorm2d(8),
                                                        Lambda(lambda x: swish(x)),
                                                        torch.nn.ConvTranspose2d(8, 6, 3, 2), #19x19
                                                        torch.nn.BatchNorm2d(6),
                                                        Lambda(lambda x: swish(x)),
                                                        torch.nn.ConvTranspose2d(6, self.max_predict_range, 5, 3),
                                                        CameraBias(59, self.device)) #59x59
        self.cam_map_size = 59

        #modules['dynamics'] = torch.nn.LSTM(self.action_dim, self.args.hidden, 2)
        modules_dynamics = [torch.nn.Linear(self.args.hidden+self.action_dim*self.max_predict_range, self.args.hidden),
                                                         Lambda(lambda x: swish(x)),
                                                         torch.nn.Linear(self.args.hidden, self.args.hidden),
                                                         Lambda(lambda x: swish(x)),
                                                         torch.nn.Linear(self.args.hidden, self.args.hidden+(1 if not self.distributional else self.num_support_reward))] #also returns reward
        if not self.distributional:
            modules_dynamics.append(Lambda(lambda x: normalize_part(x)))
        else:
            modules_dynamics.append(Lambda(lambda x: normalize_part_dist(x)))
        modules['dynamics'] = torch.nn.Sequential(*modules_dynamics)
        rand_crop = kornia.augmentation.RandomCrop((self.state_shape['pov'][0], self.state_shape['pov'][1]))


        self.revidx = [i for i in range(self.args.num_past_times-1, -1, -1)]
        self.revidx = torch.tensor(self.revidx).long()

        self.gamma = np.array([[(GAMMA**n if n >= 0 else 0) for n in range(-s,self.max_steps_predict+1-s)] for s in range(self.max_steps_predict)])
        self.gamma = torch.tensor(self.gamma, dtype=torch.float)

        if self.distributional:
            self.reward_support = torch.arange(self.reward_start, self.reward_end, (self.reward_end-self.reward_start)/self.num_support_reward).unsqueeze(0).unsqueeze(0)
            self.value_support = torch.arange(self.value_start, self.value_end, (self.value_end-self.value_start)/self.num_support_value).unsqueeze(0)

        if INFODIM and self.args.ganloss == "fisher":
            lam = torch.zeros((1,), requires_grad=True)

        if torch.cuda.is_available() and self.CUDA:
            modules = modules.cuda()
            self.revidx = self.revidx.cuda()
            self.gamma = self.gamma.cuda()
            self.reward_support = self.reward_support.cuda()
            self.value_support = self.value_support.cuda()
            if INFODIM and self.args.ganloss == "fisher":
                lam = lam.cuda()
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        
        ce_loss = prio_utils.WeightedCELoss()
        loss = prio_utils.WeightedMSELoss()
        targ_loss = prio_utils.WeightedMSETargetLoss()
        l1loss = torch.nn.L1Loss()
        l2loss = torch.nn.MSELoss()
        
        model_params = []
        nec_value_params = []
        for name, p in modules.named_parameters():
            if 'values.weight' in name:
                nec_value_params.append(p)
            else:
                model_params.append(p)

        optimizer = RAdam(model_params, self.args.lr, weight_decay=1e-6)

        model = {'modules': modules, 
                 'opt': optimizer, 
                 'ce_loss': ce_loss,
                 'l1loss': l1loss,
                 'l2loss': l2loss,
                 'loss': loss,
                 'targ_loss': targ_loss,
                 'rand_crop': rand_crop}
        if len(nec_value_params) > 0:
            nec_optimizer = torch.optim.Adam(nec_value_params, self.args.nec_lr)
            model['nec_opt'] = nec_optimizer
        if INFODIM and self.args.ganloss == "fisher":
            model['Lambda'] = lam
        return model

    def inverse_map_camera(self, x):
        x = x / 180.0
        #x = torch.sign(x)*torch.pow(torch.abs(x), 1.0/2.0)
        b = -self.cam_coeffs[1] +\
           torch.sqrt(torch.abs(self.cam_coeffs[1]**2-4*self.cam_coeffs[0]*(self.cam_coeffs[2]-torch.abs(x))))
        b /= 2.0 * self.cam_coeffs[0]
        x = torch.where(torch.abs(x) < self.y_1, x*self.x_1/self.y_1, 
                        torch.sign(x)*b)
        return x

    def map_camera(self, x):
        #x = np.sign(x) * x ** 2
        x = np.where(np.abs(x) < self.x_1, x*self.y_1/self.x_1, 
                     np.sign(x)*(self.cam_coeffs[0]*x**2+self.cam_coeffs[1]*np.abs(x)+self.cam_coeffs[2]))
        return x * 180.0

    def map_camera_torch(self, x):
        #x = np.sign(x) * x ** 2
        x = torch.where(torch.abs(x) < self.x_1, x*self.y_1/self.x_1, 
                     torch.sign(x)*(self.cam_coeffs[0]*x**2+self.cam_coeffs[1]*torch.abs(x)+self.cam_coeffs[2]))
        return x * 180.0

    def get_complete_embed(self, model, pov_state, state, past_embeds=None):
        #get pov embed
        pov_embed = model['modules']['pov_embed'](pov_state)
        embed = [pov_embed]

        if self.MEMORY_MODEL:
            #make memory embed
            memory_embed = model['modules']['memory'](past_embeds, pov_embed)
            embed.append(memory_embed)

        #combine all embeds
        for o in self.state_shape:
            if o in self.state_keys and o != 'pov':
                embed.append(state[o])
        embed = torch.cat(embed, dim=-1)
        return embed

    def sample_camera(self, cam_map):
        cam_choose = torch.distributions.Categorical(logits=cam_map).sample().reshape(-1, self.max_predict_range) #[batch_size,self.max_predict_range]
        cam_choose_x = ((cam_choose % self.cam_map_size) + torch.zeros((cam_choose.shape[0], cam_choose.shape[1]), device=self.device).uniform_()) * (2.0/self.cam_map_size) - 1.0
        cam_choose_y = ((cam_choose // self.cam_map_size) + torch.zeros((cam_choose.shape[0], cam_choose.shape[1]), device=self.device).uniform_()) * (2.0/self.cam_map_size) - 1.0
        return torch.stack([cam_choose_x, cam_choose_y], dim=-1), cam_choose

    def pretrain(self, input, worker_num=None):
        return self.train({}, input, worker_num)

    def variable(self, x, ignore_type=False):
        return variable(x, ignore_type, cuda=self.CUDA)

    def train(self, input, expert_input, worker_num=None):
        assert self.CUDA
        if 'done' not in input:
            pretrain = True
        else:
            pretrain = False
        for n in self._model:
            if 'opt' in n:
                self._model[n].zero_grad()
        embed_keys = []
        for k in self.embed_desc():
            embed_keys.append(k)
        states = {}
        for k in expert_input:
            if k in self.state_keys or k in embed_keys:
                if pretrain:
                    states[k] = self.variable(expert_input[k])
                else:
                    states[k] = torch.cat([self.variable(input[k]),self.variable(expert_input[k])],dim=0)
        if pretrain:
            actions = {'camera': self.inverse_map_camera(self.variable(expert_input['camera']))}
        else:
            actions = {'camera': self.inverse_map_camera(torch.cat([self.variable(input['camera']),self.variable(expert_input['camera'])],dim=0))}
        #quantize camera actions
        actions['camera_quant'] = torch.clamp(((actions['camera'] * 0.5 + 0.5) * self.cam_map_size).long(), max=self.cam_map_size-1)
        actions['camera_quant'] = actions['camera_quant'][:,:,0] + actions['camera_quant'][:,:,1] * self.cam_map_size
        for a in self.action_dict:
            if a != 'camera':
                if pretrain:
                    actions[a] = self.variable(expert_input[a], True).long()
                else:
                    actions[a] = torch.cat([self.variable(input[a], True).long(),self.variable(expert_input[a], True).long()],dim=0)
        if pretrain:
            reward = self.variable(expert_input['reward'])
            value_b = self.variable(expert_input['value'])
            done = self.variable(expert_input['done'], True).int()
        else:
            reward = torch.cat([self.variable(input['reward']),self.variable(expert_input['reward'])],dim=0)
            value_b = torch.cat([self.variable(input['value']),self.variable(expert_input['value'])],dim=0)
            done = torch.cat([self.variable(input['done'], True).int(),self.variable(expert_input['done'], True).int()],dim=0)
        future_done = torch.cumsum(done[:,self.zero_time_point:],1).clamp(max=1).bool()
        reward[:,(self.zero_time_point+1):] = reward[:,(self.zero_time_point+1):].masked_fill(future_done[:,:-1], 0.0)
        reward = reward[:,self.zero_time_point:]
        value_b = value_b[:,self.zero_time_point:]
        batch_size = reward.shape[0]

        if 'is_weight' in expert_input:
            if pretrain:
                is_weight = self.variable(expert_input['is_weight'])
                bc_is_weight = self.variable(expert_input['is_weight'])
            else:
                is_weight = torch.cat([self.variable(input['is_weight']),self.variable(expert_input['is_weight'])],dim=0)
                bc_is_weight = torch.cat([self.variable(input['is_weight']),self.variable(expert_input['is_weight'])],dim=0)
        else:
            is_weight = torch.ones((reward.shape[0],), device=self.device).float()
            bc_is_weight = torch.ones((reward.shape[0],), device=self.device).float()
            bc_is_weight[:(is_weight.shape[0]//2)] *= min(self.train_iter / 50000.0, 1.0)
        states['pov'] = (states['pov'].float()/255.0)*2.0 - 1.0
        states['pov'] = states['pov'].permute(0, 1, 4, 2, 3)
        current_state = {}
        next_state = {}
        for k in states:
            if k != 'pov':
                current_state[k] = states[k][:,self.zero_time_point]
                next_state[k] = states[k][:(batch_size//2),(self.zero_time_point+self.max_steps_predict)]
        current_actions = {}
        for k in actions:
            current_actions[k] = actions[k][:,self.zero_time_point:(self.zero_time_point+self.max_steps_predict)]

        if self.MEMORY_MODEL:
            with torch.no_grad():
                past_embeds = states['pov_embed'][:,(self.zero_time_point-self.args.num_past_times):self.zero_time_point]
                #mask out from other episode
                past_done = done[:,(self.zero_time_point-self.args.num_past_times):self.zero_time_point]
                past_done = past_done[:,self.revidx]
                past_done = torch.cumsum(past_done,1).clamp(max=1).bool()
                past_done = past_done[:,self.revidx].unsqueeze(-1).expand(-1,-1,past_embeds.shape[-1])
                past_embeds.masked_fill_(past_done, 0.0)

                if ON_POLICY:
                    with torch.no_grad():
                        assert False #not implemented
        else:
            past_embeds = None
            next_embeds = None

                
        #get complete embeds of input
        state_pov = states['pov'][:,0]
        if REGULARIZE:
            state_pov = F.pad(state_pov, (PAD_SIZE, PAD_SIZE, PAD_SIZE, PAD_SIZE), mode='replicate')
            state_pov = self._model['rand_crop'](state_pov)
        tembed = self.get_complete_embed(self._model, state_pov, current_state, past_embeds)

        hidden_state = self._model['modules']['state_hidden'](tembed)

        if ON_POLICY and not pretrain:
            with torch.no_grad():
                if REGULARIZE:
                    value_b = value_b[:,:self.max_steps_predict]
                    value_b[:(batch_size//2)] = 0.0
                    for i in range(NUM_REG_SAMPLE):
                        state_pov = states['pov'][:(batch_size//2),1].clone()
                        state_pov = F.pad(state_pov, (PAD_SIZE, PAD_SIZE, PAD_SIZE, PAD_SIZE), mode='replicate')
                        state_pov = self._model['rand_crop'](state_pov)
                        next_tembed = self.get_complete_embed(self._model, state_pov, next_state, next_embeds)

                        next_hidden_state = self._model['modules']['state_hidden'](next_tembed)
                        actions_hidden = self._model['modules']['actions_predict_hidden'](next_hidden_state)
                        actions_logits = self._model['modules']['V_actions_predict'](actions_hidden)
                        if not self.distributional:
                            V_value = actions_logits[:,0]
                        else:
                            V_value = actions_logits[:,:self.num_support_value]
                            V_value = F.softmax(V_value-V_value.logsumexp(dim=-1, keepdim=True), -1)
                            V_value = torch.sum((V_value * self.value_support),-1)
                        value_bpre = torch.cat([reward[:(batch_size//2),:self.max_steps_predict],V_value.unsqueeze(-1)], -1)
                        value_b[:(batch_size//2)] += torch.matmul(self.gamma.unsqueeze(0),value_bpre.unsqueeze(-1))[:,:,0]
                    value_b[:(batch_size//2)] /= NUM_REG_SAMPLE
                    value_b = value_b.detach()
                else:
                    next_tembed = self.get_complete_embed(self._model, states['pov'][:(batch_size//2),1], next_state, next_embeds)

                    next_hidden_state = self._model['modules']['state_hidden'](next_tembed)
                    actions_hidden = self._model['modules']['actions_predict_hidden'](next_hidden_state)
                    actions_logits = self._model['modules']['V_actions_predict'](actions_hidden)
                    if not self.distributional:
                        V_value = actions_logits[:,0]
                    else:
                        V_value = actions_logits[:,:self.num_support_value]
                        V_value = F.softmax(V_value-V_value.logsumexp(dim=-1, keepdim=True), -1)
                        V_value = torch.sum((V_value * self.value_support),-1)
                    value_bpre = torch.cat([reward[:(batch_size//2),:self.max_steps_predict],V_value.unsqueeze(-1)], -1)
                    value_b = value_b[:,:self.max_steps_predict]
                    value_b[:(batch_size//2)] = torch.matmul(self.gamma.unsqueeze(0),value_bpre.unsqueeze(-1))[:,:,0]
                    value_b = value_b.detach()

            
        value_b_map = self.map_target_reward_V(value_b)
        value_b_map = value_b_map.detach()
        if self.distributional:
            #quantise
            reward = ((reward - self.reward_start)*self.num_support_reward/(self.reward_end - self.reward_start)).long()
            reward = torch.clamp(reward, 0, self.num_support_reward-1)
            value_b_disc = ((value_b_map - self.value_start)*self.num_support_value/(self.value_end - self.value_start)).long()
            value_b_disc = torch.clamp(value_b_disc, 0, self.num_support_value-1)

        bc_loss = None
        v_loss = None
        r_loss = None
        if ON_POLICY and not pretrain:
            advantage = None
        for i in range(self.max_steps_predict):
            #get action logits and value
            actions_hidden = self._model['modules']['actions_predict_hidden'](hidden_state)
            actions_logits = self._model['modules']['V_actions_predict'](actions_hidden)
            if not self.distributional:
                V_value, dis_logits, zero_logits = actions_logits[:,0], actions_logits[:,1:(self.action_dim-2+1)], actions_logits[:,(self.action_dim-2+1):(self.action_dim+1)]
            else:
                V_value, dis_logits, zero_logits = actions_logits[:,:self.num_support_value], actions_logits[:,self.num_support_value:(self.action_dim-2+self.num_support_value)], actions_logits[:,(self.action_dim-2+self.num_support_value):(self.action_dim+self.num_support_value)]
            dis_logits = torch.split(dis_logits, self.action_split, -1)
            cam_map = self._model['modules']['camera_predict'](actions_hidden).reshape(-1,self.cam_map_size*self.cam_map_size)

            if ON_POLICY and not pretrain and advantage is None:
                with torch.no_grad():
                    V_value_num = F.softmax(V_value-V_value.logsumexp(dim=-1, keepdim=True), -1)
                    V_value_num = torch.sum((V_value_num * self.value_support),-1)
                    advantage = value_b[:,0] - self.inverse_map_reward_V_torch(V_value_num)
                    advantage = advantage.detach()
                    bc_is_weight *= advantage#F.relu(advantage)

            #get bc_loss
            ii = 0
            for a in self.action_dict:
                if a != 'camera':
                    if bc_loss is None:
                        bc_loss = self._model['targ_loss'](dis_logits[ii].reshape(-1, self.action_dict[a].n), current_actions[a][:,i], bc_is_weight)
                    else:
                        bc_loss += self._model['targ_loss'](dis_logits[ii].reshape(-1, self.action_dict[a].n), current_actions[a][:,i], bc_is_weight) / (self.max_steps_predict - 1)
                    ii += 1
                else:
                    bc_loss += self._model['targ_loss'](cam_map, current_actions['camera_quant'][:,i], bc_is_weight)
                    current_zero = ((torch.abs(current_actions[a][:,i,0]) < 1e-5) & (torch.abs(current_actions[a][:,i,1]) < 1e-5)).long()
                    bc_loss += self._model['targ_loss'](zero_logits, current_zero, bc_is_weight)

            #get v_loss
            if not self.distributional:
                if v_loss is None:
                    v_loss = self._model['loss'](V_value, (value_b_disc[:,i] * 0.5 + V_value.detach()* 0.5), is_weight)
                else:
                    v_loss += self._model['loss'](V_value, (value_b_disc[:,i] * 0.5 + V_value.detach()* 0.5), is_weight) / (self.max_steps_predict - 1)
            else:
                if v_loss is None:
                    v_loss = self._model['ce_loss'](V_value, value_b_disc[:,i], is_weight)
                else:
                    v_loss += self._model['ce_loss'](V_value, value_b_disc[:,i], is_weight) / (self.max_steps_predict - 1)
        
            #make next_hidden
            #encode actual actions
            current_actions_cat = []
            for a in self.action_dict:
                if a != 'camera':
                    current_actions_cat.append(one_hot(current_actions[a][:,i], self.action_dict[a].n, self.device))
                else:
                    current_actions_cat.append(current_actions[a][:,i])
            current_actions_cat = torch.cat(current_actions_cat, dim=1)

            tmp = self._model['modules']['dynamics'](torch.cat([hidden_state, current_actions_cat], -1))
            if not self.distributional:
                hidden_state, reward_p = tmp[:,:-1], tmp[:,-1]
            else:
                hidden_state, reward_p = tmp[:,:self.args.hidden], tmp[:,self.args.hidden:]
            hidden_state.register_hook(lambda grad: grad * 0.5)

            #get reward loss
            if not self.distributional:
                name_rloss = 'loss'
            else:
                name_rloss = 'ce_loss'
            if r_loss is None:
                r_loss = self._model[name_rloss](reward_p, reward[:,i], is_weight)
            else:
                r_loss += self._model[name_rloss](reward_p, reward[:,i], is_weight) / (self.max_steps_predict - 1)

        bc_loss /= 2.0#self.max_steps_predict
        v_loss /= 2.0#self.max_steps_predict
        r_loss /= 2.0#self.max_steps_predict
        loss = bc_loss + v_loss + r_loss
        rdict = {'loss': loss,
                 'bc_loss': bc_loss,
                 'v_loss': v_loss,
                 'r_loss': r_loss}
        if ON_POLICY and not pretrain:
            rdict.update({'adv': torch.mean(advantage)})
        if not self.distributional:
            rdict.update({'v_mean': torch.mean(V_value)})

        if 'Lambda' in self._model:
            if self.args.ganloss == 'fisher':
                self._model['Lambda'].retain_grad()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self._model['modules'].parameters(), 1000.0)
        for n in self._model:
            if 'opt' in n:
                self._model[n].step()
                
        if 'Lambda' in self._model and self.args.ganloss == 'fisher':
            self._model['Lambda'].data += self.args.rho * self._model['Lambda'].grad.data
            self._model['Lambda'].grad.data.zero_()
            rdict['Lambda'] = self._model['Lambda']

        #priority buffer stuff
        with torch.no_grad():
            if 'tidxs' in expert_input:
                #prioritize using bc_loss
                if pretrain:
                    rdict['prio_upd'] = [{'error': bc_loss.data.cpu().clone(),
                                          'replay_num': 0,
                                          'worker_num': worker_num[0]}]
                else:
                    rdict['prio_upd'] = [{'error': bc_loss[:batch_size].data.cpu().clone(),
                                          'replay_num': 0,
                                          'worker_num': worker_num[0]},
                                         {'error': bc_loss[batch_size:].data.cpu().clone(),
                                          'replay_num': 1,
                                          'worker_num': worker_num[1]}]

        for d in rdict:
            if d != 'prio_upd' and not isinstance(rdict[d], str) and not isinstance(rdict[d], int):
                rdict[d] = rdict[d].data.cpu().numpy()

        self.train_iter += 1
        rdict['train_iter'] = self.train_iter / 10000
        return rdict

    def reset(self):
        if self.MEMORY_MODEL:
            max_past = 1-min(self.past_time)
            self.past_embeds = torch.zeros((max_past,self.cnn_embed_dim), device=self.device, dtype=torch.float32, requires_grad=False)

    def select_action(self, state, last_reward, batch=False):
        tstate = {}
        if not batch:
            for k in state:
                tstate[k] = state[k][None, :].copy()
        tstate = self.variable(tstate)
        with torch.no_grad():
            #initial projection
            #pov_embed
            tstate['pov'] = (tstate['pov'].permute(0, 3, 1, 2)/255.0)*2.0 - 1.0
            pov_embed = self._model['modules']['pov_embed'](tstate['pov'])
            tembed = [pov_embed]
            if self.MEMORY_MODEL:
                assert False #not implemented past embeds

            #sample action
            for o in self.state_shape:
                if o in self.state_keys and o != 'pov':
                    tembed.append(tstate[o])
            tembed = torch.cat(tembed, dim=-1)

            #get initial hidden state
            init_state = self._model['modules']['state_hidden'](tembed)
            root = Node(1.0/self.num_policy_samples)
            root.hidden_state = init_state
            if self.CUDA:
                self.mcts.calc_all_V(root.hidden_state,[root])
                root.value_sum += root.value
                root.visit_count += 1
                self.mcts.expand_node(root)
                self.mcts.run_mcts(root)
            else:
                self.mcts.expand_node(root, self.mcts.recurrent_inference(root.hidden_state, None, True))
                self.mcts.run_mcts(root)
            visit_counts = np.array([child.visit_count for child in root.children.values()])
            best_i = np.argmax(visit_counts)
            action = OrderedDict()
            if self.p_count % 50 == 0:
                print("Value", root.value_())
                print("Visit count", visit_counts)
            self.p_count += 1
            if self.CUDA:
                for a in root.action:
                    if a == 'camera':
                        action[a] = self.map_camera(root.action[a][0,best_i].data.cpu().numpy())
                    else:
                        action[a] = root.action[a][0,best_i].item()
            else:
                action = root.action_samples[best_i]
                action['camera'] = self.map_camera(action['camera'])
            if self.MEMORY_MODEL:
                return action, {'pov_embed': pov_embed.data.cpu(), 'value': root.value_()}
            else:
                return action, {'value': root.value_()}

def main():
    #test for TreeSampler
    #np.random.seed(0)
    test_probs = {'A': [0.7,0.1,0.1,0.1],
                  'B': [0.2,0.8],
                  'C': [0.1,0.5,0.4]}
    treesampler = TreeSampler(test_probs)
    for i in range(np.prod([len(p) for p in test_probs.values()])):
        sample, p = treesampler.sample()
        treesampler.remove_sample(sample)
        print(sample, p)


if __name__ == '__main__':
    main()