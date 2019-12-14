import torch
import numpy as np
from all_experiments import HashingMemory, RAdam, prio_utils
from torch.nn import functional as F
from collections import OrderedDict
from utils import variable, Lambda, swish, Flatten, sample_gaussian, one_hot
import copy
import math
from modules import VisualNet, MemoryModel, CameraBias

GAMMA = 0.99
INFODIM = False


class MultiStep():
    def __init__(self, state_shape, action_dict, args, time_deltas, only_visnet=False):
        assert isinstance(state_shape, dict)
        self.state_shape = state_shape
        self.action_dict = copy.deepcopy(action_dict)
        self.state_keys = ['state', 'pov', 'history_action', 'history_reward', 'orientation', 'env_type']
        self.args = args
        self.MEMORY_MODEL = self.args.needs_embedding

        self.action_dim = 2
        self.action_split = []
        for a in action_dict:
            if a != 'camera':
                self.action_dim += action_dict[a].n
                self.action_split.append(action_dict[a].n)
        self.time_deltas = time_deltas
        self.zero_time_point = 0
        for i in range(0,len(self.time_deltas)):
            if self.time_deltas[len(self.time_deltas)-i-1] == 0:
                self.zero_time_point = len(self.time_deltas)-i-1
                break
        self.max_predict_range = 0
        for i in range(self.zero_time_point+1,len(self.time_deltas)):
            if self.time_deltas[i] - self.max_predict_range == 1:
                self.max_predict_range = self.time_deltas[i]
            else:
                break
        self.max_predict_range //= 2
        print("MAX PREDICT RANGE:",self.max_predict_range)

        self.train_iter = 0

        self.past_time = self.time_deltas[self.args.num_past_times:self.zero_time_point]
        self.future_time = self.time_deltas[(self.zero_time_point+1):]
        
        self.disc_embed_dims = []
        self.sum_logs = float(sum([np.log(d) for d in self.disc_embed_dims]))
        self.total_disc_embed_dim = sum(self.disc_embed_dims)
        self.continuous_embed_dim = self.args.hidden-self.total_disc_embed_dim

        self.running_Q_diff = None
        self.num_policy_samples = 32

        self.only_visnet = only_visnet
        self._model = self._make_model()
        self._target = None

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
        
    def build_target(self):
        self._target = self._make_model()
        self._last_target = self._make_model()
        self._target_interpolation = 0.0
        self._interpolation_speed = 50.0
        self._stage = 0

    
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
        self._model['modules'].load_state_dict(state_dict, strict)

    def loadstore(self, filename, load=True):
        if load:
            print('Load actor')
            self.train_iter = torch.load(filename + 'train_iter')
            state_dict = torch.load(filename + 'multistep')
            if self.only_visnet:
                state_dict = {n: v for n, v in state_dict.items() if n.startswith('pov_embed.')}
            self.load_state_dict(state_dict)
            if 'Lambda' in self._model:
                self._model['Lambda'] = torch.load(filename + 'multistepLam')
            if self._target is not None:
                self._stage = 2
                self._target['modules'].load_state_dict(state_dict)
                self._last_target['modules'].load_state_dict(state_dict)
        else:
            torch.save(self._model['modules'].state_dict(), filename + 'multistep')
            torch.save(self.train_iter, filename + 'train_iter')
            if 'Lambda' in self._model:
                torch.save(self._model['Lambda'], filename + 'multistepLam')
             
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
            time_skip = 10
            time_deltas = past_times + list(range(0, (time_skip*2)+1, 1))
            next_past_times = [-int(delta_a*n**2+n+1) + time_skip for n in range(num_past_times)]
            next_past_times.reverse()
            time_deltas = next_past_times + time_deltas
        else:
            args.num_past_times = 0
            time_skip = 10 #number of steps in the multi-step model
            time_deltas = list(range(0, (time_skip*2)+1, 1))
        pov_time_deltas = [0, time_skip]
        return {'min_num_future_rewards': 50}, time_deltas, pov_time_deltas

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
        desc = {}
        if self.MEMORY_MODEL:
            desc['pov_embed'] = {'compression': False, 'shape': [self.continuous_embed_dim+sum(self.disc_embed_dims)], 'dtype': np.float32}
            if len(self.disc_embed_dims) == 0:
                return desc
            else:
                assert False #not implemented
        else:
            return desc

    def _make_model(self):
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        modules = torch.nn.ModuleDict()
        modules['pov_embed'] = VisualNet(self.state_shape, self.args, self.continuous_embed_dim, self.disc_embed_dims, False)
        self.cnn_embed_dim = self.continuous_embed_dim+sum(self.disc_embed_dims)
        if self.only_visnet:
            if torch.cuda.is_available():
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

        modules['actions_predict_hidden'] = torch.nn.Sequential(torch.nn.Linear(self.complete_state, self.args.hidden),
                                                         Lambda(lambda x: swish(x)),
                                                         torch.nn.Linear(self.args.hidden, self.args.hidden),
                                                         Lambda(lambda x: swish(x)))

        modules['actions_predict'] = torch.nn.Linear(self.args.hidden, (self.action_dim-2)*self.max_predict_range+2*self.max_predict_range)#discrete probs + zero cont. probs

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

        modules['Q_values'] = torch.nn.Sequential(torch.nn.Linear(self.complete_state + (self.action_dim + 2) * self.max_predict_range, self.args.hidden*2),
                                                  Lambda(lambda x: swish(x)),
                                                  torch.nn.Linear(self.args.hidden*2, self.args.hidden),
                                                  Lambda(lambda x: swish(x)),
                                                  #HashingMemory.build(self.args.hidden, 1, self.args))
                                                  torch.nn.Linear(self.args.hidden, 1))

        self.revidx = [i for i in range(self.args.num_past_times-1, -1, -1)]
        self.revidx = torch.tensor(self.revidx).long()

        if INFODIM and self.args.ganloss == "fisher":
            lam = torch.zeros((1,), requires_grad=True)

        if torch.cuda.is_available():
            modules = modules.cuda()
            self.revidx = self.revidx.cuda()
            if INFODIM and self.args.ganloss == "fisher":
                lam = lam.cuda()
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        
        ce_loss = prio_utils.WeightedCELoss()
        loss = prio_utils.WeightedMSELoss()
        l1loss = torch.nn.L1Loss()
        l2loss = torch.nn.MSELoss()
        
        model_params = []
        nec_value_params = []
        for name, p in modules.named_parameters():
            if 'values.weight' in name:
                nec_value_params.append(p)
            else:
                model_params.append(p)

        self.gammas = GAMMA ** torch.arange(0, self.max_predict_range, device=self.device).float()#.unsqueeze(0).expand(self.args.er, -1)

        optimizer = RAdam(model_params, self.args.lr, weight_decay=1e-6)

        model = {'modules': modules, 
                 'opt': optimizer, 
                 'ce_loss': ce_loss,
                 'l1loss': l1loss,
                 'l2loss': l2loss,
                 'loss': loss}
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
        assert False #not implemented

    def train(self, input, expert_input, worker_num=None):
        for n in self._model:
            if 'opt' in n:
                self._model[n].zero_grad()
        embed_keys = []
        for k in self.embed_desc():
            embed_keys.append(k)
        states = {}
        for k in input:
            if k in self.state_keys or k in embed_keys:
                states[k] = torch.cat([variable(input[k]),variable(expert_input[k])],dim=0)
        actions = {'camera': self.inverse_map_camera(torch.cat([variable(input['camera']),variable(expert_input['camera'])],dim=0))}
        #quantize camera actions
        actions['camera_quant'] = torch.clamp(((actions['camera'] * 0.5 + 0.5) * self.cam_map_size).long(), max=self.cam_map_size-1)
        actions['camera_quant'] = actions['camera_quant'][:,:,0] + actions['camera_quant'][:,:,1] * self.cam_map_size
        for a in self.action_dict:
            if a != 'camera':
                actions[a] = torch.cat([variable(input[a], True).long(),variable(expert_input[a], True).long()],dim=0)
        batch_size = variable(input['reward']).shape[0]
        reward = torch.cat([variable(input['reward']),variable(expert_input['reward'])],dim=0)
        done = torch.cat([variable(input['done'], True).int(),variable(expert_input['done'], True).int()],dim=0)
        assert not 'r_offset' in input
        future_done = torch.cumsum(done[:,self.zero_time_point:],1).clamp(max=1).bool()
        reward[:,(self.zero_time_point+1):] = reward[:,(self.zero_time_point+1):].masked_fill(future_done[:,:-1], 0.0)

        is_weight = None
        if 'is_weight' in input:
            is_weight = torch.cat([variable(input['is_weight']),variable(expert_input['is_weight'])],dim=0)
        else:
            is_weight = torch.ones((reward.shape[0],), device=self.device).float()
        states['pov'] = (states['pov'].float()/255.0)*2.0 - 1.0
        states['pov'] = states['pov'].permute(0, 1, 4, 2, 3)
        current_state = {}
        for k in states:
            if k != 'pov':
                current_state[k] = states[k][:,self.zero_time_point]
        next_state = {}
        for k in states:
            if k != 'pov':
                next_state[k] = states[k][:,self.zero_time_point+self.max_predict_range]
        current_actions = {}
        for k in actions:
            current_actions[k] = actions[k][:,self.zero_time_point:(self.zero_time_point+self.max_predict_range)]
        next_actions = {}
        for k in actions:
            next_actions[k] = actions[k][:,(self.zero_time_point+self.max_predict_range):(self.zero_time_point+self.max_predict_range*2)]
        assert states['pov'].shape[1] == 2

        if self.MEMORY_MODEL:
            with torch.no_grad():
                past_embeds = states['pov_embed'][:,(self.zero_time_point-self.args.num_past_times):self.zero_time_point]
                #mask out from other episode
                past_done = done[:,(self.zero_time_point-self.args.num_past_times):self.zero_time_point]
                past_done = past_done[:,self.revidx]
                past_done = torch.cumsum(past_done,1).clamp(max=1).bool()
                past_done = past_done[:,self.revidx].unsqueeze(-1).expand(-1,-1,past_embeds.shape[-1])
                past_embeds.masked_fill_(past_done, 0.0)

            
                next_past_embeds = states['pov_embed'][:,:(self.zero_time_point-self.args.num_past_times)]
                #mask out from other episode
                past_done = done[:,:(self.zero_time_point-self.args.num_past_times)]
                past_done = past_done[:,self.revidx]
                past_done = torch.cumsum(past_done,1).clamp(max=1).bool()
                past_done = past_done[:,self.revidx].unsqueeze(-1).expand(-1,-1,past_embeds.shape[-1])
                next_past_embeds.masked_fill_(past_done, 0.0)
        else:
            past_embeds = None
            next_past_embeds = None

                
        #get complete embeds of input
        tembed = self.get_complete_embed(self._model, states['pov'][:,0], current_state, past_embeds)
        nembed = self.get_complete_embed(self._model, states['pov'][:,1], next_state, next_past_embeds)
        with torch.no_grad():
            nembed_target = self.get_complete_embed(self._target, states['pov'][:,1], next_state, next_past_embeds)
            nembed_last_target = self.get_complete_embed(self._last_target, states['pov'][:,1], next_state, next_past_embeds)
        
        #encode actual actions
        current_actions_cat = []
        for a in self.action_dict:
            if a != 'camera':
                current_actions_cat.append(one_hot(current_actions[a], self.action_dict[a].n, self.device).view(-1, self.action_dict[a].n*self.max_predict_range))
            else:
                current_actions_cat.append(current_actions[a].reshape(-1, 2*self.max_predict_range))
                current_zero = ((torch.abs(current_actions[a].view(-1, self.max_predict_range, 2)[:,:,0]) < 1e-5) & (torch.abs(current_actions[a].view(-1, self.max_predict_range, 2)[:,:,1]) < 1e-5)).float()
                current_actions_cat.append(1.0 - current_zero)
                current_actions_cat.append(current_zero)
        current_actions_cat = torch.cat(current_actions_cat, dim=1)

        actions_hidden = self._model['modules']['actions_predict_hidden'](nembed)
        actions_logits = self._model['modules']['actions_predict'](actions_hidden).view(-1, self.max_predict_range, self.action_dim)
        dis_logits, zero_logits = actions_logits[:,:,:(self.action_dim-2)], actions_logits[:,:,(self.action_dim-2):self.action_dim]
        dis_logits = torch.split(dis_logits, self.action_split, -1)
        cam_map = self._model['modules']['camera_predict'](actions_hidden).reshape(-1,self.max_predict_range,self.cam_map_size*self.cam_map_size)

        #get best (next) action
        next_Qs = None
        mean_Q = None
        loss = None
        all_actions = []
        all_Qs = []
        for s in range(self.num_policy_samples):
            ii = 0
            actions_cat = []
            c_actions = {}
            for a in self.action_dict:
                if a != 'camera':
                    c_actions[a] = torch.distributions.Categorical(logits=dis_logits[ii]).sample() # [batch, max_range]
                    actions_cat.append(one_hot(c_actions[a], self.action_dict[a].n, self.device).view(-1, self.action_dict[a].n*self.max_predict_range))
                    ii += 1
                else:
                    camera, camera_quant = self.sample_camera(cam_map)
                    zero_camera = torch.distributions.Categorical(logits=zero_logits).sample()
                    camera = torch.where((zero_camera == 1).unsqueeze(-1).expand(-1,-1,2), torch.zeros_like(camera), camera)
                    c_actions['camera'] = camera.clone()
                    c_actions['camera_quant'] = camera_quant.clone()
                    actions_cat.append(camera.view(-1, 2*self.max_predict_range))
                    actions_cat.append(1.0 - zero_camera.float())
                    actions_cat.append(zero_camera.float())
            actions_cat = torch.cat(actions_cat, dim=1)
            all_actions.append(c_actions)
            if self._stage != 0:
                with torch.no_grad():
                    if self._stage == 0:
                        c_next_Qs = torch.zeros((reward.shape[0], 1), device = self.device).float()
                    elif self._stage == 1:
                        c_next_Qs = self._target['modules']['Q_values'](torch.cat([nembed_target, actions_cat], dim=1)) * self._target_interpolation
                    elif self._stage == 2:
                        c_next_Qs = self._target['modules']['Q_values'](torch.cat([nembed_target, actions_cat], dim=1)) * self._target_interpolation +\
                            self._last_target['modules']['Q_values'](torch.cat([nembed_last_target, actions_cat], dim=1)) * (1.0 - self._target_interpolation)
                all_Qs.append(c_next_Qs.clone())
                if next_Qs is None:
                    next_Qs = c_next_Qs
                    #best_actions = c_actions
                    mean_Q = c_next_Qs[:,0]
                else:
                    #Q_mask = (next_Qs[:,0] < c_next_Qs[:,0]).view(-1, 1, 1)
                    #for a in best_actions:
                    #    best_actions[a] = torch.where(Q_mask.expand(-1, c_actions[a].shape[1], c_actions[a].shape[2]), c_actions[a], best_actions[a])
                    next_Qs = torch.max(next_Qs, c_next_Qs)
                    mean_Q += c_next_Qs[:,0]
            else:
                next_Qs = torch.zeros((reward.shape[0], 1), device = self.device).float()

        if self.train_iter % 1 == 0:
            #add bc loss
            entropy_bonus = 0.0
            bc_loss = None
            ii = 0
            for a in self.action_dict:
                if a != 'camera':
                    if bc_loss is None:
                        bc_loss = self._model['ce_loss'](dis_logits[ii][batch_size:].reshape(-1, self.action_dict[a].n), next_actions[a][batch_size:].reshape(-1), is_weight[batch_size:].reshape(-1, 1))
                    else:
                        bc_loss += self._model['ce_loss'](dis_logits[ii][batch_size:].reshape(-1, self.action_dict[a].n), next_actions[a][batch_size:].reshape(-1), is_weight[batch_size:].reshape(-1, 1))
                    ii += 1
                else:
                    bc_loss += self._model['ce_loss'](cam_map[batch_size:].reshape(-1,self.cam_map_size*self.cam_map_size), next_actions['camera_quant'][batch_size:].reshape(-1), is_weight[batch_size:].reshape(-1, 1))
                    current_zero = ((torch.abs(next_actions[a].view(-1, self.max_predict_range, 2)[batch_size:,:,0]) < 1e-5) & (torch.abs(next_actions[a].view(-1, self.max_predict_range, 2)[batch_size:,:,1]) < 1e-5)).long()
                    bc_loss += self._model['ce_loss'](zero_logits[batch_size:].reshape(-1, 2), current_zero.reshape(-1), is_weight[batch_size:].reshape(-1, 1))
            loss = bc_loss
            rdict = {'bc_loss': bc_loss}
            
            if self._stage != 0:
                dis_logits = [torch.nn.LogSoftmax(-1)(d) for d in dis_logits]
                zero_logits = torch.nn.LogSoftmax(-1)(zero_logits)
                cam_map = torch.nn.LogSoftmax(-1)(cam_map)

                #learn policy
                policy_loss = None
                mean_Q /= self.num_policy_samples
                for j in range(self.num_policy_samples):
                    advantage = ((all_Qs[j][:,0] - mean_Q)*is_weight).unsqueeze(-1).expand(-1, self.max_predict_range).detach()
                    ii = 0
                    for a in self.action_dict:
                        if a != 'camera':
                            if policy_loss is None:
                                policy_loss = -(advantage * torch.gather(dis_logits[ii], -1, all_actions[j][a].unsqueeze(-1))[:,:,0]).mean() + entropy_bonus * (dis_logits[ii].exp()*dis_logits[ii]).sum(-1).mean()
                            else:
                                policy_loss += -(advantage * torch.gather(dis_logits[ii], -1, all_actions[j][a].unsqueeze(-1))[:,:,0]).mean() + entropy_bonus * (dis_logits[ii].exp()*dis_logits[ii]).sum(-1).mean()
                            ii += 1
                        else:
                            policy_loss += -(advantage * torch.gather(cam_map, -1, all_actions[j]['camera_quant'].unsqueeze(-1))[:,:,0]).mean() + entropy_bonus * (cam_map.exp()*cam_map).sum(-1).mean()
                            current_zero = ((torch.abs(all_actions[j][a].view(-1, self.max_predict_range, 2)[:,:,0]) < 1e-5) & (torch.abs(all_actions[j][a].view(-1, self.max_predict_range, 2)[:,:,1]) < 1e-5)).long()
                            policy_loss += -(advantage * torch.gather(zero_logits, -1, current_zero.unsqueeze(-1))[:,:,0]).mean() + entropy_bonus * (zero_logits.exp()*zero_logits * is_weight.view(-1, 1, 1)).sum(-1).mean()
                loss += policy_loss / self.num_policy_samples
                rdict['policy_loss'] = policy_loss
        else:
            rdict = {}

        current_Qs = self._model['modules']['Q_values'](torch.cat([tembed, current_actions_cat], dim=1))
        next_Qs = next_Qs[:,0]
        target = (reward[:,self.zero_time_point:(self.zero_time_point+self.max_predict_range)] * self.gammas.unsqueeze(0).expand(reward.shape[0], -1)).sum(1) + (GAMMA ** self.max_predict_range) * next_Qs
        if is_weight is None:
            Q_loss = (current_Qs[:,0] - target.detach()).pow(2).mean()
        else:
            Q_loss = ((current_Qs[:,0] - target.detach()).pow(2) * is_weight.view(-1)).mean()
        if loss is None:
            loss = Q_loss.clone()
        else:
            loss += Q_loss
        with torch.no_grad():
            rdict['Q_diff'] = (torch.abs(current_Qs[:,0] - target)*is_weight).mean()/is_weight.mean()
            rdict['Q_std'] = current_Qs[:,0].std()
            rdict['current_Q_mean'] = current_Qs.mean()
            rdict['next_Q_mean'] = next_Qs.mean()

        
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
            if 'tidxs' in input:
                #prioritize using Q_diff
                rdict['prio_upd'] = [{'error': torch.abs(current_Qs[:,0] - target)[:batch_size].data.cpu().clone(),
                                      'replay_num': 0,
                                      'worker_num': worker_num[0]},
                                     {'error': torch.abs(current_Qs[:,0] - target)[batch_size:].data.cpu().clone(),
                                      'replay_num': 1,
                                      'worker_num': worker_num[1]}]

        for d in rdict:
            if d != 'prio_upd' and not isinstance(rdict[d], str) and not isinstance(rdict[d], int):
                rdict[d] = rdict[d].data.cpu().numpy()
        if self.train_iter > 1e3:
            int_tmp = np.power(rdict['Q_diff']/0.12, -8.0) / self._interpolation_speed
            if int_tmp > 0.1:
                int_tmp = 0.1
            self._target_interpolation += int_tmp
        if self._target_interpolation > 1.0:
            self._target_interpolation = 0.0
            if self._stage == 0:
                self._stage = 1
                self._target['modules'].load_state_dict(self._model['modules'].state_dict())
            else:
                self._stage = 2
                self._last_target['modules'].load_state_dict(self._target['modules'].state_dict())
                self._target['modules'].load_state_dict(self._model['modules'].state_dict())
        self.train_iter += 1
        rdict['train_iter'] = self.train_iter / 10000
        rdict['target_interpolation'] = self._target_interpolation
        return rdict

    def normal_pre_kl_divergence(self, mu1, log_var1):
        log_diff = log_var1 + 5.0
        nvar2 = np.exp(5.0)
        kl_values = -0.5 * (1 + log_diff - nvar2*(mu1.pow(2)) - log_diff.exp())
        kl_values = torch.sum(kl_values, dim=-1)
        kl_loss = torch.mean(kl_values)
        return kl_loss

    def normal2_kl_divergence(self, mu1, log_var1, mu2, log_var2):
        log_diff = log_var1 - log_var2
        nvar2 = (-log_var2).exp()
        kl_values = -0.5 * (1 + log_diff - nvar2*((mu1-mu2).pow(2)) - log_diff.exp())
        kl_values = torch.sum(kl_values, dim=-1)
        kl_loss = torch.mean(kl_values)
        return kl_loss
        
    def normal1_kl_divergence(self, mu, log_var):
        kl_values = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp())
        kl_values = torch.sum(kl_values, dim=-1)
        kl_loss = torch.mean(kl_values)
        #kl_loss = kl_values
        return kl_loss

    def normal_log_prob_dens(self, mu, log_var, loc):
        #log_var = log(sigma^2)
        sigma_m2 = (-log_var).exp()
        return -0.5*((loc-mu).pow(2)*sigma_m2+np.log(2.0*np.pi)+log_var)
    
    def reset(self):
        if self.MEMORY_MODEL:
            max_past = 1-min(self.past_time)
            self.past_embeds = torch.zeros((max_past,self.cnn_embed_dim), device=self.device, dtype=torch.float32, requires_grad=False)

    def select_action(self, state, last_reward, batch=False):
        tstate = {}
        if not batch:
            for k in state:
                tstate[k] = state[k][None, :].copy()
        tstate = variable(tstate)
        with torch.no_grad():
            #pov_embed
            tstate['pov'] = (tstate['pov'].permute(0, 3, 1, 2)/255.0)*2.0 - 1.0
            pov_embed = self._model['modules']['pov_embed'](tstate['pov'])
            tembed = [pov_embed]
            #memory_embed
            if self.MEMORY_MODEL:
                memory_inp = self.past_embeds[self.past_time].unsqueeze(0)
                memory_embed = self._model['modules']['memory'](memory_inp, pov_embed)
                self.past_embeds = self.past_embeds.roll(1, dims=0)
                self.past_embeds[0] = pov_embed[0]
                tembed.append(memory_embed)

            #sample action
            for o in self.state_shape:
                if o in self.state_keys and o != 'pov':
                    tembed.append(tstate[o])
            tembed = torch.cat(tembed, dim=-1)
            actions_hidden = self._model['modules']['actions_predict_hidden'](tembed)
            current_best_actions = self._model['modules']['actions_predict'](actions_hidden).view(-1, self.max_predict_range, self.action_dim)
            dis_logits, zero_logits = current_best_actions[:,:,:(self.action_dim-2)], current_best_actions[:,:,(self.action_dim-2):self.action_dim]
            dis_logits = torch.split(dis_logits, self.action_split, -1)
            action = OrderedDict()
            ii = 0
            for a in self.action_dict:
                if a != 'camera':
                    action[a] = torch.distributions.Categorical(logits=dis_logits[ii][:,0,:]).sample()[0].item()
                    ii += 1
                else:
                    cam_map = self._model['modules']['camera_predict'](actions_hidden)[:,0].reshape(1,-1)
                    cam_choose = torch.distributions.Categorical(logits=cam_map).sample()[0].item()
                    cam_choose_x = (((cam_choose % self.cam_map_size) + np.random.sample()) * (2.0/self.cam_map_size)) - 1.0
                    cam_choose_y = (((cam_choose // self.cam_map_size) + np.random.sample()) * (2.0/self.cam_map_size)) - 1.0
                    action[a] = np.array([cam_choose_x, cam_choose_y])
                    zero_act = torch.distributions.Categorical(logits=zero_logits[:,0,:]).sample()[0].item()
                    if zero_act == 1:
                        action[a] *= 0.0
                    action[a] = self.map_camera(action[a])
            if self.MEMORY_MODEL:
                return action, {'pov_embed': pov_embed.data.cpu()}
            else:
                return action, {}
