import torch
import numpy as np
from all_experiments import HashingMemory, RAdam, prio_utils
from torch.nn import functional as F
from collections import OrderedDict
from utils import variable, Lambda, swish, Flatten, sample_gaussian, one_hot
import copy
import math

GAMMA = 0.99
INFODIM = False

class VisualNet(torch.nn.Module):
    def __init__(self, state_shape, args, cont_dim=None, disc_dims=None, variational=False):
        super(VisualNet, self).__init__()
        self.args = args
        self.state_shape = state_shape
        self.variational = variational

        self.cont_dim = cont_dim
        self.disc_dims = disc_dims
        if disc_dims is None:
            self.disc_dims = []
        encnn_layers = []
        self.embed_dim = self.args.embed_dim
        self.temperature = self.args.temperature
        if self.cont_dim is not None:
            assert self.embed_dim == self.cont_dim + sum(self.disc_dims)
        assert len(self.state_shape['pov']) > 1
        if self.args.cnn_type == 'atari':
            white = False
            sizes = [8, 4, 3]
            strides = [4, 2, 1]
            batchnorm = [True, True, True]
            pooling = [1, 1, 1]
            filters = [32, 64, 32]
            padding = [0, 0, 0]
            end_size = 4
        elif self.args.cnn_type == 'mnist':
            white = False
            sizes = [3, 3, 3]
            strides = [1, 1, 1]
            batchnorm = [False, False, False]
            pooling = [2, 2, 2]
            filters = [32, 32, 32]
            padding = [0, 0, 0]
            end_size = 4
        elif self.args.cnn_type == 'adv':
            white = False
            sizes = [4, 3, 3, 4]
            strides = [2, 2, 2, 1]
            batchnorm = [True, True, True, True]
            pooling = [1, 1, 1, 1]
            filters = [16, 32, 64, 32]
            padding = [0, 0, 0, 0]
            end_size = 4
        elif self.args.cnn_type == 'dcgan':
            white = True
            filters = [32, 64, 128, 256, 400]
            sizes = [4, 4, 4, 4, 4]
            strides = [2, 2, 2, 2, 2]
            padding = [1, 1, 1, 1, 0]
            batchnorm = [False, True, True, True, False]
            pooling = [1, 1, 1, 1, 1]
            end_size = 1



        in_channels = self.state_shape['pov'][-1]

        #make encoder
        for i in range(len(sizes)):
            tencnn_layers = []
            if white and i == 0:
                tencnn_layers.append(torch.nn.BatchNorm2d(in_channels))

            tencnn_layers.append(torch.nn.Conv2d(
                in_channels,
                filters[i],
                sizes[i],
                stride=strides[i],
                padding=padding[i],
                bias=True
            ))

            if batchnorm[i]:
                tencnn_layers.append(torch.nn.BatchNorm2d(filters[i]))
            tencnn_layers.append(Lambda(lambda x: swish(x)))

            if pooling[i] > 1:
                tencnn_layers.append(torch.nn.MaxPool2d(pooling[i]))

            in_channels = filters[i]
            encnn_layers.append(torch.nn.Sequential(*tencnn_layers))

        tencnn_layers = []
        tencnn_layers.append(Flatten())
        if self.variational:
            tencnn_layers.append(torch.nn.Linear(in_channels*(end_size**2), self.cont_dim*2+sum(self.disc_dims)))
        else:
            tencnn_layers.append(torch.nn.Linear(in_channels*(end_size**2), self.embed_dim))
            tencnn_layers.append(Lambda(lambda x: swish(x)))
        encnn_layers.append(torch.nn.Sequential(*tencnn_layers))
        self.encnn_layers = torch.nn.ModuleList(encnn_layers)
        self.log_softmax = torch.nn.LogSoftmax(-1)

    def forward(self, state_pov):
        time_tmp = None
        if len(state_pov.shape) == 5:
            time_tmp = state_pov.shape[1]
            x = state_pov.contiguous().view(-1, self.state_shape['pov'][2], self.state_shape['pov'][0], self.state_shape['pov'][1])
        else:
            x = state_pov
        for _layer in self.encnn_layers:
            x = _layer(x)
        if self.variational:
            if time_tmp is not None:
                x = x.view(-1, time_tmp, self.cont_dim*2+sum(self.disc_dims))
            if len(self.disc_dims) > 0:
                cont, discs = torch.split(x, self.cont_dim*2, dim=-1)
            else:
                cont = x
            mu, log_var = torch.split(cont, self.cont_dim, dim=-1)
            if len(self.disc_dims) > 0:
                sdiscs = list(torch.split(discs, self.disc_dims, dim=-1))
                for i in range(len(sdiscs)):
                    sdiscs[i] = self.log_softmax(sdiscs[i])
                discs = torch.cat(sdiscs, dim=-1)
                return mu, log_var, discs
            else:
                return mu, log_var
        else:
            if time_tmp is not None:
                x = x.view(-1, time_tmp, self.embed_dim)
            return x

    def sample(self, mu, log_var, discs=None, training=True, hard=False):
        #sample continuous
        if training:
            cont_sample = sample_gaussian(mu, log_var)
        else:
            cont_sample = mu

        #sample discrete
        if len(self.disc_dims) > 0:
            sdiscs = list(torch.split(discs, self.disc_dims, dim=-1))
            disc_samples = [F.gumbel_softmax(d, self.temperature, (not training) or hard or ST_ENABLE) for d in sdiscs]
        else:
            disc_samples = []

        return torch.cat([cont_sample]+disc_samples, dim=-1)

class MemoryModel(torch.nn.Module):
    def __init__(self, args, embed_dim, past_time_deltas):
        super(MemoryModel, self).__init__()
        self.args = args

        #generate inner embeds
        max_len = len(past_time_deltas)
        self.conv1 = torch.nn.Sequential(Lambda(lambda x: x.transpose(-2, -1)),
                                         torch.nn.Conv1d(embed_dim, embed_dim, 1, 1),
                                         Lambda(lambda x: swish(x)),
                                         Lambda(lambda x: x.transpose(-2, -1)))
        self.conv_len = ((max_len - 1) // 1) + 1
        #add positional encoding
        self.pos_enc_dim = 16
        pe = torch.zeros((self.conv_len, self.pos_enc_dim))
        for j, pos in enumerate(past_time_deltas):
            for i in range(0, self.pos_enc_dim, 2):
                pe[j, i] = \
                math.sin(pos / (10000 ** (i/self.pos_enc_dim)))
                pe[j, i + 1] = \
                math.cos(pos / (10000 ** ((i + 1)/self.pos_enc_dim)))
                
        self.concat_or_add = False
        if self.concat_or_add:
            pe = pe.unsqueeze(0).expand(self.args.er, -1, -1).transpose(0, 1)
        else:
            pe = pe.unsqueeze(0).transpose(0, 1) / math.sqrt(float(embed_dim))
        self.register_buffer('pe', pe)

        #generate query using current pov_embed
        #TODO add also state of inventory?
        self.num_heads = 2
        self.query_gen = torch.nn.Sequential(torch.nn.Linear(embed_dim, embed_dim*2),
                                             torch.nn.LayerNorm(embed_dim*2),
                                             Lambda(lambda x: swish(x)),
                                             torch.nn.Linear(embed_dim*2, embed_dim*self.num_heads))
        if self.concat_or_add:
            self.attn1 = torch.nn.MultiheadAttention(embed_dim*self.num_heads, self.num_heads, vdim=embed_dim+self.pos_enc_dim, kdim=embed_dim)
        else:
            self.attn1 = torch.nn.MultiheadAttention(embed_dim*self.num_heads, self.num_heads, vdim=embed_dim, kdim=embed_dim)
        if self.concat_or_add:
            self.fclast = torch.nn.Sequential(torch.nn.Linear((embed_dim+self.pos_enc_dim)*self.num_heads, embed_dim*self.num_heads),
                                              torch.nn.LayerNorm(self.args.hidden),
                                              Lambda(lambda x: swish(x)))
            self.memory_embed_dim = embed_dim*self.num_heads
        else:
            self.fclast = torch.nn.Sequential(torch.nn.Linear(embed_dim*self.num_heads, embed_dim*self.num_heads),
                                              torch.nn.LayerNorm(embed_dim*self.num_heads),
                                              Lambda(lambda x: swish(x)))
            self.memory_embed_dim = embed_dim*self.num_heads

    def forward(self, x, current_pov_embed):
        assert len(x.shape) == 3
        x = self.conv1(x)
        if self.concat_or_add:
            value = torch.cat([x.transpose(0, 1), self.pe], dim=-1) #length, batch, embed_dim + position_dim
        else:
            value = x.transpose(0, 1)
            value[:,:,:self.pos_enc_dim] += self.pe
        key = x.transpose(0, 1) #length, batch, embed_dim
        query = self.query_gen(current_pov_embed).unsqueeze(0) #length, batch, embed_dim

        attn1_out, _ = self.attn1(query, key, value) #length, batch, (embed_dim + position_dim)*numheads
        assert attn1_out.size(0) == 1
        memory_embed = self.fclast(attn1_out[0,:]) #batch, self.args.hidden
        return memory_embed

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
        print('TODO add state_dim to memory')
        self.complete_state = self.cnn_embed_dim + self.state_dim
        if self.MEMORY_MODEL:
            modules['memory'] = MemoryModel(self.args, self.cnn_embed_dim, self.past_time[-self.args.num_past_times:])
            self.memory_embed = modules['memory'].memory_embed_dim
            self.complete_state += self.memory_embed

        modules['actions_predict'] = torch.nn.Sequential(torch.nn.Linear(self.complete_state, self.args.hidden),
                                                         Lambda(lambda x: swish(x)),
                                                         torch.nn.Linear(self.args.hidden, self.args.hidden),
                                                         Lambda(lambda x: swish(x)),
                                                         torch.nn.Linear(self.args.hidden, (self.action_dim-2)*self.max_predict_range+4*self.max_predict_range+2*self.max_predict_range))#discrete probs + mu/var of continuous actions + zero cont. probs

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
        return x / 180.0

    def map_camera(self, x):
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

        actions_logits = self._model['modules']['actions_predict'](nembed).view(-1, self.max_predict_range, self.action_dim + 4)
        dis_logits, zero_logits, camera_mu, camera_log_var = actions_logits[:,:,:(self.action_dim-2)], actions_logits[:,:,(self.action_dim-2):self.action_dim], actions_logits[:,:,self.action_dim:(self.action_dim+2)], actions_logits[:,:,(self.action_dim+2):]
        camera_mu *= 0.1
        camera_log_var = 5.0 - torch.nn.Softplus()(5.0 - torch.nn.Softplus()(camera_log_var + 5.0) + 5.0)
        camera_log_var -= 5.0
        dis_logits = torch.split(dis_logits, self.action_split, -1)

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
                    camera = sample_gaussian(camera_mu, camera_log_var)
                    zero_camera = torch.distributions.Categorical(logits=zero_logits).sample()
                    camera = torch.where((zero_camera == 1).unsqueeze(-1).expand(-1,-1,2), torch.zeros_like(camera), camera)
                    c_actions['camera'] = camera.clone()
                    camera = torch.tanh(camera)
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
            entropy_bonus = 0.001
            bc_loss = None
            ii = 0
            for a in self.action_dict:
                if a != 'camera':
                    if bc_loss is None:
                        bc_loss = self._model['ce_loss'](dis_logits[ii][batch_size:].reshape(-1, self.action_dict[a].n), next_actions[a][batch_size:].reshape(-1), is_weight[batch_size:].reshape(-1, 1, 1))
                    else:
                        bc_loss += self._model['ce_loss'](dis_logits[ii][batch_size:].reshape(-1, self.action_dict[a].n), next_actions[a][batch_size:].reshape(-1), is_weight[batch_size:].reshape(-1, 1, 1))
                    ii += 1
                else:
                    bc_loss += -(self.normal_log_prob_dens(camera_mu[batch_size:], camera_log_var[batch_size:], next_actions[a][batch_size:]) * is_weight[batch_size:].reshape(-1, 1, 1)).mean()
                    current_zero = ((torch.abs(next_actions[a].view(-1, self.max_predict_range, 2)[batch_size:,:,0]) < 1e-5) & (torch.abs(next_actions[a].view(-1, self.max_predict_range, 2)[batch_size:,:,1]) < 1e-5)).long()
                    bc_loss += self._model['ce_loss'](zero_logits[batch_size:].reshape(-1, 2), current_zero.reshape(-1), is_weight[batch_size:].reshape(-1, 1, 1))
            loss = bc_loss * 0.1
            rdict = {'bc_loss': bc_loss}
            
            if self._stage != 0:
                dis_logits = [torch.nn.LogSoftmax(-1)(d) for d in dis_logits]
                zero_logits = torch.nn.LogSoftmax(-1)(zero_logits)

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
                            policy_loss += -(advantage * self.normal_log_prob_dens(camera_mu, camera_log_var, all_actions[j][a].detach()).sum(-1)).mean() #+ entropy_bonus * self.normal_pre_kl_divergence(camera_mu, camera_log_var + 5.0)
                            current_zero = ((torch.abs(all_actions[j][a].view(-1, self.max_predict_range, 2)[:,:,0]) < 1e-5) & (torch.abs(all_actions[j][a].view(-1, self.max_predict_range, 2)[:,:,1]) < 1e-5)).long()
                            policy_loss += -(advantage * torch.gather(zero_logits, -1, current_zero.unsqueeze(-1))[:,:,0]).mean() + entropy_bonus * (zero_logits.exp()*zero_logits * is_weight.view(-1, 1, 1)).sum(-1).mean()
                    loss += (all_actions[j]['camera']).pow(2).mean() * 10.0 / self.num_policy_samples
                loss += policy_loss / self.num_policy_samples
                rdict['policy_loss'] = policy_loss
            else:
                for j in range(self.num_policy_samples):
                    loss += (all_actions[j]['camera']).pow(2).mean() * 10.0 / self.num_policy_samples
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
        if self.train_iter > 1e3 and rdict['Q_diff'] < 0.15:
            int_tmp = np.power(rdict['Q_diff']/0.05, -8.0) / self._interpolation_speed
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

    def select_action(self, state, batch=False):
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
            current_best_actions = self._model['modules']['actions_predict'](tembed).view(-1, self.max_predict_range, self.action_dim + 4)
            dis_logits, zero_logits, camera_mu, camera_log_var = current_best_actions[:,:,:(self.action_dim-2)], current_best_actions[:,:,(self.action_dim-2):self.action_dim], current_best_actions[:,:,self.action_dim:(self.action_dim+2)], current_best_actions[:,:,(self.action_dim+2):]
            camera_mu *= 0.1
            camera_log_var = 5.0 - torch.nn.Softplus()(5.0 - torch.nn.Softplus()(camera_log_var + 5.0) + 5.0)
            camera_log_var -= 5.0
            dis_logits = torch.split(dis_logits, self.action_split, -1)
            action = OrderedDict()
            ii = 0
            for a in self.action_dict:
                if a != 'camera':
                    action[a] = torch.distributions.Categorical(logits=dis_logits[ii][:,0,:]).sample()[0].item()
                    ii += 1
                else:
                    action[a] = np.tanh(sample_gaussian(camera_mu[:,0], camera_log_var[:,0])[0,:].data.cpu().numpy())
                    zero_act = torch.distributions.Categorical(logits=zero_logits[:,0,:]).sample()[0].item()
                    if zero_act == 1:
                        action[a] *= 0.0
                    action[a] = self.map_camera(action[a])
            if self.MEMORY_MODEL:
                return action, {'pov_embed': pov_embed.data.cpu()}
            else:
                return action, {}
