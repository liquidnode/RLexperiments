import torch
import numpy as np
from all_experiments import HashingMemory, RAdam, prio_utils
from torch.nn import functional as F
from collections import OrderedDict
from utils import variable, Lambda, swish, Flatten, sample_gaussian, one_hot, GAMMA
import copy
import math
from modules import VisualNet, MemoryModel, CameraBias

INFODIM = False


#implementation of upside down RL
class UpsideDownModel():
    def __init__(self, state_shape, action_dict, args, time_deltas, only_visnet=False):
        assert isinstance(state_shape, dict)
        assert not args.needs_embedding #not implemented
        self.MEMORY_MODEL = False
        assert len(time_deltas) == 1 and time_deltas[0] == 0 #this method only needs present state

        self.state_shape = state_shape
        self.action_dict = copy.deepcopy(action_dict)
        self.state_keys = ['state', 'pov', 'history_action', 'history_reward', 'orientation', 'env_type']
        self.args = args

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
        
        self.train_iter = 0
        self.time_left = 0
        self.last_time_left = 0

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
        if torch.cuda.is_available():
            for p in state_dict:
                state_dict[p] = state_dict[p].cuda()
        self._model['modules'].load_state_dict(state_dict, strict)

    def loadstore(self, filename, load=True):
        if load:
            print('Load actor')
            self.train_iter = torch.load(filename + 'train_iter')
            state_dict = torch.load(filename + 'upsidedown')
            if self.only_visnet:
                state_dict = {n: v for n, v in state_dict.items() if n.startswith('pov_embed.')}
            self.load_state_dict(state_dict)
            if 'Lambda' in self._model:
                self._model['Lambda'] = torch.load(filename + 'upsidedownLam')
            if self._target is not None:
                self._stage = 2
                self._target['modules'].load_state_dict(state_dict)
                self._last_target['modules'].load_state_dict(state_dict)
        else:
            torch.save(self._model['modules'].state_dict(), filename + 'upsidedown')
            torch.save(self.train_iter, filename + 'train_iter')
            if 'Lambda' in self._model:
                torch.save(self._model['Lambda'], filename + 'upsidedownLam')

    @staticmethod
    def needed_state_info(args):
        return {'needs_future_reward_and_time_left': True, 
                'min_num_future_rewards': 1}, [0], [0]

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
            desc['pov_embed'] = {'compression': False, 'shape': [self.args.hidden], 'dtype': np.float32}
            if len(self.disc_embed_dims) == 0:
                return desc
            else:
                assert False #not implemented
        else:
            return desc

    def _make_model(self):
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        modules = torch.nn.ModuleDict()
        modules['pov_embed'] = VisualNet(self.state_shape, self.args, self.args.hidden, [], False)
        self.cnn_embed_dim = self.args.hidden
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

        self.fr_tl_num = 64
        modules['fr_tl_hidden'] = torch.nn.Sequential(torch.nn.Linear(self.fr_tl_num, self.args.hidden),
                                                      Lambda(lambda x: swish(x)))

        modules['actions_predict_hidden'] = torch.nn.Sequential(torch.nn.Linear(self.complete_state, self.args.hidden),
                                                                Lambda(lambda x: swish(x)))

        modules['actions_predict'] = torch.nn.Sequential(torch.nn.Linear(self.args.hidden, self.args.hidden),
                                                         Lambda(lambda x: swish(x)),
                                                         torch.nn.Linear(self.args.hidden, self.action_dim))#discrete probs + zero cont. probs

        modules['camera_predict'] = torch.nn.Sequential(torch.nn.Linear(self.args.hidden, 8*4*4),
                                                        Lambda(lambda x: swish(x.reshape(-1, 8, 4, 4))),
                                                        torch.nn.ConvTranspose2d(8, 8, 3, 2), #9x9
                                                        torch.nn.BatchNorm2d(8),
                                                        Lambda(lambda x: swish(x)),
                                                        torch.nn.ConvTranspose2d(8, 6, 3, 2), #19x19
                                                        torch.nn.BatchNorm2d(6),
                                                        Lambda(lambda x: swish(x)),
                                                        torch.nn.ConvTranspose2d(6, 1, 5, 3),
                                                        CameraBias(59, self.device)) #59x59
        self.cam_map_size = 59

        self.ran = torch.arange(0, self.fr_tl_num//2, 1).float()
        self.ran = torch.cat([self.ran, self.ran], dim=0).view(1,self.fr_tl_num)

        if INFODIM and self.args.ganloss == "fisher":
            lam = torch.zeros((1,), requires_grad=True)

        if torch.cuda.is_available():
            modules = modules.cuda()
            if INFODIM and self.args.ganloss == "fisher":
                lam = lam.cuda()
            self.device = torch.device("cuda")
            self.ran = self.ran.cuda()
        else:
            self.device = torch.device("cpu")

        
        ce_loss = torch.nn.CrossEntropyLoss(reduction='none')
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

    def get_complete_embed(self, model, pov_state, state, future_reward, time_left, past_embeds=None):
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
        
        actions_hidden = self._model['modules']['actions_predict_hidden'](embed)

        #embed future rewards and time_left
        fr_tl_embed = torch.cat([(future_reward / 64.0).expand(-1, self.fr_tl_num//2), 
                                 (time_left / 8000.0).expand(-1, self.fr_tl_num//2)], dim=1)
        fr_tl_embed = torch.cos(np.pi*fr_tl_embed*self.ran)
        fr_tl_embed = self._model['modules']['fr_tl_hidden'](fr_tl_embed)
        return actions_hidden*fr_tl_embed, pov_embed
    
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
        future_reward = torch.cat([variable(input['future_reward']),variable(expert_input['future_reward'])],dim=0)
        time_left = torch.cat([variable(input['time_left']),variable(expert_input['time_left'])],dim=0).float()
        batch_size = variable(input['future_reward']).shape[0]

        is_weight = None
        if 'is_weight' in input:
            is_weight = torch.cat([variable(input['is_weight']),variable(expert_input['is_weight'])],dim=0)
        else:
            is_weight = torch.ones((future_reward.shape[0],), device=self.device).float()

        states['pov'] = (states['pov'].float()/255.0)*2.0 - 1.0
        states['pov'] = states['pov'].permute(0, 1, 4, 2, 3)
        current_state = {}
        for k in states:
            if k != 'pov':
                current_state[k] = states[k][:,self.zero_time_point]
        current_actions = {}
        for k in actions:
            current_actions[k] = actions[k][:,self.zero_time_point]
        future_reward = future_reward[:,self.zero_time_point:(self.zero_time_point+1)]
        time_left = time_left[:,self.zero_time_point:(self.zero_time_point+1)]
        assert states['pov'].shape[1] == 1

        actions_hidden, _ = self.get_complete_embed(self._model, states['pov'][:,0], current_state, future_reward, time_left)

        #get logits
        actions_logits = self._model['modules']['actions_predict'](actions_hidden).view(-1, self.action_dim)
        dis_logits, zero_logits = actions_logits[:,:(self.action_dim-2)], actions_logits[:,(self.action_dim-2):self.action_dim]
        dis_logits = torch.split(dis_logits, self.action_split, -1)
        cam_map = self._model['modules']['camera_predict'](actions_hidden).reshape(-1,self.cam_map_size*self.cam_map_size)

        #get loss
        entropy_bonus = 0.0
        bc_loss = None
        ii = 0
        for a in self.action_dict:
            if a != 'camera':
                if bc_loss is None:
                    bc_loss = self._model['ce_loss'](dis_logits[ii].reshape(-1, self.action_dict[a].n), current_actions[a])
                else:
                    bc_loss += self._model['ce_loss'](dis_logits[ii].reshape(-1, self.action_dict[a].n), current_actions[a])
                ii += 1
            else:
                bc_loss += self._model['ce_loss'](cam_map, current_actions['camera_quant'])
                current_zero = ((torch.abs(current_actions[a].view(-1, 2)[:,0]) < 1e-5) & (torch.abs(current_actions[a].view(-1, 2)[:,1]) < 1e-5)).long()
                bc_loss += self._model['ce_loss'](zero_logits, current_zero)
        loss = torch.mean(bc_loss * is_weight)
        rdict = {'loss': loss}


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
                #prioritize using bc_loss
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
        self.future_reward = 64.0
        tmp = self.time_left
        self.time_left = (8000.0 - self.last_time_left) * 0.85 + (8000.0 - self.time_left) * 0.1
        self.last_time_left = tmp

    def select_action(self, state, last_reward, batch=False):
        self.future_reward -= last_reward
        self.time_left -= 1
        self.time_left = max(1.0, self.time_left)
        tstate = {}
        if not batch:
            for k in state:
                tstate[k] = state[k][None, :].copy()
        tstate = variable(tstate)
        future_reward = variable(np.array([self.future_reward])).view(1,1)
        time_left = variable(np.array([self.time_left])).view(1,1)
        with torch.no_grad():
            #pov_embed
            tstate['pov'] = (tstate['pov'].permute(0, 3, 1, 2)/255.0)*2.0 - 1.0
            pov_embed = self._model['modules']['pov_embed'](tstate['pov'])
            actions_hidden, pov_embed = self.get_complete_embed(self._model, tstate['pov'], tstate, future_reward, time_left)

            current_best_actions = self._model['modules']['actions_predict'](actions_hidden).view(-1, self.action_dim)
            dis_logits, zero_logits = current_best_actions[:,:(self.action_dim-2)], current_best_actions[:,(self.action_dim-2):self.action_dim]
            dis_logits = torch.split(dis_logits, self.action_split, -1)
            action = OrderedDict()
            ii = 0
            for a in self.action_dict:
                if a != 'camera':
                    action[a] = torch.distributions.Categorical(logits=dis_logits[ii]).sample()[0].item()
                    ii += 1
                else:
                    cam_map = self._model['modules']['camera_predict'](actions_hidden).reshape(1,-1)
                    cam_choose = torch.distributions.Categorical(logits=cam_map).sample()[0].item()
                    cam_choose_x = (((cam_choose % self.cam_map_size) + np.random.sample()) * (2.0/self.cam_map_size)) - 1.0
                    cam_choose_y = (((cam_choose // self.cam_map_size) + np.random.sample()) * (2.0/self.cam_map_size)) - 1.0
                    action[a] = np.array([cam_choose_x, cam_choose_y])
                    zero_act = torch.distributions.Categorical(logits=zero_logits).sample()[0].item()
                    if zero_act == 1:
                        action[a] *= 0.0
                    action[a] = self.map_camera(action[a])
            if self.MEMORY_MODEL:
                return action, {'pov_embed': pov_embed.data.cpu()}
            else:
                return action, {}