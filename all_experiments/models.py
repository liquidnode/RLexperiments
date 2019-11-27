import torch
import numpy as np
from memory import HashingMemory
from torch.nn import functional as F
from collections import OrderedDict
from radam import RAdam

CUDA = True
tGAMMA = 0.99
GAMMA = 0.999
KAPPA = 0.1

if torch.cuda.is_available() and CUDA:
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.fastest = True

def swish(x):
    return x * torch.sigmoid(x)

class WeightedMSELoss(torch.nn.Module):
    def __init__(self):
        super(WeightedMSELoss, self).__init__()
        self.mse = torch.nn.MSELoss(reduction='none')

    def forward(self, x, target, weight=None):
        if weight is None:
            return torch.mean(self.mse(x, target))
        else:
            return torch.mean(weight*self.mse(x, target))


class Lambda(torch.nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)

class LearnerModel(torch.nn.Module):
    def __init__(self, args, state_shape):
        super(LearnerModel, self).__init__()
        self.args = args
        self.state_shape = state_shape
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
                    state_layers.append(torch.nn.Linear(self.state_shape['state'][0] if i == 0 else self.args.hidden, self.args.hidden))
                    state_layers.append(torch.nn.Tanh())
                self.state_model = torch.nn.Sequential(*state_layers)
                inp_size += self.args.hidden
            else:
                inp_size += self.state_shape['state'][0]

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
            body_out.append(self.state_model(inp['state']))
        else:
            if 'state' in inp:
                body_out.append(inp['state'])
        x = torch.cat(body_out, dim=1)
        return self.head_model(x)

#this class is modified code from https://github.com/vub-ai-lab/bdpi
class Learner():
    def __init__(self, state_shape, num_actions, args, is_critic):
        assert isinstance(state_shape, dict)
        self.state_shape = state_shape
        self.num_actions = num_actions
        self.args = args
        self.is_critic = is_critic
        
        self._setup()

    def _setup(self):
        if self.is_critic:
            # Clipped DQN requires two models
            self._models = [self._make_model() for i in range(2)]
        else:
            # The actor only needs one model
            self._models = [self._make_model()]

    def state_dict(self):
        """ Return the state_dict of the model
        """
        if self.is_critic:
            return [self._models[0]['model'].state_dict(), self._models[1]['model'].state_dict()]
        else:
            return self._models[0]['model'].state_dict()

    def load_state_dict(self, s, strict=True):
        """ Set the state of the model
        """
        if self.is_critic:
            self._models[0]['model'].load_state_dict(s[0], strict)
            self._models[1]['model'].load_state_dict(s[1], strict)
        else:
            for m in self._models:
                m['model'].load_state_dict(s, strict)

    def _predict_model(self, model, inp):
        with torch.no_grad():
            if not self.is_critic:
                return model['softmax'](model['model'](variable(inp))).data.cpu().numpy()
            else:
                return model['model'](variable(inp)).data.cpu().numpy()

    def _train_model(self, model, inp, target, epochs, is_weights=None):
        assert not isinstance(target, dict)

        v_inp = variable(inp)
        v_target = variable(target)
        if not is_weights is None:
            v_is_weights = variable(is_weights[:,None])

        def closure():
            model['opt'].zero_grad()
            out = model['model'](v_inp)
            if not self.is_critic:
                out = model['softmax'](out)
            if is_weights is None:
                loss = model['loss'](out, v_target)
            else:
                loss = model['loss'](out, v_target, v_is_weights)
            loss.backward()
            return loss
        
        loss = model['opt'].step(closure)
        if 'nec_layer' in model:
            #only iterate on nec
            def closure():
                model['opt'].zero_grad()
                out = model['nec_layer'].precalc_forward()
                if not self.is_critic:
                    out = model['softmax'](out)
                if is_weights is None:
                    loss = model['loss'](out, v_target)
                else:
                    loss = model['loss'](out, v_target, v_is_weights)
                loss.backward()
                return loss

        for i in range(epochs-1):
            loss = model['opt'].step(closure)

    def _pretrain_model(self, model, inp, target):
        assert not isinstance(target, dict)
        assert not self.is_critic
        assert inp['pov'].shape[1:] == self.state_shape['pov'], print(inp['pov'].shape[1:], )

        v_inp = variable(inp)
        v_target = variable(target, ignore_type=True)
        v_target = v_target.long()

        def closure():
            model['opt'].zero_grad()
            out = model['model'](v_inp)
            loss = model['ce_loss'](out, v_target)
            loss.backward()
            return loss

        return model['opt'].step(closure).data.cpu().numpy()

    def _make_model(self):
        layers = [LearnerModel(self.args, self.state_shape)]
        nec_layer = None
        if self.args.neural_episodic_control:
            nec_layer = HashingMemory.build(self.args.hidden, self.num_actions, self.args)
            if self.is_critic:
                nec_layer.values.weight.data.zero_()
            layers.append(nec_layer)
        else:
            layers.append(torch.nn.Linear(self.args.hidden, self.num_actions))

        smax = None
        if not self.is_critic:
            smax = torch.nn.Softmax(1)

        model = torch.nn.Sequential(*layers)

        if torch.cuda.is_available() and CUDA:
            if nec_layer is not None:
                nec_layer = nec_layer.cuda()
            if smax is not None:
                smax = smax.cuda()
            model = model.cuda()

        scale = 1.0 if self.is_critic else 0.1
        if nec_layer is None:
            optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr*scale)
        else:
            model_params = []
            nec_value_params = []
            for name, p in model.named_parameters():
                if 'values.weight' in name:
                    nec_value_params.append(p)
                else:
                    model_params.append(p)
            optimizer = torch.optim.Adam([{'params': model_params}, 
                                          {'params': nec_value_params, 'lr': self.args.nec_lr*scale}], lr=self.args.lr*scale)
            if self.args.pretrain and self.is_critic:
                for p in model_params:
                    p.requires_grad = False
        #TODO add skynet penalty
        #if self.args.pursuit_variant == 'mimic' and not self.is_critic:
        #    # Use the Actor-Mimic loss
        #    loss = CELoss()
        #else:
        if self.args.per:
            loss = WeightedMSELoss()
        else:
            loss = torch.nn.MSELoss()

        if not self.is_critic:
            ce_loss = torch.nn.CrossEntropyLoss()
            if nec_layer is not None:
                return {'model': model, 'opt': optimizer, 'loss': loss, 'ce_loss': ce_loss, 'nec_layer': nec_layer, 'softmax': smax}
            else:
                return {'model': model, 'opt': optimizer, 'loss': loss, 'ce_loss': ce_loss, 'softmax': smax}
        else:
            if nec_layer is not None:
                return {'model': model, 'opt': optimizer, 'loss': loss, 'nec_layer': nec_layer}
            else:
                return {'model': model, 'opt': optimizer, 'loss': loss}


def _variable(inp, ignore_type=False, half=False):
    if torch.is_tensor(inp):
        rs = inp
    else:
        rs = torch.from_numpy(np.asarray(inp))

    if torch.cuda.is_available() and CUDA:
        rs = rs.cuda()

    if not ignore_type:
        # Ensure we have floats
        if half:
            rs = rs.half()
        else:
            rs = rs.float()

    return rs

def variable(inp, ignore_type=False, half=False):
    if isinstance(inp, dict):
        for k in inp:
            inp[k] = _variable(inp[k], ignore_type, half)
        return inp
    else:
        return _variable(inp, ignore_type, half)



class Flatten(torch.nn.Module):
    """ Flatten an input, used to map a convolution to a Dense layer
    """
    def forward(self, x):
        return x.view(x.size()[0], -1)


class AdvValStream(torch.nn.Module):
    def __init__(self, num_actions):
        super(AdvValStream, self).__init__()
        self.num_actions = num_actions

    def forward(self, x):
        advantage, value = torch.split(x, [self.num_actions, 1], dim=1)
        advantage = advantage - torch.max(advantage, dim=1, keepdim=True)[0]
        return advantage + value

class AdvValMemory(torch.nn.Module):
    def __init__(self, hidden, num_actions, args):
        super(AdvValMemory, self).__init__()
        self.num_actions = num_actions
        self.nec_layer_adv = HashingMemory.build(hidden, self.num_actions, args)#torch.nn.Linear(hidden, self.num_actions)#
        self.nec_layer_val = HashingMemory.build(hidden, 1, args)
        self.disable_val = False

    def forward(self, x):
        advantage = self.nec_layer_adv(x)
        advantage = advantage - torch.mean(advantage, dim=1, keepdim=True)
        if not self.disable_val:
            value = self.nec_layer_val(x)
            return advantage + value
        else:
            return advantage

class SoftmaxScaled(torch.nn.Module):
    def __init__(self, args):
        super(SoftmaxScaled, self).__init__()
        self.smax = torch.nn.Softmax(1)
        self.eps_t = torch.nn.Parameter(torch.tensor(np.log(np.exp(args.eps)-1.0), requires_grad=True))
        self.softplus = torch.nn.Softplus()

    def forward(self, x):
        x = x/self.softplus(self.eps_t)
        x = x - x.mean(1)[:,None]
        return self.smax(x)
    

class SoftQ():
    def __init__(self, state_shape, num_actions, args):
        assert isinstance(state_shape, dict)
        self.state_shape = state_shape
        self.num_actions = num_actions
        self.args = args
        
        self._setup()
        self._a, self._b = self._models

    def _setup(self):
        self._models = [self._make_model() for i in range(2)]

    def state_dict(self):
        return [self._models[0]['Q_values'].state_dict(), self._models[1]['Q_values'].state_dict()]

    def load_state_dict(self, s, strict=True):
        if isinstance(s, list):
            self._models[0]['Q_values'].load_state_dict(s[0], strict)
            self._models[1]['Q_values'].load_state_dict(s[1], strict)
        else:
            self._models[0]['Q_values'].load_state_dict(s, strict)

    def loadstore(self, filename, load=True):
        if self.args.pretrain:
            if load:
                self._models[0]['Q_values'].load_state_dict(torch.load(filename + '-pretrain-softq'))
            else:
                torch.save(self._models[0]['Q_values'].state_dict(), filename + '-pretrain-softq')
        else:
            if load:
                self.load_state_dict([torch.load(filename + '-softqA'),
                                      torch.load(filename + '-softqB')])
                self._models[0]['softmax'].load_state_dict(torch.load(filename + '-softmaxA'))
                self._models[1]['softmax'].load_state_dict(torch.load(filename + '-softmaxB'))
            else:
                torch.save(self._models[0]['Q_values'].state_dict(), filename + '-softqA')
                torch.save(self._models[1]['Q_values'].state_dict(), filename + '-softqB')
                torch.save(self._models[0]['softmax'].state_dict(), filename + '-softmaxA')
                torch.save(self._models[1]['softmax'].state_dict(), filename + '-softmaxB')

    def time_train(self, input, expert_input):
        if self.args.per:
            assert False #not implemented yet
        else:
            is_weights = None
        self._a['opt'].zero_grad()

        state_keys = ['state', 'pov']

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
                states[k] = torch.cat([states[k],expert_states[k]], dim=0)
            for k in next_states:
                next_states[k] = torch.cat([next_states[k],expert_states[k]], dim=0)
            actions = torch.cat([actions, expert_actions], dim=0)
            rewards = torch.cat([rewards, expert_rewards], dim=0)

            done = np.concatenate([done, expert_done], axis=0)
            next_indexes = np.nonzero(np.logical_not(done))[0]
            
        with torch.no_grad():
            next_timeA,_ = torch.min(self._a['Q_values'](next_states), dim=1)
            next_timeB,_ = torch.min(self._b['Q_values'](next_states), dim=1)
            next_time = 0.01 + torch.max(next_timeA, next_timeB)
            next_time = (1.0-rewards)*(next_time)+rewards*0.01

        out = self._a['Q_values'](states)
        time = out.detach()
        time = torch.clamp(time, min=0.0)
        try:
            time[next_indexes, actions[next_indexes]] = time[next_indexes, actions[next_indexes]] + self.args.clr * (next_time[next_indexes] - time[next_indexes, actions[next_indexes]])
        except:
            print(time.shape)
            print(next_time.shape)
            print(next_indexes.shape)

        if is_weights is not None:
            loss = self._a['loss'](out, time, is_weights)
        else:
            loss = self._a['loss'](out, time)
        
        probas = out.data.cpu().numpy()
        min_p = np.min(probas, axis=1, keepdims=True)
        probas = np.maximum(0.01 - (probas - min_p), 1e-8)
        probas = probas/np.sum(probas, axis=1, keepdims=True)
        entropy = np.mean(-np.sum(probas*np.log(probas), axis=1))

        loss.backward()
        self._a['opt'].step()

        self._a, self._b = self._b, self._a

        return [out.data.cpu().numpy(), entropy, loss.data.cpu().numpy(), 0.0]

    def train(self, input, expert_input):
        if self.args.time:
            return self.time_train(input, expert_input)
        if self.args.per:
            assert False #not implemented yet
        else:
            is_weights = None
        self._a['opt'].zero_grad()

        state_keys = ['state', 'pov']

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

        if expert_input is not None and True:
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
                states[k] = torch.cat([states[k],expert_states[k]], dim=0)
            for k in next_states:
                next_states[k] = torch.cat([next_states[k],expert_states[k]], dim=0)
            actions = torch.cat([actions, expert_actions], dim=0)
            rewards = torch.cat([rewards, expert_rewards], dim=0)

            done = np.concatenate([done, expert_done], axis=0)
            next_indexes = np.nonzero(np.logical_not(done))[0]


        #soft Q learning
        with torch.no_grad():
            if False:
                valsA = self._a['Q_values'](next_states)/self.args.eps
                probasA = self._a['softmax'](valsA)
                meanA = torch.sum(probasA * valsA, dim=1)
                valsB = self._b['Q_values'](next_states)/self.args.eps
                probasB = self._b['softmax'](valsB)
                meanB = torch.sum(probasB * valsB, dim=1)

                next_valuesA = self.args.eps*(meanA+torch.log(torch.sum(torch.exp(valsA-meanA[:,None]),dim=1)))
                next_valuesB = self.args.eps*(meanB+torch.log(torch.sum(torch.exp(valsB-meanB[:,None]),dim=1)))
            else:
                if False:
                    #das ist falsch
                    #man muss dieselben indices nehmen
                    next_valuesA, _ = torch.max(self._a['Q_values'](next_states), dim=1)
                    next_valuesB, _ = torch.max(self._b['Q_values'](next_states), dim=1)
                else:
                    next_out = self._a['Q_values'](next_states)
                    next_probs = self._a['softmax'](next_out)
                    next_valuesA = torch.sum(next_probs*next_out, dim=1)
                    next_valuesB = torch.sum(next_probs*self._b['Q_values'](next_states), dim=1)
            #next_values = rewards.clone()
            #if next_indexes.shape[0] > 0:
            #    next_values[next_indexes] += GAMMA * torch.min(next_valuesA, next_valuesB)[next_indexes]
            next_values = torch.zeros_like(rewards)
            if next_indexes.shape[0] > 0:
                next_values[next_indexes] = (1.0 - rewards[next_indexes]) * GAMMA * torch.min(next_valuesA, next_valuesB)[next_indexes] + rewards[next_indexes]
            #next_values = torch.clamp(next_values, min=0.0)
        #needed?
        next_values = next_values.detach()

        out = self._a['Q_values'](states)
        #make learning rate
        critic_qvalues = out.detach()
        
        if KAPPA < 1.0:
            with torch.no_grad():
                #max_a,_ = torch.max(critic_qvalues, dim=1)
                probas = self._a['softmax'](critic_qvalues)
                max_a = torch.sum(probas*critic_qvalues, dim=1)
                max_b = torch.sum(probas*self._b['Q_values'](states), dim=1)
                max_a = torch.min(max_a, max_b)
                next_values -= max_a
                next_values = max_a + (next_values / KAPPA)

        next_values = critic_qvalues[:, actions] + self.args.clr * (next_values - critic_qvalues[:, actions])


        if is_weights is not None:
            loss = self._a['loss'](out[:, actions], next_values, is_weights)
        else:
            loss = self._a['loss'](out[:, actions], next_values)
        probas = self._a['softmax'](out)#/self.args.eps)
            

        if expert_input is not None:
            #pretain from expert
            if False:
                expert_critic_qvalues = critic_qvalues[-expert_done.shape[0]:]
                expert_probas = probas[-expert_done.shape[0]:]
                other_probas = probas[:batch_size]
                expert_entropy = torch.mean(-torch.sum(expert_probas*torch.log(expert_probas), dim=1))
                other_entropy = torch.mean(-torch.sum(other_probas*torch.log(other_probas), dim=1))

                #t_action = torch.distributions.Categorical(expert_probas).sample()
                #loss_policy = torch.mean(-expert_critic_qvalues[:,t_action]*torch.log(expert_probas[:,t_action]))
                loss_policy = torch.mean(torch.sum(-expert_critic_qvalues*expert_probas*torch.log(expert_probas),dim=1))
                loss_policy += self.args.eps*(other_entropy - expert_entropy)
                loss_policy *= self.args.eps

                expert_probas_detach = expert_probas.detach()
                expert_out = out[-expert_done.shape[0]:]
                #TODO max(0,x) einbauen oder nicht??
                loss_Q = torch.mean(torch.clamp(torch.sum(expert_probas_detach * expert_out, dim=1) - expert_out[:,expert_actions], min=0.0))
                loss += 0.1*(loss_policy + loss_Q)
            else:
                if True:
                    #just use ce loss
                    logits = out/self._a['softmax'].softplus(self._a['softmax'].eps_t)
                    logits = logits[-expert_done.shape[0]:]
                    logits = logits - torch.mean(logits, dim=1, keepdim=True)
                    loss_policy = self._a['ce_loss'](logits, expert_actions)
                    loss += 0.1*loss_policy
                else:
                    expert_states = {}
                    for k in state_keys:
                        if k in input:
                            expert_states[k] = variable(expert_input[k])
                    expert_actions = variable(expert_input['action'], True).long()
                    logits = self._a['Q_values'](expert_states)/self._a['softmax'].softplus(self._a['softmax'].eps_t)#/self.args.eps
                    logits = logits - torch.mean(logits, dim=1, keepdim=True)
                    loss_policy = self._a['ce_loss'](logits, expert_actions)#expert prior
                    loss += loss_policy

        loss.backward()
        self._a['opt'].step()

        self._a, self._b = self._b, self._a

        entropy = torch.mean(-torch.sum(probas*torch.log(probas), dim=1))
        return [critic_qvalues.data.cpu().numpy(), entropy.data.cpu().numpy(), loss_policy.data.cpu().numpy(), self._a['softmax'].softplus(self._a['softmax'].eps_t).data.cpu().numpy()]

    def pretrain(self, expert_input):
        state_keys = ['state', 'pov']
        expert_states = {}
        for k in state_keys:
            if k in expert_input:
                expert_states[k] = variable(expert_input[k])
        expert_actions = variable(expert_input['action'], True).long()

        def closure():
            self._a['opt'].zero_grad()
            logits = self._a['Q_values'](expert_states)
            loss = self._a['ce_loss'](logits, expert_actions)
            loss.backward()
            with torch.no_grad():
                probas = self._a['softmax'](logits)
                self.max_probs = torch.mean(torch.max(probas,dim=1)[0])
                self.entropy = torch.mean(-torch.sum(probas*torch.log(probas), dim=1))
            return loss

        loss = self._a['opt'].step(closure).data.cpu().numpy()

        return [loss, self.entropy.data.cpu().numpy(), self.max_probs.data.cpu().numpy()]

    def _make_model(self):
        layers = [LearnerModel(self.args, self.state_shape)]
        nec_layer = None
        if self.args.neural_episodic_control:
            #nec_layer = HashingMemory.build(self.args.hidden, self.num_actions+1, self.args)
            #nec_layer.values.weight.data *= self.args.eps
            nec_layer = AdvValMemory(self.args.hidden, self.num_actions, self.args)
            nec_layer.disable_val = self.args.pretrain
            #nec_layer.nec_layer_val.values.weight.data.zero_()
            nec_layer.nec_layer_adv.values.weight.data *= self.args.eps
            #nec_layer.nec_layer_adv.weight.data *= self.args.eps
            #nec_layer.nec_layer_adv.bias.data *= self.args.eps
            if self.args.pretrain:
                nec_layer.disable_val = True
            layers.append(nec_layer)
        else:
            layers.append(torch.nn.Linear(self.args.hidden, self.num_actions+1))
            layers.append(AdvValStream(self.num_actions))

        if self.args.pretrain:
            smax = torch.nn.Softmax(1)
        else:
            smax = SoftmaxScaled(self.args)

        model = torch.nn.Sequential(*layers)

        if torch.cuda.is_available() and CUDA:
            if nec_layer is not None:
                nec_layer = nec_layer.cuda()
            if smax is not None:
                smax = smax.cuda()
            model = model.cuda()
            

        scale = 1.0# if self.is_critic else 0.1
        if nec_layer is None:
            optimizer = torch.optim.Adam(model.parameters()+smax.parameters(), lr=self.args.lr*scale)
        else:
            model_params = []
            nec_value_params = []
            for name, p in model.named_parameters():
                if 'values.weight' in name:
                    nec_value_params.append(p)
                else:
                    model_params.append(p)
            nec_value_params.extend(smax.parameters())
            optimizer = torch.optim.Adam([{'params': model_params}, 
                                          {'params': nec_value_params, 'lr': self.args.nec_lr*scale}], lr=self.args.lr*scale)
            
        #TODO add skynet penalty
        #if self.args.pursuit_variant == 'mimic' and not self.is_critic:
        #    # Use the Actor-Mimic loss
        #    loss = CELoss()
        #else:
        if self.args.per:
            loss = WeightedMSELoss()
        else:
            loss = torch.nn.MSELoss()#torch.nn.SmoothL1Loss()#

        ce_loss = torch.nn.CrossEntropyLoss()
        return {'Q_values': model, 'opt': optimizer, 'loss': loss, 'ce_loss': ce_loss, 'softmax': smax}

    def get_features(self, state):
        with torch.no_grad():
            return self._a['Q_values'][0](variable(state))

    def predict(self, state, batch=False):
        if not batch:
            for k in state:
                state[k] = state[k][None,:]
        #vals = self._a['Q_values'](variable(state))/self._a['softplus'](self._a['eps_t'])#self.args.eps
        #mean = torch.mean(vals, dim=1)
        with torch.no_grad():
            vals = self._a['Q_values'](variable(state))
            return self._a['softmax'](vals).data.cpu().numpy()

    def select_action(self, state):
        if self.args.time:
            return self.time_select_action(state)
        probas = self.predict(state)[0]

        action_index = int(np.random.choice(range(self.num_actions), p=probas))
        try:
            entropy = -np.sum(probas*np.log(probas))
        except:
            print(probas)
        return action_index, entropy

    def time_select_action(self, state, batch=False):
        if not batch:
            for k in state:
                state[k] = state[k][None,:]
                
        with torch.no_grad():
            time = self._a['Q_values'](variable(state)).data.cpu().numpy()

        
        probas = time
        min_p = np.min(probas, axis=1)
        entropy = [min_p[0], np.max(probas, axis=1)[0]-min_p[0]]
        probas = np.maximum(0.01 - (probas - min_p), 0.0)
        probas = probas/np.sum(probas, axis=1, keepdims=True)
        action_index = int(np.random.choice(range(self.num_actions), p=probas[0]))

        return action_index, entropy





#mostly from https://github.com/dannysdeng/dqn-pytorch/blob/master/DQN_network.py
class IQNHead(torch.nn.Module):
    def __init__(self, input_dim, num_actions, args):
        super(IQNHead, self).__init__()
        self.num_actions = num_actions
        self.args = args
        self.quantile_embedding_dim = 64
        self.nec_dim = 32
        self.pi = np.pi

        tmp_mem_n_keys = self.args.mem_n_keys
        self.args.mem_n_keys = 128
        self.nec1 = HashingMemory.build(input_dim, self.nec_dim, self.args)
        self.args.mem_n_keys = tmp_mem_n_keys
        self.fc2 = torch.nn.Linear(self.args.hidden+self.nec_dim+self.num_actions, self.num_actions)

        self.quantile_fc0 = torch.nn.Linear(self.quantile_embedding_dim, input_dim)
        self.quantile_fc1 = torch.nn.Linear(input_dim, self.args.hidden)

        self.quantile_fc_value = torch.nn.Linear(self.args.hidden+self.nec_dim, 1)

    def forward(self, x, num_quantiles, prior=None, tau=None):
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
        
        nec_fea      = self.nec1(x).unsqueeze(1).expand(-1, num_quantiles, -1) # [Batch x Np x hidden]
        nec_fea      = nec_fea.contiguous()
        nec_fea      = nec_fea.view(BATCH_SIZE*num_quantiles, -1)

        out          = F.relu(self.quantile_fc0(cos_tau))                   # [Batch*Np x feaSize]
        fea_tile     = x.unsqueeze(1).expand(-1, num_quantiles, -1)         # [Batch x Np x feaSize]
        out          = out.view(BATCH_SIZE, num_quantiles, -1)              # [Batch x Np x feaSize]
        product      = (fea_tile * out).view(BATCH_SIZE*num_quantiles, -1)
        combined_fea = F.relu(self.quantile_fc1(product))                   # (Batch*atoms, hidden)
        #combined_fea = combined_fea + nec_fea                               # skip connection
        combined_fea = torch.cat([combined_fea,nec_fea], dim=1)

        values   = self.quantile_fc_value(combined_fea)
        values   = values.view(-1, num_quantiles).unsqueeze(1)
        
        combined_fea = torch.cat([combined_fea,prior.unsqueeze(1).expand(-1, num_quantiles, -1).contiguous().view(BATCH_SIZE*num_quantiles, -1)], dim=1)
        x        = self.fc2(combined_fea)

        if True:#prior is None:
            x_batch  = x.view(BATCH_SIZE, num_quantiles, self.num_actions)
            x_batch  = x_batch.transpose(1, 2).contiguous()

            action_component = x_batch - x_batch.mean(1, keepdim=True)
            y = values + action_component # [batch x actions x atoms]
        else:
            x_batch  = x.view(BATCH_SIZE, num_quantiles, 1)#self.num_actions)
            x_batch  = x_batch.transpose(1, 2).contiguous()

            action_component = F.softplus(x_batch) * 0.01
            prior = prior - torch.mean(prior, dim=1, keepdim=True)
            prior = prior.unsqueeze(2)
            
            y = values + prior * action_component

        return y, tau, action_component

class IQN():
    def __init__(self, state_shape, num_actions, args):
        assert isinstance(state_shape, dict)
        self.state_shape = state_shape
        self.num_actions = num_actions
        self.args = args
        
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
        head_model = IQNHead(self.args.hidden + self.num_actions, self.num_actions, self.args)
        if self.args.neural_episodic_control:
            actor_model = HashingMemory.build(self.args.hidden, self.num_actions, self.args)
        else:
            actor_model = torch.nn.Linear(self.args.hidden, self.num_actions)

        smax = torch.nn.Softmax(1)

        if torch.cuda.is_available() and CUDA:
            if smax is not None:
                smax = smax.cuda()
            tail_model = tail_model.cuda()
            head_model = head_model.cuda()
            actor_model = actor_model.cuda()
            

        scale = 1.0# if self.is_critic else 0.1
        if not self.args.neural_episodic_control:
            optimizer = torch.optim.Adam(tail_model.parameters()+head_model.parameters()+actor_model.parameters(), lr=self.args.lr*scale)
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
            for name, p in actor_model.named_parameters():
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
        modules['actor_model'] = actor_model
        return {'modules': modules, 'opt': optimizer, 'loss': loss, 'ce_loss': ce_loss, 'softmax': smax}

    def predict(self, state, batch=False):
        if not batch:
            for k in state:
                state[k] = state[k][None,:]

        with torch.no_grad():
            vals = self._a['modules']['actor_model'](self._a['modules']['tail_model'](variable(state)))
            return self._a['softmax'](vals).data.cpu().numpy()

    def select_action(self, state):
        if np.random.sample() > 0.9:
            if self.args.time:
                return self.time_select_action(state)
            probas = self.predict(state)[0]

            action_index = int(np.random.choice(range(self.num_actions), p=probas))
            try:
                entropy = -np.sum(probas*np.log2(probas))
            except:
                print(probas)
            return action_index, entropy
        else:
            for k in state:
                state[k] = state[k][None,:]
            with torch.no_grad():
                dist,_,_,_ = self._get_distribution(self._a, variable(state), self.args.N_tau)
                vals = torch.mean(dist, dim=2)
                action_index = vals.max(1)[1][0].data.cpu().numpy()
            if action_index == 0 and self.last_action_index == 0:
                return np.random.randint(0, self.num_actions), -1.0
            self.last_action_index = action_index
            return action_index, -1.0

    def _get_distribution(self, model, state, num_tau, tau=None, tail_train=True):
        if not tail_train:
            with torch.no_grad():
                tail = model['modules']['tail_model'](state)
                orig_tail = tail
                probas = model['modules']['actor_model'](tail)
                tail = torch.cat([tail, probas], dim=1)
            tail = tail.detach()
        else:
            tail = model['modules']['tail_model'](state)
            orig_tail = tail
            with torch.no_grad():
                probas = model['modules']['actor_model'](tail)
            probas = probas.detach()
            tail = torch.cat([tail, probas], dim=1)
        dist, Ttau, action_comp = model['modules']['head_model'](tail, num_tau, probas, tau)
        return dist, Ttau, orig_tail, action_comp

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

            quantiles_next[next_indices] += ((GAMMA * next_y.gather(1, action_argmax).squeeze(1)))[next_indices]#(1.0 - quantiles_next)*
            #quantiles_next = quantiles_next.clamp(0.0, 1.0)
            return quantiles_next.detach()

    def train(self, input, expert_input=None, tail_train=False):
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

        y, my_tau, tail, action_comp = self._get_distribution(self._a, states, self.args.N_tau, tail_train=tail_train)
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

        loss.backward()
        self._a['opt'].step()

        Qval = Q_sa.cpu().detach().numpy().squeeze()

        self._a, self._b = self._b, self._a

        return [loss.data.cpu().numpy(), 
                np.mean(Qval), 
                loss_actor.data.cpu().numpy(), 
                entropy.data.cpu().numpy(),
                max_probs.data.cpu().numpy(),
                accuracy.data.cpu().numpy()]




class AdvancedBC():
    def __init__(self, state_shape, num_actions, args):
        assert isinstance(state_shape, dict)
        self.state_shape = state_shape
        self.num_actions = num_actions
        self.args = args
        
        self._setup()
        self._a = self._models[0]

    def _setup(self):
        self._models = [self._make_model()]

    def state_dict(self):
        return [self._models[0]['modules'].state_dict()]

    def load_state_dict(self, s, strict=True):
        if isinstance(s, list):
            self._models[0]['modules'].load_state_dict(s[0], strict)
            self._models[1]['modules'].load_state_dict(s[1], strict)
        else:
            self._models[0]['modules'].load_state_dict(s, strict)

    def loadstore(self, filename, load=True):
        if load:
            self.load_state_dict(torch.load(filename + '-advBE'))
        else:
            torch.save(self._models[0]['modules'].state_dict(), filename + '-advBE')

    def _make_model(self):
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

        #cnn_layers.append(Flatten())
        self.flatten = Flatten()
        self.cnn_model = torch.nn.Sequential(*cnn_layers)

        inp_size = self.cnn_model(torch.zeros((1,self.state_shape['pov'][2], self.state_shape['pov'][0], self.state_shape['pov'][1]))).shape[1:] #CHW
        self.inp_size = inp_size
        all_inp_size = inp_size[0]*inp_size[1]*inp_size[2]+self.state_shape['state'][0]

        #make OPTIONS NET
        self.num_options = 5 #mid, left, right, top, down
        self.options_model = torch.nn.Sequential(torch.nn.Linear(all_inp_size, self.num_options), torch.nn.Softmax(1))

        #make OBJECTS NET and DICTIONARY
        self.num_objects = 3
        self.objects_model = torch.nn.Sequential(torch.nn.Linear(all_inp_size, self.num_objects), torch.nn.Softmax(1), torch.nn.Linear(self.num_objects, inp_size[0]))

        self.smax = torch.nn.Softmax(1)

        self.action_heads = []
        for i in range(self.num_options):
            tmp = self.args.mem_n_keys
            self.args.mem_n_keys = 64
            t_head = HashingMemory.build(inp_size[0]+self.state_shape['state'][0]+inp_size[1]*inp_size[2], self.num_actions, self.args)
            self.action_heads.append(t_head)
            self.args.mem_n_keys = tmp
            #self.action_heads.append(torch.nn.Sequential(torch.nn.Linear(inp_size[0]+self.state_shape['state'][0]+inp_size[1]*inp_size[2], self.args.hidden),
            #                                             torch.nn.Tanh(),
            #                                             torch.nn.Linear(self.args.hidden, self.num_actions)))

        self.masks = []
        self.masks.append([[0, 0, 0, 0],
                           [0, 1, 1, 0],
                           [0, 1, 1, 0],
                           [0, 0, 0, 0]])
        self.masks.append([[1, 0, 0, 0],
                           [1, 0, 0, 0],
                           [1, 0, 0, 0],
                           [1, 0, 0, 0]])
        self.masks.append([[0, 0, 0, 1],
                           [0, 0, 0, 1],
                           [0, 0, 0, 1],
                           [0, 0, 0, 1]])
        self.masks.append([[1, 1, 1, 1],
                           [0, 0, 0, 0],
                           [0, 0, 0, 0],
                           [0, 0, 0, 0]])
        self.masks.append([[0, 0, 0, 0],
                           [0, 0, 0, 0],
                           [0, 0, 0, 0],
                           [1, 1, 1, 1]])
        self.masks = np.array(self.masks, dtype=np.float32)
        self.masks = variable(self.masks)


        if torch.cuda.is_available() and CUDA:
            self.smax = self.smax.cuda()
            self.cnn_model = self.cnn_model.cuda()
            self.flatten = self.flatten.cuda()
            self.options_model = self.options_model.cuda()
            self.objects_model = self.objects_model.cuda()
            for i in range(self.num_options):
                self.action_heads[i] = self.action_heads[i].cuda()

        ce_loss = torch.nn.CrossEntropyLoss()
        modules = torch.nn.ModuleDict()
        modules['cnn_model'] = self.cnn_model
        modules['flatten'] = self.flatten
        modules['options_model'] = self.options_model
        modules['objects_model'] = self.objects_model
        for i in range(self.num_options):
            modules['action_heads'+str(i)] = self.action_heads[i]

        model_params = []
        nec_value_params = []
        for name, p in modules.named_parameters():
            if 'values.weight' in name:
                nec_value_params.append(p)
            else:
                model_params.append(p)
        optimizer = torch.optim.Adam([{'params': model_params}, 
                                      {'params': nec_value_params, 'lr': self.args.nec_lr}], lr=self.args.lr)

        return {'modules': modules, 
                'opt': optimizer, 
                'ce_loss': ce_loss, 
                'softmax': self.smax}

    def forward(self, state, options=None):
        x = (state['pov'].permute(0, 3, 1, 2)/255.0)*2.0 - 1.0
        cnn_fea = self._a['modules']['cnn_model'](x) # [batch, channels, 4, 4]
        all_fea = torch.cat([self._a['modules']['flatten'](cnn_fea), state['state']], dim=1)
        if options is None:
            options = self._a['modules']['options_model'](all_fea)  # [batch, options]
        objects = self._a['modules']['objects_model'](all_fea)  # [batch, channels]

        
        with torch.no_grad():
            intra_entropy = torch.mean(torch.sum(-options*torch.log(options), dim=1))
        inter_entropy = torch.mean(torch.std(options, dim=0))

        #compress via attention on objects
        flat_cnn_fea = cnn_fea.permute(0, 2, 3, 1).contiguous().view(-1, self.inp_size[1]*self.inp_size[2], self.inp_size[0]) #[batch, 4*4, channel]
        attention = torch.bmm(flat_cnn_fea, objects.unsqueeze(2)).squeeze(2) #[batch, 4*4]
        attention = self._a['softmax'](attention)

        action_outputs = []
        for i in range(self.num_options):
            attention_masked = self.masks[i].unsqueeze(0).view(1, -1) * attention
            attention_fea = torch.sum(attention_masked.unsqueeze(2)*flat_cnn_fea, dim=1) #[batch, channels]
            action_head_input = torch.cat([attention_fea, attention_masked.view(-1, self.inp_size[1]*self.inp_size[2]), state['state']], dim=1)
            action_outputs.append(self._a['modules']['action_heads'+str(i)](action_head_input).unsqueeze(1))

        action_outputs = torch.cat(action_outputs, dim=1)
        action = torch.sum(action_outputs * options.unsqueeze(2), dim=1)
        return action, intra_entropy, inter_entropy, options

    def pretrain(self, input):
        state_keys = ['state', 'pov']
        states = {}
        for k in state_keys:
            if k in input:
                states[k] = variable(input[k])
        actions = variable(input['action'], True).long()

        def closure():
            self._a['opt'].zero_grad()
            logits, self.intra_entropy, self.inter_entropy, _ = self.forward(states)
            self.bc_loss = self._a['ce_loss'](logits, actions)
            loss = self.bc_loss #- 0.25*self.inter_entropy
            loss.backward()
            with torch.no_grad():
                probas = self._a['softmax'](logits)
                self.max_probs = torch.mean(torch.max(probas,dim=1)[0])
                self.entropy = torch.mean(-torch.sum(probas*torch.log(probas), dim=1))
            return loss

        #loss = self._a['opt'].step(closure).data.cpu().numpy()
        self._a['opt'].step(closure)

        return [self.bc_loss.data.cpu().numpy(), 
                self.entropy.data.cpu().numpy(), 
                self.max_probs.data.cpu().numpy(),
                self.intra_entropy.data.cpu().numpy(), 
                self.inter_entropy.data.cpu().numpy()]


    def select_action(self, state, batch=False, options=None):
        if not batch:
            for k in state:
                state[k] = state[k][None,:]
        if options is not None:
            options = np.array([1.0 if i == options else 0.0 for i in range(5)])[None, :]
            options = variable(options)
        with torch.no_grad():
            probas, _, _, options = self.forward(variable(state), options)
        probas = self._a['softmax'](probas).data.cpu().numpy()[0]
        options = options.data.cpu().numpy()[0]

        action_index = int(np.random.choice(range(self.num_actions), p=probas))
        try:
            entropy = -np.sum(probas*np.log2(probas))
        except:
            print(probas)
        return action_index, entropy, options




class SimpleBC():
    def __init__(self, state_shape, num_actions, args):
        assert isinstance(state_shape, dict)
        self.state_shape = state_shape
        self.num_actions = num_actions
        self.args = args
        
        self._setup()
        self._a = self._models[0]

    def _setup(self):
        self._models = [self._make_model()]

    def state_dict(self):
        return [self._models[0]['modules'].state_dict()]

    def load_state_dict(self, s, strict=True):
        if isinstance(s, list):
            self._models[0]['modules'].load_state_dict(s[0], strict)
            self._models[1]['modules'].load_state_dict(s[1], strict)
        else:
            self._models[0]['modules'].load_state_dict(s, strict)

    def loadstore(self, filename, load=True):
        if load:
            self.load_state_dict(torch.load(filename + '-simBE'))
        else:
            torch.save(self._models[0]['modules'].state_dict(), filename + '-simBE')

    def _make_model(self):
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

        #cnn_layers.append(Flatten())
        self.flatten = Flatten()
        self.cnn_model = torch.nn.Sequential(*cnn_layers)

        inp_size = self.cnn_model(torch.zeros((1,self.state_shape['pov'][2], self.state_shape['pov'][0], self.state_shape['pov'][1]))).shape[1:] #CHW
        self.inp_size = inp_size
        all_inp_size = inp_size[0]*inp_size[1]*inp_size[2]+self.state_shape['state'][0]

        self.body = torch.nn.Sequential(torch.nn.Linear(all_inp_size, self.args.hidden), torch.nn.Tanh())

        self.action_head = HashingMemory.build(self.args.hidden, self.num_actions, self.args)

        self.smax = torch.nn.Softmax(1)


        if torch.cuda.is_available() and CUDA:
            self.smax = self.smax.cuda()
            self.cnn_model = self.cnn_model.cuda()
            self.flatten = self.flatten.cuda()
            self.body = self.body.cuda()
            self.action_head = self.action_head.cuda()

        ce_loss = torch.nn.CrossEntropyLoss()
        modules = torch.nn.ModuleDict()
        modules['cnn_model'] = self.cnn_model
        modules['flatten'] = self.flatten
        modules['body'] = self.body
        modules['action_head'] = self.action_head

        model_params = []
        nec_value_params = []
        for name, p in modules.named_parameters():
            if 'values.weight' in name:
                nec_value_params.append(p)
            else:
                model_params.append(p)
        optimizer = torch.optim.Adam([{'params': model_params}, 
                                      {'params': nec_value_params, 'lr': self.args.nec_lr}], lr=self.args.lr)

        return {'modules': modules, 
                'opt': optimizer, 
                'ce_loss': ce_loss, 
                'softmax': self.smax}

    def forward(self, state, options=None):
        x = (state['pov'].permute(0, 3, 1, 2)/255.0)*2.0 - 1.0
        cnn_fea = self._a['modules']['cnn_model'](x) # [batch, channels, 4, 4]
        all_fea = torch.cat([self._a['modules']['flatten'](cnn_fea), state['state']], dim=1)

        fea = self._a['modules']['body'](all_fea)

        action = self._a['modules']['action_head'](fea)

        return action, fea

    def pretrain(self, input):
        state_keys = ['state', 'pov']
        states = {}
        for k in state_keys:
            if k in input:
                states[k] = variable(input[k])
        actions = variable(input['action'], True).long()
        next_states = {}
        for k in state_keys:
            next_k = 'next_'+k
            if next_k in input:
                next_states[k] = variable(input[next_k])

        def closure():
            self._a['opt'].zero_grad()
            logits, fea = self.forward(states)
            self.bc_loss = self._a['ce_loss'](logits, actions)
            loss = self.bc_loss
            with torch.no_grad():
                _, next_fea = self.forward(next_states)
            next_fea = next_fea.detach() #no mode collapse?
            self.piss_loss = torch.mean((fea[:,:self.args.hidden//2]-next_fea[:,:self.args.hidden//2])**2)
            loss += self.piss_loss
            loss.backward()
            with torch.no_grad():
                probas = self._a['softmax'](logits)
                self.max_probs = torch.mean(torch.max(probas,dim=1)[0])
                self.entropy = torch.mean(-torch.sum(probas*torch.log(probas), dim=1))
            return loss

        #loss = self._a['opt'].step(closure).data.cpu().numpy()
        self._a['opt'].step(closure)

        return {'loss': self.bc_loss.data.cpu().numpy(), 
                'entropy': self.entropy.data.cpu().numpy(), 
                'max_probs': self.max_probs.data.cpu().numpy(),
                'piss_loss': self.piss_loss.data.cpu().numpy()}

    def get_features(self, state):
        with torch.no_grad():
            return self.forward(variable(state))[1][:,:self.args.hidden//2]

    def select_action(self, state, batch=False):
        if not batch:
            for k in state:
                state[k] = state[k][None,:]
        with torch.no_grad():
            probas = self.forward(variable(state))[0]
        probas = self._a['softmax'](probas).data.cpu().numpy()[0]

        action_index = int(np.random.choice(range(self.num_actions), p=probas))
        try:
            entropy = -np.sum(probas*np.log2(probas))
        except:
            print(probas)
        return action_index, entropy




class FisherBC():
    def __init__(self, state_shape, num_actions, args):
        assert isinstance(state_shape, dict)
        self.state_shape = state_shape
        self.num_actions = num_actions
        self.state_segment = 17+15*0
        self.args = args
        
        self._setup()
        self._a = self._models[0]

    def _setup(self):
        self._models = [self._make_model()]

    def state_dict(self):
        return [self._models[0]['modules'].state_dict()]

    def load_state_dict(self, s, strict=True):
        if isinstance(s, list):
            self._models[0]['modules'].load_state_dict(s[0], strict)
            self._models[1]['modules'].load_state_dict(s[1], strict)
        else:
            self._models[0]['modules'].load_state_dict(s, strict)

    def loadstore(self, filename, load=True):
        if load:
            self.load_state_dict(torch.load(filename + '-fisherBC'))
            self._models[0]['Lambda'] = torch.load(filename + '-fisherLam')
            self._models[0]['Lambda2'] = torch.load(filename + '-fisherLam2')
        else:
            torch.save(self._models[0]['modules'].state_dict(), filename + '-fisherBC')
            torch.save(self._models[0]['Lambda'], filename + '-fisherLam')
            torch.save(self._models[0]['Lambda2'], filename + '-fisherLam2')

    def _make_model(self):
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

        #cnn_layers.append(Flatten())
        self.flatten = Flatten()
        self.cnn_model = torch.nn.Sequential(*cnn_layers)

        inp_size = self.cnn_model(torch.zeros((1,self.state_shape['pov'][2], self.state_shape['pov'][0], self.state_shape['pov'][1]))).shape[1:] #CHW
        self.inp_size = inp_size
        all_inp_size = inp_size[0]*inp_size[1]*inp_size[2]+self.state_shape['state'][0] - self.state_segment # without current action

        self.macro_feature = torch.nn.Sequential(torch.nn.Linear(all_inp_size, self.args.hidden), Lambda(lambda x: swish(x)))

        #deep infomax
        self.deep_infomax = torch.nn.Sequential(torch.nn.Linear(self.args.hidden*2, self.args.hidden), Lambda(lambda x: swish(x)), torch.nn.Linear(self.args.hidden, 1))

        #unsupervised classification
        self.num_classes = 10
        self.option_class = torch.nn.Sequential(torch.nn.Linear(self.args.hidden, self.num_classes), torch.nn.Softmax(1))
        self.deep_infomax_classes = torch.nn.Sequential(torch.nn.Linear(self.args.hidden+self.num_classes, self.args.hidden), Lambda(lambda x: swish(x)), torch.nn.Linear(self.args.hidden, 1))

        self.action_head = torch.nn.Sequential(torch.nn.Linear(self.args.hidden+self.state_segment, self.args.hidden), torch.nn.Tanh(), HashingMemory.build(self.args.hidden, self.num_actions, self.args))

        self.smax = torch.nn.Softmax(1)
        lam = torch.zeros((1,), requires_grad=True)
        lam_classes = torch.zeros((1,), requires_grad=True)

        if torch.cuda.is_available() and CUDA:
            self.smax = self.smax.cuda()
            self.cnn_model = self.cnn_model.cuda()
            self.flatten = self.flatten.cuda()
            self.macro_feature = self.macro_feature.cuda()
            self.deep_infomax = self.deep_infomax.cuda()
            self.action_head = self.action_head.cuda()
            self.option_class = self.option_class.cuda()
            self.deep_infomax_classes = self.deep_infomax_classes.cuda()
            lam = lam.cuda()
            lam_classes = lam_classes.cuda()

        ce_loss = torch.nn.CrossEntropyLoss()
        modules = torch.nn.ModuleDict()
        modules['cnn_model'] = self.cnn_model
        modules['flatten'] = self.flatten
        modules['macro_feature'] = self.macro_feature
        modules['deep_infomax'] = self.deep_infomax
        modules['action_head'] = self.action_head
        modules['option_class'] = self.option_class
        modules['deep_infomax_classes'] = self.deep_infomax_classes

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
                'ce_loss': ce_loss, 
                'softmax': self.smax,
                'Lambda': lam,
                'Lambda2': lam_classes}

    def forward(self, state, options=None):
        x = (state['pov'].permute(0, 3, 1, 2)/255.0)*2.0 - 1.0
        cnn_fea = self._a['modules']['cnn_model'](x) # [batch, channels, 4, 4]
        state_wocurrentaction = state['state'][:,:-self.state_segment]
        caction = state['state'][:,-self.state_segment:]
        #print(caction.data.cpu().numpy())
        macro_fea = torch.cat([self._a['modules']['flatten'](cnn_fea), state_wocurrentaction], dim=1)

        macro_fea = self._a['modules']['macro_feature'](macro_fea)

        action = self._a['modules']['action_head'](torch.cat([macro_fea, caction], dim=1))

        return action, macro_fea

    def pretrain(self, input):
        state_keys = ['state', 'pov']
        states = {}
        for k in state_keys:
            if k in input:
                states[k] = variable(input[k])
        actions = variable(input['action'], True).long()
        next_states = {}
        for k in state_keys:
            next_k = 'next_'+k
            if next_k in input:
                next_states[k] = variable(input[next_k])

        def closure():
            self._a['opt'].zero_grad()
            logits, fea = self.forward(states)
            self.bc_loss = self._a['ce_loss'](logits, actions)
            loss = self.bc_loss.clone()

            #make fisher deep infomax
            _, next_fea = self.forward(next_states)

            real_pairs = torch.cat([fea, next_fea], dim=1)
            real_v = self._a['modules']['deep_infomax'](real_pairs)
            random_roll = np.random.randint(1, self.args.er-1)
            fake_pairs = torch.cat([fea, next_fea.roll(random_roll, 0)], dim=1)
            fake_v = self._a['modules']['deep_infomax'](fake_pairs)

            real_f, fake_f = real_v.mean(), fake_v.mean()
            real_f2, fake_f2 = (real_v**2).mean(), (fake_v**2).mean()
            self.constraint = (1 - (0.5*real_f2 + 0.5*fake_f2))

            self.fisher_infomax_loss = real_f - fake_f + self._a['Lambda'][0] * self.constraint - (self.args.rho/2.0) * self.constraint**2
            loss -= 0.1 * self.fisher_infomax_loss

            #make fisher deep infomax unsupervised classification
            classes = self._a['modules']['option_class'](fea)
            real_pairs = torch.cat([fea, classes], dim=1)
            real_v = self._a['modules']['deep_infomax_classes'](real_pairs)
            random_roll = np.random.randint(1, self.args.er-1)
            fake_pairs = torch.cat([fea, classes.roll(random_roll, 0)], dim=1)
            fake_v = self._a['modules']['deep_infomax_classes'](fake_pairs)
            
            real_f, fake_f = real_v.mean(), fake_v.mean()
            real_f2, fake_f2 = (real_v**2).mean(), (fake_v**2).mean()
            self.constraint2 = (1 - (0.5*real_f2 + 0.5*fake_f2))
            self.fisher_infomax_loss2 = real_f - fake_f + self._a['Lambda2'][0] * self.constraint2 - (self.args.rho/2.0) * self.constraint2**2
            loss -= 0.05 * self.fisher_infomax_loss2

            self._a['Lambda'].retain_grad()
            self._a['Lambda2'].retain_grad()
            loss.backward()
            with torch.no_grad():
                probas = self._a['softmax'](logits)
                self.max_probs = torch.mean(torch.max(probas,dim=1)[0])
                self.entropy = torch.mean(-torch.sum(probas*torch.log(probas), dim=1))

            return loss

        #loss = self._a['opt'].step(closure).data.cpu().numpy()
        self._a['opt'].step(closure)
        self._a['Lambda'].data += self.args.rho * self._a['Lambda'].grad.data
        self._a['Lambda'].grad.data.zero_()
        self._a['Lambda2'].data += self.args.rho * self._a['Lambda2'].grad.data
        self._a['Lambda2'].grad.data.zero_()


        return {'loss': self.bc_loss.data.cpu().numpy(), 
                'entropy': self.entropy.data.cpu().numpy(), 
                'max_probs': self.max_probs.data.cpu().numpy(),
                'fisher_loss': -self.fisher_infomax_loss.data.cpu().numpy(),
                'Lambda': self._a['Lambda'].data.cpu().numpy()[0],
                'constraint': self.constraint.data.cpu().numpy(),
                'fisher_loss2': -self.fisher_infomax_loss2.data.cpu().numpy(),
                'Lambda2': self._a['Lambda2'].data.cpu().numpy()[0],
                'constraint2': self.constraint2.data.cpu().numpy()}

    def get_features(self, state):
        with torch.no_grad():
            return self._a['modules']['option_class'](self.forward(variable(state))[1])

    def select_action(self, state, batch=False):
        if not batch:
            for k in state:
                state[k] = state[k][None,:]
        with torch.no_grad():
            probas = self.forward(variable(state))[0]
        probas = self._a['softmax'](probas).data.cpu().numpy()[0]

        action_index = int(np.random.choice(range(self.num_actions), p=probas))
        try:
            entropy = -np.sum(probas*np.log2(probas))
        except:
            print(probas)
        return action_index, entropy




class FeatureFisherBC():
    def __init__(self, state_shape, num_actions, args):
        assert isinstance(state_shape, dict)
        self.state_shape = state_shape
        self.num_actions = num_actions
        self.args = args
        
        self._setup()
        self._a = self._models[0]

    def _setup(self):
        self._models = [self._make_model()]

    def state_dict(self):
        return [self._models[0]['modules'].state_dict()]

    def load_state_dict(self, s, strict=True):
        if isinstance(s, list):
            self._models[0]['modules'].load_state_dict(s[0], strict)
            self._models[1]['modules'].load_state_dict(s[1], strict)
        else:
            self._models[0]['modules'].load_state_dict(s, strict)

    def loadstore(self, filename, load=True):
        if load:
            self.load_state_dict(torch.load(filename + '-featurefisherBC'))
            self._models[0]['Lambda'] = torch.load(filename + '-featurefisherLam')
        else:
            torch.save(self._models[0]['modules'].state_dict(), filename + '-featurefisherBC')
            torch.save(self._models[0]['Lambda'], filename + '-featurefisherLam')

    def _make_model(self):
        #make CNN
        self.micro_state = 64
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

        #cnn_layers.append(Flatten())
        self.flatten = Flatten()
        self.cnn_model = torch.nn.Sequential(*cnn_layers)

        inp_size = self.cnn_model(torch.zeros((1,self.state_shape['pov'][2], self.state_shape['pov'][0], self.state_shape['pov'][1]))).shape[1:] #CHW
        self.inp_size = inp_size
        all_inp_size = inp_size[0]*inp_size[1]*inp_size[2]#+self.state_shape['state'][0]

        self.macro_feature = torch.nn.Sequential(torch.nn.Linear(all_inp_size, self.micro_state), Lambda(lambda x: swish(x)))

        #deep infomax
        self.deep_infomax = torch.nn.Sequential(torch.nn.Linear(self.micro_state+in_channels+4, 128), Lambda(lambda x: swish(x)), torch.nn.Linear(128, 1))

        self.smax = torch.nn.Softmax(1)
        lam = torch.zeros((1,), requires_grad=True)

        if torch.cuda.is_available() and CUDA:
            self.smax = self.smax.cuda()
            self.cnn_model = self.cnn_model.cuda()
            self.flatten = self.flatten.cuda()
            self.macro_feature = self.macro_feature.cuda()
            self.deep_infomax = self.deep_infomax.cuda()
            lam = lam.cuda()

        ce_loss = torch.nn.CrossEntropyLoss()
        modules = torch.nn.ModuleDict()
        modules['cnn_model'] = self.cnn_model
        modules['flatten'] = self.flatten
        modules['macro_feature'] = self.macro_feature
        modules['deep_infomax'] = self.deep_infomax

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
                'ce_loss': ce_loss, 
                'softmax': self.smax,
                'Lambda': lam}

    def forward_fea(self, state):
        x = (state['real_pov'].permute(0, 3, 1, 2)/255.0)*2.0 - 1.0
        cnn_fea = self._a['modules']['cnn_model'](x) # [batch, channels, 4, 4]
        macro_fea = self._a['modules']['flatten'](cnn_fea)
        macro_fea = self._a['modules']['macro_feature'](macro_fea)

        #add position to cnn_features
        if torch.cuda.is_available() and CUDA:
            pos = torch.arange(0, self.inp_size[1], 1, dtype=torch.float32, device=torch.device('cuda'))
        else:
            pos = torch.arange(0, self.inp_size[1], 1, dtype=torch.float32)
        pos = torch.stack([torch.sin((pos*2.0-1.0)*np.pi), 
                           torch.cos((pos*2.0-1.0)*np.pi)], dim=0)
        posx = pos.unsqueeze(0).unsqueeze(3)
        posx = posx.expand(x.size(0), -1, -1, self.inp_size[2])
        posy = pos.unsqueeze(0).unsqueeze(2)
        posy = posy.expand(x.size(0), -1, self.inp_size[1], -1)
        cnn_fea = torch.cat([cnn_fea, posx, posy], dim=1)

        return macro_fea, cnn_fea

    def pretrain(self, input):
        state_keys = ['real_state', 'real_pov']
        states = {}
        for k in state_keys:
            if k in input:
                states[k] = variable(input[k])
        #actions = variable(input['action'], True).long()
        #next_states = {}
        #for k in state_keys:
        #    next_k = 'next_'+k
        #    if next_k in input:
        #        next_states[k] = variable(input[next_k])

        def closure():
            #make fisher deep infomax
            macro_fea, cnn_fea = self.forward_fea(states)
            macro_fea = macro_fea.unsqueeze(2).unsqueeze(3)
            macro_fea = macro_fea.expand(-1, -1, self.inp_size[1], self.inp_size[2])
            real_pairs = torch.cat([macro_fea, cnn_fea], dim=1).permute(0, 2, 3, 1).contiguous().view(-1, self.micro_state+self.inp_size[0]+4)
            real_v = self._a['modules']['deep_infomax'](real_pairs)
            random_roll = np.random.randint(1, self.args.er-1)
            fake_pairs = torch.cat([macro_fea, cnn_fea.roll(random_roll, 0)], dim=1).permute(0, 2, 3, 1).contiguous().view(-1, self.micro_state+self.inp_size[0]+4)
            fake_v = self._a['modules']['deep_infomax'](fake_pairs)
            
            real_f, fake_f = real_v.mean(), fake_v.mean()
            real_f2, fake_f2 = (real_v**2).mean(), (fake_v**2).mean()
            self.constraint = (1 - (0.5*real_f2 + 0.5*fake_f2))
            
            self.fisher_infomax_loss = real_f - fake_f + self._a['Lambda'][0] * self.constraint - (self.args.rho/2.0) * self.constraint**2
            loss = -self.fisher_infomax_loss

            self._a['Lambda'].retain_grad()
            loss.backward()

            return loss

        #loss = self._a['opt'].step(closure).data.cpu().numpy()
        self._a['opt'].step(closure)
        self._a['Lambda'].data += self.args.rho * self._a['Lambda'].grad.data
        self._a['Lambda'].grad.data.zero_()


        return {'fisher_loss': -self.fisher_infomax_loss.data.cpu().numpy(),
                'Lambda': self._a['Lambda'].data.cpu().numpy()[0],
                'constraint': self.constraint.data.cpu().numpy()}

    def get_features(self, state):
        with torch.no_grad():
            return self._a['modules']['option_class'](self.forward(variable(state))[1])

    def select_action(self, state, batch=False):
        if not batch:
            for k in state:
                state[k] = state[k][None,:]
        with torch.no_grad():
            probas = self.forward(variable(state))[0]
        probas = self._a['softmax'](probas).data.cpu().numpy()[0]

        action_index = int(np.random.choice(range(self.num_actions), p=probas))
        try:
            entropy = -np.sum(probas*np.log2(probas))
        except:
            print(probas)
        return action_index, entropy