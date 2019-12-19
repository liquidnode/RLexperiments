import torch
from torch.nn import functional as F
from utils import variable, Lambda, swish, Flatten, sample_gaussian, one_hot
import math

#Implementation of "Variational Dropout Sparsifies Deep Neural Networks" https://arxiv.org/abs/1701.05369
class VariationalDropoutLinear(torch.nn.Module):
    def __init__(self, in_size, out_size):
        super(VariationalDropoutLinear, self).__init__()
        self.linear = torch.nn.Linear(in_size, out_size)
        self.log_sigma_sqr = torch.nn.Parameter(torch.Tensor(in_size, out_size))
        self.log_sigma_sqr.data.fill_(-10.0)

    def forward(self, x):
        _wt = self.linear.weight.t()
        log_alpha = torch.clamp(self.log_sigma_sqr - 2.0 * torch.log(torch.abs(_wt)+1e-8),
                                max=8.0,
                                min=-8.0)
        if self.training:
            mean = self.linear(x)
            x2 = x * x
            _wt2 = _wt*_wt
            std = torch.sqrt(x2.matmul(log_alpha.exp()*_wt2))
            return mean + std * torch.normal(torch.zeros_like(mean), torch.ones_like(mean))
        else:
            return x.matmul(torch.where(log_alpha>3.0,
                                        torch.zeros_like(_wt),
                                        _wt))+self.linear.bias

    def to_loss(self):
        _wt = self.linear.weight.t()
        k1 = 0.63576
        k2 = 1.87320
        k3 = 1.48695
        log_alpha = torch.clamp(self.log_sigma_sqr - 2.0 * torch.log(torch.abs(_wt)+1e-8),
                                max=8.0,
                                min=-8.0)
        return -(k1 * torch.sigmoid(k2 + k3 * log_alpha) - \
            0.5 * torch.log1p((-log_alpha).exp()) - k1).sum()

def add_loss(module, loss=0):
    to_loss = getattr(module, "to_loss", None)
    if callable(to_loss):
        return loss + to_loss()
    if hasattr(module, 'children'): 
        return loss + sum([add_loss(m) for m in module.children()])
    return loss





class CameraBias(torch.nn.Module):
    def __init__(self, size, device):
        super(CameraBias, self).__init__()
        self.bias = (torch.arange(0, size, device=device).float() / ((size - 1) // 2)) - 1.0
        self.bias = self.bias.reshape((1, 1, size, 1)).expand(-1, -1, -1, size) ** 2 + \
            self.bias.reshape((1, 1, 1, size)).expand(-1, -1, size, -1) ** 2
        self.bias = - 30.0 * self.bias
        self.bias = torch.tensor(self.bias, requires_grad=True)

    def forward(self, x, interpolate=None):
        if interpolate is None:
            return x + self.bias
        else:
            return (x * interpolate.reshape(-1,1,1,1)) + self.bias

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
