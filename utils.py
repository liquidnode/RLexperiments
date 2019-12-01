import torch
import numpy as np

def swish(x):
    return x * torch.sigmoid(x)

class Lambda(torch.nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)

class Flatten(torch.nn.Module):
    """ Flatten an input, used to map a convolution to a Dense layer
    """
    def forward(self, x):
        return x.view(x.size()[0], -1)

def _variable(inp, ignore_type=False, half=False):
    if torch.is_tensor(inp):
        rs = inp
    else:
        rs = torch.from_numpy(np.asarray(inp))

    if torch.cuda.is_available():
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
        
def sample_gaussian(mu, log_var):
    std = torch.exp(0.5 * log_var)
    eps = torch.zeros(std.size(), device=mu.device).normal_()
    return mu + std * eps

def one_hot(ac, num, device):
    one_hot = torch.zeros(list(ac.shape)+[num], dtype=torch.float32, device=device)
    one_hot.scatter_(-1, ac.unsqueeze(-1), 1.0)
    return one_hot