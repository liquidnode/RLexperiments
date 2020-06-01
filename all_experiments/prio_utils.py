import numpy
import random
import numpy as np
import torch
from utils import one_hot
from torch.nn import functional as F

class WeightedMSETargetLoss(torch.nn.Module):
    def __init__(self):
        super(WeightedMSETargetLoss, self).__init__()
        self.mse = torch.nn.MSELoss(reduction='none')

    def forward(self, x, target, weight=None):
        x = F.softmax(x-x.logsumexp(dim=-1, keepdim=True), -1)
        p_target = one_hot(target, x.shape[-1], x.device)
        if weight is None:
            p_target = 0.05 * p_target + 0.95 * x.detach()
            return torch.mean(self.mse(x, p_target))
        else:
            weight = weight.unsqueeze(-1).expand(x.shape)
            mask = torch.where(weight>=0.0,  torch.ones_like(p_target), p_target)
            p_target = torch.where(weight>=0.0, 0.05 * p_target + 0.95 * x.detach(), 0.05 * (1.0 - p_target) + 0.95 * x.detach())
            weight = torch.where(weight>=0.0,  weight, -10.0*weight)
            #loss = torch.where(weight>=0.0, weight*self.mse(x, 0.05 * p_target + 0.95 * x.detach()), -weight * p_target * x)
            return torch.mean(weight*mask*self.mse(x, p_target))

class WeightedCELoss(torch.nn.Module):
    def __init__(self):
        super(WeightedCELoss, self).__init__()
        self.ce = torch.nn.CrossEntropyLoss(reduction='none')

    def forward(self, x, target, weight=None):
        if weight is None:
            return torch.mean(self.ce(x, target))
        else:
            return torch.mean(weight*self.ce(x, target))

class WeightedMSELoss(torch.nn.Module):
    def __init__(self):
        super(WeightedMSELoss, self).__init__()
        self.mse = torch.nn.MSELoss(reduction='none')

    def forward(self, x, target, weight=None):
        if weight is None:
            return torch.mean(self.mse(x, target))
        else:
            return torch.mean(weight*self.mse(x, target))


#code from https://github.com/rlcode/per

# SumTree
# a binary tree data structure where the parent’s value is the sum of its children
class SumTree:
    write = 0

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = numpy.zeros(2 * capacity - 1)
        self.n_entries = 0

    # update to the root node
    def _propagate(self, idx, change):
        parent = (idx - 1) // 2

        self.tree[parent] += change

        if parent != 0:
            self._propagate(parent, change)

    # find sample on leaf node
    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        return self.tree[0]

    # store priority and sample
    def add(self, p):
        idx = self.write + self.capacity - 1

        self.update(idx, p)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0

        if self.n_entries < self.capacity:
            self.n_entries += 1

    # update priority
    def update(self, idx, p):
        change = p - self.tree[idx]

        self.tree[idx] = p
        self._propagate(idx, change)

    # get priority and sample
    def get(self, s):
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1

        return (idx, self.tree[idx], dataIdx)
