import numpy
import random
import numpy as np
import torch

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
