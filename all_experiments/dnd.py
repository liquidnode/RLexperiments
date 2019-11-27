#from https://github.com/mjacar/pytorch-nec

"""
MIT License

Copyright (c) 2018 Michael Acar

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""


import torch
import torch.optim as optim
from torch.nn import Parameter
from pyflann import FLANN
import numpy as np


class DND(torch.nn.Module):
  def __init__(self, kernel, num_neighbors, max_memory, lr):
    super(DND, self).__init__()
    self.kernel = kernel
    self.num_neighbors = num_neighbors
    self.max_memory = max_memory
    self.lr = lr
    self.register_parameter('keys', None)
    self.register_parameter('values', None)
    self.kdtree = FLANN()

    # key_cache stores a cache of all keys that exist in the DND
    # This makes DND updates efficient
    self.key_cache = {}
    # stale_index is a flag that indicates whether or not the index in self.kdtree is stale
    # This allows us to only rebuild the kdtree index when necessary
    self.stale_index = True
    # indexes_to_be_updated is the set of indexes to be updated on a call to update_params
    # This allows us to rebuild only the keys of key_cache that need to be rebuilt when necessary
    self.indexes_to_be_updated = set()

    # Keys and value to be inserted into self.keys and self.values when commit_insert is called
    self.keys_to_be_inserted = None
    self.values_to_be_inserted = None

    # Move recently used lookup indexes
    # These should be moved to the back of self.keys and self.values to get LRU property
    self.move_to_back = set()

  def forward(self, input):
      return None

  def get_index(self, key):
    """
    If key exists in the DND, return its index
    Otherwise, return None
    """
    #if self.key_cache.get(tuple(key.data.cpu().numpy()[0])) is not None:
    #  if self.stale_index:
    #    self.commit_insert()
    #  return int(self.kdtree.nn_index(key.data.cpu().numpy(), 1)[0][0])
    #else:
    return None

  def update(self, value, index):
    """
    Set self.values[index] = value
    """
    values = self.values.data
    values[index] = value[0].data
    self.values = Parameter(values)
    self.optimizer = optim.RMSprop([self.keys, self.values], lr=self.lr)

  def insert(self, key, value):
    """
    Insert key, value pair into DND
    """
    if self.keys_to_be_inserted is None:
      # Initial insert
      self.keys_to_be_inserted = key.data
      self.values_to_be_inserted = value.data
    else:
      self.keys_to_be_inserted = torch.cat(
          [self.keys_to_be_inserted, key.data], 0)
      self.values_to_be_inserted = torch.cat(
          [self.values_to_be_inserted, value.data], 0)
    #self.key_cache[tuple(key.data.cpu().numpy()[0])] = 0
    self.stale_index = True

  def insert_batch(self, key, value):
    if self.keys_to_be_inserted is None:
      # Initial insert
      self.keys_to_be_inserted = key.data
      self.values_to_be_inserted = value.data
    else:
      self.keys_to_be_inserted = torch.cat(
          [self.keys_to_be_inserted, key.data], 0)
      self.values_to_be_inserted = torch.cat(
          [self.values_to_be_inserted, value.data], 0)
    #scheiss auf key_cache
      self.stale_index = True

  def commit_insert(self):
    if self.keys_to_be_inserted is None:
        return
    if self.keys is None:
      self.keys = Parameter(self.keys_to_be_inserted)
      self.values = Parameter(self.values_to_be_inserted)
    elif self.keys_to_be_inserted is not None:
      self.keys = Parameter(
          torch.cat([self.keys.data, self.keys_to_be_inserted], 0))
      self.values = Parameter(
          torch.cat([self.values.data, self.values_to_be_inserted], 0))

    # Move most recently used key-value pairs to the back
    if len(self.move_to_back) != 0:
      self.keys = Parameter(torch.cat([self.keys.data[list(set(range(len(
          self.keys))) - self.move_to_back)], self.keys.data[list(self.move_to_back)]], 0))
      self.values = Parameter(torch.cat([self.values.data[list(set(range(len(
          self.values))) - self.move_to_back)], self.values.data[list(self.move_to_back)]], 0))
      self.move_to_back = set()

    if len(self.keys) > self.max_memory:
      # Expel oldest key to maintain total memory
      #for key in self.keys[:-self.max_memory]:
      #  del self.key_cache[tuple(key.data.cpu().numpy())]
      self.keys = Parameter(self.keys[-self.max_memory:].data)
      self.values = Parameter(self.values[-self.max_memory:].data)
    self.keys_to_be_inserted = None
    self.values_to_be_inserted = None
    #self.optimizer = optim.RMSprop([self.keys, self.values], lr=self.lr)
    self.kdtree.build_index(self.keys.data.cpu().clone().numpy())
    self.stale_index = False

  def lookup(self, lookup_key, update_flag=False, lookup_key_cpu=None):
    """
    Perform DND lookup
    If update_flag == True, add the nearest neighbor indexes to self.indexes_to_be_updated
    """
    if self.keys is None or len(self.keys) < 2:
        return torch.zeros((1,), dtype=torch.float32).cuda()
    if lookup_key_cpu is None:
        lookup_key_cpu = lookup_key.data.cpu()
    lookup_indexes = self.kdtree.nn_index(
        lookup_key_cpu.numpy(), min(self.num_neighbors, len(self.keys)))[0][0]
    output = 0
    kernel_sum = 0
    r_lookup_indexes = []
    for i, index in enumerate(lookup_indexes):
      if i == 0 and self.key_cache.get(tuple(lookup_key[0].data.cpu().numpy())) is not None:
        # If a key exactly equal to lookup_key is used in the DND lookup calculation
        # then the loss becomes non-differentiable. Just skip this case to avoid the issue.
        continue
      r_lookup_indexes.append(index)
      if update_flag:
        self.indexes_to_be_updated.add(int(index))
      else:
        self.move_to_back.add(int(index))
    r_lookup_indexes = np.array(r_lookup_indexes)
    kernel_val = self.kernel(self.keys[r_lookup_indexes], lookup_key.expand(len(r_lookup_indexes), -1))
    output = kernel_val * self.values[r_lookup_indexes]
    kernel_sum = kernel_val.sum()
    output = (output / kernel_sum).sum(0)[None]
    return output

  def lookup_batch(self, lookup_key, update_flag=False, lookup_key_cpu=None):
    if lookup_key_cpu is None:
        lookup_key_cpu = lookup_key.data.cpu()
    if self.keys is None or len(self.keys) < 2:
        return torch.zeros((lookup_key.shape[0],), dtype=torch.float32).cuda()
    lookup_indexes = self.kdtree.nn_index(
        lookup_key_cpu.numpy(), min(self.num_neighbors, self.keys.shape[0]))[0]
    for i, index in enumerate(lookup_indexes):
      index = np.copy(np.array(index))
      index = np.where(index > len(self.keys), -1, index)
      index = np.where(index < 0, -1, index)
      if update_flag:
        self.indexes_to_be_updated.update(index)
        self.indexes_to_be_updated.discard(-1)
      else:
        self.move_to_back.update(index)
        self.move_to_back.discard(-1)
    lookup_indexes = np.array(lookup_indexes) # batch, indices
    if np.any(lookup_indexes > self.keys.shape[0]) or np.any(lookup_indexes < 0):
        print("STRANGE occurrance")
        print(np.min(lookup_indexes))
        print(np.max(lookup_indexes))
        print(self.keys.shape[0])
    lookup_indexes = np.where(lookup_indexes > self.keys.shape[0], 0, lookup_indexes)
    lookup_indexes = np.where(lookup_indexes < 0, 0, lookup_indexes)
    assert self.values.shape[0] == self.keys.shape[0]
    batch_size = lookup_indexes.shape[0]
    ind_num = lookup_indexes.shape[1]
    lookup_indexes = np.reshape(lookup_indexes, [-1])
    kernel_val = self.kernel(self.keys[lookup_indexes].view(batch_size, ind_num, -1), 
                             lookup_key.unsqueeze(1).expand(-1, ind_num, -1)) # batch, indices
    output = kernel_val * self.values[lookup_indexes].view(batch_size, -1)
    kernel_sum = kernel_val.sum(1, keepdim=True)
    output = (output / kernel_sum).sum(1)
    return output



  def update_params(self, with_rebuild=True):
    """
    Update self.keys and self.values via backprop
    Use self.indexes_to_be_updated to update self.key_cache accordingly and rebuild the index of self.kdtree
    """
    #for index in self.indexes_to_be_updated:
    #  del self.key_cache[tuple(self.keys[index].data.cpu().numpy())]
    self.optimizer.step()
    self.optimizer.zero_grad()
    #for index in self.indexes_to_be_updated:
    #  self.key_cache[tuple(self.keys[index].data.cpu().numpy())] = 0
    if with_rebuild:
        self.indexes_to_be_updated = set()
        self.kdtree.build_index(self.keys.data.cpu().numpy())
        self.stale_index = False
