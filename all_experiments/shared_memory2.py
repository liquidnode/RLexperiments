import torch
import numpy as np

class SharedMemory():
    def __init__(self, A):
        if torch.is_tensor(A):
            self.rA = A
        else:
            self.rA = torch.from_numpy(np.asarray(A))

        self.rA = self.rA.share_memory_()

    def shared_memory(self):
        return self.rA

    def numpy(self):
        return self.rA.numpy()

    def delete(self):
        del self.rA