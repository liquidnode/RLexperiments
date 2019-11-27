import numpy as np
import lzo
import zarr





A = np.random.uniform(size=(10))

print(len(A.tobytes()))

s = np.stack([np.concatenate([A[(i%10):],A[:(i%10)]],axis=0) for i in range(0, 1000)], axis=0)

z = zarr.array(s, chunks=(1000, 10))

s = s.tobytes()
print(len(s))

compressed = lzo.compress(s)

print(len(compressed))

print(z.info)