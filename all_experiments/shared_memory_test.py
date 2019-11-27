import torch
import numpy as np
from torch.multiprocessing import Process, Queue, Pipe
import datetime
import torch.multiprocessing
        
torch.multiprocessing.set_sharing_strategy('file_system')


def main():
    if False:
        #import numpy as np
        #from shared_memory import SharedMemory
        #from multiprocessing import Process

        #create shared_memory
        A = np.array([1, 2, 3, 5, 5])

        bytesA = A.tobytes()
        shA = SharedMemory(create=True, size=len(bytesA))
        shA.buf = bytesA

        def worker(name, dtype):
            shB = SharedMemory(name)
            B = np.frombuffer(shB.buf, dtype=dtype)
            B[1] = 0
            shB.close()

        p = Process(target=worker, args=(shA.name, A.dtype))
        p.start()
        p.join()
    

        print(A)
        print(np.frombuffer(shA.buf, A.dtype))
    else:

        A = np.array([1, 2, 3, 5, 5])
        rA = torch.from_numpy(A).share_memory_()

        p = Process(target=worker, args=(rA))
        p.start()
        p.join()

        print(A)
        print(rA.numpy())
        
def worker(rB):
    B = rB.numpy()
    B[1] = 0
    del rB

def producer(use_shared, queue):
    if not use_shared:
        for i in range(0, int(1e3)):
            C = np.zeros((100, 64, 64))
            queue.put(C)
            #queue[1].send(C)
    else:
        for i in range(0, int(1e3)):
            C = torch.zeros(100, 64, 64).share_memory_()
            queue.put(C)
            #queue[1].send(C)

def consumer(use_shared, queue):
    for i in range(0, int(1e3)):
        D = queue.get()
        #D = queue[0].recv()
        del D

if __name__ == '__main__':
    A = np.array([1, 2, 3, 5, 5])
    rA = torch.from_numpy(A).share_memory_()

    p = Process(target=worker, args=(rA,))
    p.start()
    p.join()

    print(A)
    print(rA.numpy())

    nrA = rA.numpy()
    nrA[0] = -1

    print(rA.numpy()) #.numpy() does not copy!


    #Queue is faster than pipe for some reason.
    #performance check
    queue = Queue()#Pipe(False)#
    #no shared memory
    use_shared = False

    pp = Process(target=producer, args=(use_shared, queue))
    pc = Process(target=consumer, args=(use_shared, queue))

    
    s = datetime.datetime.now()
    pp.start()
    pc.start()
    pp.join()
    pc.join()
    ns = datetime.datetime.now()
    d = (ns - s).total_seconds()

    print("Without shared memory", d)
    
    queue = Queue()#Pipe(False)#
    #shared memory
    use_shared = True

    pp = Process(target=producer, args=(use_shared, queue))
    pc = Process(target=consumer, args=(use_shared, queue))

    
    s = datetime.datetime.now()
    pp.start()
    pc.start()
    pp.join()
    pc.join()
    ns = datetime.datetime.now()
    d2 = (ns - s).total_seconds()
    print("With shared memory", d2)