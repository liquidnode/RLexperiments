from torch.multiprocessing import Process, Queue
import torch.multiprocessing
import zarr
import numpy as np
from shared_memory2 import SharedMemory
import time
from prio_utils import SumTree
import threading
import random

torch.multiprocessing.set_sharing_strategy('file_system')

class VirtualReplayBuffer():
    def __init__(self, args, data_description, input_queue, time_deltas, max_range=0, blocking=True, pov_time_deltas=None):
        num_workers = args.num_replay_workers
        if not blocking:
            num_workers = 1
            #TODO demultiplexer einbauen
        #every worker has own input and output queue!! output for prio. replay updates
        assert isinstance(input_queue, list)
        self.input_queue = input_queue
        self.batch_queues = [Queue(maxsize=10) for i in range(num_workers)]

        params = [(args, i, data_description, input_queue[i], self.batch_queues[i], time_deltas, max_range, blocking, pov_time_deltas) 
                  for i in range(num_workers)]

        #start processes
        self.proc_pool = [Process(target=VirtualReplayBuffer.worker, args=params[i]) for i in range(num_workers)]
        for p in self.proc_pool:
            p.start()



    @staticmethod
    def worker(args, pid, data_description, input_queue, batch_queue, time_deltas, max_range, blocking, pov_time_deltas):
        try:
            num_workers = args.num_replay_workers
            sample_efficiency = args.sample_efficiency
            chunk_size = args.chunk_size
            batch_size = args.er
            trainstart = args.trainstart
            max_skip = max(max(time_deltas),max_range)
            avoid_done = True
            time_zero = 0
            for i in range(len(time_deltas)):
                if time_deltas[len(time_deltas) - i - 1] == 0:
                    time_zero = len(time_deltas) - i - 1
                    break

            buffer_size = args.erpoolsize // num_workers
            buffers = {}

            for name in data_description:
                desc = data_description[name]
                pre_size = [buffer_size]
                pre_chunk = [chunk_size]
                if desc['compression']:
                    buffers[name] = zarr.zeros(pre_size + desc['shape'], chunks=pre_chunk + desc['shape'], dtype=desc['dtype'])
                else:
                    buffers[name] = np.zeros(pre_size + desc['shape'], dtype=desc['dtype'])
                    
            num_samples = 0
            write_pos = 0
            while True:
                #load new input
                data_shs = None
                if not blocking:
                    try:
                        data_shs = input_queue.get_nowait()
                    except:
                        pass
                else:
                    data_shs = input_queue.get()
                if data_shs is not None:
                    l = None
                    for k in data_description:
                        data_sh = SharedMemory(data_shs[k])
                        #write to buffer
                        data_np = data_sh.numpy()
                        start = write_pos
                        end = write_pos + data_np.shape[0]
                        data_sh_next = None
                        if end > buffer_size:
                            buffers[k][start:] = data_np[:(buffer_size-start)]
                            buffers[k][:(end-buffer_size)] = data_np[(buffer_size-start):]
                        else:
                            buffers[k][start:end] = data_np
                        l = data_np.shape[0]
                        data_sh.delete()
                        del data_sh
                        if data_sh_next is not None:
                            data_sh_next.delete()
                            del data_sh_next
                    write_pos = (write_pos + l) % buffer_size
                    num_samples = min(num_samples + l, buffer_size)

                #send batches
                if (num_samples >= (trainstart // num_workers) or not blocking) and num_samples >= batch_size*2 and num_samples > max_skip:
                    if not blocking:
                        if batch_queue.qsize() > 2:
                            continue
                    j = 0
                    while j < sample_efficiency:
                        #sample indices
                        idxs = (np.random.randint(0, num_samples-max_skip, size=(batch_size,))+write_pos-num_samples)%buffer_size
                        #idx+max_skip should not skip over write_pos
                        assert np.all((idxs - write_pos) % buffer_size < (idxs - write_pos + max_skip) % buffer_size)
                        #TODO remove duplicates (very unlikely when buffer_size is big)

                        batch = {}
                        for k in data_description:
                            n_time_deltas = time_deltas.copy()
                            r_offset = None
                            if k == 'pov' and pov_time_deltas is not None:
                                n_time_deltas = pov_time_deltas
                                if args.needs_random_future:
                                    r_offset = np.random.randint(1, pov_time_deltas[-1]+1, size=(batch_size,))
                                    batch['r_offset'] = SharedMemory(r_offset)
                                    batch['r_offset'] = batch['r_offset'].shared_memory()
                            elif 'dontsendpast' in data_description[k] and data_description[k]['dontsendpast']:
                                n_time_deltas = [d for d in time_deltas if d >= 0]
                            if not data_description[k]['compression']:
                                if k == 'reward' or k == 'done':
                                    trange = range(0, max(max(n_time_deltas),max_range))
                                    trange = n_time_deltas[:time_zero] + list(trange)
                                    ridxs = np.stack([idxs+s for s in trange], axis=1).reshape([-1])%buffer_size
                                    rbatch = buffers[k][ridxs]
                                    rbatch = rbatch.reshape([batch_size, len(trange)]+list(rbatch.shape[1:]))
                                    if k == 'done' and np.any(rbatch[:,time_zero:(time_zero+max(n_time_deltas)+1)]) and avoid_done:
                                        break
                                else:
                                    ridxs = np.stack([idxs+s for s in n_time_deltas], axis=1).reshape([-1])%buffer_size
                                    rbatch = buffers[k][ridxs]
                                    rbatch = rbatch.reshape([batch_size, len(n_time_deltas)]+list(rbatch.shape[1:]))
                                batch[k] = SharedMemory(rbatch)
                                batch[k] = batch[k].shared_memory()
                            else:
                                if r_offset is None:
                                    ridxs = np.stack([idxs+s for s in n_time_deltas], axis=1).reshape([-1])%buffer_size
                                else:
                                    ridxs = np.stack([idxs+s for s in n_time_deltas[:-1]]+[idxs+r_offset], axis=1).reshape([-1])%buffer_size
                                rbatch = buffers[k].oindex[ridxs,:]
                                rbatch = rbatch.reshape([batch_size, len(n_time_deltas)]+list(rbatch.shape[1:]))
                                batch[k] = SharedMemory(rbatch)
                                batch[k] = batch[k].shared_memory()
                        if 'done' in batch:
                            #send batch
                            if not blocking:
                                try:
                                    batch_queue.put_nowait(batch)
                                except:
                                    pass
                            else:
                                batch_queue.put(batch)
                            j += 1

        except:
            print("FATAL error in ReplayBuffer ", pid)
            traceback.print_exc()






class VirtualReplayBufferPER():
    def __init__(self, args, data_description, input_queue, prio_queues, time_deltas, max_range=0, blocking=True, pov_time_deltas=None):
        num_workers = args.num_replay_workers
        if not blocking:
            num_workers = 1
            #TODO demultiplexer einbauen
        #every worker has own input and output queue!! output for prio. replay updates
        assert isinstance(input_queue, list)
        self.input_queue = input_queue
        self.batch_queues = [Queue(maxsize=10) for i in range(num_workers)]

        params = [(args, i, data_description, input_queue[i], self.batch_queues[i], prio_queues[i], time_deltas, max_range, blocking, pov_time_deltas) 
                  for i in range(num_workers)]

        #start processes
        self.proc_pool = [Process(target=VirtualReplayBufferPER.worker, args=params[i]) for i in range(num_workers)]
        for p in self.proc_pool:
            p.start()



    @staticmethod
    def worker(args, pid, data_description, input_queue, batch_queue, prio_queue, time_deltas, max_range, blocking, pov_time_deltas):
        try:
            num_workers = args.num_replay_workers
            sample_efficiency = args.sample_efficiency
            chunk_size = args.chunk_size
            batch_size = args.er
            trainstart = args.trainstart
            if trainstart < int(1e5) and pid != 0:
                trainstart = 0
            max_skip = max(max(time_deltas),max_range)
            avoid_done = True
            time_zero = 0
            for i in range(len(time_deltas)):
                if time_deltas[len(time_deltas) - i - 1] == 0:
                    time_zero = len(time_deltas) - i - 1
                    break

            buffer_size = args.erpoolsize // num_workers

            eps = 0.01
            alpha = 0.7
            beta = 0.5
            beta_increment_per_sampling = 3e-7
            global max_error
            max_error = eps ** alpha
            global sum_tree
            sum_tree = SumTree(args.erpoolsize // num_workers)
            sum_tree_lock = threading.Lock()

            #make prio update thread
            def prio_update(p_queue):
                global max_error
                global sum_tree
                global beta
                while True:
                    pr_up = p_queue.get()
                    for a in pr_up:
                        pr_up[a] = pr_up[a].data.cpu().numpy()
                    sum_tree_lock.acquire()
                    #print(pr_up['error'].shape)
                    #print(pr_up['tidxs'].shape)
                    #print(max_error)
                    for i in range(len(pr_up['tidxs'])):
                        tidx = pr_up['tidxs'][i]
                        error = pr_up['error'][i]
                        #prio = (np.abs(error) + eps) ** alpha
                        max_error = max(max_error, error)
                        sum_tree.update(tidx, error)
                    sum_tree_lock.release()
            prio_th = threading.Thread(target=prio_update, args=(prio_queue,))
            prio_th.start()

            buffers = {}

            for name in data_description:
                desc = data_description[name]
                pre_size = [buffer_size]
                pre_chunk = [chunk_size]
                if desc['compression']:
                    buffers[name] = zarr.zeros(pre_size + desc['shape'], chunks=pre_chunk + desc['shape'], dtype=desc['dtype'])
                else:
                    buffers[name] = np.zeros(pre_size + desc['shape'], dtype=desc['dtype'])
                    
            num_samples = 0
            write_pos = 0
            while True:
                #load new input
                data_shs = None
                if not blocking:
                    try:
                        data_shs = input_queue.get_nowait()
                    except:
                        pass
                else:
                    data_shs = input_queue.get()
                if data_shs is not None:
                    l = None
                    for k in data_description:
                        data_sh = SharedMemory(data_shs[k])
                        #write to buffer
                        data_np = data_sh.numpy()
                        start = write_pos
                        end = write_pos + data_np.shape[0]
                        data_sh_next = None
                        if end > buffer_size:
                            buffers[k][start:] = data_np[:(buffer_size-start)]
                            buffers[k][:(end-buffer_size)] = data_np[(buffer_size-start):]
                        else:
                            buffers[k][start:end] = data_np
                        l = data_np.shape[0]
                        data_sh.delete()
                        del data_sh
                        if data_sh_next is not None:
                            data_sh_next.delete()
                            del data_sh_next
                    #update PER
                    sum_tree_lock.acquire()
                    for jj in range(l):
                        sum_tree.add(max_error)
                    sum_tree_lock.release()

                    write_pos = (write_pos + l) % buffer_size
                    num_samples = min(num_samples + l, buffer_size)

                #send batches
                if (num_samples >= (trainstart // num_workers) or not blocking) and num_samples >= batch_size and num_samples > max_skip:
                    if not blocking:
                        if batch_queue.qsize() > 2:
                            continue
                    j = 0
                    while j < sample_efficiency:
                        #sample indices
                        idxs = []
                        tree_idxs = []
                        priorities = []
                        rpriorities = []
                        sum_tree_lock.acquire()
                        segment = sum_tree.total() / batch_size
                        for i in range(batch_size):
                            a = segment * i
                            b = segment * (i + 1)

                            notFound = True
                            num_attempts = 0
                            while notFound and num_attempts < 10:
                                s = random.uniform(a, b)
                                (tidx, p, idx) = sum_tree.get(s)
                                is_done = np.any(buffers['done'][np.array(range(idx, idx+max(time_deltas)+1))%buffer_size])
                                if not ((idx - write_pos) % buffer_size < (idx - write_pos + max_skip) % buffer_size) or (is_done and avoid_done):
                                    num_attempts += 1
                                    continue
                                priorities.append(p)
                                rpriorities.append(p)
                                idxs.append(idx)
                                tree_idxs.append(tidx)
                                notFound = False
                                break
                            if notFound:
                                #fall back to uniform sampling
                                idx = (np.random.randint(0, num_samples-max_skip)+write_pos-num_samples)%buffer_size
                                tidx = idx + sum_tree.capacity - 1
                                idxs.append(idx)
                                tree_idxs.append(tidx)
                                priorities.append(-123456)
                        sum_tree_lock.release()

                        mean = np.mean(rpriorities)
                        priorities = np.where(np.array(priorities)<-123455,mean,priorities)
                        sampling_probabilities = priorities / sum_tree.total()
                        is_weight = np.power(sum_tree.n_entries * sampling_probabilities, -beta)
                        if np.any(np.isnan(is_weight)):
                            print(sum_tree.n_entries)
                            print(sampling_probabilities.min())
                            print(sampling_probabilities.max())
                            print(beta)
                        is_weight /= is_weight.max()
                        
                        beta = min(1.0, beta + beta_increment_per_sampling * len(idxs))
                        idxs = np.array(idxs)
                        tidxs = np.array(tree_idxs)
                        #idx+max_skip should not skip over write_pos
                        assert np.all((idxs - write_pos) % buffer_size < (idxs - write_pos + max_skip) % buffer_size)
                        #TODO remove duplicates (very unlikely when buffer_size is big)

                        batch = {}
                        for k in data_description:
                            n_time_deltas = time_deltas.copy()
                            r_offset = None
                            if k == 'pov' and pov_time_deltas is not None:
                                n_time_deltas = pov_time_deltas
                                if args.needs_random_future:
                                    r_offset = np.random.randint(1, pov_time_deltas[-1]+1, size=(batch_size,))
                                    batch['r_offset'] = SharedMemory(r_offset)
                                    batch['r_offset'] = batch['r_offset'].shared_memory()
                            elif 'dontsendpast' in data_description[k] and data_description[k]['dontsendpast']:
                                n_time_deltas = [d for d in time_deltas if d >= 0]
                            if not data_description[k]['compression']:
                                if k == 'reward' or k == 'done':
                                    trange = range(0, max(max(n_time_deltas),max_range))
                                    trange = n_time_deltas[:time_zero] + list(trange)
                                    ridxs = np.stack([idxs+s for s in trange], axis=1).reshape([-1])%buffer_size
                                    rbatch = buffers[k][ridxs]
                                    rbatch = rbatch.reshape([batch_size, len(trange)]+list(rbatch.shape[1:]))
                                    if k == 'done' and np.any(rbatch[:,time_zero:(time_zero+max(n_time_deltas)+1)]) and avoid_done:
                                        break
                                else:
                                    ridxs = np.stack([idxs+s for s in n_time_deltas], axis=1).reshape([-1])%buffer_size
                                    rbatch = buffers[k][ridxs]
                                    rbatch = rbatch.reshape([batch_size, len(n_time_deltas)]+list(rbatch.shape[1:]))
                                batch[k] = SharedMemory(rbatch)
                                batch[k] = batch[k].shared_memory()
                            else:
                                if r_offset is None:
                                    ridxs = np.stack([idxs+s for s in n_time_deltas], axis=1).reshape([-1])%buffer_size
                                else:
                                    ridxs = np.stack([idxs+s for s in n_time_deltas[:-1]]+[idxs+r_offset], axis=1).reshape([-1])%buffer_size
                                rbatch = buffers[k].oindex[ridxs,:]
                                rbatch = rbatch.reshape([batch_size, len(n_time_deltas)]+list(rbatch.shape[1:]))
                                batch[k] = SharedMemory(rbatch)
                                batch[k] = batch[k].shared_memory()
                        if 'done' in batch:
                            #add PER data
                            batch['is_weight'] = SharedMemory(is_weight).shared_memory()
                            batch['idxs'] = SharedMemory(idxs).shared_memory()
                            batch['tidxs'] = SharedMemory(tidxs).shared_memory()

                            #send batch
                            if not blocking:
                                try:
                                    batch_queue.put_nowait(batch)
                                except:
                                    pass
                            else:
                                batch_queue.put(batch)
                            j += 1
        except:
            print("FATAL error in ReplayBuffer ", pid)
            traceback.print_exc()
