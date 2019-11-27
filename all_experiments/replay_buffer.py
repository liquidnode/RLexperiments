from torch.multiprocessing import Process, Queue
import torch.multiprocessing
import zarr
import numpy as np
from shared_memory2 import SharedMemory

torch.multiprocessing.set_sharing_strategy('file_system')

class ReplayBuffer():
    def __init__(self, args, data_description, input_queue, blocking=True):
        num_workers = args.num_replay_workers
        if not blocking:
            num_workers = 1
            #TODO demultiplexer einbauen
        #every worker has own queue!! for prio. replay updates
        self.input_queue = input_queue
        self.batch_queues = [Queue(maxsize=10) for i in range(num_workers)]

        params = [(args, i, data_description, input_queue, self.batch_queues[i], blocking) 
                  for i in range(num_workers)]

        #start processes
        self.proc_pool = [Process(target=ReplayBuffer.worker, args=params[i]) for i in range(num_workers)]
        for p in self.proc_pool:
            p.start()



    @staticmethod
    def worker(args, pid, data_description, input_queue, batch_queue, blocking):
        try:
            num_workers = args.num_replay_workers
            sample_efficiency = args.sample_efficiency
            chunk_size = args.chunk_size
            batch_size = args.er
            trainstart = args.trainstart

            buffer_size = args.erpoolsize // num_workers
            buffers = {}

            for name in data_description:
                desc = data_description[name]
                pre_size = [buffer_size]
                pre_chunk = [chunk_size]
                if 'has_next' in desc:
                    pre_size = [buffer_size, 2]
                    pre_chunk = [chunk_size, 2]
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
                    #if not blocking and sample_efficiency == 1:
                    #    print("data rec",num_samples, batch_queue.qsize())
                    l = None
                    for k in data_description:
                        data_sh = SharedMemory(data_shs[k])
                        #write to buffer
                        data_np = data_sh.numpy()
                        start = write_pos
                        end = write_pos + data_np.shape[0]
                        data_sh_next = None
                        if 'has_next' in data_description[k]:
                            k_next = 'next_'+k
                            data_sh_next = SharedMemory(data_shs[k_next])
                            data_np_next = data_sh_next.numpy()
                            data_np = np.stack([data_np, data_np_next], axis=1)
                        if end > buffer_size:
                            buffers[k][start:] = data_np[:(buffer_size-start)]
                            buffers[k][:(end-buffer_size)] = data_np[(buffer_size-start):]
                            #buffers[k][(start-buffer_size):(end-buffer_size)] = data_np
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
                if (num_samples >= (trainstart // num_workers) or not blocking) and num_samples >= batch_size*2:
                    if not blocking:
                        if batch_queue.qsize() > 2:
                            continue
                    for j in range(sample_efficiency):
                        #sample indices
                        idxs = np.random.randint(0, num_samples, size=(batch_size,))
                        #TODO remove duplicates (very unlikely when buffer_size is big)

                        batch = {}
                        for k in data_description:
                            if not data_description[k]['compression']:
                                if 'has_next' in data_description[k]:
                                    batch[k] = SharedMemory(buffers[k][idxs,0])
                                    batch[k] = batch[k].shared_memory()
                                    batch['next_'+k] = SharedMemory(buffers[k][idxs,1])
                                    batch['next_'+k] = batch['next_'+k].shared_memory()
                                else:
                                    batch[k] = SharedMemory(buffers[k][idxs])
                                    batch[k] = batch[k].shared_memory()
                            else:
                                if 'has_next' in data_description[k]:
                                    both_data = buffers[k].oindex[idxs,:]
                                    batch[k] = SharedMemory(both_data[:,0])
                                    batch[k] = batch[k].shared_memory()
                                    batch['next_'+k] = SharedMemory(both_data[:,1])
                                    batch['next_'+k] = batch['next_'+k].shared_memory()
                                else:
                                    batch[k] = SharedMemory(buffers[k].oindex[idxs,:])
                                    batch[k] = batch[k].shared_memory()
                        #send batch
                        if not blocking:
                            try:
                                batch_queue.put_nowait(batch)
                            except:
                                pass
                        else:
                            batch_queue.put(batch)

        except:
            print("FATAL error in ReplayBuffer ", pid)
            traceback.print_exc()

