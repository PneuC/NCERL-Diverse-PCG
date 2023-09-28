import time
import importlib
import multiprocessing as mp
from src.smb.level import *
from src.smb.proxy import MarioProxy
from queue import Queue, Full as FullExpection


def _simlt_worker(remote, parent_remote, rfunc, resource):
    rfunc = importlib.import_module('src.env.rfuncs').__getattribute__(rfunc)()
    # W = MarioLevel.seg_width
    simulator = MarioProxy()
    parent_remote.close()
    refs = [MarioLevel(lvl) for lvl in resource.get('refs', [])]
    simlt_k = 150 if resource.get('test', False) else 100
    while True:
        try:
            cmd, data = remote.recv()
            if cmd == 'evaluate':
                tid, strlvl = data
                lvl = MarioLevel(strlvl)
                segs = lvl.to_segs()
                simlt_res = MarioProxy.get_seg_infos(simulator.simulate_complete(lvl, segTimeK=simlt_k))
                rewards = rfunc.get_rewards(segs=segs, simlt_res=simlt_res)
                remote.send((tid, rewards))
                pass
            elif cmd == 'close':
                remote.close()
                break
            elif cmd == 'check_playable':
                strlvl, item = data
                lvl = MarioLevel(strlvl)
                standable = False
                for i in range(lvl.h):
                    if lvl[i,0] in MarioLevel.solidset:
                        standable = True
                        break
                if standable:
                    simlt_res = simulator.simulate_game(lvl)
                    playable = simlt_res['status'] == 'WIN'
                    remote.send((playable, item))
                else:
                    remote.send((False, item))
            elif cmd == 'mnd_item':
                # strlvl = data
                p = MarioLevel(data)
                min_hm, min_dtw = float('inf'), float('inf')
                for q in refs:
                    vhm = hamming_dis(p, q)
                    vdtw = lvl_dtw(p, q)
                    if vhm > 0:
                        min_hm = min(min_hm, vhm)
                    if vdtw > 0:
                        min_dtw = min(min_dtw, vdtw)
                remote.send((min_hm, min_dtw))
            elif cmd == 'mpd':
                # strpairs = data
                hms, dtws = [], []
                for strlvl1, strlvl2 in data:
                    lvl1, lvl2 = MarioLevel(strlvl1), MarioLevel(strlvl2)
                    hms.append(hamming_dis(lvl1, lvl2))
                    # dtws.append(lvl_dtw(lvl1, lvl2))
                remote.send((hms, None))
                # remote.send((hms, dtws))
            else:
                raise KeyError(f'Unknown command for simulation worker: {cmd}')
        except EOFError:
            break
    pass


class AsycSimltPool:
    def __init__(self, poolsize, queuesize=None, rfunc_name='default', verbose=True, **rsrc):
        self.np, self.nq = poolsize, poolsize if queuesize is None else queuesize
        self.waiting_queue = Queue(self.nq)
        self.ready = [True] * poolsize
        resource = {'rfunc': 'default'}
        resource.update(rsrc)
        self.__init_remotes(rfunc_name, resource)
        self.res_buffer = []
        self.histlen = AsycSimltPool.get_histlen(rfunc_name)
        self.verbose = verbose

    @staticmethod
    def get_histlen(rfunc_name):
        rfunc = importlib.import_module('src.env.rfuncs').__getattribute__(rfunc_name)()
        return rfunc.get_n()
        pass

    def put(self, cmd, args):
        """
            Put a new evaluation task into the pool. If the pool and waiting queue is full,
            wait until a process is free
        """
        putted = False
        for i, remote in enumerate(self.remotes):
            if self.ready[i]:
                remote.send((cmd, args))
                self.ready[i] = False
                putted = True
                break
        while not putted:
            try:
                self.waiting_queue.put((cmd, args), timeout=0.01)
                putted = True
            except FullExpection:
                self.refresh()

    def get(self, wait=False):
        if wait:
            self.__wait()
        self.refresh()
        occp, occq = self.get_occupied()
        if self.verbose:
            print(f'Workers: {occp}/{self.np}, Queue: {occq}/{self.nq}, Buffer: {len(self.res_buffer)}')
        res = self.res_buffer
        self.res_buffer = []
        return res

    def get_occupied(self):
        process_occupied = sum(0 if r else 1 for r in self.ready)
        return process_occupied, self.waiting_queue.qsize()

    def refresh(self):
        """ Recive ready results and cache them in buffer, then assign tasks in waiting queue to free workers """
        for i, remote in enumerate(self.remotes):
            if remote.poll():
                self.res_buffer.append(remote.recv())
                self.ready[i] = True
        for i, remote in enumerate(self.remotes):
            if self.waiting_queue.empty():
                break
            if self.ready[i]:
                cmd, args = self.waiting_queue.get()
                remote.send((cmd, args))
                self.ready[i] = False

    def blocking(self):
        self.refresh()
        return self.waiting_queue.full()

    def __init_remotes(self, rfunc, resource):
        forkserver_available = "forkserver" in mp.get_all_start_methods()
        start_method = "forkserver" if forkserver_available else "spawn"
        ctx = mp.get_context(start_method)

        self.remotes, self.work_remotes = zip(*[ctx.Pipe() for _ in range(self.np)])
        self.processes = []
        for work_remote, remote in zip(self.work_remotes, self.remotes):
            args = (work_remote, remote, rfunc, resource)
            # daemon=True: if the main process crashes, we should not cause things to hang
            process = ctx.Process(target=_simlt_worker, args=args, daemon=True)  # pytype:disable=attribute-error
            process.start()
            self.processes.append(process)
            work_remote.close()

    def __wait(self):
        finish = False
        while not finish:
            self.refresh()
            finish = all(r for r in self.ready)
            time.sleep(0.01)

    def close(self):
        # finish = False
        # while not finish:
        #     self.refresh()
        #     finish = all(r for r in self.ready)
        #     time.sleep(0.01)
        # self.__wait()
        res = self.get(True)
        for remote, p in zip(self.remotes, self.processes):
            remote.send(('close', None))
            p.join()
        return res


