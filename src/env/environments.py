import random
import time

import gym
import numpy as np
from typing import Tuple, List, Dict, Callable, Optional

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.vec_env.base_vec_env import VecEnvStepReturn, VecEnvObs
from stable_baselines3.common.vec_env.subproc_vec_env import _flatten_obs

from src.env.logger import InfoCollector
from src.env.rfunc import RewardFunc
from src.smb.asyncsimlt import AsycSimltPool
from src.smb.level import lvlhcat
from src.gan.gans import SAGenerator
from src.gan.gankits import *
from src.smb.proxy import MarioProxy, MarioJavaAgents
from src.utils.datastruct import RingQueue


def get_padded_obs(vecs, histlen, add_batch_dim=False):
    if len(vecs) < histlen:
        lack = histlen - len(vecs)
        pad = [np.zeros([nz], np.float32) for _ in range(lack)]
        obs = np.concatenate(pad + vecs)
    else:
        obs = np.concatenate(vecs)
    if add_batch_dim:
        obs = np.reshape(obs, [1, -1])
    return obs


class SingleProcessOLGenEnv(gym.Env):
    def __init__(self, rfunc, decoder: SAGenerator, eplen: int=50, device='cuda:0'):
        self.rfunc = rfunc
        self.initvec_set = np.load(getpath('smb/init_latvecs.npy'))
        self.decoder = decoder
        self.decoder.to(device)
        self.hist_len = self.rfunc.get_n()
        self.eplen = eplen
        self.device = device
        self.action_space = gym.spaces.Box(-1, 1, (nz,))
        self.observation_space = gym.spaces.Box(-1, 1, (self.hist_len * nz,))
        # self.obs_queue = RingQueue(self.hist_len)
        self.lat_vecs = []
        self.simulator = MarioProxy()
        pass

    def step(self, action):
        self.lat_vecs.append(action)
        done = len(self.lat_vecs) == (self.eplen + 1)
        info = {}
        if done:
            rewss = self.__evalute()
            info['rewss'] = rewss
            rewsums = [sum(items) for items in zip(*rewss.values())]
            info['transitions'] = self.__process_traj(rewsums[-self.eplen:])
            self.reset()
        return self.getobs(), 0, done, info

    def __evalute(self):
        z = torch.tensor(np.stack(self.lat_vecs).reshape([-1, nz, 1, 1]), device=self.device, dtype=torch.float)
        # print(z.shape)
        segs = process_onehot(self.decoder(z))
        lvl = lvlhcat(segs)
        simlt_res = MarioProxy.get_seg_infos(self.simulator.simulate_complete(lvl))
        rewardss = self.rfunc.get_rewards(segs=segs, simlt_res=simlt_res)
        return rewardss

    def __process_traj(self, rewards):
        obs = []
        for i in range(1, len(self.lat_vecs) + 1):
            ob = get_padded_obs(self.lat_vecs[max(0, i - self.hist_len): i], self.hist_len)
            obs.append(ob)
        traj = [(obs[i], self.lat_vecs[i+1], rewards[i], obs[i+1]) for i in range(len(self.lat_vecs) - 1)]
        return traj

    def reset(self):
        self.lat_vecs.clear()
        z0 = self.initvec_set[random.randrange(0, len(self.initvec_set))]
        self.lat_vecs.append(z0)
        return self.getobs()

    def getobs(self):
        s = max(0, len(self.lat_vecs) - self.hist_len)
        return get_padded_obs(self.lat_vecs[s:], self.hist_len, True)

    def render(self, mode="human"):
        pass

    def generate_levels(self, agent, n=1, max_parallel=None):
        if max_parallel is None:
            max_parallel = min(n, 512)
        levels = []
        latvecs = []
        obs_queues = [RingQueue(self.hist_len) for _ in range(max_parallel)]
        while len(levels) < n:
            veclists = [[] for _ in range(min(max_parallel, n - len(levels)))]
            for queue, veclist in zip(obs_queues, veclists):
                queue.clear()
                init_latvec = self.initvec_set[random.randrange(0, len(self.initvec_set))]
                queue.push(init_latvec)
                veclist.append(init_latvec)
            for _ in range(self.eplen):
                obs = np.stack([get_padded_obs(queue.to_list(), self.hist_len) for queue in obs_queues])
                actions = agent.make_decision(obs)
                for queue, veclist, action in zip(obs_queues, veclists, actions):
                    queue.push(action)
                    veclist.append(action)
            for veclist in veclists:
                latvecs.append(np.stack(veclist))
                z = torch.tensor(latvecs[-1], device=self.device).view(-1, nz, 1, 1)
                lvl = lvlhcat(process_onehot(self.decoder(z)))
                levels.append(lvl)
        return levels, latvecs


class AsyncOlGenEnv:
    def __init__(self, histlen, decoder: SAGenerator, eval_pool: AsycSimltPool, eplen: int=50, device='cuda:0'):
        self.initvec_set = np.load(getpath('smb/init_latvecs.npy'))
        self.decoder = decoder
        self.decoder.to(device)
        self.device = device
        self.eval_pool = eval_pool
        self.eplen = eplen
        self.tid = 0
        self.histlen = histlen

        self.cur_vectraj = []
        self.buffer = {}

    def reset(self):
        if len(self.cur_vectraj) > 0:
            self.buffer[self.tid] = self.cur_vectraj
            self.cur_vectraj = []
            self.tid += 1
        z0 = self.initvec_set[random.randrange(0, len(self.initvec_set))]
        self.cur_vectraj.append(z0)
        return self.getobs()

    def step(self, action):
        self.cur_vectraj.append(action)
        done = len(self.cur_vectraj) == (self.eplen + 1)
        if done:
            self.__submit_eval_task()
            self.reset()
        return self.getobs(), done

    def getobs(self):
        s = max(0, len(self.cur_vectraj) - self.histlen)
        return get_padded_obs(self.cur_vectraj[s:], self.histlen, True)

    def __submit_eval_task(self):
        z = torch.tensor(np.stack(self.cur_vectraj).reshape([-1, nz, 1, 1]), device=self.device)
        segs = process_onehot(self.decoder(torch.clamp(z, -1, 1)))
        lvl = lvlhcat(segs)
        args = (self.tid, str(lvl))
        self.eval_pool.put('evaluate', args)

    def refresh(self):
        if self.eval_pool is not None:
            self.eval_pool.refresh()

    def rollout(self, close=False, wait=False) -> Tuple[List[Tuple], List[Dict[str, List]]]:
        transitions, rewss = [], []
        if close:
            eval_res = self.eval_pool.close()
        else:
            eval_res = self.eval_pool.get(wait)
        for tid, rewards in eval_res:
            rewss.append(rewards)
            rewsums = [sum(items) for items in zip(*rewards.values())]
            vectraj = self.buffer.pop(tid)
            transitions += self.__process_traj(vectraj, rewsums[-self.eplen:])
        return transitions, rewss

    def __process_traj(self, vectraj, rewards):
        obs = []
        for i in range(1, len(vectraj) + 1):
            ob = get_padded_obs(vectraj[max(0, i - self.histlen): i], self.histlen)
            obs.append(ob)
        traj = [(obs[i], vectraj[i+1], rewards[i], obs[i+1]) for i in range(len(vectraj) - 1)]
        return traj

    def close(self):
        res = self.rollout(True)
        self.eval_pool = None
        return res

    def generate_levels(self, agent, n=1, max_parallel=None):
        if max_parallel is None:
            max_parallel = min(n, 512)
        levels = []
        latvecs = []
        obs_queues = [RingQueue(self.histlen) for _ in range(max_parallel)]
        while len(levels) < n:
            veclists = [[] for _ in range(min(max_parallel, n - len(levels)))]
            for queue, veclist in zip(obs_queues, veclists):
                queue.clear()
                init_latvec = self.initvec_set[random.randrange(0, len(self.initvec_set))]
                queue.push(init_latvec)
                veclist.append(init_latvec)
            for _ in range(self.eplen):
                obs = np.stack([get_padded_obs(queue.to_list(), self.histlen) for queue in obs_queues])
                actions = agent.make_decision(obs)
                for queue, veclist, action in zip(obs_queues, veclists, actions):
                    queue.push(action)
                    veclist.append(action)
            for veclist in veclists:
                latvecs.append(np.stack(veclist))
                z = torch.tensor(latvecs[-1], device=self.device).view(-1, nz, 1, 1)
                lvl = lvlhcat(process_onehot(self.decoder(z)))
                levels.append(lvl)
        return levels, latvecs


######### Adopt from https://github.com/SUSTechGameAI/MFEDRL #########
class SyncOLGenWorkerEnv(gym.Env):
    def __init__(self, rfunc=None, hist_len=5, eplen=25, return_lvl=False, init_one=False, play_style='Runner'):
        self.rfunc = RewardFunc() if rfunc is None else rfunc
        self.mario_proxy = MarioProxy() if self.rfunc.require_simlt else None
        self.action_space = gym.spaces.Box(-1, 1, (nz,))
        self.hist_len = hist_len
        self.observation_space = gym.spaces.Box(-1, 1, (hist_len * nz,))
        self.segs = []
        self.latvec_archive = RingQueue(hist_len)
        self.eplen = eplen
        self.counter = 0
        # self.repairer = DivideConquerRepairer()
        self.init_one = init_one
        self.backup_latvecs = None
        self.backup_strsegs = None
        self.return_lvl = return_lvl
        self.jagent = MarioJavaAgents.__getitem__(play_style)
        self.simlt_k = 80 if play_style == 'Runner' else 320

    def receive(self, **kwargs):
        for key in kwargs.keys():
            setattr(self, key, kwargs[key])

    def step(self, data):
        action, strseg = data
        seg = MarioLevel(strseg)
        self.latvec_archive.push(action)

        self.counter += 1
        self.segs.append(seg)
        done = self.counter >= self.eplen
        if done:
            full_level = lvlhcat(self.segs)
            # full_level = self.repairer.repair(full_level)
            w = MarioLevel.seg_width
            segs = [full_level[:, s: s + w] for s in range(0, full_level.w, w)]
            if self.mario_proxy:
                raw_simlt_res = self.mario_proxy.simulate_complete(lvlhcat(segs), self.jagent, self.simlt_k)
                simlt_res = MarioProxy.get_seg_infos(raw_simlt_res)
            else:
                simlt_res = None
            rewards = self.rfunc.get_rewards(segs=segs, simlt_res=simlt_res)
            info = {}
            total_score = 0
            if self.return_lvl:
                info['LevelStr'] = str(full_level)
            for key in rewards:
                info[f'{key}_reward_list'] = rewards[key][-self.eplen:]
                info[f'{key}'] = sum(rewards[key][-self.eplen:])
                total_score += info[f'{key}']
            info['TotalScore'] = total_score
            info['EpLength'] = self.counter
        else:
            info = {}
        return self.__get_obs(), 0, done, info

    def reset(self):
        self.segs.clear()
        self.latvec_archive.clear()
        for latvec, strseg in zip(self.backup_latvecs, self.backup_strsegs):
            self.latvec_archive.push(latvec)
            self.segs.append(MarioLevel(strseg))

        self.backup_latvecs, self.backup_strsegs = None, None
        self.counter = 0
        return self.__get_obs()

    def __get_obs(self):
        lack = self.hist_len - len(self.latvec_archive)
        pad = [np.zeros([nz], np.float32) for _ in range(lack)]
        return np.concatenate([*pad, *self.latvec_archive.to_list()])

    def render(self, mode='human'):
        pass


class VecOLGenEnv(SubprocVecEnv):
    def __init__(
        self, env_fns: List[Callable[[], gym.Env]], start_method: Optional[str] = None, hist_len=5, eplen=50,
        init_one=True, log_path=None, log_itv=-1, log_targets=None, device='cuda:0'
    ):
        super(VecOLGenEnv, self).__init__(env_fns, start_method)
        self.decoder = get_decoder(device=device)

        if log_path:
            self.logger = InfoCollector(log_path, log_itv, log_targets)
        else:
            self.logger = None
        self.hist_len = hist_len
        self.total_steps = 0
        self.start_time = time.time()
        self.eplen = eplen
        self.device = device
        self.init_one = init_one
        self.latvec_set = np.load(getpath('smb/init_latvecs.npy'))

    def step_async(self, actions: np.ndarray) -> None:
        with torch.no_grad():
            z = torch.tensor(actions.astype(np.float32), device=self.device).view(-1, nz, 1, 1)
            segs = process_onehot(self.decoder(z))
        for remote, action, seg in zip(self.remotes, actions, segs):
            remote.send(("step", (action, str(seg))))
        self.waiting = True

    def step_wait(self) -> VecEnvStepReturn:
        self.total_steps += self.num_envs
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, rews, dones, infos = zip(*results)

        envs_to_send = [i for i in range(self.num_envs) if dones[i]]
        self.send_reset_data(envs_to_send)

        if self.logger is not None:
            for i in range(self.num_envs):
                if infos[i]:
                    infos[i]['TotalSteps'] = self.total_steps
                    infos[i]['TimePassed'] = time.time() - self.start_time
            self.logger.on_step(dones, infos)
        return _flatten_obs(obs, self.observation_space), np.stack(rews), np.stack(dones), infos

    def reset(self) -> VecEnvObs:
        self.send_reset_data()
        for remote in self.remotes:
            remote.send(("reset", None))
        obs = [remote.recv() for remote in self.remotes]
        self.send_reset_data()
        return _flatten_obs(obs, self.observation_space)

    def send_reset_data(self, env_ids=None):
        if env_ids is None:
            env_ids = [*range(self.num_envs)]
        target_remotes = self._get_target_remotes(env_ids)

        n_inits = 1 if self.init_one else self.hist_len
        # latvecs = [sample_latvec(n_inits, tensor=False) for _ in range(len(env_ids))]

        latvecs = [self.latvec_set[random.sample(range(len(self.latvec_set)), n_inits)] for _ in range(len(env_ids))]
        with torch.no_grad():
            segss = [[] for _ in range(len(env_ids))]
            for i in range(len(env_ids)):
                z = torch.tensor(latvecs[i]).view(-1, nz, 1, 1).to(self.device)
                # print(self.decoder(z).shape)
                segss[i] = [process_onehot(self.decoder(z))] if self.init_one else process_onehot(self.decoder(z))
        for remote, latvec, segs in zip(target_remotes, latvecs, segss):
            kwargs = {'backup_latvecs': latvec, 'backup_strsegs': [str(seg) for seg in segs]}
            remote.send(("env_method", ('receive', [], kwargs)))
        for remote in target_remotes:
            remote.recv()

    def close(self) -> None:
        super().close()
        if self.logger is not None:
            self.logger.close()


def make_vec_offrew_env(
        num_envs, rfunc=None, log_path=None, eplen=25, log_itv=-1, hist_len=5, init_one=True,
        play_style='Runner', device='cuda:0', log_targets=None, return_lvl=False
    ):
    return make_vec_env(
        SyncOLGenWorkerEnv, n_envs=num_envs, vec_env_cls=VecOLGenEnv,
        vec_env_kwargs={
            'log_path': log_path,
            'log_itv': log_itv,
            'log_targets': log_targets,
            'device': device,
            'eplen': eplen,
            'hist_len': hist_len,
            'init_one': init_one
        },
        env_kwargs={
            'rfunc': rfunc,
            'eplen': eplen,
            'return_lvl': return_lvl,
            'play_style': play_style,
            'hist_len': hist_len,
            'init_one': init_one
        }
    )
