import time
import torch
from math import ceil
from src.drl.rep_mem import ReplayMem
from src.utils.filesys import getpath
from src.env.environments import AsyncOlGenEnv, SingleProcessOLGenEnv


class AsyncOffpolicyTrainer:
    def __init__(self, rep_mem:ReplayMem=None, update_per=1, batch=256):
        self.rep_mem = ReplayMem() if rep_mem is None else rep_mem
        self.update_per = update_per
        self.batch = batch
        self.loggers = []

        self.start_time = 0.
        self.steps = 0
        self.num_trans = 0
        self.num_updates = 0
        pass

    def train(self, env:AsyncOlGenEnv, agent, budget, path, check_points=None):
        if check_points is None: check_points = []
        check_points.sort(reverse=True)
        self._reset()
        o = env.reset()
        for logger in self.loggers:
            if logger.__class__.__name__ == 'GenResLogger':
                logger.on_episode(env, agent, 0)
        while self.steps < budget:
            agent.actor.eval()
            a = agent.make_decision(o)
            o, done = env.step(a)
            self.steps += 1
            if done:
                model_credits = ceil(1.25 * env.eplen / self.update_per)
                # agent.actor.train()
                self._update(model_credits, env, agent)
            if len(check_points) and self.steps >= check_points[-1]:
                torch.save(agent.actor.net, getpath(f'{path}/policy{self.steps}.pth'))
                check_points.pop()

        self._update(0, env, agent, close=True)
        torch.save(agent.actor.net, getpath(f'{path}/policy.pth'))

    def _update(self, model_credits, env, agent, close=False):
        transitions, rewss = env.close() if close else env.rollout()
        self.rep_mem.add_transitions(transitions)
        self.num_trans += len(transitions)
        if len(self.rep_mem) > self.batch:
            t = 1
            while self.num_trans >= (self.num_updates + 1) * self.update_per:
                agent.update(*self.rep_mem.sample(self.batch))
                self.num_updates += 1
                if not close and t == model_credits:
                    break
                t += 1
        for logger in self.loggers:
            loginfo = self._pack_loginfo(rewss)
            if logger.__class__.__name__ == 'GenResLogger':
                logger.on_episode(env, agent, self.steps)
            else:
                logger.on_episode(**loginfo, close=close)

    def _pack_loginfo(self, rewss):
        return {
            'steps': self.steps, 'time': time.time() - self.start_time, 'rewss': rewss,
            'trans': self.num_trans, 'updates': self.num_updates
        }

    def _reset(self):
        self.start_time = time.time()
        self.steps = 0
        self.num_trans = 0
        self.num_updates = 0

    def set_loggers(self, *loggers):
        self.loggers = loggers


class SinProcOffpolicyTrainer:
    def __init__(self, rep_mem:ReplayMem=None, update_per=2, batch=256):
        self.rep_mem = ReplayMem() if rep_mem is None else rep_mem
        self.update_per = update_per
        self.batch = batch
        self.loggers = []

        self.start_time = 0.
        self.steps = 0
        self.num_trans = 0
        self.num_updates = 0

    def train(self, env:SingleProcessOLGenEnv, agent, budget, path):
        self.__reset()
        o = env.reset()
        for logger in self.loggers:
            if logger.__class__.__name__ == 'GenResLogger':
                logger.on_episode(env, agent, 0)
        while self.steps < budget:
            agent.actor.eval()
            a = agent.make_decision(o)
            o, _, done, info = env.step(a)
            self.steps += 1
            if done:
                self.__update(env, agent, info)

        self.__update(env, agent, {'transitions': [], 'rewss': []}, True)
        torch.save(agent.actor.net, getpath(f'{path}/policy.pth'))

    def __update(self, env, agent, info, close=False):
        transitions, rewss = info['transitions'], info['rewss']
        # print(rewss)
        self.rep_mem.add_transitions(transitions)
        self.num_trans += len(transitions)
        if len(self.rep_mem) > self.batch:
            t = 1
            while self.num_trans >= (self.num_updates + 1) * self.update_per:
                agent.update(*self.rep_mem.sample(self.batch))
                self.num_updates += 1
                t += 1
        for logger in self.loggers:
            loginfo = self.__pack_loginfo(rewss)
            if logger.__class__.__name__ == 'GenResLogger':
                logger.on_episode(env, agent, self.steps)
            else:
                logger.on_episode(**loginfo, close=close)

    def __pack_loginfo(self, rewss):
        if len(rewss):
            rewss = [rewss]
        return {
            'steps': self.steps, 'time': time.time() - self.start_time, 'rewss': rewss,
            'trans': self.num_trans, 'updates': self.num_updates
        }

    def __reset(self):
        self.start_time = time.time()
        self.steps = 0
        self.num_trans = 0
        self.num_updates = 0

    def set_loggers(self, *loggers):
        self.loggers = loggers
