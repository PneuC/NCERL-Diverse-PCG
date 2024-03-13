import time
import torch
import random
import numpy as np
from math import ceil
from src.drl.rep_mem import ReplayMem
from src.env.environments import AsyncOlGenEnv
from src.rlkit.samplers.rollout_functions import get_ucb_std
from src.rlkit.torch.sac.neurips20_sac_ensemble import NeurIPS20SACEnsembleTrainer
from src.utils.datastruct import RingQueue
from src.utils.filesys import getpath


class AsyncOffPolicyALgo:
    def __init__(self, rep_mem=None, update_per=1, batch=256, device='cpu'):
        self.rep_mem = rep_mem
        self.update_per = update_per
        self.batch = batch
        self.loggers = []
        self.device = device

        self.start_time = 0.
        self.steps = 0
        self.num_trans = 0
        self.num_updates = 0
        self.env = None
        self.trainer = None
        self.proxy_agent = None
        pass

    def train(self, env:AsyncOlGenEnv, trainer:NeurIPS20SACEnsembleTrainer, budget, inference_type, path):
        assert trainer.device == self.device
        self.__reset(env, trainer)
        env.reset()
        actors, critic1, critic2 = trainer.policy, trainer.qf1, trainer.qf2
        for logger in self.loggers:
            if logger.__class__.__name__ == 'GenResLogger':
                logger.on_episode(env, self.proxy_agent, 0)
        while self.steps < budget:
            if inference_type > 0.:
                self.ucb_interaction(inference_type, critic1, critic2)
            else:
                self.rand_choice_iteraction()
            update_credits = ceil(1.25 * env.eplen / self.update_per)
            self.__update(update_credits)

        self.__update(0, close=True)
        for i, actor in enumerate(actors):
            torch.save(actor, getpath(path, f'policy{i}.pth'))

    def __update(self, model_credits, close=False):
        transitions, rewss = self.env.close() if close else self.env.rollout()
        self.rep_mem.add_transitions(transitions)
        self.num_trans += len(transitions)
        if len(self.rep_mem) > self.batch:
            t = 1
            while self.num_trans >= (self.num_updates + 1) * self.update_per:
                # agent.update(*self.rep_mem.sample(self.batch))
                self.trainer.train_from_torch(self.rep_mem.sample(self.batch))
                self.num_updates += 1
                if not close and t == model_credits:
                    break
                t += 1
        for logger in self.loggers:
            loginfo = self.__pack_loginfo(rewss)
            if logger.__class__.__name__ == 'GenResLogger':
                logger.on_episode(self.env, self.proxy_agent, self.steps)
            else:
                logger.on_episode(**loginfo, close=close)

    def __pack_loginfo(self, rewss):
        return {
            'steps': self.steps, 'time': time.time() - self.start_time, 'rewss': rewss,
            'trans': self.num_trans, 'updates': self.num_updates
        }

    def __reset(self, env:AsyncOlGenEnv, trainer:NeurIPS20SACEnsembleTrainer):
        self.start_time = time.time()
        self.steps = 0
        self.num_trans = 0
        self.num_updates = 0
        self.env = env
        self.trainer = trainer
        self.proxy_agent = SunriseProxyAgent(trainer.policy, self.device)
        if self.rep_mem is None:
            self.rep_mem = MaskedRepMem(trainer.num_ensemble)
        assert self.rep_mem.m == trainer.num_ensemble


    def set_loggers(self, *loggers):
        self.loggers = loggers

    def ucb_interaction(self, inference_type, critic1, critic2, feedback_type=1):
        """
            Adapted from rlkit.samplers.rollout_functions.ensemble_ucb_rollout.
            Mask generation is moved to replay memory (MaskedRepMem)
            Noise flag is ignored since it does not useful for our experiments.
            feedback_type is fixed to 1 as original code does not change it.
        """
        o = self.env.getobs()
        policy = self.trainer.policy
        for subpolicy in policy:
            subpolicy.reset()
        while True:
            a_max, ucb_max, agent_info_max = None, None, None
            for i, subpolicy in enumerate(policy):
                _a, agent_info = subpolicy.get_action(o)
                ucb_score = get_ucb_std(
                    o, _a, inference_type, critic1, critic2, feedback_type, i, len(policy)
                )
                if i == 0:
                    a_max = _a
                    ucb_max = ucb_score
                else:
                    if ucb_score > ucb_max:
                        ucb_max = ucb_score
                        a_max = _a
            # print(a_max)
            o, d = self.env.step(a_max.squeeze())
            self.steps += 1
            if d:
                break
        pass

    def rand_choice_iteraction(self):
        o = self.env.getobs()
        policy = self.trainer.policy
        for subpolicy in policy:
            subpolicy.reset()
        choiced_policy = random.choice(policy)
        while True:
            _a, _ = choiced_policy.get_action(o)
            o, d = self.env.step(_a.squeeze())
            self.steps += 1
            if d:
                break
        pass


class SunriseProxyAgent:
    # Refer to rlkit.samplers.rollout_functions.ensemble_eval_rollout
    def __init__(self, actors, device):
        self.actors = actors
        self.device = device

    def make_decision(self, obs):
        o = torch.tensor(obs, device=self.device, dtype=torch.float32)
        actions = []
        with torch.no_grad():
            for m in self.actors:
                a = torch.clamp(m(o)[0], -1, 1)
                actions.append(a.cpu().numpy())
        actions = np.array(actions)
        selections = [random.choice(range(len(self.actors))) for _ in range(len(obs))]
        selected = [actions[s, i, :] for i, s in enumerate(selections)]
        return np.array(selected)

        # if len(obs.shape) == 1:
        #     obs = obs.reshape(1, -1)
        # with torch.no_grad():
        #     actions = np.stack([actor.get_action(obs)[0].squeeze() for actor in self.actors])
        # return actions.mean(axis=0)

    def reset(self):
        for actor in self.actors:
            actor.reset()


class MaskedRepMem:
    def __init__(self, num_ensemble, capacity=500000, ber_mean=0.0, device='cpu'):
        self.base = ReplayMem(capacity, device)
        self.mask_queue = RingQueue(capacity)
        self.ber_mean = ber_mean
        self.device = device
        self.m = num_ensemble

    def add(self, o, a, r, op):
        mask = torch.bernoulli(torch.Tensor([self.ber_mean] * self.m))
        if mask.sum() == 0:
            rand_index = np.random.randint(self.m, size=1)
            mask[rand_index] = 1
        self.mask_queue.push(mask.to(self.base.device))
        self.base.add(o, a, r, op)
        pass

    def add_transitions(self, trainsitions):
        for t in trainsitions:
            self.add(*t)

    def __len__(self):
        return len(self.base)

    def sample(self, n):
        indexes = random.sample(range(len(self.base)), n)
        base_mem = self.base.queue.main
        mask_mem = self.mask_queue.main
        obs, acts, rews, ops, masks = [], [], [], [], []
        for i in indexes:
            o, a, r, op = base_mem[i]
            obs.append(o)
            acts.append(a)
            rews.append([r])
            ops.append(op)
            masks.append(mask_mem[i])
        return {
            'observations': torch.stack(obs),
            'actions': torch.stack(acts),
            'rewards': torch.tensor(rews, device=self.device, dtype=torch.float),
            'next_observations': torch.stack(ops),
            'masks': torch.stack(masks)
        }
    pass
