import glob
import random
from abc import abstractmethod, abstractstaticmethod
import numpy as np
import torch
from src.utils.filesys import getpath
from src.gan.gans import nz
from src.gan.gankits import sample_latvec
from src.drl.drl_uses import load_cfgs
from src.drl.sunrise.sunrise_adaption import SunriseProxyAgent


def process_obs(obs, device='cpu'):
    obs = torch.tensor(obs, device=device, dtype=torch.float32)
    if len(obs.shape) == 1:
        obs = obs.unsqueeze(0)
    return obs


class GenPolicy:
    def __init__(self, n=5):
        self.n = n # Number of segments in an observation

    @abstractmethod
    def step(self, obs):
        pass

    @staticmethod
    @abstractmethod
    def from_path(path, **kwargs):
        pass

    def reset(self):
        pass


class RLGenPolicy(GenPolicy):
    def __init__(self, model, n, device='cuda:0'):
        self.model = model
        super(RLGenPolicy, self).__init__(n)
        self.model.eval()
        self.model.to(device)
        self.device = device
        self.meregs = []

    def step(self, obs):
        obs = process_obs(obs, device=self.device)
        b, d = obs.shape
        if d < nz * self.n:
            obs = torch.cat([torch.zeros([b, nz * self.n - d], device=self.device), obs], dim=-1)
        with torch.no_grad():
            # mus, sigmas, betas = self.model.get_intermediate(obs)
            # print(mus[0].cpu().numpy(), '\n', betas[0].cpu().numpy(), '\n')
            model_output, _ = self.model(obs)
        return torch.clamp(model_output, -1, 1).squeeze().cpu().numpy()

    @staticmethod
    def from_path(path, device='cuda:0'):
        model = torch.load(getpath(f'{path}/policy.pth'), map_location=device)
        n = load_cfgs(path, 'N')
        return RLGenPolicy(model, n, device)

#
# class SunriseGenPolicy(GenPolicy):
#     def __init__(self, models, n, device='cpu'):
#         super(SunriseGenPolicy, self).__init__(n)
#         for model in models:
#             model.to(device)
#         self.models = models
#         self.m = len(self.models)
#
#         self.agent = SunriseProxyAgent(models, device)
#
#     def step(self, obs):
#         actions = [m(obs.unsqueeze()).squeeze().cpu().numpy() for m in self.models]
#         if len(obs.shape) == 1:
#             return random.choice(actions)
#         else:
#             actions = np.array(actions)
#             selections = [random.choice(range(self.m)) for _ in range(len(obs))]
#             selected = [actions[s, i, :] for i, s in enumerate(selections)]
#             return np.array(selected)
#     #
#     # def reset(self):
#     #     # self.agent.reset()
#     #     pass
#
#     @staticmethod
#     def from_path(path, device='cpu'):
#         models = [
#             torch.load(p, map_location=device)
#             for p in glob.glob(getpath(path, 'policy*.pth'))
#         ]
#         n = load_cfgs(path, 'N')
#         return SunriseGenPolicy(models, n, device)
#
#     # @property
#     # def device(self):
#     #     return self.agent.device


class EnsembleGenPolicy(GenPolicy):
    def __init__(self, models, n, device='cpu'):
        super(EnsembleGenPolicy, self).__init__(n)
        for model in models:
            model.to(device)
        self.device = device
        self.models = models
        self.m = len(models)


    def step(self, obs):
        o = torch.tensor(obs, device=self.device, dtype=torch.float32)
        actions = []
        with torch.no_grad():
            for m in self.models:
                a = m(o)
                if type(a) == tuple:
                    a = a[0]
                actions.append(torch.clamp(a, -1, 1).cpu().numpy())
        if len(obs.shape) == 1:
            return random.choice(actions)
        else:
            actions = np.array(actions)
            selections = [random.choice(range(self.m)) for _ in range(len(obs))]
            selected = [actions[s, i, :] for i, s in enumerate(selections)]
            return np.array(selected)

    @staticmethod
    def from_path(path, device='cpu'):
        models = [
            torch.load(p, map_location=device)
            for p in glob.glob(getpath(path, 'policy*.pth'))
        ]
        n = load_cfgs(path, 'N')
        return EnsembleGenPolicy(models, n, device)


class RandGenPolicy(GenPolicy):
    def __init__(self):
        super(RandGenPolicy, self).__init__(1)

    def step(self, obs):
        # if len(obs.shape) == 1:
        #     return sample_latvec(1).squeeze().numpy()
        # else:
        n = obs.shape[0]
        return sample_latvec(n).squeeze().numpy()

    @staticmethod
    def from_path(path, **kwargs):
        return RandGenPolicy()


class DvDGenPolicy(GenPolicy):
    def __init__(self, learner, n, rand_switch=False):
        super(DvDGenPolicy, self).__init__(n)
        self.master = learner
        self.rand_switch = rand_switch
        self.working_policy = None
        self.reset()

    def step(self, obs):
        return self.working_policy.forward(obs).astype(np.float32)

    def reset(self):
        if self.rand_switch:
            self.working_policy = random.choice(self.master.agents)
        else:
            self.working_policy = self.master.agents[self.master.agent]

    @staticmethod
    def from_path(path, device='cpu'):
        """
            We don't find loading function in DvD-ES codes and we have no idea about
            how to implement it :-(
        """
        return None
