from itertools import chain
from math import ceil

import torch
import random
import numpy as np
from typing import List
from src.gan.gans import nz
from src.gan.gankits import process_onehot, get_decoder
from src.smb.level import MarioLevel, lvlhcat
from src.utils.datastruct import RingQueue
from src.utils.filesys import getpath


class OnlineGenerator:
    def __init__(self, policy, decoder=None, g_device='cuda:0'):
        self.init_vecs = np.load(getpath('smb/init_latvecs.npy'))
        self.policy = policy
        self.decoder = get_decoder() if decoder is None else decoder
        self.decoder.to(g_device)
        self.g_device = g_device
        self.obs_buffer = RingQueue(policy.n)
        self.re_init()
        self.init_seg = None

    def re_init(self, condition=None):
        for _ in range(self.policy.n):
            self.obs_buffer.push(np.zeros([nz]))
        if condition is not None:
            for item in condition:
                self.obs_buffer.push(condition)
        else:
            latvec = random.choice(self.init_vecs)
            self.obs_buffer.push(latvec)
        init_onehot = torch.tensor(self.obs_buffer.rear(), device=self.g_device).view(1, -1, 1, 1)
        self.init_seg = process_onehot(self.decoder(init_onehot))

    def step(self):
        obs = np.concatenate(self.obs_buffer.to_list(), axis=-1)
        latvec = self.policy.step(obs)
        self.obs_buffer.push(latvec)
        z = torch.tensor(latvec, device=self.g_device).view(-1, nz, 1, 1)
        seg = process_onehot(self.decoder(z))
        return seg

    def forward(self, l) -> List[MarioLevel]:
        self.re_init()
        self.policy.reset()
        return [self.init_seg, *(self.step() for _ in range(l))]

    def generate(self, n, l):
        return [lvlhcat(self.forward(l)) for _ in range(n)]


class VecOnlineGenerator(OnlineGenerator):
    def __init__(self, policy, decoder=None, g_device='cuda:0', vec_num=50):
        self.vec_num = vec_num
        super().__init__(policy, decoder, g_device)

    def re_init(self, condition=None):
        for _ in range(self.policy.n):
            self.obs_buffer.push(np.zeros([self.vec_num, nz]))
        if condition is not None:
            self.obs_buffer.push(condition)
        else:
            latvecs = self.init_vecs[random.sample(range(len(self.init_vecs)), self.vec_num)]
            self.obs_buffer.push(latvecs)
        init_onehot = torch.tensor(self.obs_buffer.rear(), device=self.g_device).view(-1, nz, 1, 1)
        self.init_seg = process_onehot(self.decoder(init_onehot))
        # for lvl in self.init_seg[:2]:
        #     print(lvl)

    def forward(self, l, rand_init=True):
        if rand_init: self.re_init()
        self.policy.reset()
        lvls = [[item] for item in self.init_seg]
        for _ in range(l):
            for lvl, seg in zip(lvls, self.step()):
                lvl.append(seg)
        return lvls

    def generate(self, n, l, rand_init=True):
        batchs = [[lvlhcat(item) for item in self.forward(l, rand_init)] for _ in range(ceil(n / self.vec_num))]
        res = list(chain(*batchs))
        return res[:n]


if __name__ == '__main__':
    a = np.random.rand(10, 3)
    print(a)
    print(a[random.sample(range(len(a)), 2)])
    pass
