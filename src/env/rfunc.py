import numpy as np
from math import ceil
from abc import abstractmethod
from src.utils.mymath import a_clip
from src.smb.level import *

defaults = {'n': 5, 'gl': 0.14, 'gg': 0.30, 'wl': 2, 'wg': 10}


class RewardFunc:
    def __init__(self, *args):
        self.terms = args
        self.require_simlt = any(term.require_simlt for term in self.terms)

    def get_rewards(self, **kwargs):
        return {
            term.get_name(): term.compute_rewards(**kwargs)
            for term in self.terms
        }

    def get_n(self):
        n = 1
        for term in self.terms:
            try:
                n = max(n, term.n)
            except AttributeError:
                pass
        return n

    def __str__(self):
        return 'Reward Function:\n' + ',\n'.join('\t' + str(term) for term in self.terms)


class RewardTerm:
    def __init__(self, require_simlt):
        self.require_simlt = require_simlt

    def get_name(self):
        return self.__class__.__name__

    @abstractmethod
    def compute_rewards(self, **kwargs):
        pass


class Playability(RewardTerm):
    def __init__(self, magnitude=1):
        super(Playability, self).__init__(True)
        self.magnitude=magnitude

    def compute_rewards(self, **kwargs):
        simlt_res = kwargs['simlt_res']
        return [0 if item['playable'] else -self.magnitude for item in simlt_res[1:]]

    def __str__(self):
        return f'{self.magnitude} * Playability'


class MeanDivergenceFun(RewardTerm):
    def __init__(self, goal_div, n=defaults['n'], s=8):
        super().__init__(False)
        self.l = goal_div * 0.26 / 0.6
        self.u = goal_div * 0.94 / 0.6
        self.n = n
        self.s = s

    def compute_rewards(self, **kwargs):
        segs = kwargs['segs']
        rewards = []
        for i in range(1, len(segs)):
            seg = segs[i]
            histroy = lvlhcat(segs[max(0, i - self.n): i])
            k = 0
            divergences = []
            while k * self.s <= (min(self.n, i) - 1) * MarioLevel.seg_width:
                cmp_seg = histroy[:, k * self.s: k * self.s + MarioLevel.seg_width]
                # print(i, nd, cmp_seg.shape)
                divergences.append(tile_pattern_js_div(seg, cmp_seg))
                k += 1
            mean_d = sum(divergences) / len(divergences)
            if mean_d < self.l:
                rewards.append(-(mean_d - self.l) ** 2)
            elif mean_d > self.u:
                rewards.append(-(mean_d - self.u) ** 2)
            else:
                rewards.append(0)
        return rewards


class SACNovelty(RewardTerm):
    def __init__(self, magnitude, goal_div, require_simlt, n):
        super().__init__(require_simlt)
        self.g = goal_div
        self.magnitude = magnitude
        self.n = n

    def compute_rewards(self, **kwargs):
        n_segs = len(kwargs['segs'])
        rewards = []
        for i in range(1, n_segs):
            reward = 0
            r_sum = 0
            for k in range(1, self.n + 1):
                j = i - k
                if j < 0:
                    break
                r = 1 - k / (self.n + 1)
                r_sum += r
                reward += a_clip(self.disim(i, j, **kwargs), self.g, r)
            rewards.append(reward * self.magnitude / r_sum)
        return rewards

    @abstractmethod
    def disim(self, i, j, **kwargs):
        pass


class LevelSACN(SACNovelty):
    def __init__(self, magnitude=1, g=defaults['gl'], w=defaults['wl'], n=defaults['n']):
        super(LevelSACN, self).__init__(magnitude, g, False, n)
        self.w = w

    def disim(self, i, j, **kwargs):
        segs = kwargs['segs']
        seg1, seg2 = segs[i], segs[j]
        return tile_pattern_js_div(seg1, seg2, self.w)

    def __str__(self):
        s = f'{self.magnitude} * LevelSACN(g={self.g:.3g}, w={self.w}, n={self.n})'
        return s


class GameplaySACN(SACNovelty):
    def __init__(self, magnitude=1, g=defaults['gg'], w=defaults['wg'], n=defaults['n']):
        super(GameplaySACN, self).__init__(magnitude, g, True, n)
        self.w = w

    def disim(self, i, j, **kwargs):
        simlt_res = kwargs['simlt_res']
        trace1, trace2 = simlt_res[i]['trace'], simlt_res[j]['trace']
        return trace_div(trace1, trace2, self.w)

    def __str__(self):
        s = f'{self.magnitude} * GameplaySACN(g={self.g:.3g}, w={self.w}, n={self.n})'
        return s


class Fun(RewardTerm):
    def __init__(self, magnitude=1., num_windows=3, lb=0.26, ub=0.94, stride=8):
        super().__init__(False)
        self.lb, self.ub = lb, ub
        self.magnitude = magnitude
        self.stride = stride
        self.num_windows = num_windows
        self.n = ceil(num_windows * stride / MarioLevel.seg_width - 1e-8)

    def compute_rewards(self, **kwargs):
        n_segs = len(kwargs['segs'])
        lvl = lvlhcat(kwargs['segs'])
        W = MarioLevel.seg_width
        rewards = []
        for i in range(1, n_segs):
            seg = lvl[:, W*i: W*(i+1)]
            divs = []
            for k in range(0, self.num_windows + 1):
                s = W * i - k * self.stride
                if s < 0:
                    break
                cmp_seg = lvl[:, s:s+W]
                divs.append(tile_pattern_kl_div(seg, cmp_seg))
            mean_div = np.mean(divs)
            rew = 0
            if mean_div > self.ub:
                rew = -(self.ub - mean_div) ** 2
            if mean_div < self.lb:
                rew = -(self.lb - mean_div) ** 2
            rewards.append(rew * self.magnitude)
        return rewards

    def __str__(self):
        s = f'{self.magnitude} * Fun(lb={self.lb:.2f}, ub={self.ub:.2f}, n={self.num_windows}, stride={self.stride})'
        return s


class HistoricalDeviation(RewardTerm):
    def __init__(self, magnitude=1., m=3, n=10):
        super().__init__(False)
        self.magnitude = magnitude
        self.m = m
        self.n = n

    def compute_rewards(self, **kwargs):
        segs = kwargs['segs']
        n_segs = len(kwargs['segs'])
        rewards = []
        for i in range(1, n_segs):
            divs = []
            for k in range(1, self.n+1):
                j = i - k
                if j < 0:
                    break
                divs.append(tile_pattern_kl_div(segs[i], segs[j]))
            divs.sort()
            m = min(i, self.m)
            rew = np.mean(divs[:m])
            rewards.append(rew * self.magnitude)
        return rewards

    def __str__(self):
        return f'{self.magnitude} * HistoricalDeviation(m={self.m}, n={self.n})'


if __name__ == '__main__':
    # print(type(ceil(0.2)))
    # arr = [1., 3., 2.]
    # arr.sort()
    # print(arr)
    rfunc = HistoricalDeviation()

