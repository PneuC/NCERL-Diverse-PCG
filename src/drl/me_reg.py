import torch
from math import sqrt
from abc import abstractmethod
from itertools import combinations
from src.gan.gankits import nz


class ExclusionReg:
    # NOTE: To be maximised
    def __init__(self, lbd):
        self.lbd = lbd

    @abstractmethod
    def forward(self, muss, stdss, betas):
        pass


class WassersteinExclusion(ExclusionReg):
    def forward(self, muss, stdss, betas):
        b, m, d = muss.shape
        rho = torch.zeros([b], device=muss.device)
        for i, j in combinations(range(m), 2):
            x = torch.square((muss[:, i, :] - muss[:, j, :])).sum(dim=-1)
            y = torch.sum((stdss[:, i, :] + stdss[:, j, :]), dim=-1)
            z = torch.sqrt((stdss[:, i, :] * stdss[:, j, :]).sum(dim=-1))
            w = (x + y - 2 * z).sqrt()
            rho += betas[:, i] * betas[:, j] * w
        return self.lbd * rho


class LogWassersteinExclusion(ExclusionReg):
    def forward(self, muss, stdss, betas):
        b, m, d = muss.shape
        rho = torch.zeros([b], device=muss.device)

        for i, j in combinations(range(m), 2):
            x = torch.square((muss[:, i, :] - muss[:, j, :])).sum(dim=-1)
            y = torch.sum((stdss[:, i, :] + stdss[:, j, :]), dim=-1)
            z = torch.sqrt((stdss[:, i, :] * stdss[:, j, :]).sum(dim=-1))
            w = (x + y - 2 * z).sqrt()
            rho += betas[:, i] * betas[:, j] * torch.log(w + 1)
        return self.lbd * rho


class ClipExclusion(ExclusionReg):
    def __init__(self, lbd, wbar=0.6 * sqrt(nz)):
        super(ClipExclusion, self).__init__(lbd)
        self.wbar = wbar

    def forward(self, muss, stdss, betas):
        b, m, d = muss.shape
        rho = torch.zeros([b], device=muss.device)
        for i, j in combinations(range(m), 2):
            x = torch.square((muss[:, i, :] - muss[:, j, :])).sum(dim=-1)
            y = torch.sum((stdss[:, i, :] + stdss[:, j, :]), dim=-1)
            z = torch.sqrt((stdss[:, i, :] * stdss[:, j, :]).sum(dim=-1))
            w = (x + y - 2 * z).sqrt()
            rho += betas[:, i] * betas[:, j] * torch.clip(w, max=self.wbar)
        return self.lbd * rho


class LogClipExclusion(ExclusionReg):
    def __init__(self, lbd, wbar=0.6 * sqrt(nz)):
        super(LogClipExclusion, self).__init__(lbd)
        self.wbar = wbar

    def forward(self, muss, stdss, betas):
        b, m, d = muss.shape
        rho = torch.zeros([b], device=muss.device)

        for i, j in combinations(range(m), 2):
            x = torch.square((muss[:, i, :] - muss[:, j, :])).sum(dim=-1)
            y = torch.sum((stdss[:, i, :] + stdss[:, j, :]), dim=-1)
            z = torch.sqrt((stdss[:, i, :] * stdss[:, j, :]).sum(dim=-1))
            w = (x + y - 2 * z).sqrt()
            rho += betas[:, i] * betas[:, j] * torch.log(torch.clip(w, max=self.wbar) + 1)
        return self.lbd * rho


# class SurrogateDistReg:
#     def __init__(self, lbd, clip=30.):
#         self.lbd = lbd
#         self.clip = clip
#
#     def forward(self, muss, stdss, betas):


if __name__ == '__main__':
    pass
