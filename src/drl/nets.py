import torch
from abc import abstractmethod
from typing import Callable, Tuple
from torch import nn
from torch._C._te import Tensor
from torch.distributions import Normal, MixtureSameFamily, Categorical, Independent


def mlp(sizes, activation, output_activation:Callable=nn.Identity):
    if not len(sizes) or sizes[0] <= 0:
        return nn.Identity()
    layers = []
    for i in range(len(sizes) - 1):
        act = activation if i < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[i], sizes[i + 1]), act()]
    return nn.Sequential(*layers)

def conv_esmbs(obs_dim, hdims, act_dim, m, activation):
    hiddens = [nn.Conv1d(obs_dim, hdims[0] * m, 1), activation()]
    for i in range(len(hdims) - 1):
        hiddens.append(nn.Conv1d(hdims[i] * m, hdims[i + 1] * m, 1, groups=m))
        hiddens.append(activation())
    head_convs = nn.Sequential(*hiddens)
    mu_layers= nn.Sequential(nn.Conv1d(hdims[-1] * m, act_dim * m, 1, groups=m))
    logstd_layers = nn.Conv1d(hdims[-1] * m, act_dim * m, 1, groups=m)
    return head_convs, mu_layers, logstd_layers

LOG_STD_MAX = 2
LOG_STD_MIN = -20
MUSCALE = 1


class GaussianMLP(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation=nn.ReLU):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.net = mlp([obs_dim, *hidden_sizes], activation)
        self.mu_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.log_std_layer = nn.Linear(hidden_sizes[-1], act_dim)

    def forward(self, obs, deterministic=False):
        net_out = self.net(obs)
        mu = MUSCALE * torch.tanh(self.mu_layer(net_out))
        if deterministic:
            return mu, None
        log_std = self.log_std_layer(net_out)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)
        pi_distribution = Normal(mu, std)
        a = pi_distribution.rsample()
        logp_pi = pi_distribution.log_prob(a).sum(dim=-1)
        return torch.clamp(a, -1, 1), logp_pi   # Clamp is somehow crucial.


class ObsActMLP(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation=nn.ReLU):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.main = mlp([obs_dim + act_dim, *hidden_sizes, 1], activation)

    def forward(self, obs, act):
        return self.main(torch.cat([obs, act], dim=-1)).squeeze()


class LearnableLogCoeffient(nn.Module):
    def __init__(self, v0=-1.0):
        super(LearnableLogCoeffient, self).__init__()
        self.val = nn.Parameter(torch.tensor(v0), requires_grad=True)

    def forward(self, x, grad=True):
        if grad:
            return torch.exp(self.val) * x
        return torch.exp(self.val.detach()) * x


def esmb_sample(muss, stdss, betas, mono=True):
    subpolicies = Independent(Normal(muss, stdss), 1)

    if mono:
        policy = MixtureSameFamily(Categorical(betas), subpolicies)
        acts = torch.clamp(policy.sample(), -1, 1)
        return acts, policy.log_prob(acts)
    else:
        actss = torch.clamp(subpolicies.rsample(), -1, 1)
        logpss = subpolicies.log_prob(actss)
        return actss, logpss, betas  # acts: B * m * nz; logp: B * m; betas: B * m


class MixGaussianModule(nn.Module):
    @abstractmethod
    def get_intermediate(self, obs) -> Tuple[Tensor, Tensor, Tensor]:
        pass

    def forward(self, obs: Tensor, mono=True):
        muss, stdss, betas = self.get_intermediate(obs)
        return esmb_sample(muss, stdss, betas, mono)


class EsmbGaussianMLP(MixGaussianModule):
    def __init__(self, obs_dim, act_dim, head_sizes, beta_sizes, m=2, activation=nn.ReLU):
        super().__init__()
        for i in range(m):
            self.__setattr__(f'head{i}', mlp([obs_dim, *head_sizes], activation, activation))
            self.__setattr__(f'mu_layer{i}', nn.Linear(head_sizes[-1], act_dim))
            self.__setattr__(f'logstd_layer{i}', nn.Linear(head_sizes[-1], act_dim))
        self.beta_layer = mlp([obs_dim, *beta_sizes, m], activation, lambda : nn.Softmax(dim=-1))
        self.m = m

    def get_intermediate(self, obs) -> Tuple[Tensor, Tensor, Tensor]:
        betas = self.beta_layer(obs)
        heads = [self.__getattr__(f'head{i}') for i in range(self.m)]
        head_outs = [head(obs) for head in heads]

        mu_layers = [self.__getattr__(f'mu_layer{i}') for i in range(self.m)]
        logstd_layers = [self.__getattr__(f'logstd_layer{i}') for i in range(self.m)]
        muss = [
            MUSCALE * torch.tanh(mu_layer(head_out))
            for mu_layer, head_out in zip(mu_layers, head_outs)
        ]
        logstdss = [
            torch.clamp(logstd_layer(head_out), LOG_STD_MIN, LOG_STD_MAX)
            for logstd_layer, head_out in zip(logstd_layers, head_outs)
        ]
        stdss = [torch.exp(logstds) for logstds in logstdss]
        return torch.stack(muss, 1), torch.stack(stdss, 1), betas
    pass


class EsmbGaussianConv(MixGaussianModule):
    # Theoretically equivalent to multiple MLPs (EsmbGaussianMLP) but runs faster on GPU.
    def __init__(self, obs_dim, act_dim, head_sizes, beta_sizes, m=2, activation=nn.ReLU):
        super().__init__()
        self.beta_layer = mlp([obs_dim, *beta_sizes, m], activation, lambda : nn.Softmax(dim=-1))
        self.heads, self.mu_layers, self.logstd_layers = conv_esmbs(obs_dim, head_sizes, act_dim, m, activation)
        self.m = m
        self.act_dim = act_dim

    def get_intermediate(self, obs) -> Tuple[Tensor, Tensor, Tensor]:
        n = len(obs)
        betas = self.beta_layer(obs)
        head_featrs = self.heads(obs.view(n, -1, 1))
        muss = MUSCALE * torch.tanh(self.mu_layers(head_featrs))
        logstds = torch.clamp(self.logstd_layers(head_featrs), LOG_STD_MIN, LOG_STD_MAX)
        stdss = torch.exp(logstds).transpose(1, 2)
        return muss.view(n, self.m, self.act_dim), stdss.view(n, self.m, self.act_dim), betas


if __name__ == '__main__':
    model = EsmbGaussianConv(5, 3, [8, 8, 8], [8, 8, 8])
    print(model)
    model.eval()
    print(model(torch.rand([1, 5])))
    pass
