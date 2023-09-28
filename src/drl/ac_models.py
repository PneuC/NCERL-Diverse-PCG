import torch.nn.functional as F
from stable_baselines3.common.utils import polyak_update
from torch.optim import Adam
from src.drl.nets import *


class SoftActor:
    def __init__(self, net_constructor, tar_ent=None):
        self.net = net_constructor()
        self.optimiser = Adam(self.net.parameters(), 3e-4)
        self.alpha_coe = LearnableLogCoeffient()
        self.alpha_optimiser = Adam(self.alpha_coe.parameters(), 3e-4)
        self.tar_ent = -self.net.act_dim if tar_ent is None else tar_ent
        self.device = 'cpu'
        pass

    def to(self, device):
        self.net.to(device)
        self.alpha_coe.to(device)
        self.device = device

    def eval(self):
        self.net.eval()

    def train(self):
        self.net.train_NCESAC()

    def forward(self, obs, grad=True, deterministic=False):
        if grad:
            return self.net(obs, deterministic)
        with torch.no_grad():
            return self.net(obs, deterministic)

    def backward_policy(self, critic, obs, coe=1.):
        acts, logps = self.forward(obs)
        qvalues = critic.forward(obs, acts)
        a_loss = coe * (self.alpha_coe(logps, grad=False) - qvalues).mean()
        a_loss.backward()

    def backward_alpha(self, obs):
        _, logps = self.forward(obs, grad=False)
        loss_alpha = -(self.alpha_coe(logps + self.tar_ent)).mean()
        loss_alpha.backward()
        pass

    def zero_grads(self):
        self.optimiser.zero_grad()
        self.alpha_optimiser.zero_grad()

    def grad_step(self):
        self.optimiser.step()
        self.alpha_optimiser.step()

    def get_nn_arch_str(self):
        return str(self.net) + '\n'


class SoftDoubleClipCriticQ:
    def __init__(self, nn_constructor, gamma=0.99, tau=0.005):
        self.net1 = nn_constructor()
        self.net2 = nn_constructor()
        self.tar_net1 = nn_constructor()
        self.tar_net2 = nn_constructor()
        self.tar_net1.load_state_dict(self.net1.state_dict())
        self.tar_net2.load_state_dict(self.net2.state_dict())
        self.opt1 = Adam(self.net1.parameters(), 3e-4)
        self.opt2 = Adam(self.net2.parameters(), 3e-4)
        self.device = 'cpu'
        self.gamma = gamma
        self.tau = tau

    def to(self, device):
        self.net1.to(device)
        self.net2.to(device)
        self.tar_net1.to(device)
        self.tar_net2.to(device)
        self.device = device

    def forward(self, obs, acts, grad=True, tar=False):
        def foo():
            if tar:
                q1 = self.tar_net1(obs, acts)
                q2 = self.tar_net2(obs, acts)
            else:
                q1 = self.net1(obs, acts)
                q2 = self.net2(obs, acts)
            return torch.minimum(q1, q2)
        if grad:
            return foo()
        with torch.no_grad():
            return foo()

    def compute_target(self, actor, rews, ops):
        aps, logpi_aps = actor.forward(ops, grad=False)
        qps = self.forward(ops, aps, tar=True, grad=False)
        y = rews + self.gamma * (qps - actor.alpha_coe(logpi_aps, False))
        return y

    def backward_mse(self, actor, obs, acts, rews, ops):
        y = self.compute_target(actor, rews, ops)
        loss1 = F.mse_loss(self.net1(obs, acts), y)
        loss2 = F.mse_loss(self.net2(obs, acts), y)
        loss1.backward()
        loss2.backward()

    def update_tarnet(self):
        polyak_update(self.net1.parameters(), self.tar_net1.parameters(), self.tau)
        polyak_update(self.net2.parameters(), self.tar_net2.parameters(), self.tau)

    def zero_grads(self):
        self.opt1.zero_grad()
        self.opt2.zero_grad()

    def grad_step(self):
        self.opt1.step()
        self.opt2.step()

    def get_nn_arch_str(self):
        return str(self.net1) + '\n' + str(self.net2) + '\n'


class MERegMixSoftActor(SoftActor):
    def __init__(self, net_constructor, me_reg, tar_ent=None):
        super(MERegMixSoftActor, self).__init__(net_constructor, tar_ent)
        self.me_reg = me_reg

    def forward(self, obs, grad=True, mono=True):
        if grad:
            return self.net(obs, mono)
        with torch.no_grad():
            return self.net(obs, mono)

    def backward_me_reg(self, critic_W, obs):
        muss, stdss, betas = self.net.get_intermediate(obs)
        loss1 = -torch.mean(self.me_reg.forward(muss, stdss, betas))
        actss, _, _ = esmb_sample(muss, stdss, betas, mono=False)
        wvaluess = critic_W.forward(obs, actss)
        loss2 = -(betas * wvaluess).mean()
        loss = loss1 + loss2
        loss.backward()
        pass

    def backward_policy(self, critic, obs):
        actss, logpss, betas = self.forward(obs, mono=False)
        qvaluess = critic.forward(obs, actss)
        a_loss = (betas * (self.alpha_coe(logpss, grad=False) - qvaluess)).mean()
        a_loss.backward()
        pass

    def backward_alpha(self, obs):
        _, logps = self.forward(obs, grad=False)
        loss_alpha = -(self.alpha_coe(logps + self.tar_ent)).mean()
        loss_alpha.backward()
        pass


class MERegSoftDoubleClipCriticQ(SoftDoubleClipCriticQ):
    def forward(self, obs, actss, grad=True, tar=False):
        def foo():
            obss = torch.unsqueeze(obs, dim=1).expand(-1, actss.shape[1], -1)
            if tar:
                q1 = self.tar_net1(obss, actss)
                q2 = self.tar_net2(obss, actss)
            else:
                q1 = self.net1(obss, actss)
                q2 = self.net2(obss, actss)
            return torch.minimum(q1, q2)
        if grad:
            return foo()
        with torch.no_grad():
            return foo()

    def compute_target(self, actor, rews, ops):
        apss, logpss, betaps = actor.forward(ops, grad=False, mono=False)
        qpss = self.forward(ops, apss, tar=True, grad=False)
        qps = (betaps * (qpss.squeeze() - actor.alpha_coe(logpss, grad=False))).sum(dim=-1)
        y = rews + self.gamma * qps
        return y
    pass


class MERegDoubleClipCriticW(MERegSoftDoubleClipCriticQ):
    def compute_target(self, actor, rews, ops):
        with torch.no_grad():
            mupss, stdpss, betaps = actor.net.get_intermediate(ops)
            me_regps = actor.me_reg.forward(mupss, stdpss, betaps)
            apss, *_ = esmb_sample(mupss, stdpss, betaps, mono=False)
            wpss = self.forward(ops, apss, tar=True, grad=False)
            y = me_regps + (betaps * wpss).sum(dim=-1)
        return self.gamma * y
    pass



if __name__ == '__main__':
    a = torch.tensor([[[1., 1.], [2., 2.], [3., 3.]]], requires_grad=True)
    b = a.detach().mean(-1)
    print(a)
    print(b)
    pass

