import torch
from src.drl.ac_models import SoftActor
from src.drl.nets import esmb_sample


class PMOESoftActor(SoftActor):
    def __init__(self, net_constructor, tar_ent=None):
        super(PMOESoftActor, self).__init__(net_constructor, tar_ent)

    def forward(self, obs, grad=True, mono=True):
        if grad:
            return self.net(obs, mono)
        with torch.no_grad():
            return self.net(obs, mono)

    def backward_policy(self, critic, obs):
        muss, stdss, betas = self.net.get_intermediate(obs)
        actss, logpss, _ = esmb_sample(muss, stdss, betas, False)
        obss = torch.unsqueeze(obs, dim=1).expand(-1, actss.shape[1], -1)
        qvaluess = critic.forward(obss, actss)
        l_pri = (torch.sum(self.alpha_coe(logpss, grad=False) - qvaluess, dim=-1)).mean()
        t = qvaluess - torch.max(qvaluess, -1, True).values
        v = torch.where(t == 0., 1., 0.) - betas
        l_frep = (v * v).sum(-1).mean()
        l = l_frep + l_pri
        l.backward()
        pass

    def backward_alpha(self, obs):
        _, logps = self.forward(obs, grad=False)
        loss_alpha = -(self.alpha_coe(logps + self.tar_ent)).mean()
        loss_alpha.backward()
        pass

