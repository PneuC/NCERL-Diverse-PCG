from src.drl.ac_models import *


class ActCrtAgent:
    def __init__(self, actor, critic, device='cpu'):
        self.actor = actor
        self.critic = critic
        self.device = device
        self.to(device)

    def to(self, device):
        self.actor.to(device)
        self.critic.to(device)
        self.device = device

    @abstractmethod
    def update(self, obs, acts, rews, ops):
        pass

    def make_decision(self, obs, **kwargs):
        a, _ = self.actor.forward(
            torch.tensor(obs, dtype=torch.float, device=self.device),
            grad=False, **kwargs
        )
        return a.squeeze().cpu().numpy()


class SAC(ActCrtAgent):
    def __init__(self, actor: SoftActor, critic: SoftDoubleClipCriticQ, device='cpu'):
        super(SAC, self).__init__(actor, critic, device)

    def update(self, obs, acts, rews, ops):
        self.actor.zero_grads()
        self.actor.backward_policy(self.critic, obs)
        self.actor.backward_alpha(obs)
        self.actor.grad_step()
        self.critic.zero_grads()
        self.critic.backward_mse(self.actor, obs, acts, rews, ops)
        self.critic.grad_step()
        self.critic.update_tarnet()
        pass


class MESAC(ActCrtAgent):
    def __init__(self, actor: MERegMixSoftActor, critic: SoftDoubleClipCriticQ, criticU: MERegDoubleClipCriticW, device='cpu'):
        self.criticU = criticU
        super(MESAC, self).__init__(actor, critic, device)
        self.to(device)

    def to(self, device):
        self.actor.to(device)
        self.critic.to(device)
        self.criticU.to(device)
        self.device = device

    def update(self, obs, acts, rews, ops):
        self.actor.zero_grads()
        self.actor.backward_policy(self.critic, obs)
        if self.actor.me_reg.lbd > 0.:
            self.actor.backward_me_reg(self.criticU, obs)
        self.actor.backward_alpha(obs)
        self.actor.grad_step()
        self.critic.zero_grads()
        self.critic.backward_mse(self.actor, obs, acts, rews, ops)
        self.critic.grad_step()
        self.critic.update_tarnet()
        if self.actor.me_reg.lbd > 0.:
            self.criticU.zero_grads()
            self.criticU.backward_mse(self.actor, obs, acts, rews, ops)
            self.criticU.grad_step()
            self.criticU.update_tarnet()
        pass



if __name__ == '__main__':

    pass
