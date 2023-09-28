import math

import torch
from torch import nn
from torch.optim.optimizer import Optimizer


###### Borrowed from https://github.com/heykeetae/Self-Attention-GAN/blob/master/sagan_models.py ######
class SelfAttn(nn.Module):
    """ Self attention Layer"""

    def __init__(self, in_dim):
        super(SelfAttn, self).__init__()
        self.chanel_in = in_dim
        # self.activation = activation

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps(B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize, C, width, height = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B X CX(N)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)  # B X C x (*W*H)
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        attention = self.softmax(energy)  # BX (N) X (N)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)  # B X C X N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)

        out = self.gamma * out + x
        return out
        # return out, attention
#######################################################################################################

class DenseNeck(nn.Module):
    def __init__(self, n_channels, growth):
        super().__init__()
        self.main = nn.Sequential(
            nn.BatchNorm2d(n_channels),
            nn.Conv2d(n_channels, growth * 4, 1, bias=False),
            nn.BatchNorm2d(growth * 4),
            nn.Conv2d(growth * 4, growth, 3, padding=1, bias=False),
        )

    def forward(self, x):
        return torch.cat((x, self.main(x)), -3)

class DenseTransition(nn.Module):
    def __init__(self, channels, reduction=0.5):
        super().__init__()
        self.main = nn.Sequential(
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, int(channels * reduction), kernel_size=1, bias=False),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        return self.main(x)


class DenseBlock(nn.Module):
    def __init__(self, n_channels, n=8, growth=16):
        super().__init__()
        layers = []
        for i in range(n):
            layers.append(DenseNeck(n_channels, growth))
            n_channels += growth
            pass
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)


#######################################################################################################


class Lion(Optimizer):
    r"""Implements Lion algorithm."""
    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0):
        """Initialize the hyperparameters.
        Args:
            params (iterable): iterable of parameters to optimize or dicts defining parameter groups
            lr (float): learning rate (default: 1e-4)
            betas (Tuple[float, float]): coefficients used for computing running averages of gradient and its square (default: (0.9, 0.99))
            weight_decay (float): weight decay (L2 penalty) (default: 0)
        """
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super(Lion, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(Lion, self).__setstate__(state)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.
        Args:
            closure (callable): A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError('Lion does not support sparse gradients')
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)

                # Get hyperparameters
                lr = group['lr']
                beta1, beta2 = group['betas']
                weight_decay = group['weight_decay']

                # Update biased first moment estimate
                state['step'] += 1
                exp_avg = state['exp_avg']
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

                # Update biased second raw moment estimate
                exp_avg_sq = state['exp_avg_sq']
                exp_avg_sq.mul_(beta2).addcmul_(grad - exp_avg, grad - exp_avg, value=1 - beta2)

                # Compute the bias-corrected first and second moment estimates
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                step_size = lr / bias_correction1

                # Update parameters
                p.addcdiv_(exp_avg, denom, value=-step_size)

                # Weight decay
                if weight_decay != 0:
                    p.data.add_(p.data, alpha=-weight_decay * lr)

        return loss