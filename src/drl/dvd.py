import os
import json
import random
import time
import torch
import importlib
import numpy as np
from math import ceil
from torch import nn
from analysis.tests import evaluate_generator, evaluate_gen_log
from src.drl.ac_agents import SAC
from src.drl.ac_models import SoftActor, SoftDoubleClipCriticQ
from src.drl.nets import GaussianMLP, ObsActMLP
from src.drl.rep_mem import ReplayMem
from src.drl.trainer import AsyncOffpolicyTrainer
from src.env.environments import AsyncOlGenEnv
from src.env.logger import GenResLogger, AsyncCsvLogger, AsyncStdLogger
from src.gan.gankits import sample_latvec, get_decoder
from src.gan.gans import nz
from src.olgen.ol_generator import VecOnlineGenerator
from src.olgen.olg_policy import EnsembleGenPolicy
from src.smb.asyncsimlt import AsycSimltPool
from src.utils.filesys import getpath, auto_dire
from src.utils.misc import record_time


################## Borrowed from https://github.com/jjccero/DvD_TD3 ##################
class TS:
    def __init__(self, arms=None, random_choice=6):
        self.arms = [0.0, 0.5] if arms is None else arms
        self.arm_num = len(self.arms)
        self.alpha = np.ones(self.arm_num, dtype=int)
        self.beta = np.ones(self.arm_num, dtype=int)
        self.arm = 0
        self.choices = 0
        self.random_choice = random_choice

    @property
    def value(self):
        return self.arms[self.arm]

    def update(self, reward):
        self.choices += 1
        if reward:
            self.alpha[self.arm] += 1
        else:
            self.beta[self.arm] += 1

    def sample(self):
        if self.choices < self.random_choice:
            self.arm = np.random.choice(self.arm_num)
        else:
            self.arm = np.argmax(np.random.beta(self.alpha, self.beta))
        return self.value

    def clear(self):
        self.alpha[:] = 1
        self.beta[:] = 1
        self.choices = 0

def l2rbf(m_actions):
    actions = torch.stack(m_actions)
    x1 = actions.unsqueeze(0).repeat_interleave(actions.shape[0], 0)
    x2 = actions.unsqueeze(1).repeat_interleave(actions.shape[0], 1)
    d2 = torch.square(x1 - x2)
    l2 = torch.var(actions, dim=0).detach() + 1e-8
    return (d2 / (2 * l2)).mean(-1)

class LogDet(nn.Module):
    def __init__(self, beta=0.99):
        super(LogDet, self).__init__()
        self.beta = beta

    def forward(self, embeddings):
        d = l2rbf(embeddings)
        K = (-d).exp()
        K_ = self.beta * K + (1 - self.beta) * torch.eye(len(embeddings), device=K.device)
        L = torch.linalg.cholesky(K_)
        log_det = 2 * torch.log(torch.diag(L)).sum()
        return log_det
#######################################################################################


class PESACAgent:
    def __init__(self, subs, device='cpu'):
        self.subs = subs
        self.device = device
        self.to(device)
        self.i = 0
        self.test = False

    def to(self, device):
        for sub in self.subs:
            sub.to(device)
        self.device = device

    def update(self, obs, acts, rews, ops):
        for sub in self.subs:
            sub.update(obs, acts, rews, ops)
        pass

    def next(self):
        self.i += 1
        self.i %= len(self.subs)

    def make_decision(self, obs, **kwargs):
        if self.test:
            sub = self.subs[random.randrange(0, len(self.subs))]
        else:
            sub = self.subs[self.i]
        a, _ = sub.actor.forward(
            torch.tensor(obs, dtype=torch.float, device=self.device),
            grad=False, **kwargs
        )
        return a.squeeze().cpu().numpy()


class DvDAgent(PESACAgent):
    def __init__(self, subs, phi_batch=20, device='cpu'):
        super().__init__(subs, device)
        self.div_coe = 0.0
        self.phi_batch = phi_batch
        self.log_det = LogDet()
        self.bandit = TS()

    def div_loss(self, obs):
        o = obs[:self.phi_batch]
        embeddings = [sub.actor.forward(o)[0].flatten() for sub in self.subs]
        return self.log_det(embeddings).exp()

    def update(self, obs, acts, rews, ops):
        for sub in self.subs:
            sub.actor.zero_grads()
        ldiv = self.div_coe * self.div_loss(obs)
        ldiv.backward()
        for sub in self.subs:
            sub.actor.backward_policy(sub.critic, obs, (1 - self.div_coe))
            sub.actor.backward_alpha(obs)
            sub.actor.grad_step()
            sub.critic.zero_grads()
            sub.critic.backward_mse(sub.actor, obs, acts, rews, ops)
            sub.critic.grad_step()
            sub.critic.update_tarnet()

    def adapt_div_coe(self, delta):
        self.bandit.update(delta > 0)
        self.div_coe = self.bandit.sample()
        print('Lambda: %.4g' % self.div_coe)


class DvDTrainer(AsyncOffpolicyTrainer):
    def __init__(self, rep_mem:ReplayMem=None, update_per=1, batch=256, eval_itv=20000, eval_num=50):
        super().__init__(rep_mem, update_per, batch)
        self.eval_logger = None
        self.mean_return = 0
        self.env = None
        self.agent = None
        self.eval_itv = eval_itv if eval_itv >= 0 else eval_itv
        self.eval_num = eval_num if eval_num >= 0 else eval_num

    def train(self, env:AsyncOlGenEnv, agent: DvDAgent, budget, path, check_points=None):
        self._reset()
        self.env = env
        self.agent = agent

        o = self.env.reset()
        for logger in self.loggers:
            if logger.__class__.__name__ == 'GenResLogger':
                self.agent.test = True
                logger.on_episode(self.env, agent, 0)
                self.agent.test = False
        eval_horizon = self.eval_itv
        self.__eval()
        while self.steps < budget:
            a = agent.make_decision(o)
            o, done = env.step(a)
            self.steps += 1
            if done:
                model_credits = ceil(1.25 * env.eplen / self.update_per)
                self._update(model_credits, env, agent)
                agent.next()
            if self.steps >= eval_horizon:
                self.__eval()
                eval_horizon += self.eval_itv

        self._update(0, env, agent, close=True)
        for i, sub in enumerate(self.agent.subs):
            torch.save(sub.actor.net, getpath(f'{path}/policy{i}.pth'))

    def __eval(self):
        if self.steps > 0:
            transitions, rewss = self.env.rollout(wait=True)
            self.rep_mem.add_transitions(transitions)
            self.num_trans += len(transitions)
        self.agent.test = True
        it = 0
        o = self.env.reset()
        rewss = []
        while it < self.eval_num:
            a = self.agent.make_decision(o)
            o, done = self.env.step(a)
            if done:
                rewss += self.env.rollout()[1]
                it +=1
        rewss += self.env.rollout(wait=True)[1]
        rewss = np.array([[v for v in rews.values()] for rews in rewss])
        mean_return = float(np.sum(rewss, axis=1).mean())
        if self.steps > 0:
            self.agent.adapt_div_coe(mean_return - self.mean_return)
        self.mean_return = mean_return
        self.agent.test = False

    def _update(self, model_credits, env, agent, close=False):
        transitions, rewss = env.close() if close else env.rollout()
        self.rep_mem.add_transitions(transitions)
        self.num_trans += len(transitions)
        if len(self.rep_mem) > self.batch:
            t = 1
            while self.num_trans >= (self.num_updates + 1) * self.update_per:
                agent.update(*self.rep_mem.sample(self.batch))
                self.num_updates += 1
                if not close and t == model_credits:
                    break
                t += 1
        for logger in self.loggers:
            loginfo = self._pack_loginfo(rewss)
            if logger.__class__.__name__ == 'GenResLogger':
                agent.test = True
                logger.on_episode(env, agent, self.steps)
                agent.test = False
            else:
                logger.on_episode(**loginfo, close=close)

    def _reset(self):
        self.mean_return = 0
        self.start_time = time.time()
        self.steps = 0
        self.num_trans = 0
        self.num_updates = 0
        self.env = None
        self.agent = None


####################### Comand Line Configuration #######################


def set_DvDSAC_parser(parser):
    parser.add_argument('--n_workers', type=int, default=20, help='Number of max_parallel processes in the environment.')
    parser.add_argument('--queuesize', type=int, default=25, help='Size of waiting queue of the environment.')
    parser.add_argument('--eplen', type=int, default=50, help='Episode length of the environment.')
    parser.add_argument('--budget', type=int, default=int(1e6), help='Total time steps of training.')
    parser.add_argument('--gamma', type=float, default=0.9, help='RL parameter')
    parser.add_argument('--tar_entropy', type=float, default=-nz, help='SAC parameter, taget entropy')
    parser.add_argument('--tau', type=float, default=0.02, help='SAC parameter, taget net smooth coefficient')
    parser.add_argument('--update_per', type=int, default=2, help='Do one update (with one batch) per how many collected transitions')
    parser.add_argument('--batch', type=int, default=256, help='Batch size for one update')
    parser.add_argument('--mem_size', type=int, default=int(1e6), help='Size of replay memory')
    parser.add_argument('--gpuid', type=int, default=0, help='ID of GPU to train the policy. CPU will be used if gpuid < 0')
    parser.add_argument('--rfunc', type=str, default='default', help='Name of the reward function in src/env/rfuncs.py')
    parser.add_argument('--path', type=str, default='', help='Path related to \'/training_data\'to save the training logs. If not specified, a new folder named SAC{id} will be created.')
    parser.add_argument('--actor_hiddens', type=int, nargs='+', default=[256, 256], help='List of number of units in each hideen layer of actor net')
    parser.add_argument('--critic_hiddens', type=int, nargs='+', default=[256, 256], help='List of number of units in each hideen layer of critic net')
    parser.add_argument('--gen_period', type=int, default=20000, help='Period of saving level generation results')
    parser.add_argument('--periodic_gen_num', type=int, default=200, help='Number of levels to be generated for each evaluation')
    parser.add_argument('--redirect', action='store_true', help='If add this, redirect STD log to log.txt')
    parser.add_argument(
        '--check_points', type=int, nargs='+',
        help='check points to save policy, specified by the number of time steps.'
    )
    parser.add_argument('--name', type=str, default='DvDSAC', help='Name of this algorithm.')
    parser.add_argument('--m', type=int, default=5, help='Number of ensemble heads in the actor')
    parser.add_argument('--eval_itv', type=int, default=20000, help='Period of evaluating policy and adapt diversity loss coefficient')
    parser.add_argument('--eval_num', type=int, default=50, help='Number of evaluation times')
    pass


def train_DvDSAC(args):
    def _construct_agent(_args, _path, _device, _obs_dim, _act_dim):
        subs = []
        for i in range(_args.m):
            actor = SoftActor(
                lambda: GaussianMLP(_obs_dim, _act_dim, _args.actor_hiddens), tar_ent=_args.tar_entropy
            )
            critic = SoftDoubleClipCriticQ(
                lambda: ObsActMLP(_obs_dim, _act_dim, _args.critic_hiddens), gamma=_args.gamma, tau=_args.tau
            )
            subs.append(SAC(actor, critic, _device))
        with open(f'{_path}/nn_architecture.txt', 'w') as f:
            f.writelines([
                '-' * 24 + 'Actor' + '-' * 24 + '\n', subs[0].actor.get_nn_arch_str(),
                '-' * 24 + 'Critic-Q' + '-' * 24 + '\n', subs[0].critic.get_nn_arch_str()
            ])
        return DvDAgent(subs, device=_device)

    if not args.path:
        path = auto_dire('training_data', args.name)
    else:
        path = getpath('training_data', args.path)
        os.makedirs(path, exist_ok=True)
    if os.path.exists(f'{path}/policy.pth'):
        print(f'Trainning at <{path}> is skipped as there has a finished trial already.')
        return
    device = 'cpu' if args.gpuid < 0 or not torch.cuda.is_available() else f'cuda:{args.gpuid}'

    evalpool = AsycSimltPool(args.n_workers, args.queuesize, args.rfunc, verbose=False)
    rfunc = importlib.import_module('src.env.rfuncs').__getattribute__(f'{args.rfunc}')()
    env = AsyncOlGenEnv(rfunc.get_n(), get_decoder('models/decoder.pth'), evalpool, args.eplen, device=device)
    loggers = [
        AsyncCsvLogger(f'{path}/log.csv', rfunc),
        AsyncStdLogger(rfunc, 2000, f'{path}/log.txt' if args.redirect else '')
    ]
    if args.periodic_gen_num > 0:
        loggers.append(GenResLogger(path, args.periodic_gen_num, args.gen_period))
    with open(path + '/run_configuration.txt', 'w') as f:
        f.write(time.strftime('%Y-%m-%d %H:%M') + '\n')
        f.write(f'---------{args.name}---------\n')
        args_strlines = [
            f'{key}={val}\n' for key, val in vars(args).items()
            if key not in {'name', 'rfunc', 'path', 'entry'}
        ]
        f.writelines(args_strlines)
        f.write('-' * 50 + '\n')
        f.write(str(rfunc))
    N = rfunc.get_n()
    with open(f'{path}/cfgs.json', 'w') as f:
        data = {'N': N, 'gamma': args.gamma, 'h': args.eplen, 'rfunc': args.rfunc, 'm': args.m}
        json.dump(data, f)
    obs_dim, act_dim = env.histlen * nz, nz

    agent = _construct_agent(args, path, device, obs_dim, act_dim)

    agent.to(device)
    trainer = DvDTrainer(
        ReplayMem(args.mem_size, device=device), update_per=args.update_per, batch=args.batch,
        eval_itv=args.eval_itv, eval_num=args.eval_num
    )
    trainer.set_loggers(*loggers)
    _, timecost = record_time(trainer.train)(env, agent, args.budget, path, check_points=args.check_points)

