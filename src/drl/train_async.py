import importlib
from src.env.logger import *
from src.drl.ac_agents import *
from src.drl.rep_mem import ReplayMem
from src.utils.misc import record_time
from src.utils.filesys import auto_dire
from src.env.environments import AsyncOlGenEnv
from src.drl.trainer import AsyncOffpolicyTrainer
from src.drl.pmoe import PMOESoftActor
from analysis.tests import *
from src.gan.gankits import get_decoder


def set_common_args(parser):
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

def drl_train(foo):
    def __inner(args):
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
            data = {'N': N, 'gamma': args.gamma, 'h': args.eplen, 'rfunc': args.rfunc}
            if args.name == 'MESAC':
                data.update({'m': args.m, 'lambda': args.lbd, 'me_type': args.me_type})
            json.dump(data, f)
        obs_dim, act_dim = env.histlen * nz, nz

        agent = foo(args, path, device, obs_dim, act_dim)

        agent.to(device)
        trainer = AsyncOffpolicyTrainer(
            ReplayMem(args.mem_size, device=device), update_per=args.update_per, batch=args.batch
        )
        trainer.set_loggers(*loggers)
        _, timecost = record_time(trainer.train)(env, agent, args.budget, path, check_points=args.check_points)
    return __inner

############### AsyncSAC ###############
def set_AsyncSAC_parser(parser):
    set_common_args(parser)
    parser.add_argument('--name', type=str, default='AsyncSAC', help='Name of this algorithm.')

@drl_train
def train_AsyncSAC(args, path, device, obs_dim, act_dim):
    actor = SoftActor(
        lambda: GaussianMLP(obs_dim, act_dim, args.actor_hiddens), tar_ent=args.tar_entropy
    )
    critic = SoftDoubleClipCriticQ(
        lambda : ObsActMLP(obs_dim, act_dim, args.critic_hiddens), gamma=args.gamma, tau=args.tau
    )
    with open(f'{path}/nn_architecture.txt', 'w') as f:
        f.writelines([
            '-' * 24 + 'Actor' + '-' * 24 + '\n', actor.get_nn_arch_str(),
            '-' * 24 + 'Critic-Q' + '-' * 24 + '\n', critic.get_nn_arch_str()
        ])
    return SAC(actor, critic, device)

############## NCESAC ##############
def set_NCESAC_parser(parser):
    set_common_args(parser)
    parser.add_argument('--name', type=str, default='NCESAC', help='Name of this algorithm.')
    parser.add_argument('--lbd', type=float, default=0.2, help='Weight of mutual exlusion regularisation')
    parser.add_argument('--m', type=int, default=2, help='Number of ensemble heads in the actor')
    parser.add_argument('--me_type', type=str, default='clip', choices=['log', 'clip', 'logclip'], help='Type of mutual exclusion regularisation')
    parser.add_argument('--actor_net_type', type=str, default='mlp', choices=['mlp', 'conv'], help='Type of actor\'s NN')

@drl_train
def train_NCESAC(args, path, device, obs_dim, act_dim):
    me_reg, actor_nn_constructor = None, None
    if args.me_type == 'log':
        me_reg = LogWassersteinExclusion(args.lbd)
    elif args.me_type == 'clip':
        me_reg = ClipExclusion(args.lbd)
    elif args.me_type == 'logclip':
        me_reg = LogClipExclusion(args.lbd)
    if args.actor_net_type == 'conv':
        actor_nn_constructor = lambda: EsmbGaussianConv(
            obs_dim, act_dim, args.actor_hiddens, args.actor_hiddens, args.m
        )
    elif args.actor_net_type == 'mlp':
        actor_nn_constructor = lambda: EsmbGaussianMLP(
            obs_dim, act_dim, args.actor_hiddens, args.actor_hiddens, args.m
        )
    actor = MERegMixSoftActor(actor_nn_constructor, me_reg, tar_ent=args.tar_entropy)
    critic = MERegSoftDoubleClipCriticQ(
        lambda : ObsActMLP(obs_dim, act_dim, args.critic_hiddens),
        gamma=args.gamma, tau=args.tau
    )
    critic_U = MERegDoubleClipCriticW(
        lambda : ObsActMLP(obs_dim, act_dim, args.critic_hiddens),
        gamma=args.gamma, tau=args.tau
    )
    with open(f'{path}/nn_architecture.txt', 'w') as f:
        f.writelines([
            '-' * 24 + 'Actor' + '-' * 24 + '\n', actor.get_nn_arch_str(),
            '-' * 24 + 'Critic-Q' + '-' * 24 + '\n', critic.get_nn_arch_str(),
            '-' * 24 + 'Critic-U' + '-' * 24 + '\n', critic_U.get_nn_arch_str()
        ])
    return MESAC(actor, critic, critic_U, device)


############## PMOESAC ##############
def set_PMOESAC_parser(parser):
    set_common_args(parser)
    parser.add_argument('--name', type=str, default='PMOESAC', help='Name of this algorithm.')
    parser.add_argument('--m', type=int, default=5, help='Number of ensemble heads in the actor')

@drl_train
def train_PMOESAC(args, path, device, obs_dim, act_dim):
    actor = PMOESoftActor(
        lambda: EsmbGaussianMLP(obs_dim, act_dim, args.actor_hiddens, args.actor_hiddens, args.m),
        tar_ent=args.tar_entropy
    )
    critic = SoftDoubleClipCriticQ(
        lambda : ObsActMLP(obs_dim, act_dim, args.critic_hiddens),
        gamma=args.gamma, tau=args.tau
    )
    with open(f'{path}/nn_architecture.txt', 'w') as f:
        f.writelines([
            '-' * 24 + 'Actor' + '-' * 24 + '\n', actor.get_nn_arch_str(),
            '-' * 24 + 'Critic-Q' + '-' * 24 + '\n', critic.get_nn_arch_str()
        ])
    return SAC(actor, critic, device)

