import importlib
from src.utils.filesys import auto_dire
from src.utils.misc import record_time
from src.env.logger import *
from src.env.environments import SingleProcessOLGenEnv
from src.drl.ac_agents import *
from src.drl.rep_mem import ReplayMem
from src.drl.trainer import SinProcOffpolicyTrainer
from analysis.tests import *
from src.gan.gankits import get_decoder

def set_common_args(parser):
    parser.add_argument('--n_workers', type=int, default=20, help='Number of workers for evaluation.')
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

        rfunc = importlib.import_module('src.env.rfuncs').__getattribute__(f'{args.rfunc}')()
        env = SingleProcessOLGenEnv(rfunc, get_decoder('models/decoder.pth'), args.eplen, device=device)
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
            json.dump(data, f)
        obs_dim, act_dim = env.hist_len * nz, nz

        agent = foo(args, path, device, obs_dim, act_dim)

        agent.to(device)
        trainer = SinProcOffpolicyTrainer(
            ReplayMem(args.mem_size, device=device), update_per=args.update_per, batch=args.batch
        )
        trainer.set_loggers(*loggers)
        _, timecost = record_time(trainer.train)(env, agent, args.budget, path)

    return __inner

############### SAC ###############
def set_SAC_parser(parser):
    set_common_args(parser)
    parser.add_argument('--name', type=str, default='SAC', help='Name of this algorithm.')

@drl_train
def train_SAC(args, path, device, obs_dim, act_dim):
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


