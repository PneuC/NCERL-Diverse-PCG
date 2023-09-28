import json
import importlib
from analysis.tests import *
from src.drl.ac_agents import SAC
from src.drl.ac_models import SoftActor, SoftDoubleClipCriticQ
from src.drl.nets import GaussianMLP, ObsActMLP
from src.drl.egsac.egsac_trainer import SyncOffPolicyTrainer
from src.gan.gans import nz
from src.drl.rep_mem import ReplayMem
from src.utils.filesys import auto_dire, getpath
from src.env.environments import make_vec_offrew_env
from src.utils.misc import record_time


def set_EGSAC_parser(parser):
    parser.add_argument(
        '--n_workers', type=int, default=20,
        help='Number of parallel environments.'
    )
    parser.add_argument(
        '--eplen', type=int, default=50,
        help='Maximum nubmer of segments to generate in the generation enviroment.'
    )
    parser.add_argument(
        '--budget', type=int, default=int(1e6),
        help='Total time steps (frames) for training SAC designer.'
    )
    parser.add_argument('--gamma', type=float, default=0.9)
    parser.add_argument('--tar_entropy', type=float, default=-nz)
    parser.add_argument('--tau', type=float, default=0.02)
    parser.add_argument('--update_freq', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--mem_size', type=int, default=int(1e6))
    parser.add_argument(
        '--gpuid', type=int, default=0,
        help='ID of GPU to train the SAC designer. CPU will be used if gpuid < 0'
    )
    parser.add_argument(
        '--rfunc_name', type=str, default='default',
        help='Name of the file where the reward function located. '
             'The file must be put in the \'src.reward_functions\' package.'
    )
    parser.add_argument(
        '--path', type=str, default='',
        help='Path relateed to \'/training_data\'to save the training log. '
             'If not specified, a new folder named exp{id} will be created.'
    )
    parser.add_argument(
        '--play_style', type=str, default='Runner',
        help='Path relateed to \'/training_data\'to save the training log. '
             'If not specified, a new folder named exp{id} will be created.'
    )
    parser.add_argument('--init_n', action='store_true')
    parser.add_argument(
        '--check_points', type=int, nargs='+',
        help='check points to save deisigner, specified by the number of time steps.'
    )
    parser.add_argument('--gen_period', type=int, default=20000, help='Period of saving level generation results')
    parser.add_argument('--periodic_gen_num', type=int, default=200, help='Number of levels to be generated for each evaluation')
    parser.add_argument('--actor_hiddens', type=int, nargs='+', default=[256, 256], help='List of number of units in each hideen layer of actor net')
    parser.add_argument('--critic_hiddens', type=int, nargs='+', default=[256, 256], help='List of number of units in each hideen layer of critic net')

def train_EGSAC(args):
    if not args.path:
        res_path = auto_dire('training_data', 'EGSAC')
    else:
        res_path = getpath('training_data/' + args.path)
        os.makedirs(res_path, exist_ok=True)
    if os.path.exists(f'{res_path}/policy.pth'):
        print(f'Training is skipped due to \`{res_path}\` has been occupied')
        return
    device = 'cpu' if args.gpuid < 0 or not torch.cuda.is_available() else f'cuda:{args.gpuid}'

    rfunc = importlib.import_module('src.env.rfuncs').__getattribute__(f'{args.rfunc_name}')()
    with open(res_path + '/run_config.txt', 'w') as f:
        f.write(time.strftime('%Y-%m-%d %H:%M') + '\n')
        f.write('---------EGSAC---------\n')
        args_strlines = [
            f'{key}={val}\n' for key, val in vars(args).items()
            if key not in {'rfunc_name', 'res_path', 'entry', 'check_points'}
        ]
        f.writelines(args_strlines)
        f.write('-' * 50 + '\n')
        f.write(str(rfunc))
    hist_len = rfunc.get_n()
    # with open(f'{res_path}/hist_len.json', 'w') as f:
    #     json.dump(hist_len, f)
    with open(f'{res_path}/cfgs.json', 'w') as f:
        data = {'N': hist_len, 'gamma': args.gamma, 'h': args.eplen, 'rfunc': args.rfunc_name}
        json.dump(data, f)

    env = make_vec_offrew_env(
        args.n_workers, rfunc, res_path, args.eplen, hist_len=hist_len, play_style=args.play_style,
        log_itv=args.n_workers * 2, device=device, log_targets=['file', 'std'], init_one=not args.init_n
    )

    obs_dim, act_dim = env.observation_space.shape[0], env.action_space.shape[0]
    actor = SoftActor(
        lambda: GaussianMLP(obs_dim, act_dim, args.actor_hiddens), tar_ent=args.tar_entropy
    )
    critic = SoftDoubleClipCriticQ(
        lambda : ObsActMLP(obs_dim, act_dim, args.critic_hiddens), gamma=args.gamma, tau=args.tau
    )
    with open(f'{res_path}/nn_architecture.txt', 'w') as f:
        f.writelines([
            '-' * 24 + 'Actor' + '-' * 24 + '\n', actor.get_nn_arch_str(),
            '-' * 24 + 'Critic-Q' + '-' * 24 + '\n', critic.get_nn_arch_str()
        ])
    agent = SAC(actor, critic, device)

    d_trainer = SyncOffPolicyTrainer(
        env, args.budget, args.update_freq, args.batch_size, ReplayMem(args.mem_size, device=device),
        res_path, args.check_points
    )
    _, timecost = record_time(d_trainer.train)(agent, args.gen_period, args.periodic_gen_num)

    if args.periodic_gen_num > 0:
        evaluate_gen_log(res_path, args.n_workers)

