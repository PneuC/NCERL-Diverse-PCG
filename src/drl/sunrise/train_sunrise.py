import json
import importlib
import src.rlkit.torch.pytorch_util as ptu
from analysis.tests import *
from src.env.environments import AsyncOlGenEnv
from src.env.logger import AsyncCsvLogger, AsyncStdLogger, GenResLogger
from src.gan.gankits import *
from src.drl.sunrise.sunrise_adaption import AsyncOffPolicyALgo, MaskedRepMem
from src.olgen.ol_generator import OnlineGenerator
from src.olgen.olg_policy import EnsembleGenPolicy
from src.rlkit.torch.sac.policies import TanhGaussianPolicy, MakeDeterministic
from src.rlkit.torch.sac.neurips20_sac_ensemble import NeurIPS20SACEnsembleTrainer
from src.rlkit.torch.networks import FlattenMlp
from src.smb.asyncsimlt import AsycSimltPool
from src.utils.filesys import auto_dire, getpath
from src.utils.misc import record_time

variant = dict(
    algorithm="SAC",
    version="normal",
    layer_size=256,
    replay_buffer_size=int(1e6),
    # algorithm_kwargs=dict(
    #     num_epochs=210,
    #     num_eval_steps_per_epoch=1000,
    #     num_trains_per_train_loop=1000,
    #     num_expl_steps_per_train_loop=1000,
    #     min_num_steps_before_training=1000,
    #     max_path_length=1000,
    #     batch_size=args.batch_size,
    #     save_frequency=args.save_freq,
    # ),
    trainer_kwargs=dict(
        # discount=0.99,
        # soft_target_tau=5e-3,
        target_update_period=1,
        policy_lr=3E-4,
        qf_lr=3E-4,
        reward_scale=1,
        use_automatic_entropy_tuning=True,
    ),
    # num_ensemble=args.num_ensemble,
    # num_layer=args.num_layer,
    # seed=args.seed,
    # ber_mean=args.ber_mean,
    # env=args.env,
    # inference_type=args.inference_type,
    # temperature=args.temperature,
    # log_dir="",
)


def set_SUNRISE_args(parser):
    parser.add_argument('--name', type=str, default='SUNRISE', help='Name of this algorithm')
    parser.add_argument('--n_workers', type=int, default=20, help='Number of max_parallel processes in the environment.')
    parser.add_argument('--queuesize', type=int, default=25, help='Size of waiting queue of the environment.')
    parser.add_argument('--eplen', type=int, default=50, help='Episode length of the environment.')
    parser.add_argument('--budget', type=int, default=int(1e6), help='Total time steps of training.')
    parser.add_argument('--gamma', type=float, default=0.9, help='RL parameter, discount factor')
    # parser.add_argument('--tar_entropy', type=float, default=-nz, help='SAC parameter, taget entropy')
    parser.add_argument('--tau', type=float, default=0.02, help='SAC parameter, taget net smooth coefficient')
    parser.add_argument('--update_per', type=int, default=2,
                        help='Do one update (with one batch) per how many collected transitions')
    # parser.add_argument('--batch', type=int, default=256, help='Batch size for one update')
    parser.add_argument('--mem_size', type=int, default=int(1e6), help='Size of replay memory')
    parser.add_argument('--gpuid', type=int, default=0,
                        help='ID of GPU to train the policy. CPU will be used if gpuid < 0')
    parser.add_argument('--rfunc', type=str, default='default', help='Name of the reward function in src/env/rfuncs.py')
    parser.add_argument('--path', type=str, default='',
                        help='Path related to \'/training_data\'to save the training logs. If not specified, a new folder named SAC{id} will be created.')
    # parser.add_argument('--actor_hiddens', type=int, nargs='+', default=[256, 256], help='List of number of units in each hideen layer of actors net')
    # parser.add_argument('--critic_hiddens', type=int, nargs='+', default=[256, 256], help='List of number of units in each hideen layer of critic net')
    parser.add_argument('--gen_period', type=int, default=20000, help='Period of saving level generation results')
    parser.add_argument('--periodic_gen_num', type=int, default=200,
                        help='Number of levels to be generated for each evaluation')
    parser.add_argument('--redirect', action='store_true', help='If add this, redirect STD log to log.txt')

    # architecture
    parser.add_argument('--num_layer', default=2, type=int)

    # train
    parser.add_argument('--batch_size', default=256, type=int)

    # ensemble
    parser.add_argument('--num_ensemble', default=5, type=int)
    parser.add_argument('--ber_mean', default=0.5, type=float)

    # inference
    parser.add_argument('--inference_type', default=1.0, type=float)

    # corrective feedback
    parser.add_argument('--temperature', default=10.0, type=float)


def get_env(args, device):
    evalpool = AsycSimltPool(args.n_workers, args.queuesize, args.rfunc, verbose=False)
    rfunc = importlib.import_module('src.env.rfuncs').__getattribute__(f'{args.rfunc}')()
    env = AsyncOlGenEnv(rfunc.get_n(), get_decoder('models/decoder.pth'), evalpool, args.eplen, device=device)
    return env

def get_trainer(args, obs_dim, action_dim, path, device):
    M = variant['layer_size']
    num_layer = args.num_layer
    network_structure = [M] * num_layer


    L_qf1, L_qf2, L_target_qf1, L_target_qf2, L_policy = [], [], [], [], []

    for _ in range(args.num_ensemble):
        qf1 = FlattenMlp(
            input_size=obs_dim + action_dim,
            output_size=1,
            hidden_sizes=network_structure,
        )
        qf2 = FlattenMlp(
            input_size=obs_dim + action_dim,
            output_size=1,
            hidden_sizes=network_structure,
        )
        target_qf1 = FlattenMlp(
            input_size=obs_dim + action_dim,
            output_size=1,
            hidden_sizes=network_structure,
        )
        target_qf2 = FlattenMlp(
            input_size=obs_dim + action_dim,
            output_size=1,
            hidden_sizes=network_structure,
        )
        policy = TanhGaussianPolicy(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_sizes=network_structure,
        )

        L_qf1.append(qf1)
        L_qf2.append(qf2)
        L_target_qf1.append(target_qf1)
        L_target_qf2.append(target_qf2)
        L_policy.append(policy)


    trainer = NeurIPS20SACEnsembleTrainer(
        # env=eval_env,
        policy=L_policy,
        qf1=L_qf1,
        qf2=L_qf2,
        target_qf1=L_target_qf1,
        target_qf2=L_target_qf2,
        num_ensemble=args.num_ensemble,
        discount=args.gamma,
        soft_target_tau=args.tau,
        feedback_type=1,
        temperature=args.temperature,
        temperature_act=0,
        expl_gamma=0,
        log_dir=path,
        device=device,
        **variant['trainer_kwargs']
    )
    return trainer
    pass

def get_algo(args, rfunc, device, path):
    algorithm = AsyncOffPolicyALgo(
        MaskedRepMem(args.num_ensemble, args.mem_size, args.ber_mean, device),
        update_per=args.update_per,
        batch=args.batch_size,
        device=device
    )
    loggers = [
        AsyncCsvLogger(f'{path}/log.csv', rfunc),
        AsyncStdLogger(rfunc, 2000, f'{path}/log.txt' if args.redirect else '')
    ]
    if args.periodic_gen_num > 0:
        loggers.append(GenResLogger(path, args.periodic_gen_num, args.gen_period))
    algorithm.set_loggers(*loggers)
    return algorithm

def train_SUNRISE(args):
    if not args.path:
        path = auto_dire('training_data', args.name)
    else:
        path = getpath('training_data/' + args.path)
        try:
            os.makedirs(path)
        except FileExistsError:
            if os.path.exists(f'{path}/policy.pth'):
                print(f'Training is cancelled due to \`{path}\` has been occupied')
                return
    if os.path.exists(f'{path}/policy.pth'):
        print(f'Trainning at <{path}> is skipped as there has a finished trial already.')
        return
    device = 'cpu' if args.gpuid < 0 or not torch.cuda.is_available() else f'cuda:{args.gpuid}'
    rfunc = importlib.import_module('src.env.rfuncs').__getattribute__(f'{args.rfunc}')()

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
        data = {'N': N, 'gamma': args.gamma, 'h': args.eplen, 'rfunc': args.rfunc, 'm': args.num_ensemble}
        json.dump(data, f)

    ptu.set_gpu_mode(True)

    env = get_env(args, device)

    obs_dim, action_dim = env.histlen * nz, nz
    trainer = get_trainer(args, obs_dim, action_dim, path, device)
    algorithm = get_algo(args, rfunc, device, path)
    _, timecost = record_time(algorithm.train)(env, trainer, args.budget, args.inference_type, path)

    pass


if __name__ == '__main__':
    pass
