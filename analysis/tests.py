import os
import csv
import time
import random
from src.smb.level import *
from src.drl.me_reg import *
from src.drl.nets import esmb_sample
from src.utils.filesys import getpath
from src.utils.datastruct import RingQueue
from src.smb.asyncsimlt import AsycSimltPool
from src.env.environments import get_padded_obs
from src.olgen.ol_generator import VecOnlineGenerator, OnlineGenerator
from src.drl.drl_uses import load_cfgs, load_performance
from src.olgen.olg_policy import process_obs, RandGenPolicy, RLGenPolicy, EnsembleGenPolicy


def evaluate_rewards(lvls, rfunc='default', dest_path='', parallel=1, eval_pool=None):
    internal_pool = eval_pool is None
    if internal_pool:
        eval_pool = AsycSimltPool(parallel, rfunc_name=rfunc, verbose=False, test=True)
    res = []
    for lvl in lvls:
        eval_pool.put('evaluate', (0, str(lvl)))
        buffer = eval_pool.get()
        for _, item in buffer:
            res.append([sum(r) for r in zip(*item.values())])
    if internal_pool:
        buffer = eval_pool.close()
    else:
        buffer = eval_pool.get(True)
    for _, item in buffer:
        res.append([sum(r) for r in zip(*item.values())])
    if len(dest_path):
        np.save(dest_path, res)
    return res

def evaluate_mnd(lvls, refs, parallel=2):
    eval_pool = AsycSimltPool(parallel, verbose=False, refs=[str(ref) for ref in refs])
    # m, _ = len(lvls), len(refs)
    res = []
    for lvl in lvls:
        eval_pool.put('mnd_item', str(lvl))
        res += eval_pool.get()
    res += eval_pool.get(wait=True)
    res = np.array(res)
    eval_pool.close()
    return np.mean(res[:, 0]), np.mean(res[:, 1])

def evaluate_mpd(lvls, parallel=2):
    task_datas = [[] for _ in range(parallel)]
    for i, (A, B) in enumerate(combinations(lvls, 2)):
        # lvlA, lvlB = lvls[i * 2], lvls[i * 2 + 1]
        task_datas[i % parallel].append((str(A), str(B)))

    hms, dtws = [], []
    eval_pool = AsycSimltPool(parallel, verbose=False)
    for task_data in task_datas:
        eval_pool.put('mpd', task_data)
    res = eval_pool.get(wait=True)
    for task_hms, _ in res:
        hms += task_hms
        # dtws += task_dtws
    return np.mean(hms) #, np.mean(dtws)

def evaluate_gen_log(path, parallel=5):
    rfunc_name = load_cfgs(path, 'rfunc')
    f = open(getpath(f'{path}/step_tests.csv'), 'w', newline='')
    wrtr = csv.writer(f)
    cols = ['step', 'r-avg', 'r-std', 'mnd-hm', 'mnd-dtw', 'mpd-hm', 'mpd-dtw', '']
    wrtr.writerow(cols)
    start_time = time.time()
    for lvls, name in traverse_batched_level_files(f'{path}/gen_log'):
        step = name[4:]
        rewards = [sum(item) for item in evaluate_rewards(lvls, rfunc_name, parallel=parallel)]
        r_avg, r_std = np.mean(rewards), np.std(rewards)
        # mpd_hm, mpd_dtw = evaluate_mpd(lvls, parallel=parallel)
        mpd = evaluate_mpd(lvls, parallel=parallel)
        line = [step, r_avg, r_std, mpd, '']
        wrtr.writerow(line)
        f.flush()
        print(
            f'{path}: step{step} evaluated in {time.time()-start_time:.1f}s -- '
            + '; '.join(f'{k}: {v}' for k, v in zip(cols, line))
        )
    f.close()
    pass

def evaluate_generator(generator, nr=200, h=50, parallel=5, dest_path=None, additional_info=None, rfunc_name='default'):
    if additional_info is None: additional_info = {}
    ''' Test Reward '''
    lvls = generator.generate(nr, h)
    rewards = [sum(item) for item in evaluate_rewards(lvls, parallel=parallel, rfunc=rfunc_name)]
    r_avg, r_std = np.mean(rewards), np.std(rewards)
    ''' Test MPD '''
    # mpd, _ = evaluate_mpd(lvls, parallel=parallel)
    mpd, *_ = evaluate_mpd(generator.generate(3000*2, h), parallel=parallel)
    res = {
        'r-avg': r_avg, 'r-std': r_std, 'div': mpd,
    }
    res.update(additional_info)
    if dest_path:
        with open(getpath(dest_path), 'w', newline='') as f:
            keys = [k for k in res.keys()]
            wrtr = csv.writer(f)
            wrtr.writerow(keys + [''])
            wrtr.writerow([res[k] for k in keys] + [''])
    return res
    pass

def evaluate_jmer(training_path, n=1000, max_parallel=None, device='cuda:0'):
    init_vecs = np.load(getpath('smb/init_latvecs.npy'))
    try:
        m, histlen, h, gamma, me_type = load_cfgs(training_path, 'm', 'N', 'h', 'gamma', 'me_type')
    except KeyError:
        return 0.
    mereg_func = LogWassersteinExclusion(1.) if me_type == 'logw' else WassersteinExclusion(1.)
    model = torch.load(getpath(training_path, 'policy.pth'), map_location=device)
    model.requires_grad_(False)
    if max_parallel is None:
        max_parallel = min(n, 512)
    me_regs = []
    obs_queues = [RingQueue(histlen) for _ in range(max_parallel)]
    while len(me_regs) < n:
        size = min(max_parallel, n - len(me_regs))
        mereg_vals, discount = np.zeros([size]), 1.
        veclists = [[] for _ in range(size)]
        for queue, veclist in zip(obs_queues, veclists):
            queue.clear()
            init_latvec = init_vecs[random.randrange(0, len(init_vecs))]
            queue.push(init_latvec)
            veclist.append(init_latvec)
        for _ in range(h):
            obs = np.stack([get_padded_obs(queue.to_list(), histlen) for queue in obs_queues[:size]])
            muss, stdss, betas = model.get_intermediate(process_obs(obs, device))
            mereg_vals += discount * mereg_func.forward(muss, stdss, betas).squeeze().cpu().numpy()
            discount *= gamma
            actions, _ = esmb_sample(muss, stdss, betas)
            for queue, veclist, action in zip(obs_queues, veclists, actions.cpu().numpy()):
                queue.push(action)
                veclist.append(action)
        me_regs += mereg_vals.tolist()
    return me_regs

def evaluate_baseline(*rfuncs, parallel=4):
    nr, md, nd, h = 100, 1000, 200, 50
    gen_policy = RandGenPolicy()
    olgenerator = OnlineGenerator(gen_policy)
    lvls, refs = olgenerator.generate(md, h), olgenerator.generate(nd, h)
    divs_h, divs_js = evaluate_mnd(lvls, refs, parallel=parallel)
    keys, vals = ['d-h', 'd-js'], [divs_h, divs_js]
    print(f'Diversity of baseline generator: Hamming {divs_h:.2f}; TPJS {divs_js:.2f}')
    for rfunc in rfuncs:
        try:
            print(f'Start to evaluate {rfunc}')
            start_time = time.time()
            lvls = olgenerator.generate(nr, h)
            rewards = [sum(item) for item in evaluate_rewards(lvls, parallel=parallel, rfunc=rfunc)]
            keys.append(rfunc)
            vals.append(np.mean(rewards))
            print(f'Evaluation for {rfunc} finished in {time.time()-start_time:.2f}s')
            print(f'Evaluation results for {rfunc}: {vals[-1]:.2f}')
        except AttributeError:
            continue
    with open(getpath('training_data', 'baselines.csv'), 'w', newline='') as f:
        wrtr = csv.writer(f)
        wrtr.writerow(keys)
        wrtr.writerow(vals)

def sample_initial():
    playable_latvecs = np.load(getpath('smb/init_latvecs.npy'))
    indexes = random.sample([*range(len(playable_latvecs))], 500)
    z = playable_latvecs[indexes, :]

    np.save(getpath('analysis/initial_seg.npy'), z)
    pass

def generate_levels_for_test(h=25):
    init_set = np.load(getpath('analysis/initial_seg.npy'))
    def _generte_one(policy, path):
        try:
            start = time.time()
            generator = VecOnlineGenerator(policy, vec_num=len(init_set))
            fd, _ = os.path.split(getpath(path))
            os.makedirs(fd, exist_ok=True)
            generator.re_init(init_set)
            lvls = generator.generate(len(init_set), h, rand_init=False)
            save_batch(lvls, path)
            print('Save to', path, '%.2fs' % (time.time() - start))
        except FileNotFoundError as e:
            print(e)
    for l, m in product(['0.0', '0.1', '0.2', '0.3', '0.4', '0.5'], [2, 3, 4, 5]):
        for i in range(1, 6):
            pi_path = f'training_data/varpm-fhp/l{l}_m{m}/t{i}'
            _generte_one(RLGenPolicy.from_path(pi_path), f'test_data/varpm-fhp/l{l}_m{m}/t{i}/samples.lvls')
            pi_path = f'training_data/varpm-lgp/l{l}_m{m}/t{i}'
            _generte_one(RLGenPolicy.from_path(pi_path), f'test_data/varpm-lgp/l{l}_m{m}/t{i}/samples.lvls')
    for algo in ['sac', 'egsac', 'asyncsac', 'pmoe']:
        for i in range(1, 6):
            pi_path = f'training_data/{algo}/fhp/t{i}'
            _generte_one(RLGenPolicy.from_path(pi_path), f'test_data/{algo}/fhp/t{i}/samples.lvls')
            pi_path = f'training_data/{algo}/lgp/t{i}'
            _generte_one(RLGenPolicy.from_path(pi_path), f'test_data/{algo}/lgp/t{i}/samples.lvls')
    for algo in ['sunrise', 'dvd']:
        for i in range(1, 5):
            pi_path = f'training_data/{algo}/fhp/t{i}'
            _generte_one(EnsembleGenPolicy.from_path(pi_path), f'test_data/{algo}/fhp/t{i}/samples.lvls')
            pi_path = f'training_data/{algo}/lgp/t{i}'
            _generte_one(EnsembleGenPolicy.from_path(pi_path), f'test_data/{algo}/lgp/t{i}/samples.lvls')
        pass


if __name__ == '__main__':
    generate_levels_for_test()
