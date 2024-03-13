import csv
import time

import torch

from plots import print_compare_tab_nonrl
from src.gan.gankits import *
from src.smb.level import *
from itertools import combinations, chain
from src.utils.filesys import getpath
from src.smb.asyncsimlt import AsycSimltPool


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
    return np.mean(hms)

def evaluate_gen_log(path, rfunc_name, parallel=5):
    f = open(getpath(f'{path}/step_tests.csv'), 'w', newline='')
    wrtr = csv.writer(f)
    cols = ['step', 'r-avg', 'r-std', 'diversity']
    wrtr.writerow(cols)
    start_time = time.time()
    for lvls, name in traverse_batched_level_files(f'{path}/gen_log'):
        step = name[4:]
        rewards = [sum(item) for item in evaluate_rewards(lvls, rfunc_name, parallel=parallel)]
        r_avg, r_std = np.mean(rewards), np.std(rewards)
        mpd = evaluate_mpd(lvls, parallel=parallel)
        line = [step, r_avg, r_std, mpd]
        wrtr.writerow(line)
        f.flush()
        print(
            f'{path}: step{step} evaluated in {time.time()-start_time:.1f}s -- '
            + '; '.join(f'{k}: {v}' for k, v in zip(cols, line))
        )
    f.close()
    pass




if __name__ == '__main__':
    print_compare_tab_nonrl()

    # arr = [[1, 2], [1, 2]]
    # arr = [*chain(*arr)]
    # print(arr)
    # for i in range(5):
    #     path = f'training_data/GAN{i}'
        # lvls = []
        # init_lateves = torch.tensor(np.load(getpath('analysis/initial_seg.npy')), device='cuda:0')
        # decoder = get_decoder(device='cuda:0')
        # init_seg_onehots = decoder(init_lateves.view(*init_lateves.shape, 1, 1))
        # gan = get_decoder(f'{path}/decoder.pth', device='cuda:0')
        # for init_seg_onehot in init_seg_onehots:
        #     seg_onehots = gan(sample_latvec(25, device='cuda:0'))
        #     a = init_seg_onehot.view(1, *init_seg_onehot.shape)
        #     b = seg_onehots
        #     # print(a.shape, b.shape)
        #     segs = process_onehot(torch.cat([a, b], dim=0))
        #     level = lvlhcat(segs)
        #     lvls.append(level)
        # save_batch(lvls, getpath(path, 'samples.lvls'))
        # lvls = load_batch(f'{path}/samples.lvls')[:15]
        # imgs = [lvl.to_img() for lvl in lvls]
        # make_img_sheet(imgs, 1, save_path=f'generation_results/GAN/trial{i+1}/sample_lvls.png')

    # ts = torch.tensor([
    #     [[0, 0], [0, 1], [0, 2]],
    #     [[1, 0], [1, 1], [1, 2]],
    # ])
    # print(ts.shape)
    # print(ts[[*range(2)], [1, 2], :])
    # task = 'fhp'
    # parallel = 50
    # samples = []
    # for algo in ['dvd', 'egsac', 'pmoe', 'sunrise', 'asyncsac', 'sac']:
    #     for t in range(5):
    #         lvls = load_batch(getpath('test_data', algo, task, f't{t + 1}', 'samples.lvls'))
    #         samples += lvls
    # for l in ['0.0', '0.1', '0.2', '0.3', '0.4', '0.5']:
    #     for t in range(5):
    #         lvls = load_batch(getpath('test_data', f'varpm-{task}', f'l{l}_m5', f't{t + 1}', 'samples.lvls'))
    #         samples += lvls
    #
    # # task_datas = [[] for _ in range(parallel)]
    # # for i, (A, B) in enumerate(combinations(samples, 2)):
    # #     lvlA, lvlB = lvls[i * 2], lvls[i * 2 + 1]
    #     # task_datas[i % parallel].append((str(A), str(B)))
    #
    # distmat = []
    # eval_pool = AsycSimltPool(parallel, verbose=False)
    # for A in samples:
    #     eval_pool.put('mpd', [(str(A), str(B)) for B in samples])
    #     res = eval_pool.get()
    # for task_hms, _ in res:
    #     hms += task_hms
    # np.save(getpath('test_data', f'samples_dists-{task}.npy'), hms)

    # start = time.time()
    # samples = load_batch(getpath('test_data/varpm-fhp/l0.0_m2/t1/samples.lvls'))
    # distmat = []
    # for a in samples:
    #     dist_list = []
    #     for b in samples:
    #         dist_list.append(hamming_dis(a, b))
    #     distmat.append(dist_list)
    # print(time.time() - start)
    pass
