import os
import random
import numpy as np
from src.gan.gankits import *
from src.utils.filesys import getpath
from src.utils.img import make_img_sheet
from src.utils.datastruct import RingQueue
from src.olgen.olg_policy import RLGenPolicy, RandGenPolicy
from src.smb.level import lvlhcat, save_batch


def rand_gen_levels(n=100, h=50, dest_path=''):
    levels = []
    latvecs = []
    decoder = get_decoder('models/decoder.pth', 'cuda:0')
    init_arxv = np.load(getpath('smb/init_latvecs.npy'))
    for _ in range(n):
        z0 = init_arxv[random.randrange(0, len(init_arxv))]
        z0 = torch.tensor(z0, device='cuda:0', dtype=torch.float)
        z = torch.cat([z0, sample_latvec(h, 'cuda:0')], dim=0)
        lvl = lvlhcat(process_onehot(decoder(z)))
        levels.append(lvl)
        latvecs.append(z.cpu().numpy())
    if dest_path:
        save_batch(levels, dest_path)
        np.save(getpath(dest_path), np.stack(latvecs))
    return levels, np.stack(latvecs)

def generate_levels(policy, dest_folder='', batch_name='samples.lvls', n=200, h=50, parallel=64, save_img=False):
    levels = []
    latvecs = []
    obs_queues = [RingQueue(policy.n) for _ in range(parallel)]
    init_arxv = np.load(getpath('smb/init_latvecs.npy'))
    decoder = get_decoder('models/decoder.pth', 'cuda:0')
    while len(levels) < n:
        veclists = [[] for _ in range(parallel)]
        for queue, veclist in zip(obs_queues, veclists):
            queue.clear()
            init_latvec = init_arxv[random.randrange(0, len(init_arxv))]
            queue.push(init_latvec)
            veclist.append(init_latvec)
        for _ in range(h):
            obs = np.stack([np.concatenate(queue.to_list()) for queue in obs_queues])
            actions = policy.step(obs)
            for queue, veclist, action in zip(obs_queues, veclists, actions):
                queue.push(action)
                veclist.append(action)
        for veclist in veclists:
            latvecs.append(np.stack(veclist))
            z = torch.tensor(latvecs[-1], device='cuda:0').view(-1, nz, 1, 1)
            lvl = lvlhcat(process_onehot(decoder(z)))
            levels.append(lvl)
        # print(f'{len(levels)}/{n} generated')
    if dest_folder:
        os.makedirs(getpath(dest_folder), exist_ok=True)
        save_batch(levels[:n], getpath(dest_folder, batch_name))
        if save_img:
            for i, lvl in enumerate(levels[:n]):
                lvl.to_img(f'{dest_folder}/lvl-{i}.png')
    return levels[:n]


def make_samples(path, n=12, h=20, space=12):
    plc = RLGenPolicy.from_path(path)
    levels = generate_levels(plc, n=n, h=h)
    imgs = [lvl.to_img() for lvl in levels]
    make_img_sheet(imgs, ncols=1, y_margin=space, save_path=f'{path}/samples.png')
    pass

if __name__ == '__main__':
    pass
