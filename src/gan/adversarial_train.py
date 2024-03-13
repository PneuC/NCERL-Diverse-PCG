import os
import csv
import time
import torch
import random
import torch.nn.functional as F
from copy import deepcopy
from src.smb.level import *
from torch.optim import Adam
from src.utils.mymath import crowdivs
from src.utils.filesys import auto_dire
from src.utils.img import make_img_sheet
from src.utils.datastruct import batched_iter
from src.gan.gans import SAGenerator, SADiscriminator
from src.gan.gankits import process_onehot, sample_latvec


def get_gan_train_data():
    H, W = MarioLevel.height, MarioLevel.seg_width
    data = []
    for lvl, _ in traverse_level_files('smb/levels'):
        num_lvl = lvl.to_num_arr()
        _, length = num_lvl.shape
        for s in range(length - W):
            seg = num_lvl[:, s: s+W]
            onehot = np.zeros([MarioLevel.n_types, H, W])
            xs = [seg[i, j] for i, j in product(range(H), range(W))]
            ys = [k // W for k in range(H * W)]
            zs = [k % W for k in range(H * W)]
            onehot[xs, ys, zs] = 1
            data.append(onehot)
    return data

def set_GAN_parser(parser):
    parser.add_argument('--batch_size', type=int, default=128, help='input batch size')
    parser.add_argument('--niter', type=int, default=2000, help='number of iterations to training GAN')
    parser.add_argument('--eval_itv', type=int, default=10, help='Interval (in unit of iteration) of evaluating and logging')
    parser.add_argument('--save_itv', type=int, default=100, help='Interval (in unit of iteration) of saving agent and samples')
    parser.add_argument('--repeatD', type=int, default=5, help='repeatly training D for how many time for each iteration')
    parser.add_argument('--repeatG', type=int, default=1, help='repeatly training G for how many time for each iteration')
    parser.add_argument('--lrD', type=float, default=4e-4, help='learning rate for D, default=4e-4')
    parser.add_argument('--lrG', type=float, default=1e-4, help='learning rate for G, default=1e-4')
    parser.add_argument('--regD', type=float, default=3e-4, help='weight_decay for D, default=1e-3')
    parser.add_argument('--regG', type=float, default=0., help='weight_decay for G, default=0')
    parser.add_argument('--beta1', type=float, default=0., help='beta1 parameter for Adam optimiser, default=0.')
    parser.add_argument('--beta2', type=float, default=0.9, help='beta2 parameter for Adam optimiser, default=0.9')
    parser.add_argument('--gpuid', type=int, default=0, help='id of gpu. If smaller than 0, use cpu')
    parser.add_argument('--res_path', type=str, default='', help='root_folder to store training data')
    parser.add_argument('--weight_clip', type=float, default=0., help='clip weight of dicriminator into [-this, this] if this > 0')
    parser.add_argument('--noise', type=str, default='uniform', help='Type of noise distribution')
    parser.add_argument('--base_channels', type=int, default=32, help='Number of channels of the layer with least channels')

def train_GAN(args):
    def evaluate_diversity(levels_):
        hamming_tab = np.array([[hamming_dis(l1, l2) for l1 in levels_] for l2 in levels_])
        tpjs_tab = np.array([[tile_pattern_js_div(l1, l2) for l1 in levels_] for l2 in levels_])
        hamming_divs_ = crowdivs(hamming_tab)
        tpjs_divs_ = crowdivs(tpjs_tab)
        return hamming_divs_, tpjs_divs_

    device = 'cpu' if args.gpuid < 0 or not torch.cuda.is_available() else f'cuda:{args.gpuid}'
    netG = SAGenerator(args.base_channels).to(device)
    netD = SADiscriminator(args.base_channels).to(device)
    optG = Adam(netG.parameters(), lr=args.lrG, betas=(args.beta1, args.beta2), weight_decay=args.regG)
    optD = Adam(netD.parameters(), lr=args.lrD, betas=(args.beta1, args.beta2), weight_decay=args.regD)
    data = get_gan_train_data()
    data = [torch.tensor(item, device=device, dtype=torch.float) for item in data]
    if args.res_path == '':
        res_path = auto_dire('training_data', name='GAN')
    else:
        res_path = getpath('training_data/' + args.res_path)
        try:
            os.makedirs(res_path)
        except FileExistsError:
            print(f'Training cancelled due to decoder.pth already exists in {res_path}')
            return
    with open(getpath(f'{res_path}/NN_architectures.txt'), 'w') as f:
        f.write('=' * 24  +' Generator ' + '=' * 24 + '\n')
        f.write(str(netG))
        f.write('\n' + '=' * 22  +' Discriminator ' + '=' * 22 + '\n')
        f.write(str(netD))

    cfgs = deepcopy(vars(args))
    cfgs.pop('entry')
    cfgs['start-time'] = time.strftime('%Y-%m-%plc %H:%M:%S', time.localtime())
    with open(f'{res_path}/cfgs.csv', 'w') as f:
        w = csv.writer(f)
        w.writerow(['key', 'value', ''])
        w.writerows(list(cfgs.items()))
    # pds.DataFrame.from_dict(cfgs, orient='index', columns=['value']).to_csv(f'{path_}/cfgs.csv')

    start_time = time.time()
    log_target = open(f'{res_path}/logs.csv', 'w')
    log_writer = csv.writer(log_target)
    log_writer.writerow(['Iterations', 'D-real', 'D-fake', 'Divs-hamming', 'Divs-tpjs', 'Time', ''])
    # log_data = []
    for t in range(args.niter):
        random.shuffle(data)
        # Train
        for item, n in batched_iter(data, args.batch_size):
            real = torch.stack(item)
            for _ in range(args.repeatD):
                if args.weight_clip > 0:
                    for p in netD.parameters():
                        p.data.clamp_(-args.weight_clip, args.weight_clip)
                with torch.no_grad():
                    z = sample_latvec(n, device=device, distribuion=args.noise)
                    fake = netG(z)
                l_real = F.relu(1 - netD(real)).mean()
                l_fake = F.relu(netD(fake) + 1).mean()
                optD.zero_grad()
                l_real.backward()
                l_fake.backward()
                optD.step()

            for _ in range(args.repeatG):
                sample_latvec(n, device=device, distribuion=args.noise)
                fake = netG(z)
                optG.zero_grad()
                loss_G = -netD(fake).mean()
                loss_G.backward()
                optG.step()
        # # Evaluate
        # if t % args.eval_itv == (args.eval_itv - 1):
        #     netG.eval()
        #     netD.eval()
        #     with torch.no_grad():
        #         real = torch.stack(data[:min(100, len(data))])
        #         z = sample_latvec(100, device=device, distribuion=args.noise)
        #         fake = netG(z)
        #         y_real = netD(real).mean().item()
        #         y_fake = netD(fake).mean().item()
        #     # hamming_divs, tpjs_divs = evaluate_diversity(process_onehot(fake))
        #
        #     # items = (t+1, y_real, y_fake, hamming_divs, tpjs_divs, time.time() - start_time)
        #     # log_writer.writerow(items)
        #     print(
        #         'Iteration %d, y-real=%.3g, y-fake=%.3g, Hamming-divs: %.5g, TPJS-divs: %.5g, '
        #         'time: %.1fs' % items
        #     )
        #     netD.train()
        #     netG.train()
        if t % args.save_itv == (args.save_itv - 1):
            netG.eval()
            netD.eval()
            with torch.no_grad():
                z = sample_latvec(54, device=device, distribuion=args.noise)
                fake = netG(z)
            levels = process_onehot(fake)
            iteration_path = res_path + f'/iteration{t+1}'
            os.makedirs(iteration_path, exist_ok=True)
            imgs = [lvl.to_img() for lvl in levels]
            make_img_sheet(imgs, 9, save_path=f'{iteration_path}/samplesheet.png')
            torch.save(netG, getpath(iteration_path + '/decoder.pth'))
            # pds.DataFrame(log_data, columns=log_keys).to_csv(f'{path_}/log.csv')

            netD.train()
            netG.train()

    netG.eval()
    netD.eval()
    with torch.no_grad():
        z = sample_latvec(54, device=device, distribuion=args.noise)
        fake = netG(z)
    levels = process_onehot(fake)
    imgs = [lvl.to_img() for lvl in levels]
    torch.save(netG, f'{res_path}/decoder.pth')
    make_img_sheet(imgs, 9, save_path=f'{res_path}/samplesheet.png')
    log_target.close()