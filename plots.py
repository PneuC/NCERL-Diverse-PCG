import glob
import json
import os
import re

import numpy as np
import pandas as pds
import matplotlib
import matplotlib.pyplot as plt
from math import sqrt
import torch
from root import PRJROOT
from sklearn.manifold import TSNE
from itertools import product, chain
# from src.drl.drl_uses import load_cfgs
from src.gan.gankits import get_decoder, process_onehot
from src.gan.gans import nz
from src.smb.level import load_batch, hamming_dis, lvlhcat
from src.utils.datastruct import RingQueue
from src.utils.filesys import load_dict_json, getpath
from src.utils.img import make_img_sheet
from torch.distributions import Normal

matplotlib.rcParams["axes.formatter.limits"] = (-5, 5)


def print_compare_tab():
    rand_lgp, rand_fhp, rand_divs = load_dict_json(
        'test_data/rand_policy/performance.csv', 'lgp', 'fhp', 'diversity'
    )
    rand_performance = {'lgp': rand_lgp, 'fhp': rand_fhp, 'diversity': rand_divs}

    def _print_line(_data, minimise=False):
        means = _data.mean(axis=-1)
        stds = _data.std(axis=-1)
        max_i, min_i = np.argmax(means), np.argmin(means)
        mean_str_content = [*map(lambda x: '%.4g' % x, _data.mean(axis=-1))]
        std_str_content = [*map(lambda x: '$\pm$%.3g' % x, _data.std(axis=-1))]
        if minimise:
            mean_str_content[min_i] = r'\textbf{%s}' % mean_str_content[min_i]
            mean_str_content[max_i] = r'\textit{%s}' % mean_str_content[max_i]
            std_str_content[min_i] = r'\textbf{%s}' % std_str_content[min_i]
            std_str_content[max_i] = r'\textit{%s}' % std_str_content[max_i]
        else:
            mean_str_content[max_i] = r'\textbf{%s}' % mean_str_content[max_i]
            mean_str_content[min_i] = r'\textit{%s}' % mean_str_content[min_i]
            std_str_content[max_i] = r'\textbf{%s}' % std_str_content[max_i]
            std_str_content[min_i] = r'\textit{%s}' % std_str_content[min_i]
        print('    &', ' & '.join(mean_str_content), r'\\')
        print('    & &', ' & '.join(std_str_content), r'\\')
        pass

    def _print_block(_task):
        fds = [
            f'sac/{_task}', f'egsac/{_task}', f'asyncsac/{_task}',
            f'pmoe/{_task}', f'dvd/{_task}', f'sunrise/{_task}',
            f'varpm-{_task}/l0.0_m5', f'varpm-{_task}/l0.1_m5', f'varpm-{_task}/l0.2_m5',
            f'varpm-{_task}/l0.3_m5', f'varpm-{_task}/l0.4_m5', f'varpm-{_task}/l0.5_m5'
        ]
        rewards, divs = [], []
        for fd in fds:
            rewards.append([])
            divs.append([])
            # print(getpath())
            for path in glob.glob(getpath('test_data', fd, '**', 'performance.csv'), recursive=True):
                reward, div = load_dict_json(path, 'reward', 'diversity')
                rewards[-1].append(reward)
                divs[-1].append(div)
        rewards = np.array(rewards)
        divs = np.array(divs)

        print('    & \\multirow{2}{*}{Reward}')
        _print_line(rewards)
        print('    \\cline{2-14}')
        print('    & \\multirow{2}{*}{Diversity}')
        _print_line(divs)
        print('    \\cline{2-14}')
        print('    & \\multirow{2}{*}{G-mean}')
        gmean = np.sqrt(rewards * divs)
        _print_line(gmean)

        print('    \\cline{2-14}')
        print('    & \\multirow{2}{*}{N-rank}')
        r_rank = np.zeros_like(rewards.flatten())
        r_rank[np.argsort(-rewards.flatten())] = np.linspace(1, len(r_rank), len(r_rank))

        d_rank = np.zeros_like(divs.flatten())
        d_rank[np.argsort(-divs.flatten())] = np.linspace(1, len(r_rank), len(r_rank))
        n_rank = (r_rank.reshape([12, 5]) + d_rank.reshape([12, 5])) / (2 * 5)
        _print_line(n_rank, True)

    print('    \\multirow{8}{*}{MarioPuzzle}')
    _print_block('fhp')
    print('    \\midrule')
    print('    \\multirow{8}{*}{MultiFacet}')
    _print_block('lgp')
    pass

def print_compare_tab_nonrl():
    # rand_lgp, rand_fhp, rand_divs = load_dict_json(
    #     'test_data/rand_policy/performance.csv', 'lgp', 'fhp', 'diversity'
    # )
    # rand_performance = {'lgp': rand_lgp, 'fhp': rand_fhp, 'diversity': rand_divs}

    def _print_line(_data, minimise=False):
        means = _data.mean(axis=-1)
        stds = _data.std(axis=-1)
        max_i, min_i = np.argmax(means), np.argmin(means)
        mean_str_content = [*map(lambda x: '%.4g' % x, _data.mean(axis=-1))]
        std_str_content = [*map(lambda x: '$\pm$%.3g' % x, _data.std(axis=-1))]
        if minimise:
            mean_str_content[min_i] = r'\textbf{%s}' % mean_str_content[min_i]
            mean_str_content[max_i] = r'\textit{%s}' % mean_str_content[max_i]
            std_str_content[min_i] = r'\textbf{%s}' % std_str_content[min_i]
            std_str_content[max_i] = r'\textit{%s}' % std_str_content[max_i]
        else:
            mean_str_content[max_i] = r'\textbf{%s}' % mean_str_content[max_i]
            mean_str_content[min_i] = r'\textit{%s}' % mean_str_content[min_i]
            std_str_content[max_i] = r'\textbf{%s}' % std_str_content[max_i]
            std_str_content[min_i] = r'\textit{%s}' % std_str_content[min_i]
        print('    &', ' & '.join(mean_str_content), r'\\')
        print('    & &', ' & '.join(std_str_content), r'\\')
        pass

    def _print_block(_task):
        fds = [
            f'GAN-{_task}', f'DDPM-{_task}',
            f'varpm-{_task}/l0.0_m5', f'varpm-{_task}/l0.1_m5', f'varpm-{_task}/l0.2_m5',
            f'varpm-{_task}/l0.3_m5', f'varpm-{_task}/l0.4_m5', f'varpm-{_task}/l0.5_m5'
        ]
        rewards, divs = [], []
        for fd in fds:
            rewards.append([])
            divs.append([])
            # print(getpath())
            for path in glob.glob(getpath('test_data', fd, '**', 'performance.csv'), recursive=True):
                reward, div = load_dict_json(path, 'reward', 'diversity')
                rewards[-1].append(reward)
                divs[-1].append(div)
        rewards = np.array(rewards)
        divs = np.array(divs)

        print('    & \\multirow{2}{*}{Reward}')
        _print_line(rewards)
        print('    \\cline{2-10}')
        print('    & \\multirow{2}{*}{Diversity}')
        _print_line(divs)
        print('    \\cline{2-10}')
        # print('    & \\multirow{2}{*}{G-mean}')
        # gmean = np.sqrt(rewards * divs)
        # _print_line(gmean)
        #
        # print('    \\cline{2-10}')
        # print('    & \\multirow{2}{*}{N-rank}')
        # r_rank = np.zeros_like(rewards.flatten())
        # r_rank[np.argsort(-rewards.flatten())] = np.linspace(1, len(r_rank), len(r_rank))
        #
        # d_rank = np.zeros_like(divs.flatten())
        # d_rank[np.argsort(-divs.flatten())] = np.linspace(1, len(r_rank), len(r_rank))
        # n_rank = (r_rank.reshape([8, 5]) + d_rank.reshape([8, 5])) / (2 * 5)
        # _print_line(n_rank, True)

    print('    \\multirow{4}{*}{MarioPuzzle}')
    _print_block('fhp')
    print('    \\midrule')
    print('    \\multirow{4}{*}{MultiFacet}')
    _print_block('lgp')
    pass

def plot_cmp_learning_curves(task, save_path='', title=''):
    plt.style.use('seaborn')
    colors = [plt.plot([0, 1], [-1000, -1000])[0].get_color() for _ in range(6)]
    plt.cla()
    plt.style.use('default')

    # colors = ('#5D2CAB', '#005BD4', '#007CE4', '#0097DD', '#00ADC4', '#00C1A5')
    def _get_algo_data(fd):
        res = []
        for i in range(1, 6):
            path = getpath(fd, f't{i}', 'step_tests.csv')
            try:
                data = pds.read_csv(path)
                trajectory = [
                    [float(item['step']), float(item['r-avg']), float(item['diversity'])]
                    for _, item in data.iterrows()
                ]
                trajectory.sort(key=lambda x: x[0])
                res.append(trajectory)
                if len(trajectory) != 26:
                    print('Not complete (%d)/26:' % len(trajectory), path)
            except FileNotFoundError:
                print(path)
        res = np.array(res)
        # rdsum = res[:, :, 1] + res[:, :, 2]
        gmean = np.sqrt(res[:, :, 1] * res[:, :, 2])
        steps = res[0, :, 0]
        # r_avgs = np.mean(res[:, :, 1], axis=0)
        # r_stds = np.std(res[:, :, 1], axis=0)
        # divs = np.mean(res[:, :, 2], axis=0)
        # div_std = np.std(res[:, :, 2], axis=0)
        _performances = {
            'reward': (np.mean(res[:, :, 1], axis=0), np.std(res[:, :, 1], axis=0)),
            'diversity': (np.mean(res[:, :, 2], axis=0), np.std(res[:, :, 2], axis=0)),
            # 'rdsum': (np.mean(rdsum, axis=0), np.std(rdsum, axis=0)),
            'gmean': (np.mean(gmean, axis=0), np.std(gmean, axis=0)),
        }
        # print(_performances['gmean'])
        return steps, _performances

    def _plot_criterion(_ax, _criterion):
        i, j, k = 0, 0, 0
        for algo, (steps, _performances) in performances.items():
            avgs, stds = _performances[_criterion]
            if '\lambda' in algo:
                ls = '-'
                _c = colors[i]
                i += 1
            elif algo in {'SAC', 'EGSAC', 'ASAC'}:
                ls = ':'
                _c = colors[j]
                j += 1
            else:
                ls = '--'
                _c = colors[j]
                j += 1
            _ax.plot(steps, avgs, color=_c, label=algo, ls=ls)
            _ax.fill_between(steps, avgs - stds, avgs + stds, color=_c, alpha=0.15)
            _ax.grid(False)
            # plt.plot(steps, avgs, label=algo)
            # plt.plot(_performances, label=algo)
        pass
        _ax.set_xlabel('Time step')

    fig, ax = plt.subplots(1, 3, figsize=(9.6, 3.2), dpi=250, width_ratios=[1, 1, 1])
    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5, 4), dpi=256)
    # fig, ax1 = plt.subplots(1, 1, figsize=(8, 3), dpi=256)
    # ax2 = ax1.twinx()
    # fig = plt.plot(figsize=(4, 3), dpi=256)
    performances = {
        'SUNRISE': _get_algo_data(f'test_data/sunrise/{task}'),
        '$\lambda$=0.0': _get_algo_data(f'test_data/varpm-{task}/l0.0_m5'),
        'DvD': _get_algo_data(f'test_data/dvd/{task}'),
        '$\lambda$=0.1': _get_algo_data(f'test_data/varpm-{task}/l0.1_m5'),
        'PMOE': _get_algo_data(f'test_data/pmoe/{task}'),
        '$\lambda$=0.2': _get_algo_data(f'test_data/varpm-{task}/l0.2_m5'),
        'SAC': _get_algo_data(f'test_data/sac/{task}'),
        '$\lambda$=0.3': _get_algo_data(f'test_data/varpm-{task}/l0.3_m5'),
        'EGSAC': _get_algo_data(f'test_data/egsac/{task}'),
        '$\lambda$=0.4': _get_algo_data(f'test_data/varpm-{task}/l0.4_m5'),
        'ASAC': _get_algo_data(f'test_data/asyncsac/{task}'),
        '$\lambda$=0.5': _get_algo_data(f'test_data/varpm-{task}/l0.5_m5'),
    }
    # _plot_algo(*_get_algo_data(glob.glob(getpath('test_data/SAC', '**', 'step_tests.csv'))), 'SAC')
    # _plot_algo(*_get_algo_data(glob.glob(getpath('test_data/EGSAC', '**', 'step_tests.csv'))), 'EGSAC')
    # _plot_algo(*_get_algo_data(glob.glob(getpath('test_data/AsyncSAC', '**', 'step_tests.csv'))), 'AsyncSAC')
    # _plot_algo(*_get_algo_data(glob.glob(getpath('test_data/SUNRISE', '**', 'step_tests.csv'))), 'SUNRISE')
    # _plot_algo(*_get_algo_data(glob.glob(getpath('test_data/DvD-ES', '**', 'step_tests.csv'))), 'DvD-ES')
    # _plot_algo(*_get_algo_data(glob.glob(getpath('test_data/lbd-m-crosstest/l0.04_m5', '**', 'step_tests.csv'))), 'NCESAC')



    _plot_criterion(ax[0], 'reward')
    _plot_criterion(ax[1], 'diversity')
    # _plot_criterion(ax[2], 'rdsum')
    _plot_criterion(ax[2], 'gmean')
    # ax[0].set_title(f'{title} reward')
    ax[0].set_title(f'Cumulative Reward')
    ax[1].set_title('Diversity Score')
    # ax[2].set_title('Summation')
    ax[2].set_title('G-mean')
    # plt.title(title)

    lines, labels = fig.axes[-1].get_legend_handles_labels()
    fig.suptitle(title, fontsize=14)
    plt.tight_layout(pad=0.5)
    if save_path:
        plt.savefig(getpath(save_path))
    else:
        plt.show()

    plt.cla()
    plt.figure(figsize=(9.6, 2.4), dpi=250)
    plt.grid(False)
    plt.axis('off')
    plt.yticks([1.0])
    plt.legend(
        lines, labels, loc='lower center', ncol=6, edgecolor='white', fontsize=15,
        columnspacing=0.8, borderpad=0.16, labelspacing=0.2, handlelength=2.4, handletextpad=0.3
    )
    plt.tight_layout(pad=0.5)
    plt.show()
    pass

def plot_crosstest_scatters(rfunc, xrange=None, yrange=None, title=''):
    def get_pareto():
        all_points = list(chain(*scatter_groups.values())) + cmp_points
        res = []
        for p in all_points:
            non_dominated = True
            for q in all_points:
                if q[0] >= p[0] and q[1] >= p[1] and (q[0] > p[0] or q[1] > p[1]):
                    non_dominated = False
                    break
            if non_dominated:
                res.append(p)
        res.sort(key=lambda item:item[0])
        return np.array(res)
    def _hex_color(_c):
        return
    scatter_groups = {}
    all_lbd = set()
    # Initialise
    plt.style.use('seaborn-v0_8-dark-palette')
    # plt.figure(figsize=(4, 4), dpi=256)
    plt.figure(figsize=(2.5, 2.5), dpi=256)
    plt.axes().set_axisbelow(True)

    # Competitors' performances
    cmp_folders = ['asyncsac', 'egsac', 'sac', 'sunrise', 'dvd', 'pmoe']
    cmp_names = ['ASAC', 'EGSAC', 'SAC', 'SUNRISE', 'DvD', 'PMOE']
    cmp_labels = ['A', 'E', 'S', 'R', 'D', 'M']
    cmp_markers = ['2', 'x', '+', 'o', '*', 'D']
    cmp_sizes = [42, 20, 32, 16, 24, 10, 10]
    cmp_points = []
    for name, folder, label, mk, s in zip(cmp_names, cmp_folders, cmp_labels, cmp_markers, cmp_sizes):
        path_fmt = getpath('test_data', folder, rfunc, '*', 'performance.csv')
        # print(path_fmt)
        xs, ys = [], []
        for path in glob.glob(path_fmt, recursive=True):
            # print(path)
            try:
                x, y = load_dict_json(path, 'reward', 'diversity')
                xs.append(x)
                ys.append(y)
                cmp_points.append([x, y])
                # plt.text(x, y, label, size=7, weight='bold', va='center', ha='center', color='#202020')
            except FileNotFoundError:
                print(path)
        if label in {'A', 'E', 'S'}:
            plt.scatter(xs, ys, marker=mk, zorder=2, s=s, label=name, color='#202020')
        else:
            plt.scatter(
                xs, ys, marker=mk, zorder=2, s=s, label=name, color=[0., 0., 0., 0.],
                edgecolors='#202020', linewidths=1
            )
    # NCESAC performances
    for path in glob.glob(getpath('test_data', f'varpm-{rfunc}', '**', 'performance.csv'), recursive=True):
        try:
            x, y = load_dict_json(path, 'reward', 'diversity')
            key = path.split('\\')[-3]
            _, mtxt = key.split('_')
            ltxt, _ = key.split('_')
            lbd = float(ltxt[1:])
            # if mtxt in {'m2', 'm3', 'm4'}:
            #     continue
            all_lbd.add(lbd)
            if key not in scatter_groups.keys():
                scatter_groups[key] = []
            scatter_groups[key].append([x, y])
        except Exception as e:
            print(path)
            print(e)

    palette = plt.get_cmap('seismic')
    color_x = [0.2, 0.33, 0.4, 0.61, 0.67, 0.79]
    colors = {lbd: matplotlib.colors.to_hex(c) for c, lbd in zip(palette(color_x), sorted(all_lbd))}
    colors = {0.0: '#150080', 0.1: '#066598', 0.2: '#01E499', 0.3: '#9FD40C', 0.4: '#F3B020', 0.5: '#FA0000'}
    for lbd in sorted(all_lbd): plt.plot([-20], [-20], label=f'$\\lambda={lbd:.1f}$', lw=6, c=colors[lbd])
    markers = {2: 'o', 3: '^', 4: 'D', 5: 'p', 6: 'h'}
    msizes = {2: 25, 3: 25, 4: 16, 5: 28, 6: 32}
    for key, group in scatter_groups.items():
        ltxt, mtxt = key.split('_')
        l = float(ltxt[1:])
        m = int(mtxt[1:])
        arr = np.array(group)
        plt.scatter(
            arr[:, 0], arr[:, 1], marker=markers[m], s=msizes[m], color=[0., 0., 0., 0.], zorder=2,
            edgecolors=colors[l], linewidths=1
        )

    plt.xlim(xrange)
    plt.ylim(yrange)
    # plt.xlabel('Task Reward')
    # plt.ylabel('Diversity')
    # plt.legend(ncol=2)
    # plt.legend(
    #     ncol=2, loc='lower left', columnspacing=1.2, borderpad=0.0,
    #     handlelength=1, handletextpad=0.5, framealpha=0.
    # )
    pareto = get_pareto()
    plt.plot(
        pareto[:, 0], pareto[:, 1], color='black', alpha=0.18, lw=6, zorder=3,
        solid_joinstyle='round', solid_capstyle='round'
    )
    # plt.plot([88, 98, 98, 88, 88], [35, 35, 0.2, 0.2, 35], color='black', alpha=0.3, lw=1.5)
    # plt.xticks(fontsize=16)
    # plt.yticks(fontsize=16)
    # plt.xticks([(1+space) * (m-mlow) + 0.5 for m in ms], [f'm={m}' for m in ms])
    plt.title(title)
    plt.grid()
    plt.tight_layout(pad=0.4)
    plt.show()

def plot_varpm_heat(task, name):
    def _get_score(m, l):
        fd = getpath('test_data', f'varpm-{task}', f'l{l}_m{m}')
        rewards, divs = [], []
        for i in range(5):
            reward, div = load_dict_json(f'{fd}/t{i+1}/performance.csv', 'reward', 'diversity')
            rewards.append(reward)
            divs.append(div)
        gmean = [sqrt(r * d) for r, d in zip(rewards, divs)]
        return np.mean(rewards), np.std(rewards), \
            np.mean(divs), np.std(divs), \
            np.mean(gmean), np.std(gmean)

    def _plot_map(avg_map, std_map, criterion):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3), dpi=256, width_ratios=(1, 1))
        heat1 = ax1.imshow(avg_map, cmap='spring')
        heat2 = ax2.imshow(std_map, cmap='spring')
        ax1.set_xlim([-0.5, 5.5])
        ax1.set_xticks([0, 1, 2, 3, 4, 5], ['$\lambda$=0.0', '$\lambda$=0.1', '$\lambda$=0.2', '$\lambda$=0.3', '$\lambda$=0.4', '$\lambda$=0.5'])
        ax1.set_ylim([-0.5, 3.5])
        ax1.set_yticks([0, 1, 2, 3], ['m=5', 'm=4', 'm=3', 'm=2'])
        ax1.set_title('Average')
        for x, y in product([0, 1, 2, 3, 4, 5], [0, 1, 2, 3]):
            v = avg_map[y, x]
            s = '%.4f' % v
            if v >= 1000: s = s[:4]
            elif v >= 1: s = s[:5]
            else: s = s[1:6]
            ax1.text(x, y, s, va='center', ha='center')
        plt.colorbar(heat1, ax=ax1, shrink=0.9)
        ax2.set_xlim([-0.5, 5.5])
        ax2.set_xticks([0, 1, 2, 3, 4, 5], ['$\lambda$=0.0', '$\lambda$=0.1', '$\lambda$=0.2', '$\lambda$=0.3', '$\lambda$=0.4', '$\lambda$=0.5'])
        ax2.set_ylim([-0.5, 3.5])
        ax2.set_yticks([0, 1, 2, 3], ['m=5', 'm=4', 'm=3', 'm=2'])
        for x, y in product([0, 1, 2, 3, 4, 5], [0, 1, 2, 3]):
            v = std_map[y, x]
            s = '%.4f' % v
            if v >= 1000: s = s[:4]
            elif v >= 1: s = s[:5]
            else: s = s[1:6]
            ax2.text(x, y, s, va='center', ha='center')
        ax2.set_title('Standard Deviation')
        plt.colorbar(heat2, ax=ax2, shrink=0.9)

        fig.suptitle(f'{name}: {criterion}', fontsize=14)
        plt.tight_layout()
        # plt.show()
        plt.savefig(getpath(f'results/heat/{name}-{criterion}.png'))

    r_mean_map, r_std_map, d_mean_map, d_std_map, g_mean_map, g_std_map \
        = (np.zeros([4, 6], dtype=float) for _ in range(6))
    ms = [2, 3, 4, 5]
    ls = ['0.0', '0.1', '0.2', '0.3', '0.4', '0.5']
    for i, j in product(range(4), range(6)):
        r_mean, r_std, d_mean, d_std, g_mean, g_std = _get_score(ms[i], ls[j])
        r_mean_map[i, j] = r_mean
        r_std_map[i, j] = r_std
        d_mean_map[i, j] = d_mean
        d_std_map[i, j] = d_std
        g_mean_map[i, j] = g_mean
        g_std_map[i, j] = g_std

    _plot_map(r_mean_map, r_std_map, 'Reward')
    _plot_map(d_mean_map, d_std_map, 'Diversity')
    _plot_map(g_mean_map, g_std_map,'G-mean')
    # _plot_map(g_mean_map, g_std_map,'G-mean')

def vis_samples():
    # for l, m in product(['0.0', '0.1', '0.2', '0.3', '0.4', '0.5'], [2, 3, 4, 5]):
    #     for i in range(1, 6):
    #         lvls = load_batch(f'{PRJROOT}/test_data/varpm-fhp/l{l}_m{m}/t{i}/samples.lvls')
    #         imgs = [lvl.to_img(save_path=None) for lvl in lvls[:10]]
    #         make_img_sheet(imgs, 1, save_path=f'{PRJROOT}/test_data/varpm-fhp/l{l}_m{m}/t{i}/samples.png')
    # for algo in ['sac', 'egsac', 'asyncsac', 'dvd', 'sunrise', 'pmoe']:
    #     for i in range(1, 6):
    #         lvls = load_batch(f'{PRJROOT}/test_data/{algo}/fhp/t{i}/samples.lvls')
    #         imgs = [lvl.to_img(save_path=None) for lvl in lvls[:10]]
    #         make_img_sheet(imgs, 1, save_path=f'{PRJROOT}/test_data/{algo}/fhp/t{i}/samples.png')
    for i in range(1, 6):
        lvls = load_batch(f'{PRJROOT}/test_data/DDPM-fhp/t{i}/samples.lvls')
        imgs = [lvl.to_img(save_path=None) for lvl in lvls[:10]]
        make_img_sheet(imgs, 1, save_path=f'{PRJROOT}/test_data/DDPM-fhp/t{i}/samples.png')
        pass
    pass

def make_tsne(task, title, n=500, save_path=None):
    if not os.path.exists(getpath('test_data', f'samples_dist-{task}_{n}.npy')):
        samples = []
        for algo in ['dvd', 'egsac', 'pmoe', 'sunrise', 'asyncsac', 'sac']:
            for t in range(5):
                lvls = load_batch(getpath('test_data', algo, task, f't{t+1}', 'samples.lvls'))
                samples += lvls[:n]
        for l in ['0.0', '0.1', '0.2', '0.3', '0.4', '0.5']:
            for t in range(5):
                lvls = load_batch(getpath('test_data', f'varpm-{task}', f'l{l}_m5', f't{t+1}', 'samples.lvls'))
                samples += lvls[:n]
        distmat = []
        for a in samples:
            dist_list = []
            for b in samples:
                dist_list.append(hamming_dis(a, b))
            distmat.append(dist_list)
        distmat = np.array(distmat)
        np.save(getpath('test_data', f'samples_dist-{task}_{n}.npy'), distmat)

    labels = (
        '$\lambda$=0.0', '$\lambda$=0.1', '$\lambda$=0.2', '$\lambda$=0.3', '$\lambda$=0.4',
        '$\lambda$=0.5', 'DvD', 'EGSAC', 'PMOE', 'SUNRISE', 'ASAC', 'SAC'
    )
    tsne = TSNE(learning_rate='auto', n_components=2, metric='precomputed')
    print(np.load(getpath('test_data', f'samples_dist-{task}_{n}.npy')).shape)
    data = np.load(getpath('test_data', f'samples_dist-{task}_{n}.npy'))
    embx = np.array(tsne.fit_transform(data))

    plt.style.use('seaborn-dark-palette')
    plt.figure(figsize=(5, 5), dpi=384)
    colors = [plt.plot([-1000, -1100], [0, 0])[0].get_color() for _ in range(6)]
    for i in range(6):
        x, y = embx[i*n*5:(i+1)*n*5, 0], embx[i*n*5:(i+1)*n*5, 1]
        plt.scatter(x, y, s=10, label=labels[i], marker='x', c=colors[i])
    for i in range(6, 12):
        x, y = embx[i*n*5:(i+1)*n*5, 0], embx[i*n*5:(i+1)*n*5, 1]
        plt.scatter(x, y, s=8, linewidths=0, label=labels[i], c=colors[i-6])
    # plt.scatter(embx[100:200, 0], embx[100:200, 1], c=colors[1], s=12, linewidths=0, label='Killer')
    # plt.scatter(embx[200:, 0], embx[200:, 1], c=colors[2], s=12, linewidths=0, label='Collector')
    # for i in range(4):
    #     plt.text(embx[i+100, 0], embx[i+100, 1], str(i+1))
    #     plt.text(embx[i+200, 0], embx[i+200, 1], str(i+1))
    #     pass
    # for emb, lb, c in zip(embs, labels,colors):
    #     plt.scatter(emb[:,0], emb[:,1], c=c, label=lb, alpha=0.15, linewidths=0, s=7)

    # xspan = 1.08 * max(abs(embx[:, 0].max()), abs(embx[:, 0].min()))
    # yspan = 1.08 * max(abs(embx[:, 1].max()), abs(embx[:, 1].min()))

    xrange = [1.05 * embx[:, 0].min(), 1.05 * embx[:, 0].max()]
    yrange = [1.05 * embx[:, 1].min(), 1.25 * embx[:, 1].max()]

    plt.xlim(xrange)
    plt.ylim(yrange)
    plt.xticks([])
    plt.yticks([])
    # plt.legend(ncol=6, handletextpad=0.02, labelspacing=0.05, columnspacing=0.16)
    # plt.xticks([-xspan, -0.5 * xspan, 0, 0.5 * xspan, xspan], [''] * 5)
    # plt.yticks([-yspan, -0.5 * yspan, 0, 0.6 * yspan, yspan], [''] * 5)
    plt.title(title)
    plt.legend(loc='upper center', ncol=6, fontsize=9, handlelength=.5, handletextpad=0.5, columnspacing=0.3, framealpha=0.)
    plt.tight_layout(pad=0.2)

    if save_path:
        plt.savefig(getpath(save_path))
    else:
        plt.show()

def _prob_fmt(p, digitals=3, threshold=0.001):
    fmt = '%.' + str(digitals) + 'f'
    if p < threshold:
        return '$\\approx 0$'
    else:
        txt = '$%s$' % ((fmt % p)[1:])
        if txt == '$.000$':
            txt = '$1.00$'
        return txt

def _g_fmt(v, digitals=4):
    fmt = '%.' + str(digitals) + 'g'
    txt = (fmt % v)
    lack = digitals - len(txt.replace('-', '').replace('.', ''))
    if lack > 0 and '.' not in txt:
        txt += '.'
    return txt + '0' * lack
    pass

def print_selection_prob(path, h=15, runs=2):
    s0 = 0
    model = torch.load(getpath(f'{path}/policy.pth'), map_location='cpu')
    model.requires_grad_(False)
    model.to('cpu')
    n = 11
    # n = load_cfgs(path, 'N')
    # print(model.m)

    init_vec = np.load(getpath('analysis/initial_seg.npy'))[s0]
    decoder = get_decoder(device='cpu')
    obs_buffer = RingQueue(n)
    for r in range(runs):
        for _ in range(h): obs_buffer.push(np.zeros([nz]))
        obs_buffer.push(init_vec)
        level_latvecs = [init_vec]
        probs = np.zeros([model.m, h])
        # probs = []
        selects = []
        for t in range(h):
            # probs.append([])
            obs = torch.tensor(np.concatenate(obs_buffer.to_list(), axis=-1), dtype=torch.float).view([1, -1])
            muss, stdss, betas = model.get_intermediate(torch.tensor(obs))
            i = torch.multinomial(betas.squeeze(), 1).item()
            # print(i)
            mu, std = muss[0][i], stdss[0][i]
            action = Normal(mu, std).rsample([1]).squeeze().numpy()
            # print(action)
            # print(mu)
            # print(std)
            # print(action.numpy())
            obs_buffer.push(action)
            level_latvecs.append(action)
            # i = torch.multinomial(betas.squeeze(), 1).item()
            # print(i)
            probs[:, t] = betas.squeeze().numpy()
            selects.append(i)
            pass
        onehots = decoder(torch.tensor(level_latvecs).view(-1, nz, 1, 1))
        segs = process_onehot(onehots)
        lvl = lvlhcat(segs)
        lvl.to_img(f'figures/gen_process/run{r}-01.png')
        txts = [[_prob_fmt(p) for p in row] for row in probs]
        for t, i in enumerate(selects):
            txts[i][t] = r'$\boldsymbol{%s}$' % txts[i][t][1:-1]
        for i, txt in enumerate(txts):
            print(f'    & $\\beta_{i+1}$ &', ' & '.join(txt), r'\\')
        print(r'\midrule')

    pass

def calc_selection_freqs(task, n):
    def _count_one_init():
        counts = np.zeros([model.m])
        # init_vec = np.load(getpath('analysis/initial_seg.npy'))
        obs_buffer = RingQueue(n)
        for _ in range(runs):
            for _ in range(h): obs_buffer.push(np.zeros([len(init_vecs), nz]))
            obs_buffer.push(init_vecs)
            # level_latvecs = [init_vec]
            for _ in range(h):
                obs = np.concatenate(obs_buffer.to_list(), axis=-1)
                obs = torch.tensor(obs, device='cuda:0', dtype=torch.float)
                muss, stdss, betas = model.get_intermediate(obs)
                selects = torch.multinomial(betas.squeeze(), 1).squeeze()
                mus = muss[[*range(len(init_vecs))], selects, :]
                stds = stdss[[*range(len(init_vecs))], selects, :]
                actions = Normal(mus, stds).rsample().squeeze().cpu().numpy()
                obs_buffer.push(actions)
                for i in selects:
                    counts[i] = counts[i] + 1
        return counts
        # onehots = decoder(torch.tensor(level_latvecs).view(-1, nz, 1, 1))
        pass
    pass
    init_vecs = np.load(getpath('analysis/initial_seg.npy'))
    freqs = [[] for _ in range(30)]
    start_line = 0
    for l in ('0.0', '0.1', '0.2', '0.3', '0.4', '0.5'):
        print(r'    \midrule')
        for t, m in product(range(1, 6), (2, 3, 4, 5)):
            path = getpath(f'test_data/varpm-{task}/l{l}_m{m}/t{t}')
            model = torch.load(getpath(f'{path}/policy.pth'), map_location='cuda:0')
            model.requires_grad_(False)
            freq = np.zeros([m])
            # n = load_cfgs(path, 'N')
            runs, h = 100, 25
            freq += _count_one_init()
            freq /= (len(init_vecs) * runs * h)
            freq = np.sort(freq)[::-1]
            i = start_line + t - 1
            freqs[i] += freq.tolist()
            print(freqs[i])
        start_line += 5
        print(freqs)
    with open(getpath(f'analysis/select_freqs-{task}.json'), 'w') as f:
        json.dump(freqs, f)

def print_selection_freq():
    # task, n = 'lgp', 5
    task, n = 'fhp', 11
    if not os.path.exists(getpath(f'analysis/select_freqs-{task}.json')):
        calc_selection_freqs(task, n)
    with open(getpath(f'analysis/select_freqs-{task}.json'), 'r') as f:
        freqs = json.load(f)
    lbds = ['0.0', '0.1', '0.2', '0.3', '0.4', '0.5']
    for i, row_data in enumerate(freqs):
        if i % 5 == 0:
            print(r'  \midrule')
            print(r'  \multirow{5}{*}{$%s$}' % lbds[i//5])
        txt = ' & '.join(map(_prob_fmt, row_data))
        print(f'  & {i%5+1} &', txt, r'\\')

def print_individual_performances(task):
    for m, l in product((2, 3, 4, 5), ('0.0', '0.1', '0.2', '0.3', '0.4', '0.5')):
        values = []
        if l == '0.0':
            print(r'  \midrule')
            print(r'  \multirow{6}{*}{%d}' % m)
        for t in range(1, 6):
            path = f'test_data/varpm-{task}/l{l}_m{m}/t{t}/performance.csv'
            reward, diversity = load_dict_json(path, 'reward', 'diversity')
            values.append([reward, diversity])
        values.sort(key=lambda item: -item[0])
        values = [*chain(*values)]
        txts = [_g_fmt(v) for v in values]
        print('  &', f'${l}$ & ', ' & '.join(txts), r'\\')
    pass

if __name__ == '__main__':
    # print_selection_prob('test_data/varpm-fhp/l0.5_m5/t5')
    # print_selection_prob('test_data/varpm-fhp/l0.1_m5/t5')
    # print_selection_freq()
    # print_compare_tab_nonrl()
    # print_individual_performances('fhp')
    # print('\n\n')
    # print_individual_performances('lgp')

    # plot_cmp_learning_curves('fhp', save_path='results/learning_curves/fhp.png', title='MarioPuzzle')
    # plot_cmp_learning_curves('lgp', save_path='results/learning_curves/lgp.png', title='MultiFacet')

    # plot_crosstest_scatters('fhp', title='MarioPuzzle')
    # plot_crosstest_scatters('lgp', title='MultiFacet')
    # # plot_crosstest_scatters('fhp', yrange=(0, 2500), xrange=(20, 70), title='MarioPuzzle')
    # plot_crosstest_scatters('lgp', yrange=(0, 1500), xrange=(20, 50), title='MultiFacet')
    # plot_crosstest_scatters('lgp', yrange=(0, 800), xrange=(44, 48), title=' ')


    # plot_varpm_heat('fhp', 'MarioPuzzle')
    # plot_varpm_heat('lgp', 'MultiFacet')

    vis_samples()

    # make_tsne('fhp', 'MarioPuzzle', n=100)
    # make_tsne('lgp', 'MultiFacet', n=100)
    pass

