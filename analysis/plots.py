import os
import glob
import json
from math import sqrt

import pandas
import matplotlib
import numpy as np
import pandas as pds
from matplotlib import ticker, cm
from itertools import product, chain
from matplotlib import pyplot as plt
from matplotlib.colors import to_hex
from src.utils.filesys import getpath, load_singlerow_csv
from src.drl.drl_uses import load_performance

# plt.style.use('seaborn-v0_8-paper')
# plt.style.use('seaborn-v0_8-muted')
# plt.style.use('ggplot')
matplotlib.rcParams["axes.formatter.limits"] = (-5, 5)
colors = ('#845EC2', '#D65DB1', '#FF6F91', '#FF9671', '#FFC75F', '#F9F871')
# plt.cla()
# print(colors)

def trial_data_iter(root_folder, performance_key='r-avg', lrange=(0, 0.1), mlow=2, space=0.5):
    llow, lspan = lrange[0], lrange[1] - lrange[0]
    for path in glob.glob(getpath(root_folder, '**', 'performance.csv'), recursive=True):
        try:
            data = pds.read_csv(path)
            tmp = path.split('\\')[-3]
            ltxt, mtxt = tmp.split('_')
            lbd, m = float(ltxt[1:]), int(mtxt[1:])
            x = (m - mlow) * (1 + space) + (lbd - llow) / lspan
            y = data[performance_key][0]
            yield m, lbd, x, y
        except KeyError:
            print(f'\'{performance_key}\' not found', path)

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
    cmp_labels = ['A', 'E', 'S', 'R', 'D', 'M', 'P']
    cmp_markers = ['2', 'x', '+', 'o', '*', 'D', 's']
    cmp_sizes = [36, 16, 28, 16, 24, 10, 10]
    cmp_points = []
    for name, folder, label, mk, s in zip(cmp_names, cmp_folders, cmp_labels, cmp_markers, cmp_sizes):
        path_fmt = getpath('training_data', folder, rfunc, '*', 'performance.csv')
        # print(path_fmt)
        xs, ys = [], []
        for path in glob.glob(path_fmt, recursive=True):
            # print(path)
            try:
                x, y = load_performance(os.path.split(path)[0], 'r-avg', 'mpd-hm')
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
    for path in glob.glob(getpath('training_data', f'varpm-{rfunc}', '**', 'performance.csv'), recursive=True):
        try:
            x, y = load_performance(os.path.split(path)[0], 'r-avg', 'mpd-hm')
        except Exception:
            print(path)
        # data = pds.read_csv(path)
        key = path.split('\\')[-3]
        _, mtxt = key.split('_')
        ltxt, _ = key.split('_')
        lbd = float(ltxt[1:])
        all_lbd.add(lbd)
        if key not in scatter_groups.keys():
            scatter_groups[key] = []
        scatter_groups[key].append([x, y])

        # data = np.array(data)
        # plt.scatter(data[:, 0], data[:, 1], marker=f'${label}$', s=24, c='black', zorder=2)

    palette = cm.get_cmap('seismic')
    color_x = [0.2, 0.33, 0.4, 0.61, 0.67, 0.79]
    colors = {lbd: to_hex(c) for c, lbd in zip(palette(color_x), sorted(all_lbd))}
    colors = {0.0: '#150080', 0.1: '#066598', 0.2: '#01E499', 0.3: '#9FD40C', 0.4: '#F3B020', 0.5: '#FA0000'}
    for lbd in sorted(all_lbd): plt.plot([-20], [-20], label=f'$\\lambda={lbd:.1f}$', lw=6, c=colors[lbd])
    markers = {2: 'o', 3: '^', 4: 'D', 5: 'p', 6: 'h'}
    msizes = {2: 25, 3: 25, 4: 18, 5: 32, 6: 32}
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


def plot_crosstest_boxes(
        root_folder, performance_key='r-avg', yrange=None, lrange=(0, 0.1), mlow=2, space=0.5,
        title=None, legend=False, dest=''
):
    gray = '#383838'
    xs, ys = {}, {}
    ms = []
    for m, lbd, x, y in trial_data_iter(root_folder, performance_key, lrange, mlow, space):
        if m not in ms:
            ms.append(m)
        if lbd not in xs.keys():
            xs[lbd] = {}
            ys[lbd] = {}
        if m not in xs[lbd].keys():
            xs[lbd][m] = x
            ys[lbd][m] = []
        ys[lbd][m].append(y)
    plt.style.use('ggplot')
    plt.figure(figsize=(5, 2.5), dpi=256)
    colors = {lbd: plt.plot([-1, -1], [-1, -1],label=f'$\lambda={lbd:.2f}$', lw=5)[0].get_color() for lbd in xs.keys()}
    for lbd in xs.keys():
        groupms = list(xs[lbd].keys())
        positions = [xs[lbd][m] for m in groupms]
        data = [ys[lbd][m] for m in groupms]
        c = colors[lbd]
        plt.boxplot(
            data, positions=positions, widths=0.16, showmeans=True, patch_artist=True,
            # meanprops={'markerfacecolor': c, 'markeredgecolor': c, 'markersize': 4, 'marker': 'D'},
            # medianprops={'color': c},
            # boxprops={'edgecolor': c, 'facecolor': (1, 1, 1, 0)},
            # whiskerprops={"color": c},
            # capprops={"color": c},
            # flierprops={'markeredgecolor': c, 'markersize': 4}
            meanprops={'markerfacecolor': gray, 'markersize': 4, 'markeredgecolor': gray, 'marker': 'D'},
            medianprops={'color': gray},
            boxprops={'facecolor': c, 'edgecolor': c, 'linewidth': 0.5},
            whiskerprops={"color": gray},
            capprops={"color": gray},
            flierprops={'markeredgecolor': gray, 'markersize': 4}
        )
    plt.xlim((-0.2, (max(ms) - mlow) * (1 + space) + 1.2))
    if yrange is not None:
        plt.ylim(yrange)
    if legend:
        plt.legend(
            loc='lower left', fontsize=9, columnspacing=1.0, borderpad=0.0,
            handlelength=1, handletextpad=0.4, framealpha=0.
        )
    plt.xticks([(1+space) * (m-mlow) + 0.5 for m in ms], [f'$m={m}$' for m in ms], fontsize=12)
    plt.title(performance_key if title is None else title)
    plt.tight_layout(pad=0.2)
    if not dest:
        plt.show()
    else:
        plt.savefig(getpath(dest))
    pass

def plot_obj_scatter():
    # plt.style.use('seaborn')
    # plt.figure(figsize=(3.5, 3.5), dpi=512)
    x, y = [], []
    for l, m in product(('0.00', '0.02', '0.04', '0.06', '0.08', '0.10'), (2, 3, 4, 5)):
        fd = getpath('training_data', 'lbd-m-crosstest', f'l{l}_m{m}')
        itemx = [load_performance(f'{fd}/t{i}', 'r-avg') for i in range(1, 6)]
        itemy = [load_performance(f'{fd}/t{i}', 'divs') for i in range(1, 6)]
        x.append(np.mean(itemx))
        y.append(np.mean(itemy))
    plt.scatter(x, y)
    x, y = [], []
    for algo in ('AsyncSAC', 'DvD-ES', 'EGSAC-modified-model', 'SAC', 'SUNRISE'):
        fd = getpath('training_data', algo)
        itemx = [load_performance(f'{fd}/t{i}', 'r-avg') for i in range(1, 6)]
        itemy = [load_performance(f'{fd}/t{i}', 'divs') for i in range(1, 6)]
        x.append(np.mean(itemx))
        y.append(np.mean(itemy))
    plt.scatter(x, y)
    plt.xlim((60, 100))
    plt.ylim((0, 40))
    plt.xticks(np.linspace(60, 100, 9), ['60', '65', '70', '75', '80', '85', '90', '95', '100'])
    plt.tight_layout(pad=0.2)
    plt.show()
    pass

def plot_rew_div_curve(rt_path, label='', pareto=False):
    res = []
    for item in glob.glob(getpath(rt_path, '**', 'step_tests.csv')):
        data = pds.read_csv(item)
        trajectory = [[item['r-avg'], item['divs'], item['step']] for _, item in data.iterrows()]
        # trajectory.sort(key=lambda x: x[2])
        res += trajectory
        pass
    if pareto:
        pareto_points = []
        for x in res:
            is_pareto = True
            for xp in res:
                if (xp[0] >= x[0] and xp[1] >= x[1]) and (xp[0] > x[0] or xp[1] > x[1]):
                    is_pareto = False
                    break
            if is_pareto:
                pareto_points.append(x)
        res = pareto_points
    # print(res)
    res = np.array(res)
    reward = res[:,0]
    divs = res[:,1]
    # reward = res[:,:,0]
    # divs = res[:,:,1]
    # steps = res[0,:,2]
    # print(reward)
    # print(divs)
    # for _, item in data.iterrows():
    #     print(item)
    plt.xlabel('Quality')
    plt.ylabel('Diversity')
    plt.scatter(reward, divs, label=label, s=8)
    # plt.plot(steps, np.mean(reward, axis=0), label=label)
    # plt.plot(steps, np.mean(divs, axis=0), label=label)
    # plt.show()
    pass

def plot_train_curves_sub(rt_path, ax1, save_path='', title='', rlim=(30, 100), q=False, d=False):
    # c1, c2 = plt.plot([0, 0], [0, 1])[0].get_color(), plt.plot([0, 0], [0, 1])[0].get_color()
    # plt.close()
    def _get_algo_data(paths):
        res = []
        for item in paths:
            data = pds.read_csv(item)
            trajectory = [
                [float(item['r-avg']), float(item['r-std']), float(item['divs']), int(item['step'])]
                for _, item in data.iterrows()
            ]
            trajectory.sort(key=lambda x: x[3])
            res.append(trajectory)
            pass
        # print(res)
        res = np.array(res)
        r_avgs = np.mean(res[:, :, 0], axis=0)
        r_stds = np.std(res[:, :, 0], axis=0)
        divs = np.mean(res[:, :, 2], axis=0)
        d_std = np.std(res[:, :, 2], axis=0)
        steps = res[0, :, 3]
        # print(steps)
        return steps, r_avgs, r_stds, divs, d_std

    def _plot_algo(steps, r_avg, r_std, divs, d_std, label):#, linestyle):
        # ax1.plot(steps, r_avg, lw=1, zorder=2, ls=linestyle, label=f'{label} reward', color=c1)
        # ax1.fill_between(steps, r_avg - r_std, r_avg + r_std, alpha=0.2, linewidth=0, zorder=2, color=c1)
        # ax2.plot(steps, divs, lw=1, ls=linestyle, zorder=2, color=c2, label=f'{label} diversity')
        ax1.plot(steps, r_avg, lw=2, zorder=3.5, label=f'{label} Quality')
        ax1.fill_between(steps, r_avg - r_std, r_avg + r_std, alpha=0.2, linewidth=0, zorder=3.5)
        ax2.plot(steps, divs, lw=2, ls='--', zorder=3.5, label=f'{label} Diversity')
        ax2.fill_between(steps, divs - d_std, divs + d_std, alpha=0.2, linewidth=0, zorder=3.5)
        pass

    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5, 4), dpi=256)
    # fig, ax1 = plt.subplots(1, 1, figsize=(4, 3), dpi=256)
    ax2 = ax1.twinx()

    _plot_algo(*_get_algo_data(glob.glob(getpath(rt_path, 'AsyncSAC', '**', 'step_tests.csv'))), 'AsyncSAC')#, 'solid')
    _plot_algo(*_get_algo_data(glob.glob(getpath(rt_path, 'MESAC', '**', 'step_tests.csv'))), 'NCESAC')#, 'dashed')

    if q: ax1.set_ylabel('Quality')#, color=c1)
    ax1.set_xlabel('Time step')
    # ax1.set_xticks([0, 1e5, 2e5, 3e5, 4e5, 5e5], ['0', '100K', '200K', '300K', '400K', '500K'])
    ax1.set_ylim(rlim)
    ax1.set_yticks([int(v) for v in np.linspace(rlim[0], rlim[-1], 8)])
    # ax1.xaxis.set_major_formatter(ticker.ScalarFormatter())

    if d: ax2.set_ylabel('Diversity')#, color=c2)
    # ax2.set_xticks([0, 1e5, 2e5, 3e5, 4e5, 5e5], ['0', '100K', '200K', '300K', '400K', '500K'])
    ax2.set_yticks([0, 10, 20, 30, 40, 50, 60, 70])
    ax2.set_ylim((0, 70))
    # ax2.xaxis.set_major_formatter(ticker.ScalarFormatter())

    ax1.set_title(title)
    ax1.set_axisbelow(True)
    ax2.grid(b=None)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    return lines1 + lines2, labels1 + labels2
    # ax1.tight_layout(pad=0.5)
    # ax1.legend()
    # ax2.legend()
    # ax1.legend(loc=2, bbox_to_anchor=(1.05, 1.05), borderaxespad=0.)
    # plt.legend()
    # plt.title(title)
    # plt.tight_layout(pad=0.5)
    # if save_path:
    #     plt.savefig(getpath(save_path))
    # else:
    #     plt.show()
    # pass

def plot_cmp_learning_curves(task, save_path='', title=''):
    def _get_algo_data(fd):
        res = []
        for i in range(1, 6):
            path = getpath(fd, f't{i}', 'step_tests.csv')
            try:
                data = pds.read_csv(path)
                trajectory = [
                    [float(item['step']), float(item['r-avg']), float(item['mpd-hm'])]
                    for _, item in data.iterrows()
                ]
                trajectory.sort(key=lambda x: x[0])
                res.append(trajectory)
                if len(trajectory) != 51:
                    print('Not complete (%d)/51:' % len(trajectory), path)
            except FileNotFoundError:
                print(path)
        res = np.array(res)
        rdsum = res[:, :, 1] + res[:, :, 2]
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
                _c = colors[k]
                k += 1
            _ax.plot(steps, avgs, color=_c, label=algo, ls=ls)
            _ax.fill_between(steps, avgs - stds, avgs + stds, color=_c, alpha=0.15)
            _ax.grid(False)
            # plt.plot(steps, avgs, label=algo)
            # plt.plot(_performances, label=algo)
        _ax.set_xlabel('Time step')

    fig, ax = plt.subplots(1, 3, figsize=(9.6, 3.2), dpi=250, width_ratios=[1, 1, 1])
    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5, 4), dpi=256)
    # fig, ax1 = plt.subplots(1, 1, figsize=(8, 3), dpi=256)
    # ax2 = ax1.twinx()
    # fig = plt.plot(figsize=(4, 3), dpi=256)
    performances = {
        'sunrise': _get_algo_data(f'training_data/sunrise/{task}'),
        '$\lambda$=0.0': _get_algo_data(f'training_data/varpm-{task}/l0.0_m5'),
        'dvd': _get_algo_data(f'training_data/dvd/{task}'),
        '$\lambda$=0.1': _get_algo_data(f'training_data/varpm-{task}/l0.1_m5'),
        'pmoe': _get_algo_data(f'training_data/pmoe/{task}'),
        '$\lambda$=0.2': _get_algo_data(f'training_data/varpm-{task}/l0.2_m5'),
        'sac': _get_algo_data(f'training_data/sac/{task}'),
        '$\lambda$=0.3': _get_algo_data(f'training_data/varpm-{task}/l0.3_m5'),
        'egsac': _get_algo_data(f'training_data/egsac/{task}'),
        '$\lambda$=0.4': _get_algo_data(f'training_data/varpm-{task}/l0.4_m5'),
        'asac': _get_algo_data(f'training_data/asyncsac/{task}'),
        '$\lambda$=0.5': _get_algo_data(f'training_data/varpm-{task}/l0.5_m5'),
    }
    # _plot_algo(*_get_algo_data(glob.glob(getpath('training_data/SAC', '**', 'step_tests.csv'))), 'SAC')
    # _plot_algo(*_get_algo_data(glob.glob(getpath('training_data/EGSAC', '**', 'step_tests.csv'))), 'EGSAC')
    # _plot_algo(*_get_algo_data(glob.glob(getpath('training_data/AsyncSAC', '**', 'step_tests.csv'))), 'AsyncSAC')
    # _plot_algo(*_get_algo_data(glob.glob(getpath('training_data/SUNRISE', '**', 'step_tests.csv'))), 'SUNRISE')
    # _plot_algo(*_get_algo_data(glob.glob(getpath('training_data/DvD-ES', '**', 'step_tests.csv'))), 'DvD-ES')
    # _plot_algo(*_get_algo_data(glob.glob(getpath('training_data/lbd-m-crosstest/l0.04_m5', '**', 'step_tests.csv'))), 'NCESAC')

    _plot_criterion(ax[0], 'reward')
    _plot_criterion(ax[1], 'diversity')
    # _plot_criterion(ax[2], 'rdsum')
    _plot_criterion(ax[2], 'gmean')
    ax[0].set_title(f'{title} reward')
    ax[1].set_title('Diversity')
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

def plot_train_log(paths, lbs=None, save_path='', smooth=50):
    if lbs is None:
        lbs = list(range(len(paths)))
    plt.style.use('ggplot')
    plt.figure(figsize=(6, 4), dpi=512)
    for path, lb in zip(paths, lbs):
        data = pds.read_csv(getpath(f'{path}/log.csv'))
        x = data['steps'].to_numpy()
        yraw = data['reward_sum'].to_numpy()

        y = [np.mean(yraw[max(s, s-smooth):min(len(x), s+smooth)]) for s in range(len(x))]
        plt.plot(x, y, label=lb)
    # plt.ylim((0, 200))
    plt.legend(ncol=2, loc='lower right')
    plt.tight_layout()
    if save_path:
        plt.savefig(getpath(save_path))
    else:
        plt.show()

def plot_step_tests(paths, lbs=None, save_path=''):
    if lbs is None:
        lbs = list(range(len(paths)))
    # plt.figure(figsize=(6, 4), dpi=512)
    fig, ax1 = plt.subplots(1, 1, figsize=(8, 3), dpi=256)
    ax2 = ax1.twinx()

    for path, lb in zip(paths, lbs):
        data = pds.read_csv(getpath(f'{path}/step_tests.csv'))
        x = data['step'].to_numpy()
        yr = data['r-avg'].to_numpy()
        yrstd = data['r-std'].to_numpy()
        ydivs = data['mpd-hm'].to_numpy()
        # y = [np.mean(yraw[max(s, s-smooth):min(len(x), s+smooth)]) for s in range(len(x))]
        ax1.plot(x, yr, label=f'{lb} task reward')
        ax1.fill_between(x, yr - yrstd, yr + yrstd, alpha=0.2)
        ax2.plot(x, ydivs, ls='--', label=f'{lb} diversity')

    # ax1.set_ylabel('Task Reward')
    # ax1.set_xlabel('Time Step')
    # ax1.set_ylim(rlim)
    # ax1.set_yticks([int(v) for v in np.linspace(rlim[0], rlim[-1], 8)])

    ax2.set_ylabel('Diversity')#, color=c2)
    # ax2.set_yticks([0, 10, 20, 30, 40, 50, 60, 70])
    # ax2.set_ylim((0, 70))

    ax2.grid(b=None)
    ax1.legend(loc='upper left', bbox_to_anchor=(1.2, 1.0), borderaxespad=0.)
    ax2.legend(loc='lower left', bbox_to_anchor=(1.2, -0.15), borderaxespad=0.)

    plt.tight_layout()
    if save_path:
        plt.savefig(getpath(save_path))
    else:
        plt.show()

def plot_step_divs(path, save_path=''):
    plt.figure(figsize=(6, 4), dpi=512)
    data = pds.read_csv(getpath(f'{path}/step_tests.csv'))
    x = data['step'].to_numpy()
    # yr = data['r-avg'].to_numpy()
    # yrstd = data['r-std'].to_numpy()
    # ydivs = data['mnd'].to_numpy()
    y1 = data['mnd-hm'].to_numpy()
    y2 = data['mnd-dtw'].to_numpy()
    y3 = data['mpd-hm'].to_numpy()
    y4 = data['mpd-dtw'].to_numpy()
    # y = [np.mean(yraw[max(s, s-smooth):min(len(x), s+smooth)]) for s in range(len(x))]
    plt.plot(x, y1, label='MND-Hamming')
    plt.plot(x, y2, label='MND-DTW')
    plt.plot(x, y3, label='MPD-Hamming')
    plt.plot(x, y4, label='MPD-DTW')
    # plt.ylim((0, 200))
    plt.legend(ncol=2, loc='upper right')
    plt.tight_layout()
    if save_path:
        plt.savefig(getpath(save_path))
    else:
        plt.show()

def print_compare_tab():
    def _print_line(_data):
        means = _data.mean(axis=-1)
        stds = _data.std(axis=-1)
        max_i, min_i = np.argmax(means), np.argmin(means)
        mean_str_content = [*map(lambda x: '%.4g' % x, _data.mean(axis=-1))]
        mean_str_content[max_i] = r'\textbf{%s}' % mean_str_content[max_i]
        mean_str_content[min_i] = r'\textit{%s}' % mean_str_content[min_i]
        std_str_content = [*map(lambda x: '$\pm$%.3g' % x, _data.std(axis=-1))]
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
            for path in glob.glob(getpath('training_data', fd, '**', 'performance.csv'), recursive=True):
                reward, div = load_singlerow_csv(path, 'r-avg', 'mpd-hm')
                rewards[-1].append(reward)
                divs[-1].append(div)
        rewards = np.array(rewards)
        divs = np.array(divs)
        print('    & \\multirow{2}{*}{Reward}')
        _print_line(rewards)
        print('    \\cline{2-14}')
        print('    & \\multirow{2}{*}{Diversity}')
        _print_line(divs)
        # print('    \\cline{2-14}')
        # print('    & \\multirow{2}{*}{Sum}')
        # rdsum = rewards + divs
        # _print_line(rdsum)
        print('    \\cline{2-14}')
        print('    & \\multirow{2}{*}{G-mean}')
        gmean = np.sqrt(rewards * divs)
        _print_line(gmean)
        rank_r = np.argsort(rewards)
        print(rank_r)

    print('    \\multirow{6}{*}{MarioPuzzle}')
    _print_block('fhp')
    print('    \\midrule')
    print('    \\multirow{6}{*}{MultiFacet}')
    _print_block('lgp')
    pass

def print_param_analysis_tab(task):
    def _print_line(_data):
        means = _data.mean(axis=-1)
        stds = _data.std(axis=-1)
        max_i, min_i = np.argmax(means), np.argmin(means)
        mean_str_content = [*map(lambda x: '%.4g' % x, _data.mean(axis=-1))]
        mean_str_content[max_i] = r'\textbf{%s}' % mean_str_content[max_i]
        mean_str_content[min_i] = r'\textit{%s}' % mean_str_content[min_i]
        std_str_content = [*map(lambda x: '$\pm$%.3g' % x, _data.std(axis=-1))]
        std_str_content[max_i] = r'\textbf{%s}' % std_str_content[max_i]
        std_str_content[min_i] = r'\textit{%s}' % std_str_content[min_i]
        print('    &', ' & '.join(mean_str_content), r'\\')
        print('    & &', ' & '.join(std_str_content), r'\\')
        pass
    def _print_block(_task):
        fds = [
            f'sac/{_task}', f'egsac/{_task}', f'asyncsac/{_task}',
            f'pesac/{_task}', f'pmoe/{_task}', f'dvd/{_task}', f'sunrise/{_task}',
            f'varpm-{_task}/l0.0_m5', f'varpm-{_task}/l0.1_m5', f'varpm-{_task}/l0.2_m5',
            f'varpm-{_task}/l0.3_m5', f'varpm-{_task}/l0.4_m5', f'varpm-{_task}/l0.5_m5'
        ]
        rewards, divs = [], []
        for fd in fds:
            rewards.append([])
            divs.append([])
            # print(getpath())
            for path in glob.glob(getpath('training_data', fd, '**', 'performance.csv'), recursive=True):
                reward, div = load_singlerow_csv(path, 'r-avg', 'mpd-hm')
                rewards[-1].append(reward)
                divs[-1].append(div)
        rewards = np.array(rewards)
        divs = np.array(divs)
        print('    & \\multirow{2}{*}{Reward}')
        _print_line(rewards)
        print('    \\cline{2-14}')
        print('    & \\multirow{2}{*}{Diversity}')
        _print_line(divs)
        # print('    \\cline{2-14}')
        # print('    & \\multirow{2}{*}{Sum}')
        # rdsum = rewards + divs
        # _print_line(rdsum)
        print('    \\cline{2-14}')
        print('    & \\multirow{2}{*}{G-mean}')
        gmean = np.sqrt(rewards * divs)
        _print_line(gmean)

    print('    \\multirow{6}{*}{MarioPuzzle}')
    _print_block('fhp')
    print('    \\midrule')
    print('    \\multirow{6}{*}{MultiFacet}')
    _print_block('lgp')

def plot_varpm_heat(task, name):
    def _get_score(m, l):
        fd = getpath('training_data', f'varpm-{task}', f'l{l}_m{m}')
        rewards, divs = [], []
        for i in range(5):
            reward, div = load_performance(f'{fd}/t{i+1}', 'r-avg', 'mpd-hm')
            rewards.append(reward)
            divs.append(div)
        gmean = [sqrt(r * d) for r, d in zip(rewards, divs)]
        return np.mean(rewards), np.std(rewards), \
            np.mean(divs), np.std(divs), \
            np.mean(gmean), np.std(gmean)

    def _plot_map(avg_map, std_map, criterion):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3), dpi=256, width_ratios=(1, 1))
        heat = ax1.imshow(avg_map, cmap='spring')
        heat = ax2.imshow(std_map, cmap='spring')
        ax1.set_xlim([-0.5, 5.5])
        ax1.set_xticks([0, 1, 2, 3, 4, 5], ['$\lambda$=0.0', '$\lambda$=0.1', '$\lambda$=0.2', '$\lambda$=0.3', '$\lambda$=0.4', '$\lambda$=0.5'])
        ax1.set_ylim([-0.5, 3.5])
        ax1.set_yticks([0, 1, 2, 3], ['m=5', 'm=4', 'm=3', 'm=2'])
        ax1.set_title('Average')
        for x, y in product([0, 1, 2, 3, 4, 5], [0, 1, 2, 3]):
            v = avg_map[y, x]
            ax1.text(x, y, '%.3g' % v, va='center', ha='center')
        plt.colorbar(heat, ax=ax1, shrink=0.9)
        ax2.set_xlim([-0.5, 5.5])
        ax2.set_xticks([0, 1, 2, 3, 4, 5], ['$\lambda$=0.0', '$\lambda$=0.1', '$\lambda$=0.2', '$\lambda$=0.3', '$\lambda$=0.4', '$\lambda$=0.5'])
        ax2.set_ylim([-0.5, 3.5])
        ax2.set_yticks([0, 1, 2, 3], ['m=5', 'm=4', 'm=3', 'm=2'])
        for x, y in product([0, 1, 2, 3, 4, 5], [0, 1, 2, 3]):
            v = std_map[y, x]
            ax2.text(x, y, '%.3g' % v, va='center', ha='center')
        ax2.set_title('Standard Deviation')
        plt.colorbar(heat, ax=ax2, shrink=0.9)

        fig.suptitle(f'{name}: {criterion}', fontsize=14)
        plt.tight_layout()
        plt.show()
        pass
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

def expressive_range(levels):
    pass



if __name__ == '__main__':
    # plot_crosstest_scatters('fhp', title='MarioPuzzle')
    # plot_crosstest_scatters('lgp', title='MultiFacet')

    # plot_crosstest_scatters('fhp', yrange=(0, 100), xrange=(20, 120), title='MarioPuzzle')
    # plot_crosstest_scatters('lgp', yrange=(0, 60), xrange=(40, 100), title='MultiFacet')
    # plot_crosstest_scatters('lgp', yrange=(0, 35), xrange=(90, 98), title=' ')

    # plot_step_divs('training_data/param-sens/l0.1_m5/t5')
    # plot_step_tests(
    #     ['training_data/param-sens/l0.5_m5/t1', 'training_data/param-sens/l0.5_m5/t2'],
    #     ['Trial1', 'Trial2']
    # )
    print_compare_tab()

    # plot_cmp_learning_curves('fhp', save_path='analysis/results/learning_curves/fhp.png', title='MarioPuzzle')
    # plot_cmp_learning_curves('lgp', save_path='analysis/results/learning_curves/lgp.png', title='MultiFacet')

    # plot_varpm_heat('fhp', 'MarioPuzzle')
    # plot_varpm_heat('lgp', 'MultiFacet')
    pass
