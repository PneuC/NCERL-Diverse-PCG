import json
import os
import csv
import numpy as np
from itertools import product
from src.smb.level import save_batch
from src.utils.filesys import getpath

############### Loggers for async environment ###############
class AsyncCsvLogger:
    def __init__(self, target, rfunc, buffer_size=50):
        self.rterms = tuple(term.get_name() for term in rfunc.terms)
        self.cols = ('steps', *self.rterms, 'reward_sum', 'time', 'trans', 'updates', '')
        self.buffer = []
        self.buffer_size = buffer_size
        self.ftarget = open(getpath(target), 'w', newline='')
        self.writer = csv.writer(self.ftarget)
        self.writer.writerow(self.cols)

    def on_episode(self, **kwargs):
        for rews in kwargs['rewss']:
            rews_list = [sum(rews[key]) for key in self.rterms]
            self.buffer.append([
                kwargs['steps'], *rews_list, sum(rews_list),
                kwargs['time'], kwargs['trans'], kwargs['updates']
            ])
        self.__try_write()
        if kwargs['close']:
            self.close()

    def __try_write(self):
        if len(self.buffer) < self.buffer_size:
            return
        self.writer.writerows(self.buffer)
        self.ftarget.flush()
        self.buffer.clear()

    def close(self):
        self.writer.writerows(self.buffer)
        self.ftarget.close()
        pass


class AsyncStdLogger:
    def __init__(self, rfunc, itv=2000, path=''):
        self.rterms = tuple(term.get_name() for term in rfunc.terms)
        self.rews = {rterm: 0. for rterm in self.rterms}
        self.n = 0
        self.itv = itv
        self.horizon = itv
        if not len(path):
            self.f = None
        else:
            self.f = open(getpath(path), 'w')
        self.last_steps = 0
        self.last_trans = 0
        self.last_updates = 0
        self.buffer = []
        pass

    def on_episode(self, **kwargs):
        newrews = {k: self.rews[k] for k in self.rterms}
        for rews, k in product(kwargs['rewss'], self.rterms):
            newrews[k] = newrews[k] + sum(rews[k])
        self.rews = newrews
        self.n += len(kwargs['rewss'])
        if kwargs['steps'] >= self.horizon or kwargs['close']:
            self.__output(**kwargs)
            self.horizon += self.itv
            self.rews = {k: 0 for k in self.rews.keys()}
            self.n = 0
            self.last_steps = kwargs['steps']
            self.last_trans = kwargs['trans']
            self.last_updates = kwargs['updates']
        if kwargs['close'] and self.f is not None:
            self.f.close()

    def __output(self, **kwargs):
        steps = kwargs['steps']
        if kwargs['close']:
            head = '-' * 20 + 'Closing rollouts' + '-' * 20
        else:
            head = '-' * 20 + f'Rollout of {self.last_steps}-{steps} steps' + '-' * 20
        self.buffer.append(head)
        rsum = 0
        for t in self.rterms:
            v = 0 if self.n == 0 else self.rews[t] / self.n
            rsum += v
            self.buffer.append(f'{t}: {v:.2f}')
        self.buffer.append(f'Reward sum: {rsum: .2f}')
        self.buffer.append('Time elapsed: %.1fs' % kwargs['time'])
        self.buffer.append('Transitions collected: %d (%d in total)' % (kwargs['trans'] - self.last_trans, kwargs['trans']))
        self.buffer.append('Number of updates: %d (%d in total)' % (kwargs['updates']- self.last_updates, kwargs['updates']))
        if self.f is None:
            print('\n'.join(self.buffer) + '\n')
        else:
            self.f.write('\n'.join(self.buffer) + '\n')
            self.f.flush()
        self.buffer.clear()
        pass

    def __reset(self):
        pass


class GenResLogger:
    def __init__(self, root_path, k, itv=5000):
        self.k = k
        self.itv = itv
        self.horizon = 0
        self.path = getpath(f'{root_path}/gen_log')
        os.makedirs(self.path, exist_ok=True)

    def on_episode(self, env, agent, steps):
        if steps >= self.horizon:
            lvls, vectraj = env.generate_levels(agent, self.k)
            # np.save(f'{self.path}/step{steps}', vectraj)
            if len(lvls):
                save_batch(lvls, f'{self.path}/step{steps}')
            self.horizon += self.itv
        pass
    pass


############### Loggers for sync environment, from https://github.com/SUSTechGameAI/MFEDRL ###############
class InfoCollector:
    ignored_keys = {'episode', 'terminal_observation'}
    save_itv = 1000

    def __init__(self, path, log_itv=100, log_targets=None):
        self.data = []
        self.path = path
        self.msg_itv = log_itv
        self.time_before_save = InfoCollector.save_itv
        self.msg_ptr = 0
        self.log_targets = [] if log_targets is None else log_targets
        if 'file' in log_targets:
            with open(f'{self.path}/log.txt', 'w') as f:
                f.write('')
        self.recent_time = 0

    def on_step(self, dones, infos):
        for done, info in zip(dones, infos):
            if done:
                self.data.append({
                    key: val for key, val in info.items()
                    if key not in InfoCollector.ignored_keys and 'reward_list' not in key
                })
                self.time_before_save -= 1
        if self.time_before_save <= 0:
            with open(f'{self.path}/ep_infos.json', 'w') as f:
                json.dump(self.data, f)
            self.time_before_save += InfoCollector.save_itv

        if self.log_targets and 0 < self.msg_itv <= (len(self.data) - self.msg_ptr):
            keys = set(self.data[-1].keys()) - {'TotalSteps', 'TimePassed', 'TotalScore', 'EpLen'}

            msg = '%sTotal steps: %d%s\n' % ('-' * 16, self.data[-1]['TotalSteps'], '-' * 16)
            msg += 'Time passed: %ds\n' % self.data[-1]['TimePassed']
            t = self.data[-1]['TimePassed'] - self.recent_time
            self.recent_time = self.data[-1]['TimePassed']
            f = sum(item['EpLength'] for item in self.data[self.msg_ptr:])
            msg += 'fps: %.3g\n' % (f/t)
            for key in keys:
                values = [item[key] for item in self.data[self.msg_ptr:]]
                values = np.array(values)
                msg += '%s: %.2f +- %.2f\n' % (key, values.mean(), values.std())
            values = [item['TotalScore'] for item in self.data[self.msg_ptr:]]
            values = np.array(values)
            msg += 'TotalScore: %.2f +- %.2f\n' % (values.mean(), values.std())

            if 'file' in self.log_targets:
                with open(f'{self.path}/log.txt', 'a') as f:
                    f.write(msg + '\n')
            if 'std' in self.log_targets:
                print(msg)
            self.msg_ptr = len(self.data)
            pass

    def close(self):
        with open(self.path, 'w') as f:
            json.dump(self.data, f)
