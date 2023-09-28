import torch
from analysis.generate import generate_levels
from src.drl.ac_agents import SAC
from src.drl.rep_mem import ReplayMem
from src.olgen.olg_policy import RLGenPolicy
from src.utils.filesys import getpath


class SyncOffPolicyTrainer:
    def __init__(self, env, step_budget, update_freq=2, batch_size=256, rep_mem=None, save_path = '.', check_points = None):
        self.env = env
        self.n_parallel = env.num_envs
        self.step_budget = step_budget
        self.update_freq = update_freq
        self.batch_size = batch_size
        self.rep_mem = ReplayMem() if rep_mem is None else rep_mem
        self.steps = 0
        # self.check_points = [] if not check_points else check_points
        # self.check_points.sort(reverse=True)
        self.save_path = save_path

    def train(self, agent: SAC, gen_period=10000, gen_num=200):
        self.steps = 0
        obs = self.env.reset()
        print('Start to train SAC')
        obs_buffer = [[] for _ in range(self.env.num_envs)]
        action_buffer = [[] for _ in range(self.env.num_envs)]
        next_obs_buffer = [[] for _ in range(self.env.num_envs)]
        new_transitions = 0

        gen_horizon = gen_period
        if gen_num > 0:
            generate_levels(
                RLGenPolicy(agent.actor.net, self.env.hist_len), getpath(self.save_path, 'gen_log'),
                f'step{self.steps}', gen_num, self.env.eplen
            )
        while self.steps < self.step_budget:
            actions = agent.make_decision(obs)
            next_obs, _, dones, infos = self.env.step(actions)
            for i, (ob, action, next_ob, done, info) in enumerate(zip(obs, actions, next_obs, dones, infos)):
                obs_buffer[i].append(ob)
                action_buffer[i].append(action)
                next_obs_buffer[i].append(next_ob)
                if done:
                    next_obs_buffer[i].append(info['terminal_observation'])
                else:
                    next_obs_buffer[i].append(next_ob)

            del obs
            obs = next_obs
            for i, (done, info) in enumerate(zip(dones, infos)):
                if not done:
                    continue
                reward_lists = []
                for key in info.keys():
                    if 'reward_list' not in key:
                        continue
                    reward_lists.append(info[key])
                rewards = []

                for j in range(len(reward_lists[0])):
                    step_reward = 0
                    for item in reward_lists:
                        step_reward += item[j]
                    rewards.append(step_reward)
                self.rep_mem.add_batched(obs_buffer[i], action_buffer[i], rewards, next_obs_buffer[i])
                obs_buffer[i].clear()
                action_buffer[i].clear()
                next_obs_buffer[i].clear()

                new_transitions += len(reward_lists[0])
            if new_transitions > self.update_freq and len(self.rep_mem) > self.batch_size:
                update_times = new_transitions // self.update_freq
                for _ in range(update_times):
                    batch_data = self.rep_mem.sample(self.batch_size)
                    agent.update(*batch_data)
                new_transitions = new_transitions % self.update_freq
            self.steps += self.n_parallel

            # generate levels
            if self.steps >= gen_horizon and gen_num > 0:
                genpolicy = RLGenPolicy(agent.actor.net, self.env.hist_len)
                generate_levels(genpolicy, getpath(self.save_path, 'gen_log'), f'step{self.steps}', gen_num, self.env.eplen)
                gen_horizon += gen_period

            # if len(self.check_points) and self.steps >= self.check_points[-1]:
            #     check_point_path = getpath(self.save_path + f'/model_at_{self.steps}')
            #     # os.makedirs(check_point_path, exist_ok=True)
            #     # agent.save(check_point_path)
            #     self.check_points.pop()
            #     pass
        torch.save(agent.actor.net, getpath(f'{self.save_path}/policy.pth'))

        # agent.save(self.save_path)
        pass