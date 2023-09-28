import torch
import random
from src.utils.datastruct import RingQueue


class ReplayMem:
    def __init__(self, capacity=int(2e5), device='cpu'):
        self.queue = RingQueue(capacity)
        self.device = device

    def add(self, o, a, r, op):
        transition = (
            torch.tensor(o, device=self.device, dtype=torch.float),
            torch.tensor(a, device=self.device, dtype=torch.float),
            r,
            torch.tensor(op, device=self.device, dtype=torch.float)
        )
        self.queue.push(transition)

    def add_transitions(self, transitions):
        for transition in transitions:
            self.add(*transition)

    def add_batched(self, obs, actions, rewards, next_obs):
        for o, a, r, op in zip(obs, actions, rewards, next_obs):
            self.add(o, a, r, op)

    def sample(self, n):
        tuples = random.sample(self.queue.main, n)
        obs, acts, rews, ops = [], [], [], []
        for o, a, r, op in tuples:
            obs.append(o)
            acts.append(a)
            rews.append(r)
            ops.append(op)
        obs = torch.stack(obs)
        acts = torch.stack(acts)
        rews = torch.tensor(rews, device=self.device, dtype=torch.float)
        ops = torch.stack(ops)
        return obs, acts, rews, ops

    def __len__(self):
        return len(self.queue)

    def clear(self):
        self.queue.clear()


if __name__ == '__main__':
    # mem = ReplayMem()
    # mem.add_batched(
    #     [np.zeros([5]), np.zeros([5]), np.zeros([5]), np.zeros([5])],
    #     [np.zeros([5]), np.zeros([5]), np.zeros([5]), np.zeros([5])],
    #     [np.zeros([5]), np.zeros([5]), np.zeros([5]), np.zeros([5])],
    #     [np.zeros([5]), np.zeros([5]), np.zeros([5]), np.zeros([5])]
    # )
    # mem.sample(2, 'cuda:0')
    # mem.sample(2, 'cuda:0')
    # mem.sample(2, 'cuda:0')
    # mem.sample(2, 'cuda:0')
    # mem.sample(2, 'cuda:0')
    # mem.sample(2, 'cuda:0')
    pass
