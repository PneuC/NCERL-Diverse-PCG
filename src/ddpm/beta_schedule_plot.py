import torch
import matplotlib.pyplot as plt

def linear_beta_schedule(timesteps):
    beta_start = 1e-4
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timesteps)

def quadratic_beta_schedule(timesteps):
    beta_start = 1e-4
    beta_end = 0.02
    return torch.linspace(beta_start**0.5, beta_end**0.5, timesteps) ** 2

def sigmoid_beta_schedule(timesteps):
    beta_start = 1e-4
    beta_end = 0.02
    s = torch.tensor(6.0)
    betas = torch.sigmoid(torch.linspace(-s, s, timesteps))
    return betas * (beta_end - beta_start) + beta_start

def plot_beta_schedules(timesteps):
    plt.figure(figsize=(10, 6))
    plt.plot(linear_beta_schedule(timesteps), label='Linear')
    plt.plot(quadratic_beta_schedule(timesteps), label='Quadratic')
    plt.plot(sigmoid_beta_schedule(timesteps), label='Sigmoid')
    plt.xlabel('Timestep')
    plt.ylabel('Beta')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    timesteps = 1000
    plot_beta_schedules(timesteps)
