import torch
from torch.distributions import Categorical
from tqdm import trange
from src.ddpm.dataset import create_dataloader
import numpy as np
import os
from pathlib import Path
from src.smb.level import MarioLevel

DATAPATH = os.path.join(Path(__file__).parent.resolve(), "levels", "ground", "unique_onehot.npz")

class Diffusion:
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, img_size=MarioLevel.seg_width, device="cuda", schedule='linear'):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.img_size = img_size
        self.device = device
        self.schedule = schedule

        # Prepare noise schedule according to the selected schedule type
        self.beta = self.prepare_noise_schedule().to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

    def prepare_noise_schedule(self):
        # Linear beta schedule
        if self.schedule == 'linear':
            return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)
        
        # Quadratic beta schedule
        elif self.schedule == 'quadratic':
            return (torch.linspace(self.beta_start**0.5, self.beta_end**0.5, self.noise_steps) ** 2)
        
        # Sigmoid beta schedule
        elif self.schedule == 'sigmoid':
            s = torch.tensor(10.0)
            betas = torch.sigmoid(torch.linspace(-s, s, self.noise_steps))
            return betas * (self.beta_end - self.beta_start) + self.beta_start

        else:
            raise ValueError("Invalid schedule type. Supported schedules: 'linear', 'quadratic', and 'sigmoid'.")

    def noise_images(self, x_0, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1. - self.alpha_hat[t])[:, None, None, None]
        eps = torch.randn_like(x_0)
        return sqrt_alpha_hat * x_0 + sqrt_one_minus_alpha_hat * eps, eps

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))
    
    def sample_only_final(self, x_t, t, noise, temperature=None):
        '''
        using the reparameterization trick we represent x_0 and return categorical distribution
        '''
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1. - self.alpha_hat[t])[:, None, None, None]
        logits = (1.0/sqrt_alpha_hat) * (x_t - sqrt_one_minus_alpha_hat * noise)

        # Check if a temperature tensor is provided
        if temperature is not None:
            # Ensure the temperature tensor has the same shape as the logits tensor
            temperature = temperature.view(1, MarioLevel.n_types, 1, 1)
            # Divide the logits by the temperature tensor element-wise
            logits = logits / temperature

        return Categorical(logits=logits.permute(0, 2, 3, 1))

    def sample(self, model, n):
        print(f"Sampling {n} new images....")
        imgs = []
        model.eval()
        with torch.no_grad():
            x = torch.randn((n, MarioLevel.n_types, self.img_size, self.img_size)).to(self.device)
            imgs.append(x)
            for i in trange(self.noise_steps - 1, 0, -1, position=0):
                t = (torch.ones(n) * i).long().to(self.device)
                predicted_noise = model(x, t)
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
                if i % 100 == 1:
                    imgs.append(x)
        model.train()
        return imgs
    
if __name__ == '__main__':
    diffusion = Diffusion(schedule='quadratic')
    data = create_dataloader(DATAPATH).dataset.data_
    noise = torch.randn((MarioLevel.n_types, MarioLevel.seg_width, MarioLevel.height))
    level = data[-15]
    for t in [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]:
        noised_level = diffusion.noise_images(torch.tensor(level).to("cuda"), torch.tensor([t]).to("cuda"))[0]
        noised_level = noised_level[0].cpu().numpy()
        noised_level = np.argmax(noised_level, axis=0)
        # plt.imshow(get_img_from_level(noised_level))
        # plt.show()