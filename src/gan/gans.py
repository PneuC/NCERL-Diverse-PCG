import torch
from torch import nn
from src.smb.level import MarioLevel
from src.utils.dl import SelfAttn

nz = 20


class SAGenerator(nn.Module):
    def __init__(self, base_channels=32):
        super(SAGenerator, self).__init__()
        self.main = nn.Sequential(
            nn.utils.spectral_norm(nn.ConvTranspose2d(nz, base_channels * 4, 4)),
            nn.BatchNorm2d(base_channels * 4), nn.ReLU(),
            nn.utils.spectral_norm(nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 4, 2, 1)),
            nn.BatchNorm2d(base_channels * 2), nn.ReLU(),
            SelfAttn(base_channels * 2),
            nn.utils.spectral_norm(nn.ConvTranspose2d(base_channels * 2, base_channels, 4, 2, 1)),
            nn.BatchNorm2d(base_channels), nn.ReLU(),
            SelfAttn(base_channels),
            nn.utils.spectral_norm(nn.ConvTranspose2d(base_channels, MarioLevel.n_types, 3, 1, 1)),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.main(x)


class SADiscriminator(nn.Module):
    def __init__(self, base_channels=32):
        super(SADiscriminator, self).__init__()
        self.main = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(MarioLevel.n_types, base_channels, 3, 1, 1)),
            nn.BatchNorm2d(base_channels), nn.LeakyReLU(0.1),
            SelfAttn(base_channels),
            nn.utils.spectral_norm(nn.Conv2d(base_channels, base_channels * 2, 4, 2, 1)),
            nn.BatchNorm2d(base_channels * 2), nn.LeakyReLU(0.1),
            SelfAttn(base_channels * 2),
            nn.utils.spectral_norm(nn.Conv2d(base_channels * 2, base_channels * 4, 4, 2, 1)),
            nn.BatchNorm2d(base_channels * 4), nn.LeakyReLU(0.1),
            nn.utils.spectral_norm(nn.Conv2d(base_channels * 4, 1, 4)),
            nn.Flatten()
        )

    def forward(self, x):
        return self.main(x)


if __name__ == '__main__':
    noise = torch.rand(2, nz, 1, 1) * 2 - 1
    netG = SAGenerator()
    netD = SADiscriminator()
    # print(netG)
    X = netG(noise)
    Y = netD(X)
    print(X.shape, Y.shape)
    pass

