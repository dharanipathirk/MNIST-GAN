import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.model = nn.Sequential(
            nn.ConvTranspose2d(config['latent_dim'], 1024, 3, 1, 0, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(True),
            nn.ConvTranspose2d(1024, 512, 3, 2, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, config['channels'], 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, z):
        return self.model(z)
