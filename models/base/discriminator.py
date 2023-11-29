import torch
import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.model = nn.Sequential(
            nn.Conv2d(config['channels'], 256, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1024, 3, 2, 0, bias=False),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(1024, 1, 3, 1, 0, bias=False),
        )

    def forward(self, img):
        out = self.model(img)
        return (
            out.view(-1, 1)
            if self.config['use_wasserstein']
            else torch.sigmoid(out).view(-1, 1)
        )
