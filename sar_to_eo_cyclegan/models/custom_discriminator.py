import torch
import torch.nn as nn

class SmallDiscriminator(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        features = 32

        self.model = nn.Sequential(
            nn.Conv2d(in_channels, features, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(features, features * 2, 4, 2, 1),
            nn.InstanceNorm2d(features * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(features * 2, features * 4, 4, 2, 1),
            nn.InstanceNorm2d(features * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(features * 4, 1, 4, 1, 0)
        )

    def forward(self, x):
        return self.model(x)