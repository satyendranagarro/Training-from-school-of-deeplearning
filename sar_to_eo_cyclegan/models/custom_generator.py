import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_norm=True):
        super().__init__()
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)]
        if use_norm:
            layers.append(nn.InstanceNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)

class SmallGenerator(nn.Module):
    def __init__(self, in_channels=2, out_channels=3):
        super().__init__()
        features = 64

        self.encoder = nn.Sequential(
            ConvBlock(in_channels, features),
            ConvBlock(features, features * 2),
            ConvBlock(features * 2, features * 4),
        )

        self.middle = nn.Sequential(
            ConvBlock(features * 4, features * 4),
            ConvBlock(features * 4, features * 4)
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(features * 4, features * 2, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(features * 2),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(features * 2, features, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(features),
            nn.ReLU(inplace=True),

            nn.Conv2d(features, out_channels, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.middle(x)
        x = self.decoder(x)
        return x