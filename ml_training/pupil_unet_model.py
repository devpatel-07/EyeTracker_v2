"""Tiny U-Net architecture used for pupil-mask segmentation."""

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, inputs):
        return self.layers(inputs)


class TinyUNet(nn.Module):
    def __init__(self, base_channels=16):
        super().__init__()
        channels = base_channels
        self.encoder1 = ConvBlock(1, channels)
        self.encoder2 = ConvBlock(channels, channels * 2)
        self.encoder3 = ConvBlock(channels * 2, channels * 4)
        self.bottleneck = ConvBlock(channels * 4, channels * 8)
        self.pool = nn.MaxPool2d(2)

        self.up3 = nn.ConvTranspose2d(channels * 8, channels * 4, 2, stride=2)
        self.decoder3 = ConvBlock(channels * 8, channels * 4)
        self.up2 = nn.ConvTranspose2d(channels * 4, channels * 2, 2, stride=2)
        self.decoder2 = ConvBlock(channels * 4, channels * 2)
        self.up1 = nn.ConvTranspose2d(channels * 2, channels, 2, stride=2)
        self.decoder1 = ConvBlock(channels * 2, channels)
        self.output = nn.Conv2d(channels, 1, 1)

    def forward(self, inputs):
        encoder1 = self.encoder1(inputs)
        encoder2 = self.encoder2(self.pool(encoder1))
        encoder3 = self.encoder3(self.pool(encoder2))
        bottleneck = self.bottleneck(self.pool(encoder3))

        decoder3 = self.decoder3(torch.cat((self.up3(bottleneck), encoder3), dim=1))
        decoder2 = self.decoder2(torch.cat((self.up2(decoder3), encoder2), dim=1))
        decoder1 = self.decoder1(torch.cat((self.up1(decoder2), encoder1), dim=1))
        return self.output(decoder1)
