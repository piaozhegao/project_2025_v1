import torch
import torch.nn as nn
import numpy as np
from monai.networks.blocks import Convolution

class UNet3DEncoder(nn.Module):
    def __init__(self,
                 in_channels=1,
                 channels=(16, 32, 64, 128, 256),
                 strides=(2,2,2,2),
                 num_res_units=2):
        super().__init__()
        self.context_blocks = nn.ModuleList()
        self.pooling = nn.ModuleList()
        c_in = in_channels
        for i, c_out in enumerate(channels):
            self.context_blocks.append(
                Convolution(
                    spatial_dims=3,
                    in_channels=c_in,
                    out_channels=c_out,
                    strides=1,
                    norm="batch"
                )
            )
            if i < len(strides):
                self.pooling.append(nn.MaxPool3d(strides[i]))
            c_in = c_out

    def forward(self, x):
        for i, block in enumerate(self.context_blocks):
            x = block(x)
            if i < len(self.pooling):
                x = self.pooling[i](x)
        return x

class UNet3DClassifier(nn.Module):
    def __init__(self,
                 in_channels=3,
                 channels=(16, 32, 64, 128, 256),
                 strides=(2,2,2,2),
                 num_res_units=2,
                 hidden_dim=256,
                 num_classes=2):
        super().__init__()

        self.encoder = UNet3DEncoder(in_channels, channels, strides, num_res_units)
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        self.classifier = nn.Sequential(
            nn.Linear(channels[-1], hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        logits = self.classifier(x)
        return logits