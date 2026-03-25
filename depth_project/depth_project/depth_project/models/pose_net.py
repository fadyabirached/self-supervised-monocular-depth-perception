import torch
import torch.nn as nn


def conv_block(in_ch, out_ch, k=3, s=2, p=1):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p),
        nn.ReLU(inplace=True),
    )


class PoseNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            conv_block(6, 32, 7, 2, 3),
            conv_block(32, 64, 5, 2, 2),
            conv_block(64, 128, 3, 2, 1),
            conv_block(128, 256, 3, 2, 1),
            conv_block(256, 256, 3, 2, 1),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(256, 6)

    def forward(self, tgt, src):
        x = torch.cat([tgt, src], dim=1)
        x = self.net(x)
        x = self.pool(x).flatten(1)
        x = 0.01 * self.fc(x)
        axisangle = x[:, :3]
        translation = x[:, 3:]
        return axisangle, translation
