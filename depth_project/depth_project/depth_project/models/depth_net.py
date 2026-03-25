import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_block(in_ch, out_ch, k=3, s=1, p=1):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class DepthNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.e1 = nn.Sequential(conv_block(3, 32, 7, 2, 3), conv_block(32, 32))
        self.e2 = nn.Sequential(conv_block(32, 64, 5, 2, 2), conv_block(64, 64))
        self.e3 = nn.Sequential(conv_block(64, 128, 3, 2, 1), conv_block(128, 128))
        self.e4 = nn.Sequential(conv_block(128, 256, 3, 2, 1), conv_block(256, 256))

        self.up3 = nn.ConvTranspose2d(256, 128, 4, 2, 1)
        self.d3 = nn.Sequential(conv_block(256, 128), conv_block(128, 128))

        self.up2 = nn.ConvTranspose2d(128, 64, 4, 2, 1)
        self.d2 = nn.Sequential(conv_block(128, 64), conv_block(64, 64))

        self.up1 = nn.ConvTranspose2d(64, 32, 4, 2, 1)
        self.d1 = nn.Sequential(conv_block(64, 32), conv_block(32, 32))

        self.up0 = nn.ConvTranspose2d(32, 16, 4, 2, 1)
        self.out = nn.Conv2d(16, 1, 3, 1, 1)

        self.embed_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
        )

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(e1)
        e3 = self.e3(e2)
        e4 = self.e4(e3)

        u3 = self.up3(e4)
        d3 = self.d3(torch.cat([u3, e3], dim=1))

        u2 = self.up2(d3)
        d2 = self.d2(torch.cat([u2, e2], dim=1))

        u1 = self.up1(d2)
        d1 = self.d1(torch.cat([u1, e1], dim=1))

        u0 = self.up0(d1)
        disp = torch.sigmoid(self.out(u0))
        emb = F.normalize(self.embed_head(e4), dim=1)
        return disp, emb
