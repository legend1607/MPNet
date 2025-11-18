import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------- ResBlock ----------------
class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.InstanceNorm2d(out_ch),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.InstanceNorm2d(out_ch)
        )
        self.shortcut = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.conv(x) + self.shortcut(x))


# ---------------- Encoder ----------------
class Encoder_CNN_2D(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        self.enc1 = nn.Sequential(ResBlock(1,32), nn.MaxPool2d(2))   # /2
        self.enc2 = nn.Sequential(ResBlock(32,64), nn.MaxPool2d(2))  # /4
        self.enc3 = nn.Sequential(ResBlock(64,128), nn.MaxPool2d(2)) # /8
        self.enc4 = nn.Sequential(ResBlock(128,256), nn.MaxPool2d(2))# /16

        self.spatial_size = None  # 由输入自动推断

        self.fc1 = nn.Linear(256, 256)
        self.fc2 = nn.Linear(256, latent_dim)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)        # [B, 256, H/16, W/16]

        # 推断 spatial size
        if self.spatial_size is None:
            self.spatial_size = e4.shape[-1]

        # GAP
        h = F.adaptive_avg_pool2d(e4, 1).flatten(1)

        # 全连接
        latent = self.fc2(F.silu(self.fc1(h)))

        return latent, [e1, e2, e3, e4]


# ---------------- Decoder ----------------
class Decoder_CNN_2D(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()

        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.SiLU()
        )

        self.up1 = ResBlock(256 + 128, 128)
        self.up2 = ResBlock(128 + 64, 64)
        self.up3 = ResBlock(64 + 32, 32)
        self.up4 = ResBlock(32, 16)
        self.final = nn.Conv2d(16, 1, 3, padding=1)

    def forward(self, latent, enc_feats):
        e1, e2, e3, e4 = enc_feats

        B = latent.size(0)
        spatial = e4.shape[-1]

        # 还原卷积特征图
        x = self.fc(latent).unsqueeze(-1).unsqueeze(-1)
        x = x.repeat(1, 1, spatial, spatial)

        x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = self.up1(torch.cat([x, e3], dim=1))

        x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = self.up2(torch.cat([x, e2], dim=1))

        x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = self.up3(torch.cat([x, e1], dim=1))

        x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = self.up4(x)

        return torch.sigmoid(self.final(x))
