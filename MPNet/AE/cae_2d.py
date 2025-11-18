import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------- Residual Block ----------------
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
    def __init__(self, mask_size=160, latent_dim=128):
        super().__init__()
        self.enc1 = nn.Sequential(ResBlock(1,32), nn.MaxPool2d(2))
        self.enc2 = nn.Sequential(ResBlock(32,64), nn.MaxPool2d(2))
        self.enc3 = nn.Sequential(ResBlock(64,128), nn.MaxPool2d(2))
        self.enc4 = nn.Sequential(ResBlock(128,256), nn.MaxPool2d(2))

        conv_out_size = (mask_size // 16) ** 2 * 256
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.SiLU(),
            nn.Linear(512, latent_dim)
        )

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        latent = self.fc(e4.flatten(1))
        return latent, [e1,e2,e3,e4]

# ---------------- Decoder ----------------
class Decoder_CNN_2D(nn.Module):
    def __init__(self, latent_dim=128, feature_map_size=10):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 256*feature_map_size*feature_map_size),
            nn.SiLU()
        )
        self.up1 = ResBlock(256+128,128)
        self.up2 = ResBlock(128+64,64)
        self.up3 = ResBlock(64+32,32)
        self.up4 = ResBlock(32,16)
        self.final = nn.Conv2d(16,1,3,padding=1)

    def forward(self, latent, enc_feats):
        e1,e2,e3,e4 = enc_feats
        x = self.fc(latent).view(latent.size(0),256,e4.size(2),e4.size(3))
        x = F.interpolate(x, scale_factor=2)
        x = self.up1(torch.cat([x,e3],dim=1))
        x = F.interpolate(x, scale_factor=2)
        x = self.up2(torch.cat([x,e2],dim=1))
        x = F.interpolate(x, scale_factor=2)
        x = self.up3(torch.cat([x,e1],dim=1))
        x = F.interpolate(x, scale_factor=2)
        x = self.up4(x)
        x = self.final(x)
        return x
