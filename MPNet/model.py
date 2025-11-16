import torch
import torch.nn as nn
import torch.nn.functional as F

class MLPPathPredictor(nn.Module):
    def __init__(self, input_size, num_sector_theta=8, num_sector_phi=16, dropout_p=0.0):
        super().__init__()
        # 共享特征
        self.shared_fc = nn.Sequential(
            nn.Linear(input_size, 1280), nn.PReLU(), nn.Dropout(p=dropout_p),
            nn.Linear(1280, 1024), nn.PReLU(), nn.Dropout(p=dropout_p),
            nn.Linear(1024, 896), nn.PReLU(), nn.Dropout(p=dropout_p),
            nn.Linear(896, 768), nn.PReLU(), nn.Dropout(p=dropout_p),
            nn.Linear(768, 512), nn.PReLU(), nn.Dropout(p=dropout_p),
            nn.Linear(512, 384), nn.PReLU()
        )
        # 步长回归
        self.fc_norm = nn.Sequential(
            nn.Linear(384, 1280), nn.PReLU(), nn.Dropout(p=dropout_p),
            nn.Linear(1280, 640), nn.PReLU(), nn.Dropout(p=dropout_p),
            nn.Linear(640, 320), nn.PReLU(), nn.Dropout(p=dropout_p),
            nn.Linear(320, 160), nn.PReLU(), nn.Dropout(p=dropout_p),
            nn.Linear(160, 32), nn.PReLU(), nn.Dropout(p=dropout_p),
            nn.Linear(32, 16), nn.PReLU(),
            nn.Linear(16,1)
        )
        # theta 分支
        self.fc_orient_theta = nn.Sequential(
            nn.Linear(384, 1280), nn.PReLU(), nn.Dropout(p=dropout_p),
            nn.Linear(1280, 640), nn.PReLU(), nn.Dropout(p=dropout_p),
            nn.Linear(640, 320), nn.PReLU(), nn.Dropout(p=dropout_p),
            nn.Linear(320, 160), nn.PReLU(), nn.Linear(160, num_sector_theta)
        )
        # phi 分支
        self.fc_orient_phi = nn.Sequential(
            nn.Linear(384, 1280), nn.PReLU(), nn.Dropout(p=dropout_p),
            nn.Linear(1280, 640), nn.PReLU(), nn.Dropout(p=dropout_p),
            nn.Linear(640, 320), nn.PReLU(), nn.Dropout(p=dropout_p),
            nn.Linear(320, 160), nn.PReLU(), nn.Linear(160, num_sector_phi)
        )

    def forward(self, x):
        B, L, F = x.shape
        x_flat = x.view(B*L, F)
        feat = self.shared_fc(x_flat)
        out_norm = self.fc_norm(feat).view(B,L,1)
        out_theta = self.fc_orient_theta(feat).view(B,L,-1)
        out_phi = self.fc_orient_phi(feat).view(B,L,-1)
        return out_norm, out_theta, out_phi
