import torch
import torch.nn as nn

class MLP(nn.Module):
    """
    基于原始深层 MLP，输出预测均值 + 预测标准差（置信度）
    输入: [CAE latent + 当前点 + 目标点]
    输出: [mu_x, mu_y, log_sigma_x, log_sigma_y]
    """
    def __init__(self, input_size, hidden_sizes=[512, 384, 256, 128, 64, 32]):
        super(MLP, self).__init__()

        layers = []
        in_size = input_size
        for h in hidden_sizes:
            layers.append(nn.Linear(in_size, h))
            layers.append(nn.PReLU())
            layers.append(nn.Dropout(p=0.2))  # lighter dropout
            in_size = h
        
        layers.append(nn.Linear(hidden_sizes[-1], 4))  # Output layer: mu_x, mu_y, log_sigma_x, log_sigma_y

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        out = self.model(x)
        mu = out[:, :2]
        sigma = torch.exp(out[:, 2:])  # ensure positivity
        return mu, sigma
    
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP_branch(nn.Module):
    def __init__(self, input_size, output_size, hidden_sizes=None, dropout_rate=0.5):
        super().__init__()
        if hidden_sizes is None:
            hidden_sizes = [512, 256, 128]
        layers = []
        in_size = input_size
        for h in hidden_sizes:
            layers.append(nn.Linear(in_size, h))
            layers.append(nn.PReLU())
            layers.append(nn.Dropout(dropout_rate))
            in_size = h
        layers.append(nn.Linear(in_size, output_size))
        self.fc = nn.Sequential(*layers)

    def forward(self, x):
        return self.fc(x)


class MLP_fusion_gaussian(nn.Module):
    """
    输出 (num_candidates, joint_dim) 的 μ 和 σ
    """
    def __init__(self, cae_input_size, joint_input_size,
                 fusion_hidden_sizes=None, branch_hidden_sizes=None,
                 joint_dim=2, num_candidates=5, dropout_rate=0.5):
        super().__init__()
        self.num_candidates = num_candidates
        self.joint_dim = joint_dim
        
        self.cae_branch = MLP_branch(cae_input_size, 128, branch_hidden_sizes, dropout_rate)
        self.joint_branch = MLP_branch(joint_input_size, 128, branch_hidden_sizes, dropout_rate)
        
        if fusion_hidden_sizes is None:
            fusion_hidden_sizes = [256, 128]

        layers = []
        in_size = 128 * 2
        for h in fusion_hidden_sizes:
            layers.append(nn.Linear(in_size, h))
            layers.append(nn.PReLU())
            layers.append(nn.Dropout(dropout_rate))
            in_size = h
        
        layers.append(nn.Linear(in_size, num_candidates * joint_dim * 2))
        self.fc_fusion = nn.Sequential(*layers)
    
    def forward(self, cae_input, joint_input):
        F_e = self.cae_branch(cae_input)
        F_j = self.joint_branch(joint_input)

        F_cat = torch.cat([F_e, F_j], dim=-1)
        out = self.fc_fusion(F_cat)

        out = out.view(out.size(0), self.num_candidates, self.joint_dim, 2)
        mu = out[..., 0]
        sigma = F.softplus(out[..., 1]) + 1e-6
        return mu, sigma

class MLP_original(nn.Module):
    def __init__(self, input_size, output_size, hidden_sizes=None, dropout_rate=0.5):
        super(MLP_original, self).__init__()
        
        # Default 6 hidden layers if not provided
        if hidden_sizes is None:
            hidden_sizes = [1024, 768, 512, 384, 256, 128]
        
        layers = []
        in_size = input_size
        for h in hidden_sizes:
            layers.append(nn.Linear(in_size, h))
            layers.append(nn.PReLU())
            layers.append(nn.Dropout(dropout_rate))
            in_size = h
        
        # Output layer
        layers.append(nn.Linear(in_size, output_size))
        self.fc = nn.Sequential(*layers)

    def forward(self, x):
        return self.fc(x)
    