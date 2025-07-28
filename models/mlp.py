import torch
import torch.nn as nn

class Mlp(nn.Module):
    """Standard two-layer MLP used in ViT blocks"""
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class MlpDecomp(nn.Module):
    """Decomposed MLP with latent bottleneck matrices"""
    def __init__(self, in_features, hidden_features=None, out_features=None, latent_dim=64, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.w1a = nn.Linear(in_features, latent_dim, bias=False)
        self.w1b = nn.Linear(latent_dim, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.w2a = nn.Linear(hidden_features, latent_dim, bias=False)
        self.w2b = nn.Linear(latent_dim, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.w1a(x)
        x = self.w1b(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.w2a(x)
        x = self.w2b(x)
        x = self.drop2(x)
        return x
