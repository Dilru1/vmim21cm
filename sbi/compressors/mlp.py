"""MLP compressor for already-computed summaries (Case 1: 15-D PDF -> t_dim)."""
import torch
import torch.nn as nn
import torch.nn.functional as F

_ACT = {"leaky_relu": lambda: nn.LeakyReLU(0.1), "relu": nn.ReLU, "tanh": nn.Tanh}


class SummaryCompressor(nn.Module):
    def __init__(self, n_input, t_dim=8, dense_layers=(256, 256, 128, 64),
                 dropout=0.0, activation="leaky_relu", use_resnet=False, n_params=4,
                 direct=False):
        super().__init__()
        if direct and t_dim != n_params:
            raise ValueError(f"direct_regression needs t_dim == n_params, got {t_dim} vs {n_params}")
        self.use_resnet = use_resnet
        self.direct = bool(direct)
        act = _ACT[activation]
        layers, prev = [], n_input
        for h in dense_layers:
            layers += [nn.Linear(prev, h), nn.LayerNorm(h), act(), nn.Dropout(dropout)]
            prev = h
        self.body = nn.Sequential(*layers)
        self.fc_summary = nn.Linear(prev, t_dim)
        self.fc_aux = nn.Linear(prev, n_params)

    def features(self, x):
        return self.body(x)

    def forward(self, x):
        f = self.features(x)
        if self.direct:
            t = self.fc_summary(f)
            return t, t
        return self.fc_summary(f), self.fc_aux(f)
