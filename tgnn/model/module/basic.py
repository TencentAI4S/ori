# Copyright (c) 2025, Tencent Inc. All rights reserved.
# Data: 2025/1/9 15:58
# Author: chenchenqin
import torch
import torch.nn as nn

from tgnn.model.layer import Linear


class ResidualBlock1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.linear_1 = Linear(self.dim, self.dim, init="relu")
        self.linear_2 = Linear(self.dim, self.dim, init="final")
        self.relu = nn.ReLU()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.relu(x)
        x = self.linear_1(x)

        x = self.relu(x)
        x = self.linear_2(x)

        return x + residual