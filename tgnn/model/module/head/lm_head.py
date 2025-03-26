# Copyright (c) 2025, Tencent Inc. All rights reserved.
# Data: 2025/1/9 15:58
# Author: chenchenqin
import torch
import torch.nn as nn
import torch.nn.functional as F

from tgnn.model.layer import Linear, LayerNorm, GELU


class RobertaLMHead(nn.Module):
    """Head for masked language modeling."""

    def __init__(self, dim, output_dim, weight):
        super().__init__()
        self.dense = Linear(dim, dim)
        self.layer_norm = LayerNorm(dim)
        self.weight = weight
        self.bias = nn.Parameter(torch.zeros(output_dim))
        self.gelu = GELU('erf')

    def forward(self, features):
        x = self.dense(features)
        x = self.gelu(x)
        x = self.layer_norm(x)
        # project back to size of vocabulary with bias
        x = F.linear(x, self.weight) + self.bias

        return x
