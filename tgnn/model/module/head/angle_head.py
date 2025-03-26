# Copyright (c) 2025, Tencent Inc. All rights reserved.
# Data: 2025/1/9 15:58
# Author: chenchenqin
from typing import Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from tgnn.model.layer import Linear
from tgnn.model.module import ResidualBlock1d
from tgnn.protein import residue_constants as rc


class AngleHead(nn.Module):
    """
    Implements Algorithm 20, lines 11-14

    Args:
        dim: Input channel dimension
        hidden_dim: Hidden channel dimension
        num_blocks: Number of resnet blocks
        num_angles: Number of torsion angles to generate, default 7,
        eps: normalization eps
    """

    def __init__(self,
                 dim,
                 hidden_dim=128,
                 num_blocks=2,
                 num_angles=7,
                 normalize=False,
                 eps=1e-8,
                 activation=nn.ReLU()):
        super(AngleHead, self).__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.num_blocks = num_blocks
        self.num_angles = num_angles
        self.normalize = normalize
        self.eps = eps
        self.linear_in = Linear(self.dim, self.hidden_dim)
        self.linear_initial = Linear(self.dim, self.hidden_dim)

        self.layers = nn.ModuleList()
        for _ in range(self.num_blocks):
            layer = ResidualBlock1d(self.hidden_dim)
            self.layers.append(layer)

        self.linear_out = Linear(self.hidden_dim, self.num_angles * 2)
        self.act = activation

    def forward(self, s: torch.Tensor, s_initial: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            s: [*, dim] single embedding
            s_initial: [*, dim] single embedding as of the start of the StructureModule

        Returns:
            [*, num_angles, 2] predicted angles, in range (0, 1]
        """
        s_initial = self.act(s_initial)
        s_initial = self.linear_initial(s_initial)
        s = self.act(s)
        s = self.linear_in(s)
        s = s + s_initial
        for layer in self.layers:
            s = layer(s)
        s = self.act(s)
        s = self.linear_out(s)  # [*, num_angles * 2]
        s = s.view(s.shape[:-1] + (-1, 2))
        if self.normalize:
            s = F.normalize(s, dim=-1, eps=self.eps)

        return s


class AngleHeads(nn.Module):
    """decouple angle head"""

    def __init__(self, c_s, c_h=None):
        super().__init__()
        self.c_s = c_s
        self.c_h = c_h or c_s
        self.heads = nn.ModuleDict()
        for name in rc.restypes:
            self.heads[name] = AngleHead(self.c_s, self.c_h)

    def forward(self,
                aa_seqs: List[str],
                s: torch.Tensor,
                s_init: torch.Tensor):
        """
        Args:
            aa_seqs: list of aa seq
            s: [bs, seq_len, c_s]
            s_init: [bs, seq_len, c_s]
        """
        bs, seq_len = s.shape[:2]
        dtype = s.dtype
        device = s.device
        s = s.reshape(bs * seq_len, -1)
        s_init = s_init.reshape(bs * seq_len, -1)
        angles = torch.cat([
            torch.ones((bs * seq_len, 7, 1), dtype=dtype, device=device),
            torch.zeros((bs * seq_len, 7, 1), dtype=dtype, device=device),
        ], dim=3)  # cos(x) = 1 & sin(x) = 0 => zero-initialization
        aa_seq = "".join(aa_seqs)
        for res_name in rc.restypes:
            res_ids = [idx for idx, name in enumerate(aa_seq) if name == res_name]
            if len(res_ids) == 0:
                continue
            res_s = s[res_ids]
            res_s_init = s_init[res_ids]
            angles[res_ids] = self.angle_heads[res_name](res_s, res_s_init)

        angles = angles.reshape(bs, seq_len, 7, 2)
        angles = F.normalize(angles, dim=-1)
        return angles
