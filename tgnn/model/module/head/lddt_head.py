# Copyright (c) 2025, Tencent Inc. All rights reserved.
# Data: 2025/1/9 15:58
# Author: chenchenqin
from typing import Any

import torch
import torch.nn as nn

from tgnn.model.layer import Linear, LayerNorm


class LDDTHead(nn.Module):
    """per residue lDDT-Ca scores prediction"""

    def __init__(self,
                 c_in=384,
                 c_hidden=None,
                 num_bins=50,
                 num_atoms=1,
                 NormLayer: Any = LayerNorm):
        super().__init__()
        self.c_in = c_in
        self.num_bins = num_bins
        self.num_atoms = num_atoms
        self.hidden_dim = c_hidden or c_in
        self.relu = nn.ReLU()
        self.norm = NormLayer(self.c_in)
        self.linear_1 = Linear(self.c_in, self.hidden_dim, init="relu")
        self.linear_2 = Linear(self.hidden_dim, self.hidden_dim, init="relu")
        self.linear_3 = Linear(self.hidden_dim, self.num_bins * num_atoms, init="final")

    @classmethod
    def compute_plddt(cls, logits: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: [*, num_atoms, num_bins]
        """
        num_bins = logits.shape[-1]
        bin_width = 1.0 / num_bins
        bounds = torch.arange(0.5 * bin_width, 1.0,
                              step=bin_width,
                              dtype=logits.dtype,
                              device=logits.device)  # [num_bins, ]
        pred_lddt_ca = (logits.softmax(dim=-1) @ bounds[:, None]).squeeze(-1)
        return pred_lddt_ca * 100

    def forward(self, x):
        """
        Args:
            x: [*, dim], single features

        Returns:
            [*, num_atoms, num_bins], predict per-residue & full-chain lDDT-Ca scores
        """
        x = self.norm(x)
        x = self.linear_1(x)
        x = self.relu(x)
        x = self.linear_2(x)
        x = self.relu(x)
        logits = self.linear_3(x)
        return logits.reshape(logits.shape[:-1] + (self.num_atoms, self.num_bins))
