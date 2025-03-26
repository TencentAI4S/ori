# Copyright (c) 2025, Tencent Inc. All rights reserved.
# Data: 2025/1/9 15:58
# Author: chenchenqin
import torch
import torch.nn as nn

from tgnn.model.layer import Linear


class DistogramHead(nn.Module):
    """
    Computes a distogram probability distribution.

    For use in computation of distogram loss, subsection 1.9.8
    """

    def __init__(self, c_z, num_bins=64, normalize=False):
        """
        Args:
            c_z:
                Input channel dimension
            num_bins:
                Number of distogram bins
        """
        super().__init__()
        self.c_z = c_z
        self.num_bins = num_bins
        self.normalize = normalize
        self.linear = Linear(self.c_z, self.num_bins, init="final")

    def forward(self, z):
        """
        Args:
            z: [*, seq_len, seq_len, C_z] pair embedding
        Returns:
            [*, seq_len, seq_len, num_bins] distogram probability distribution
        """
        logits = self.linear(z)
        logits = logits + logits.transpose(-2, -3)
        if self.normalize:
            logits = logits / 2

        return logits

    @staticmethod
    def compute_distogram(positions, num_bins=15, min_bin=3.375, max_bin=21.375):
        """
        Args；
            coords： [bs, seq_len, 3 atoms, 3], where it's [N, CA, C] x 3 coordinates.

        Returns:
            dist bins: [bs, seq_len, seq_len], range [0, 14]
        """
        assert positions.shape[-2] == 3, f"only support bacbone atoms"
        boundaries = torch.linspace(
            min_bin,
            max_bin,
            num_bins - 1,
            device=positions.device
        )
        boundaries = boundaries ** 2
        N, CA, C = [x.squeeze(-2) for x in positions.chunk(3, dim=-2)]
        # Infer CB coordinates.
        b = CA - N
        c = C - CA
        a = b.cross(c, dim=-1)
        CB = -0.58273431 * a + 0.56802827 * b - 0.54067466 * c + CA
        dists = (CB[..., None, :, :] - CB[..., :, None, :]).pow(2).sum(dim=-1, keepdims=True)

        bins = torch.sum(dists > boundaries, dim=-1)

        return bins
