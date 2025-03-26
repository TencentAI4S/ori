# -*- coding: utf-8 -*-
# Copyright (c) 2023, Tencent Inc. All rights reserved.
# Author: chenchenqin
# Data: 2023/11/15 14:18
import torch
from torch import nn


class RelativePositionEmbeddingSS(nn.Module):
    """esmfold relative postion embedding"""

    def __init__(self, c_z, num_bins=32):
        super().__init__()
        self.c_z = c_z
        self.num_bins = num_bins
        # Note an additional offset is used so that the 0th position
        # is reserved for masked pairs.
        self.embedding = nn.Embedding(2 * num_bins + 2, c_z)

    def forward(self, residue_index: torch.LongTensor, mask=None):
        """
        Args:
            residue_index: [*, seq_len], indices
            mask: [*, seq_len], seq mask or padding mask, false is padding

        Returns:
            z: [*, seq_len, seq_len, c_z] pairwise tensor of embeddings
        """
        assert residue_index.dtype == torch.long
        # [bs, 1, L] - [bs, L, 1]
        diff = residue_index[..., None, :] - residue_index[..., :, None]
        diff = diff.clamp(-self.num_bins, self.num_bins)
        diff = diff + self.num_bins + 1  # Add 1 to adjust for padding index.

        if mask is not None:
            assert residue_index.shape[-1] == mask.shape[-1]
            pair_mask = mask[..., None, :] * mask[..., None]
            diff[~pair_mask.bool()] = 0

        z = self.embedding(diff)
        return z
