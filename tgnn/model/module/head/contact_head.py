# Copyright (c) 2025, Tencent Inc. All rights reserved.
# Data: 2025/1/9 15:58
# Author: chenchenqin
import torch.nn as nn


def apc(x):
    """Perform average product correct, used for contact prediction.
    Args:
        x: [*, num_layers, seq_len, seq_len]
    """
    a1 = x.sum(-1, keepdims=True)
    a2 = x.sum(-2, keepdims=True)
    a12 = a1.sum(-2, keepdims=True)
    avg = a1 * a2
    avg.div_(a12)  # in-place to reduce memory
    normalized = x - avg
    return normalized


def symmetrize(x):
    "Make layer symmetric in final two dimensions, used for contact prediction."
    return x + x.transpose(-1, -2)


class ContactPredictionHead(nn.Module):
    """Performs symmetrization, apc, and computes a logistic regression on the output features"""

    def __init__(
            self,
            dim: int,
            hidden_dim: int = None,
            bias=True
    ):
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim
        if self.hidden_dim is not None:
            self.fusion = nn.Linear(self.dim, self.hidden_dim, bias=bias)
        else:
            self.fusion = None
            self.hidden_dim = self.dim

        self.linear = nn.Linear(self.hidden_dim, 1, bias)

    def forward(self, attentions, seq_mask=None):
        """
        Args:
            attentions: [bs, num_layers, num_heads, seq_len, seq_len]
            seq_mask: [bs, seq_len]

        Returns:
            contact map: [bs, seq_len, seq_len]
        """
        batch_size, layers, num_heads, seqlen, _ = attentions.size()
        attentions = attentions.view(batch_size, layers * num_heads, seqlen, seqlen)
        # note that apm have reduction op along axis
        if seq_mask is not None:
            pair_mask = seq_mask[..., None, :] * seq_mask[..., None]
            attentions = attentions * pair_mask[..., None, :, :]

        if self.fusion is not None:
            attentions = attentions.permute(0, 2, 3, 1)  # [bs, seq_len, seq_len, 8]
            attentions = self.fusion(attentions)
            attentions = attentions.permute(0, 3, 1, 2)  # [bs, 8, seq_len, seq_len]

        attentions = apc(symmetrize(attentions))
        attentions = attentions.permute(0, 2, 3, 1)  # [bs, seq_len, seq_len, 8]
        return self.linear(attentions).squeeze(3)
