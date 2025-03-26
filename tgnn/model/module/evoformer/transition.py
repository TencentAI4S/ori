# Copyright (c) 2025, Tencent Inc. All rights reserved.
# Data: 2025/1/9 15:58
# Author: chenchenqin
from typing import Optional

import torch
import torch.nn as nn

from tgnn.model.layer import LayerNorm, Linear
from tgnn.model.utils import chunk_layer


class Transition(nn.Module):
    """
    Implements Algorithm 9 and 15. PairTransition and MSATransition
    """
    def __init__(self, c_in, c_hidden=None, dropout=0.0):
        """
        Args:
            c: Transition channel dimension
        """
        super(Transition, self).__init__()
        self.c_in = c_in
        self.c_hidden = c_hidden or 4 * c_in
        self.mlp = nn.Sequential(
            LayerNorm(self.c_in),
            Linear(self.c_in, self.c_hidden, init="relu"),
            nn.ReLU(),
            Linear(self.c_hidden, self.c_in, init="final"),
            nn.Dropout(dropout)
        )

    @torch.jit.ignore
    def _chunk(self,
               x: torch.Tensor,
               mask: torch.Tensor,
               chunk_size: int,
               ) -> torch.Tensor:
        return chunk_layer(
            self._transition,
            {"x": x, "mask": mask},
            chunk_size=chunk_size,
            no_batch_dims=len(x.shape[:-2]),
        )

    def _transition(self, x, mask=None):
        residual = x
        x = self.mlp(x)
        if mask is not None:
            x = x * mask

        x = residual + x
        return x

    def forward(self,
                x: torch.Tensor,
                mask: Optional[torch.Tensor] = None,
                chunk_size: Optional[int] = None,
                ) -> torch.Tensor:
        """
        Args:
            x:
                [*, seq_len, seq_len, c] pair embedding
        Returns:
            [*, seq_len, seq_len, c] pair embedding update
        """
        if mask is not None:
            mask = mask[..., None]

        if chunk_size is not None:
            x = self._chunk(x, mask, chunk_size)
        else:
            x = self._transition(x=x, mask=mask)

        return x


class SequenceToPair(nn.Module):
    """like Outer-produce mean in alphafold"""
    def __init__(self, c_s, c_z, c_h=None):
        super().__init__()
        self.c_h = c_h or c_z // 2
        self.c_z = c_z
        self.c_s = c_s
        self.layernorm = LayerNorm(self.c_s)
        self.proj = Linear(self.c_s, self.c_h * 2, bias=True)
        self.o_proj = Linear(self.c_h * 2, self.c_z, bias=True, init="final")

    def forward(self, s):
        """
        Args:
            s: [bs, seq_len, c_s], sequence feature

        Returns:
            z: [bs, seq_len, seq_len, c_s], pair feature
        """
        assert len(s.shape) == 3, f"expect 3D tensor, got {s.shape}"
        s = self.layernorm(s)
        s = self.proj(s)
        q, k = s.chunk(2, dim=-1)

        prod = q[:, None, :, :] * k[:, :, None, :]
        diff = q[:, None, :, :] - k[:, :, None, :]

        x = torch.cat([prod, diff], dim=-1)
        x = self.o_proj(x)

        return x


class PairToSequence(nn.Module):
    def __init__(self, c_z, num_heads):
        super().__init__()
        self.layernorm = LayerNorm(c_z)
        self.linear = Linear(c_z, num_heads, bias=False, init="final")

    def forward(self, z):
        """
        Args:
          z: B x L x L x c_z

        Returns:
            pairwise_bias: B x L x L x num_heads
        """
        assert len(z.shape) == 4
        z = self.layernorm(z)
        bias = self.linear(z)

        return bias