# Copyright (c) 2025, Tencent Inc. All rights reserved.
# Data: 2025/1/9 15:58
# Author: chenchenqin
from functools import partialmethod
from typing import Optional

import torch
import torch.nn as nn

from tgnn.model.layer import Linear, LayerNorm
from tgnn.utils.tensor import permute_final_dims


class TriangleMultiplicativeUpdate(nn.Module):
    """
    Implements Algorithms 11 and 12.

    Args:
        c_z: Input channel dimension
        c_hidden: Hidden channel dimension
    """
    def __init__(self, c_z, c_hidden, _outgoing=True):
        super(TriangleMultiplicativeUpdate, self).__init__()
        self.c_z = c_z
        self.c_hidden = c_hidden
        self._outgoing = _outgoing
        self.linear_a_p = Linear(self.c_z, self.c_hidden)
        self.linear_a_g = Linear(self.c_z, self.c_hidden, init="gating")
        self.linear_b_p = Linear(self.c_z, self.c_hidden)
        self.linear_b_g = Linear(self.c_z, self.c_hidden, init="gating")
        self.linear_g = Linear(self.c_z, self.c_z, init="gating")
        self.linear_z = Linear(self.c_hidden, self.c_z, init="final")
        self.layer_norm_in = LayerNorm(self.c_z)
        self.layer_norm_out = LayerNorm(self.c_hidden)

    def _combine_projections(self,
                             a: torch.Tensor,
                             b: torch.Tensor
                             ) -> torch.Tensor:
        if self._outgoing:
            a = permute_final_dims(a, (2, 0, 1))
            b = permute_final_dims(b, (2, 1, 0))
        else:
            a = permute_final_dims(a, (2, 1, 0))
            b = permute_final_dims(b, (2, 0, 1))

        p = torch.matmul(a, b)
        return permute_final_dims(p, (1, 2, 0))

    def forward(self,
                z: torch.Tensor,
                mask: Optional[torch.Tensor] = None
                ) -> torch.Tensor:
        """

        Args:
            z: [*, seq_len, seq_len, c_z] input tensor
            mask: [*, N_res, N_res] input mask

        Returns:
            [*, N_res, N_res, C_z] output tensor
        """
        z = self.layer_norm_in(z)
        a = self.linear_a_p(z) * self.linear_a_g(z).sigmoid()
        b = self.linear_b_p(z) * self.linear_b_g(z).sigmoid()
        if mask is not None:
            mask = mask.unsqueeze(-1)
            a = a * mask
            b = b * mask

        # Prevents overflow of torch.matmul in combine projections in
        # reduced-precision modes
        a_std = a.std()
        b_std = b.std()
        if a_std != 0. and b_std != 0.:
            a = a / a_std
            b = b / b_std

        if a.dtype in (torch.float16, torch.bfloat16):
            x = self._combine_projections(a.float(), b.float()).to(a.dtype)
        else:
            x = self._combine_projections(a, b) # [*, L, L, dim]

        x = self.layer_norm_out(x)
        x = self.linear_z(x)
        g = self.linear_g(z).sigmoid()
        x = x * g

        return x


class TriangleMultiplicationOutgoing(TriangleMultiplicativeUpdate):
    """
    Implements Algorithm 11.
    """
    __init__ = partialmethod(TriangleMultiplicativeUpdate.__init__, _outgoing=True)


class TriangleMultiplicationIncoming(TriangleMultiplicativeUpdate):
    """
    Implements Algorithm 12.
    """
    __init__ = partialmethod(TriangleMultiplicativeUpdate.__init__, _outgoing=False)
