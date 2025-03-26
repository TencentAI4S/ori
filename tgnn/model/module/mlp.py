# Copyright (c) 2025, Tencent Inc. All rights reserved.
# Data: 2025/1/9 15:58
# Author: chenchenqin
import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from xformers.ops import swiglu, unbind

    logging.warning(f"using xformer swiglu extention")
except:
    swiglu = None


class SwiGLU(nn.Module):
    """feed forward network

    Args:
        dim: input embedding dim
        hidden_dim: inner hidden dim, also named ffn_dim in other project
        multiple_of: emsure hidden dim are divided
        ffn_dim_multiplier: config param in llama2, default none for compact with llama
        bias: linear layer bias
        _pack_weights: pack fc linear and than split, set true for faster training

    Note that MLP is also called swiglu operator in some papers, you call speed up by installing xformers
    """

    def __init__(
            self,
            dim: int,
            hidden_dim: int = None,
            multiple_of: int = 256,
            ffn_dim_multiplier: Optional[float] = None,
            _pack_weights=False,
            bias=False,
            xformer=True
    ):
        super().__init__()
        hidden_dim = hidden_dim or 4 * dim
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)

        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        self.pack_fc = _pack_weights
        if self.pack_fc:
            self.c_fc = nn.Linear(dim, 2 * hidden_dim, bias=bias)
        else:
            self.c_fc1 = nn.Linear(dim, hidden_dim, bias=bias)
            self.c_fc2 = nn.Linear(dim, hidden_dim, bias=bias)

        self.c_proj = nn.Linear(hidden_dim, dim, bias=bias)
        self.xformer = xformer

    def native_impl(self, x):
        """
        Args:
            x: [*, dim]
        """
        if self.pack_fc:
            x1, x2 = torch.chunk(self.c_fc(x), 2, dim=-1)
            x = F.silu(x1) * x2
        else:
            x = F.silu(self.c_fc1(x)) * self.c_fc2(x)
        x = self.c_proj(x)
        return x

    def swiglu_impl(self, x):
        if self.pack_fc:
            fcw = self.c_fc.weight
            fc1w, fc2w = unbind(
                fcw.view([2, fcw.shape[0] // 2, fcw.shape[1]]),
                dim=0,
            )
            fcb = self.c_fc.bias
            if fcb is not None:
                fc1b, fc2b = unbind(fcb.view([2, fcb.shape[0] // 2]), dim=0)
            else:
                fc1b, fc2b = None, None
            x = swiglu(x,
                       fc1w, fc1b,
                       fc2w, fc2b,
                       self.c_proj.weight, self.c_proj.bias)
        else:
            x = swiglu(x,
                       self.c_fc1.weight, self.c_fc1.bias,
                       self.c_fc2.weight, self.c_fc2.bias,
                       self.c_proj.weight, self.c_proj.bias)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.xformer and swiglu is not None:
            return self.swiglu_impl(x)

        return self.native_impl(x)


class MLP(nn.Module):
    """feed forward network, invert bottleneck
    """

    def __init__(self,
                 dim: int,
                 hidden_dim: int,
                 bias: bool = True,
                 dropout: float = 0.0):
        super().__init__()
        self.c_fc = nn.Linear(dim, hidden_dim, bias=bias)
        self.gelu = nn.GELU('tanh')
        self.c_proj = nn.Linear(hidden_dim, dim, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)

        return x
