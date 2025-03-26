# Copyright (c) 2025, Tencent Inc. All rights reserved.
# Data: 2025/1/9 15:58
# Author: chenchenqin
from typing import Optional

import torch


def precompute_freqs_cis(seq_len: int,
                         rotary_dim: int,
                         theta: float = 10000.0,
                         interpolation_factor=None,
                         dtype: Optional[torch.dtype] = torch.float32,
                         device: Optional[torch.device] = None,
                         complex=False) -> torch.Tensor:
    """build rope cache
    Args:
        seq_len: sequence length
        n_elem: head dim

    Returns:
        freqs_cis: [seq_len, head_dim // 2, 2]
    """
    freqs = 1.0 / (theta ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32, device=device) / rotary_dim))
    # Create position indexes `[0, 1, ..., seq_len - 1]`
    seq_idx = torch.arange(seq_len, dtype=torch.float32, device=device)
    if interpolation_factor is not None:
        seq_idx *= 1 / interpolation_factor

    freqs = torch.outer(seq_idx, freqs)
    # TODO: if pytorch2.0 compile suport complex64, delete it
    if not complex:
        freqs_cis = torch.stack([torch.cos(freqs),
                                 torch.sin(freqs)], dim=-1)
        freqs_cis = freqs_cis.to(dtype)
    else:
        low_precison_dtypes = (torch.float16, torch.bfloat16, torch.int8)
        complex_dtype = (
            torch.complex32 if dtype in low_precison_dtypes else torch.complex64
        )
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs).to(complex_dtype)

    return freqs_cis


def apply_rotary_emb(
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        interleaved: bool = True,
) -> torch.Tensor:
    """
    Args:
        x: [bs, seq_len, num_heads, head_dim], xq or xk
        freqs_cis: [seq_len, num_heads, head_dim // 2, 2]
    """
    seq_len = x.shape[-3]
    freqs_cis = freqs_cis[:seq_len]
    # TODO: wait pytorch2.0 support torch.complex32
    if freqs_cis.dtype in (torch.complex32, torch.complex64):
        # cast because `view_as_complex` does not support bfloat16 tensors
        # force convert x to complex64
        if interleaved:
            xc = x.reshape(*x.shape[:-1], -1, 2)
        else:
            x1, x2 = x.chunk(2, dim=-1)
            xc = torch.stack([x1, x2], dim=-1)

        xc = torch.view_as_complex(xc).to(freqs_cis.dtype)
        freqs_cis = freqs_cis.view(xc.size(1), 1, xc.size(3))
        out = torch.view_as_real(xc * freqs_cis).flatten(start_dim=-2)
        out = out.type_as(x)
        return out

    if interleaved:
        xc = x.reshape(*x.shape[:-1], -1, 2)  # [*, seq_len, num_heads, head_dim // 2, 2]
        x0, x1 = xc[..., 0], xc[..., 1]
    else:
        x0, x1 = x.chunk(2, dim=-1)

    freqs_cis = freqs_cis.view(seq_len, 1, x0.size(-1), 2)  # [seq_len, 1, head_dim // 2, 2]
    cos, sin = freqs_cis[..., 0], freqs_cis[..., 1]
    out = torch.stack([
        x0 * cos - x1 * sin,
        x1 * cos + x0 * sin
    ], dim=-1).flatten(start_dim=-2)

    return out


class RotaryEmbedding(torch.nn.Module):

    def __init__(self,
                 dim,
                 max_len=64,
                 theta=10000,
                 dtype=None,
                 device=None):
        factory_kwargs = {"dtype": dtype, "device": device}
        super().__init__()
        self.dim = dim
        self.max_len = max_len
        self.theta = theta
        freqs_cis = precompute_freqs_cis(self.max_len,
                                         self.dim,
                                         theta=self.theta,
                                         **factory_kwargs)
        # complex can not register buffer, this will convert buffer dtype to float
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)

    def _update_cos_sin_tables(self, x):
        seq_len = 0 if x is None else x.shape[2]
        if seq_len > self.max_len:
            freqs_cis = precompute_freqs_cis(self.max_len,
                                             self.dim,
                                             self.theta,
                                             dtype=self.freqs_cis.dtype,
                                             device=self.freqs_cis.device)
            self.register_buffer("freqs_cis", freqs_cis, persistent=False)

    def forward(self,
                q: torch.Tensor,
                k: torch.Tensor):
        """
        Args:
            q, k: [bs, seq_len, num_heads, head_dim]
        """
        self._update_cos_sin_tables(q)
        seq_len = q.shape[1]
        freqs_cis = self.freqs_cis[:seq_len]
        return apply_rotary_emb(q, freqs_cis), apply_rotary_emb(k, freqs_cis)
