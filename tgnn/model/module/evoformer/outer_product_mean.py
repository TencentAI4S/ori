# Copyright (c) 2025, Tencent Inc. All rights reserved.
# Data: 2025/1/9 15:58
# Author: chenchenqin
from functools import partial
from typing import Optional

import torch
import torch.nn as nn

from tgnn.model.layer import LayerNorm, Linear
from tgnn.model.utils import chunk_layer


class OuterProductMeanMSA(nn.Module):
    """Outer-produce mean for transforming sequence features into an update for pair features.
    Implements Algorithm 10.

    Args:
        c_m: MSA embedding channel dimension
        c_z: Pair embedding channel dimension
        c_hidden: Hidden channel dimension
    """
    def __init__(self, c_m, c_z, c_hidden=32, eps=1e-3):
        super(OuterProductMeanMSA, self).__init__()
        self.c_m = c_m
        self.c_z = c_z
        self.c_hidden = c_hidden
        self.eps = eps
        self.layer_norm = LayerNorm(c_m)
        self.linear_1 = Linear(c_m, c_hidden)
        self.linear_2 = Linear(c_m, c_hidden)
        self.linear_out = Linear(c_hidden ** 2, c_z, init="final")

    def _opm(self, a, b):
        # [*, seq_len, seq_len, C, C]
        outer = torch.einsum("...bac,...dae->...bdce", a, b)
        # [*, seq_len, seq_len, C * C]
        outer = outer.reshape(outer.shape[:-2] + (-1,))
        # [*, seq_len, seq_len, C_z]
        outer = self.linear_out(outer)

        return outer

    @torch.jit.ignore
    def _chunk(self,
               a: torch.Tensor,
               b: torch.Tensor,
               chunk_size: int
               ) -> torch.Tensor:
        a_reshape = a.reshape((-1,) + a.shape[-3:])
        b_reshape = b.reshape((-1,) + b.shape[-3:])
        out = []
        for a_prime, b_prime in zip(a_reshape, b_reshape):
            outer = chunk_layer(
                partial(self._opm, b=b_prime),
                {"a": a_prime},
                chunk_size=chunk_size,
                no_batch_dims=1,
            )
            out.append(outer)

        # For some cursed reason making this distinction saves memory
        if len(out) == 1:
            outer = out[0].unsqueeze(0)
        else:
            outer = torch.stack(out, dim=0)

        outer = outer.reshape(a.shape[:-3] + outer.shape[1:])

        return outer

    def forward(self,
                m: torch.Tensor,
                mask: Optional[torch.Tensor] = None,
                chunk_size: Optional[int] = None
                ) -> torch.Tensor:
        """
        Args:
            m: [*, num_seqs, seq_len, c_m] MSA embedding
            mask: [*, num_seqs, seq_len] MSA mask

        Returns:
            [*, seq_len, seq_len, C_z] pair embedding update
        """
        ln = self.layer_norm(m)
        a = self.linear_1(ln)
        b = self.linear_2(ln)
        if mask is not None:
            mask = mask.unsqueeze(-1)
            a = a * mask
            b = b * mask

        a = a.transpose(-2, -3)
        b = b.transpose(-2, -3)

        if chunk_size is not None:
            outer = self._chunk(a, b, chunk_size)
        else:
            outer = self._opm(a, b)

        if mask is None:
            # [*, num_seqs, seq_len]
            mask = m.new_ones(m.shape[:-1] + (1, ))

        norm = torch.einsum("...abc,...adc->...bdc", mask, mask)
        outer = outer / (norm + self.eps)

        return outer


class OuterProductMeanSS(nn.Module):
    """Outer-produce mean for transforming single features into an update for pair features.

    Args:
        c_s: Sequence embedding channel dimension
        c_z: Pair embedding channel dimension
        c_hidden: Hidden channel dimension
    """

    def __init__(self, c_s, c_z, c_hidden=32):
        super(OuterProductMeanSS, self).__init__()
        self.c_s = c_s
        self.c_z = c_z
        self.c_h = c_hidden
        self.layer_norm = LayerNorm(c_s)
        self.linear_1 = Linear(c_s, c_hidden)
        self.linear_2 = Linear(c_s, c_hidden)
        self.linear_out = Linear(c_hidden ** 2, c_z, init='final')

    def _opm(self, a, b):
        # [*, N_res, N_res, C, C]
        outer = torch.einsum("...bc,...de->...bdce", a, b)
        # [*, N_res, N_res, C * C]
        outer = outer.reshape(outer.shape[:-2] + (-1,))
        # [*, N_res, N_res, C_z]
        outer = self.linear_out(outer)
        return outer

    def forward(self, s, mask: Optional[torch.Tensor] = None):
        """
        Args:
            s: [N, L, c_s], single features
            mask: [N, L], seq mask

        Returns:
            z: update term for pair features of size N x L x L x c_z
        """
        s = self.layer_norm(s)
        a = self.linear_1(s)
        b = self.linear_2(s)
        if mask is not None:
            mask = mask.unsqueeze(-1)
            a = a * mask
            b = b * mask

        outer = self._opm(a, b)
        return outer


class OuterProductMeanSM(nn.Module):
    """
    Outer-produce mean for update complex pair feature using
    antibody single feature and antigen MSA feature

    Args:
        c_s:
            Sequence embedding channel dimension (antibody)
        c_m:
            MSA embedding channel dimension (antigen)
        c_z:
            Pair embedding channel dimension
        c_hidden:
            Hidden channel dimension
    """

    def __init__(self, c_s, c_m, c_z, c_hidden=32, eps=1e-3):
        super(OuterProductMeanSM, self).__init__()
        self.c_s = c_s
        self.c_m = c_m
        self.c_z = c_z
        self.c_hidden = c_hidden
        self.eps = eps

        self.norm_s = LayerNorm(c_s)
        self.linear_s = Linear(c_s, c_hidden)

        self.norm_m = LayerNorm(c_m)
        self.linear_m = Linear(c_m, c_hidden)
        self.linear_out = Linear(c_hidden ** 2, c_z)

    def _opm(self, a, b):
        # [*, N_res, N_res, C, C]
        outer = torch.einsum("...bac,...dae->...bdce", a, b)
        # [*, N_res, N_res, C * C]
        outer = outer.reshape(outer.shape[:-2] + (-1,))
        # [*, N_res, N_res, C_z]
        outer = self.linear_out(outer)
        return outer

    def forward(self, s, m,
                seq_mask=None,
                msa_mask=None):
        """
        Args:
            s: single feature of size N x L_1 x c_s
            m: MSA feature of size N x L_2 x c_s

        Returns:
            outer: updated pair features of size N x L_1 x L_2 x c_z
        """
        num_seqs = m.shape[1]
        s = self.norm_s(s)
        a = self.linear_s(s)

        if seq_mask is not None:
            seq_mask = a.new_ones(a.shape[:-1])

        a = a * seq_mask[..., None]
        a = a[..., None, :].repeat(1, 1, num_seqs, 1)
        m = self.norm_m(m)
        b = self.linear_m(m)
        if msa_mask is not None:
            msa_mask = b.new_ones(b.shape[:-1])

        b = b * msa_mask[..., None]
        b = b.transpose(-2, -3)  # N x L_2 x K x c
        outer = self._opm(a, b)

        if seq_mask is None and msa_mask is not None:
            seq_mask = seq_mask[..., None, :]
            seq_mask = seq_mask.repeat([1, ] * len(seq_mask[:-2]) + [num_seqs, 1])

        norm = torch.einsum("...abc,...adc->...bdc", seq_mask, msa_mask)
        norm = norm + self.eps
        outer = outer / (norm + self.eps)

        return outer
