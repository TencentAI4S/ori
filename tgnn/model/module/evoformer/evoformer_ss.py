# Copyright (c) 2025, Tencent Inc. All rights reserved.
# Data: 2025/1/9 15:58
# Author: chenchenqin
from functools import partial

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange, repeat

from tgnn.model.layer import LayerNorm, Linear
from tgnn.model.utils import checkpoint_blocks
from .transition import PairToSequence, SequenceToPair, Transition
from .triangular_multiplicative_update import TriangleMultiplicationIncoming, TriangleMultiplicationOutgoing
from ..attention import TriangleAttentionStartingNode, TriangleAttentionEndingNode


class GatingAttention(nn.Module):

    def __init__(self, dim, num_heads):
        super().__init__()
        assert dim % num_heads == 0
        self.embed_dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.proj = Linear(dim, dim * 3, bias=False)
        self.o_proj = Linear(dim, dim, bias=True)
        self.g_proj = Linear(dim, dim)
        self.rescale_factor = self.head_dim ** -0.5

    def forward(self, x, mask=None, bias=None):
        """
        Args:
          x: batch of input sequneces (.. x L x C)
          mask: batch of boolean masks where 1=valid, 0=padding position (.. x L_k). optional.
          bias: batch of scalar pairwise attention biases (.. x Lq x Lk x num_heads). optional.

        Outputs:
          sequence projection (B x L x embed_dim), attention maps (B x L x L x num_heads)
        """
        # this is different with alhpafold project
        t = rearrange(self.proj(x), "... l (h c) -> ... h l c", h=self.num_heads)
        q, k, v = t.chunk(3, dim=-1)

        q = self.rescale_factor * q
        attn_score = torch.einsum("...qc,...kc->...qk", q, k)

        # Add external attention bias.
        if bias is not None:
            attn_score = attn_score + rearrange(bias, "... lq lk h -> ... h lq lk")

        # Do not attend to padding tokens.
        if mask is not None:
            mask = repeat(
                mask, "... lk -> ... h lq lk", h=self.num_heads, lq=q.shape[-2]
            )
            attn_score = attn_score.masked_fill(mask == False, -np.inf)

        attn_score = attn_score.float().softmax(dim=-1).type_as(q)
        y = torch.einsum("...hqk,...hkc->...qhc", attn_score, v)
        y = rearrange(y, "... h c -> ... (h c)", h=self.num_heads)
        y = self.g_proj(x).sigmoid() * y
        y = self.o_proj(y)
        return y


class EvoformerBlockSS(nn.Module):
    """evoformer single sequence transformer"""

    def __init__(
            self,
            c_s,
            c_z,
            num_heads_seq=32,
            num_heads_pair=32,
            dropout=0.0
    ):
        super().__init__()
        assert c_s % num_heads_seq == 0
        assert c_z % num_heads_pair == 0
        assert c_z % 2 == 0
        self.c_s = c_s
        self.c_z = c_z
        self.layernorm_1 = LayerNorm(c_s)
        self.sequence_to_pair = SequenceToPair(c_s, c_z)
        self.pair_to_sequence = PairToSequence(c_z, num_heads_seq)
        self.seq_attention = GatingAttention(c_s, num_heads=num_heads_seq)

        self.tri_mul_out = TriangleMultiplicationOutgoing(c_z, c_z)
        self.tri_mul_in = TriangleMultiplicationIncoming(c_z, c_z)

        self.tri_att_start = TriangleAttentionStartingNode(c_z, num_heads_pair)
        self.tri_att_end = TriangleAttentionEndingNode(c_z, num_heads_pair)

        self.mlp_seq = Transition(c_s, dropout=dropout)
        self.mlp_pair = Transition(c_z, dropout=dropout)
        self.seq_drop = nn.Dropout(dropout)

    def forward(self,
                s: torch.Tensor,
                z: torch.Tensor,
                mask=None,
                chunk_size=None):
        """
        Args:
            s: B x L x c_s, seq feature
            z: B x L x L x c_z, pair feature
            mask: B x L, sequence mask

        Returns:
            s: B x L x c_s, updated seq feature
            z: B x L x L x c_z, updated pair feature
        """
        if mask is not None:
            assert len(mask.shape) == 2
            pair_mask = mask[:, :, None] * mask[:, None, :]
        else:
            pair_mask = None
        # Update sequence state
        bias = self.pair_to_sequence(z)

        # Self attention with bias + mlp.
        y = self.seq_attention(self.layernorm_1(s),
                               mask=mask,
                               bias=bias)

        s = s + self.seq_drop(y)
        s = self.mlp_seq(s)

        # Update pairwise state
        z = z + self.sequence_to_pair(s)
        # Axial attention with triangular bias. [bs, seq_len, seq_len]
        z = z + self.tri_mul_out(z, mask=pair_mask)
        z = z + self.tri_mul_in(z, mask=pair_mask)
        z = z + self.tri_att_start(z, mask=pair_mask, chunk_size=chunk_size)
        z = z + self.tri_att_end(z, mask=pair_mask, chunk_size=chunk_size)

        # MLP over pairs.
        z = self.mlp_pair(z)

        return s, z


class EvoformerStackSS(nn.Module):
    """Single sequence evoformer
    """

    def __init__(
            self,
            c_s=384,
            c_z=256,
            num_layers=8
    ):
        super().__init__()
        self.num_layers = num_layers
        self.c_s = c_s
        self.c_z = c_z
        self.activation_checkpoint = False
        self.blocks = nn.ModuleList()
        for _ in range(self.num_layers):
            block = EvoformerBlockSS(self.c_s, self.c_z)
            self.blocks.append(block)

    def set_chunk_size(self, chunk_size):
        self.chunk_size = chunk_size

    def enable_activation_checkpoint(self, enabled=True, interval=1):
        if enabled:
            self.checkpoint_interval = interval
        else:
            self.checkpoint_interval = None

    def forward(self, s, z, chunk_size=None):
        """
        Args:
            s: [bs, seq_len, c_s], single features
            z: [bs, seq_len, seq_len, c_z], pair features

        Returns:
            s: [bs, seq_len, c_s], updated single features
            z: [bs, seq_len, c_s], updated pair features
        """
        blocks = [
            partial(
                b,
                chunk_size=chunk_size
            )
            for b in self.blocks
        ]

        s, z = checkpoint_blocks(
            blocks,
            args=(s, z),
            interval=self.checkpoint_interval
        )

        return s, z
