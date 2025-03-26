# Copyright (c) 2025, Tencent Inc. All rights reserved.
# Data: 2025/1/9 15:58
# Author: chenchenqin
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from tgnn.model.layer import Linear
from .rotary_embedding import RotaryEmbedding


class MultiheadAttention(nn.Module):
    """Multi-headed attention.

    See "Attention Is All You Need" for more details.
    """

    def __init__(
            self,
            dim,
            num_heads,
            kdim=None,
            vdim=None,
            dropout=0.0,
            bias=True,
            pack_qkv=False,
            use_rotary_embeddings: bool = False
    ):
        super().__init__()
        assert (dim % num_heads == 0), "embedding dim must be divisible by num_heads"
        self.dim = dim
        self.kdim = kdim if kdim is not None else dim
        self.vdim = vdim if vdim is not None else dim
        self.qkv_same_dim = self.kdim == dim and self.vdim == dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = dim // num_heads
        self.pack_qkv = pack_qkv and self.qkv_same_dim
        if self.pack_qkv:
            self.qkv_proj = Linear(self.dim, self.dim * 3, bias=bias)
        else:
            self.k_proj = Linear(self.kdim, self.dim, bias=bias)
            self.v_proj = Linear(self.vdim, self.dim, bias=bias)
            self.q_proj = Linear(self.dim, self.dim, bias=bias)

        self.out_proj = Linear(dim, dim, bias=bias, init="final")
        self.use_rotary_embeddings = use_rotary_embeddings
        if self.use_rotary_embeddings:
            self.rot_emb = RotaryEmbedding(dim=self.head_dim)
        self.scale_factor = self.head_dim ** -0.5
        self.kv_cache = None

    def _project_qkv(self, q, k=None, v=None):
        seq_len, bs = q.shape[:2]  # [seq_len, bs, dim]
        k = q if k is None else k
        v = q if v is None else v
        if self.pack_qkv:
            assert v is q and k is q, ("k, v must be None")
            q, k, v = self.qkv_proj(q)
        else:
            q = self.q_proj(q)
            k = self.k_proj(k)
            v = self.v_proj(v)

        q = q.contiguous().view(-1, bs * self.num_heads, self.head_dim).transpose(0, 1)
        k = k.contiguous().view(-1, bs * self.num_heads, self.head_dim).transpose(0, 1)
        v = v.contiguous().view(-1, bs * self.num_heads, self.head_dim).transpose(0, 1)
        if self.rot_emb:
            q, k = self.rot_emb(q, k)

        # [bs * num_heads, seq_len, head_dim]
        return q, k, v

    def update_kv_cache(self, k, v, max_len=2048):
        bs = k.shape[0] // self.num_heads
        if self.kv_cache is None:
            dtype = k.dtype
            device = k.device
            cache_shape = (bs, self.num_heads, 0, self.head_dim)
            self.kv_cache = (
                torch.zeros(cache_shape, device=device, dtype=dtype),
                torch.zeros(cache_shape, device=device, dtype=dtype)
            )
        prev_k = self.kv_cache[0].view(bs * self.num_heads, -1, self.head_dim)
        prev_v = self.kv_cache[1].view(bs * self.num_heads, -1, self.head_dim)
        if k.shape[1] >= max_len:
            prev_k = prev_k[:, 1:]
            prev_v = prev_v[:, 1:]

        k = torch.cat([prev_k, k], dim=1)
        v = torch.cat([prev_v, v], dim=1)
        self.kv_cache = (k, v)

        return k, v

    def forward(
            self,
            query,
            key: Optional[Tensor] = None,
            value: Optional[Tensor] = None,
            attn_mask: Optional[Tensor] = None,
            attn_bias: Optional[Tensor] = None,
            key_padding_mask: Optional[Tensor] = None,
            need_weights: bool = False,
            cache_kv=False
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Args:
            q, k, v: [*, seq_len, dim]
            key_padding_mask (ByteTensor, optional): mask to exclude
                keys that are pads, of shape `(batch, src_len)`, where
                padding elements are indicated by 1s.
            need_weights (bool, optional): return the attention weights,
                averaged over heads (default: False).
            attn_bias: [bs, 1, seq_len, seq_len]

        Returns:
            attn: [seq_len, bs, dim]
        """
        q_len, bs = query.shape[:2]
        q, k, v = self._project_qkv(query, key, value)  # [bs * num_heads, seq_len, head_dim]

        if cache_kv:
            self.update_kv_cache(k, v)

        kv_len = k.size(1)
        if attn_mask is not None:
            attn_mask = attn_mask[:, None].bool()

        if key_padding_mask is not None:
            if attn_mask is None:
                attn_mask = torch.ones((bs, 1, q_len, kv_len), dtype=torch.bool, device=q.device)

            assert key_padding_mask.shape[:2] == (bs, kv_len)
            padding_mask = key_padding_mask[:, None, None].bool()
            attn_mask *= ~padding_mask

        if attn_bias is not None:
            if attn_mask is not None:
                attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
            attn_mask = attn_bias.to(q.dtype)

        if need_weights:
            # [bsz * self.num_heads, tgt_len, src_len]
            attn_weights = q @ k.transpose(-2, -1) * self.scale_factor
            attn_weights = attn_weights.view(bs, self.num_heads, q_len, kv_len)
            if attn_mask is not None:
                if attn_mask.dtype == torch.bool:
                    attn_weights = attn_weights.masked_fill(~attn_mask, float("-inf"))
                else:
                    attn_weights += attn_mask
            attn_weights = F.softmax(attn_weights.float(), dim=-1).type_as(q)
            attn_probs = F.dropout(attn_weights, p=self.dropout, training=self.training)
            attn_probs = attn_probs.view(bs * self.num_heads, q_len, kv_len)
            attn = attn_probs @ v
        else:
            if attn_mask is not None:
                attn_mask = attn_mask.view(-1, q_len, kv_len).repeat(self.num_heads, 1, 1)

            attn = F.scaled_dot_product_attention(q, k, v,
                                                  attn_mask=attn_mask,
                                                  dropout_p=self.dropout)  # [bs * num_heads, seq_len, head_dim]
            attn_weights = None

        attn = attn.transpose(0, 1).contiguous().view(q_len, bs, -1)  # [q_len, bs, dim]
        attn = self.out_proj(attn)

        if need_weights:
            attn_weights = attn_weights.transpose(0, 1)  # [bs, q_len, kv_len]

        return attn, attn_weights
