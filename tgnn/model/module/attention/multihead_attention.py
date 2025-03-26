# -*- coding: utf-8 -*-
# Copyright (c) 2023, Tencent Inc. All rights reserved.
# Author: chenchenqin
# Data: 2023/6/21 14:32
from functools import partial, partialmethod
from typing import Optional

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from tgnn.model.layer import Linear, apply_rotary_emb, precompute_freqs_cis
from tgnn.model.utils import chunk_layer
from tgnn.utils.tensor import flatten_final_dims


def repeat_kv(x: torch.Tensor, repeats: int, dim=2) -> torch.Tensor:
    if repeats == 1:
        return x

    return torch.repeat_interleave(x, dim=dim, repeats=repeats)


class MultiheadAttention(nn.Module):

    def __init__(self,
                 dim,
                 num_heads,
                 num_kv_heads=None,
                 bias=False,
                 dropout=0.0,
                 pack_qkv=True,
                 gating=False,
                 attention_mode=None):
        super().__init__()
        assert dim % num_heads == 0, f"number of heads must devide dim"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = self.dim // num_heads
        self.bias = bias
        self.num_kv_heads = num_heads if num_kv_heads is None else num_kv_heads
        self.kv_dim = self.num_kv_heads * self.head_dim
        self.repeats = self.dim // self.kv_dim
        self.pack_qkv = pack_qkv
        self.attn_dropout = dropout
        self.gating = gating
        if self.num_heads * self.head_dim != self.dim:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.dim}"
                f" and `num_heads`: {num_heads})."
            )
        if self.pack_qkv:
            self.c_attn = Linear(self.dim, self.dim + 2 * self.kv_dim, bias=self.bias)
        else:
            self.q_proj = Linear(self.dim, self.dim, bias=self.bias)
            self.k_proj = Linear(self.dim, self.kv_dim, bias=self.bias)
            self.v_proj = Linear(self.dim, self.kv_dim, bias=self.bias)

        if self.gating:
            self.g_proj = Linear(self.dim, self.dim, bias=self.bias, init="gating")
        self.c_proj = Linear(self.dim, self.dim, bias=self.bias, init="final")
        self.attention_mode = "sdpa" if attention_mode is None else attention_mode

    def _project_qkv(self,
                     x: torch.Tensor,
                     freqs_cis: torch.Tensor = None):
        """
        Args:
            x: tensor[*, seq_len, self.dim + 2 * self.kv_dim], qkv packed tensor

        Returns:
            q, k, v: tensor[*, seq_len, num_heads, head_dim]
        """
        if self.pack_qkv:
            q, k, v = self.c_attn(x).split([self.dim, self.kv_dim, self.kv_dim], dim=-1)
        else:
            q = self.q_proj(x)
            k = self.k_proj(x)
            v = self.v_proj(x)

        q = q.view(q.shape[:-1] + (-1, self.head_dim))  # [*, seq_len, num_kv_heads, head_dim]
        k = k.view(k.shape[:-1] + (-1, self.head_dim))  # [*, seq_len, num_head, head_dim]
        v = v.view(v.shape[:-1] + (-1, self.head_dim))

        if freqs_cis is not None:
            q = apply_rotary_emb(q, freqs_cis)  # [*, seq_len, num_heads, head_dim]
            k = apply_rotary_emb(k, freqs_cis)  # [*, seq_len, num_kv_heads, head_dim]

        k = repeat_kv(k, self.repeats)  # [bs, seq_len, num_heads, head_dim]
        v = repeat_kv(v, self.repeats)

        return q, k, v

    def _scaled_dot_product_attention(self,
                                      q, k, v,
                                      attn_mask=None,
                                      attn_bias=None,
                                      is_causal=False,
                                      dropout_p=0.0,
                                      return_attn_weight=False):
        """
        Args:
            q, k, v: tensor[bs, num_head, seq_len, head_dim]

        Returns:
            out: tensor[bs, seq_len, dim]
        """
        if not self.training:
            dropout_p = 0.0

        if is_causal:
            assert attn_mask is None and attn_bias is None, f"causal attention don't support custom attention mask and bias"

        if self.attention_mode in ("native", "sdpa") and not return_attn_weight:
            if attn_bias is not None:
                if isinstance(attn_bias, (list, tuple)):
                    attn_bias = sum(attn_bias)

                if attn_mask is not None:
                    attn_bias.masked_fill_(~attn_mask, torch.finfo(attn_bias.dtype).min)
                attn_mask = attn_bias.to(q.dtype)

            q = q.transpose(-2, -3)  # [*, num_heads, seq_len, dim]
            k = k.transpose(-2, -3)
            v = v.transpose(-2, -3)
            y = F.scaled_dot_product_attention(q, k, v,
                                               attn_mask=attn_mask,
                                               dropout_p=dropout_p,
                                               is_causal=is_causal)  # [bs, num_heads, seq_len, head_dim]
            y = y.transpose(-2, -3)
            attn_weights = None
        else:
            if attn_bias is not None:
                if isinstance(attn_bias, (list, tuple)):
                    attn_bias = sum(attn_bias)

                if attn_mask is not None:
                    attn_bias.masked_fill_(~attn_mask, torch.finfo(attn_bias.dtype).min)
                attn_mask = attn_bias.to(q.dtype)

            q = q.transpose(-2, -3)  # [*, num_heads, seq_len, head_dim]
            k = k.transpose(-2, -3)
            v = v.transpose(-2, -3)
            attn_weights = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            seq_len = q.size(-2)
            # causal mask to ensure that attention is only applied to the left in the input sequence
            if is_causal:
                attn_mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=q.device))
                attn_mask = attn_mask.expand_as(attn_weights)

            if attn_mask is not None:
                if attn_mask.dtype == torch.bool:
                    attn_weights = attn_weights.masked_fill(~attn_mask, torch.finfo(attn_weights.dtype).min)
                else:
                    attn_weights += attn_mask

            scores = F.softmax(attn_weights.float(), dim=-1).type_as(q)
            scores = torch.dropout(scores, dropout_p, train=self.training)
            y = scores @ v  # [bs, num_layers, num_heads, seq_len, seq_len]
            y = y.transpose(-2, -3)

        return flatten_final_dims(y, 2), attn_weights

    def forward(self,
                x: torch.Tensor,
                freqs_cis: Optional[torch.Tensor] = None,
                attn_mask: Optional[torch.Tensor] = None,
                attn_bias: Optional[torch.Tensor] = None,
                is_causal: bool = False,
                return_attn_weight: bool = False
                ):
        """
        Args:
            x: [*, seq_len, dim]
            freqs_cis: [seq_len, head_dim // 2, 2]
            attn_mask: [*, 1, seq_len, seq_len]
            attn_bias: [*, 1, seq_len, seq_len], tensor of list of attention biases
            is_causal: [seq_len, seq_len]
            return_attn_weight: whether to return the attention weights

        Returns:
            y: [*, seq_len, dim], ouptut hiddens
            attn_weight: [*, num_heads, seq_len, seq_len]
        """
        q, k, v = self._project_qkv(x, freqs_cis=freqs_cis)
        y, attn_weight = self._scaled_dot_product_attention(q, k, v,
                                                            attn_mask=attn_mask,
                                                            attn_bias=attn_bias,
                                                            is_causal=is_causal,
                                                            return_attn_weight=return_attn_weight,
                                                            dropout_p=self.attn_dropout)
        if self.gating:
            y = y * self.g_proj(x).sigmoid()

        y = self.c_proj(y)

        return y, attn_weight


class CausalMultiheadAttention(MultiheadAttention):

    def __init__(self,
                 dim,
                 num_heads,
                 max_len=1024,
                 num_kv_heads=None,
                 pack_qkv=True,
                 bias=False):
        super().__init__(dim,
                         num_heads,
                         num_kv_heads=num_kv_heads,
                         pack_qkv=pack_qkv,
                         bias=bias)
        self.kv_cache = None
        self.max_len = max_len

    def update_kv_cache(self, k, v, start_pos):
        if self.kv_cache is None:
            bs = k.shape[0]
            dtype = k.dtype
            device = k.device
            cache_shape = (bs, self.max_len, self.num_heads, self.head_dim)
            self.kv_cache = (
                torch.zeros(cache_shape, device=device, dtype=dtype),
                torch.zeros(cache_shape, device=device, dtype=dtype)
            )
        cache_k, cache_v = self.kv_cache
        if start_pos[-1] >= self.max_len:
            start_pos = torch.tensor(self.max_len - 1, device=start_pos.device)
            # shift 1 position to the left
            cache_k = torch.roll(cache_k, shifts=-1, dims=1)
            cache_v = torch.roll(cache_v, shifts=-1, dims=1)

        k = cache_k.index_copy(1, start_pos, k)
        v = cache_v.index_copy(1, start_pos, v)
        self.kv_cache = (k, v)

        return k, v

    def forward(self,
                x: torch.Tensor,
                freqs_cis: Optional[torch.Tensor] = None,
                attn_mask: Optional[torch.Tensor] = None,
                start_pos: Optional[torch.Tensor] = None,
                return_attn_weight: bool = False):
        q, k, v = self._project_qkv(x, freqs_cis=freqs_cis)
        if start_pos is not None:
            k, v = self.update_kv_cache(k, v, start_pos)
        
        y, attn_weight = self._scaled_dot_product_attention(q, k, v,
                                                            attn_mask=attn_mask,
                                                            is_causal=attn_mask is None,
                                                            return_attn_weight=return_attn_weight,
                                                            dropout_p=self.attn_dropout)
        y = self.c_proj(y)

        return y, attn_weight


class GatedMultiHeadAttention(MultiheadAttention):
    __init__ = partialmethod(MultiheadAttention.__init__, gating=True)


class MSARowAttention(nn.Module):

    def __init__(self,
                 dim,
                 num_heads,
                 num_kv_heads=None,
                 bias=False,
                 dropout=0.0,
                 pack_qkv=True,
                 attention_mode=None,
                 gating=False):
        super(MSARowAttention, self).__init__()
        self.mha = MultiheadAttention(dim,
                                      num_heads,
                                      num_kv_heads=num_kv_heads,
                                      bias=bias,
                                      dropout=dropout,
                                      pack_qkv=pack_qkv,
                                      gating=gating,
                                      attention_mode=attention_mode)
        self.head_dim = self.mha.head_dim
        self.freqs_cis = None

    def update_freqs_cis(self, seq_len, dtype=None, device=None):
        if self.freqs_cis is None or seq_len > self.freqs_cis.shape[0]:
            self.freqs_cis = precompute_freqs_cis(seq_len,
                                                  rotary_dim=self.head_dim,
                                                  dtype=dtype,
                                                  device=device)

    @torch.jit.ignore
    def _chunk(self,
               m: torch.Tensor,
               attn_bias: Optional[torch.Tensor],
               return_attn_weight: bool,
               chunk_size: int,
               ) -> torch.Tensor:
        def fn(m, biases):
            return super().forward(m, biases=biases)

        inputs = {"m": m}
        if attn_bias is not None:
            inputs["attn_bias"] = attn_bias
            fn = partial(fn,
                         return_attn_weight=return_attn_weight,
                         freqs_cis=self.freqs_cis)
        else:
            fn = partial(fn, attn_bias=None, return_attn_weight=return_attn_weight)

        return chunk_layer(
            fn,
            inputs,
            chunk_size=chunk_size,
            no_batch_dims=len(m.shape[:-2])
        )

    def forward(self,
                m: torch.Tensor,
                mask: Optional[torch.Tensor] = None,
                return_attn_weight: bool = False,
                chunk_size: Optional[int] = None
                ):
        """
        Args:
            m: [*, row, col, dim]
            mask: [*, row, col]

        Returns:
            out: [*, row, col, dim]
            attn_weight: [*, row, num_heads, col, col]
        """
        if mask is not None:
            mask = mask[..., :, None, None, :].bool()  # [*, row, num_heads(1), col(1), col]

        self.update_freqs_cis(seq_len=m.shape[-2], dtype=m.dtype, device=m.device)
        if chunk_size is not None:
            return self._chunk(
                m,
                mask,
                return_attn_weight,
                chunk_size
            )
        # attn bias is unstability
        return self.mha(m,
                        freqs_cis=self.freqs_cis,
                        attn_mask=mask,
                        return_attn_weight=return_attn_weight)


class MSAColumnAttention(MSARowAttention):

    def forward(self,
                m: torch.Tensor,
                mask: Optional[torch.Tensor] = None,
                return_attn_weight: bool = False,
                chunk_size: Optional[int] = None):
        """
        Args:
            m: [*, row, col, dim]
            mask: [*, row, col]

        Returns:
            out: [*, row, col, dim]
            attn_weights: [*, col, num_heads, row, row]
        """
        m = m.transpose(-2, -3)
        if mask is not None:
            mask = mask.transpose(-1, -2)

        m, attn_weights = super().forward(m,
                                          mask=mask,
                                          return_attn_weight=return_attn_weight,
                                          chunk_size=chunk_size)
        m = m.transpose(-2, -3)

        return m, attn_weights
