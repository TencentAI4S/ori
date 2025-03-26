# -*- coding: utf-8 -*-
# Copyright (c) 2023, Tencent Inc. All rights reserved.
# Author: chenchenqin
# Data: 2023/10/30 17:31
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from tgnn.config import get_config
from tgnn.model.layer import Linear
from tgnn.utils.tensor import permute_final_dims, flatten_final_dims

try:
    from deepspeed.ops.deepspeed4science import DS4Sci_EvoformerAttention
except ImportError:
    DS4Sci_EvoformerAttention = None


class GatedMultiheadAttention(nn.Module):
    """
    Standard multi-head attention using AlphaFold's default layer
    initialization. Allows multiple bias vectors.

    Args:
        c_q: Input dimension of query data
        c_k: Input dimension of key data
        c_v: Input dimension of value data
        dim: Per-head hidden dimension
        num_heads: Number of attention heads
        gating: Whether the output should be gated using query data
    """

    def __init__(
            self,
            c_q: int,
            *,
            c_k: int = None,
            c_v: int = None,
            dim: int = None,
            num_heads: int = 8,
            pack_qkv: bool = False,
            gating: bool = True
    ):
        super(GatedMultiheadAttention, self).__init__()
        dim = dim or c_q
        assert dim % num_heads == 0, f"Number of heads({num_heads}) must be divisible by hiddens({dim})"
        self.c_q = c_q
        self.c_k = c_k or c_q
        self.c_v = c_v or c_q
        self.pack_qkv = pack_qkv
        self.dim = dim
        self.head_dim = self.dim // num_heads
        self.num_heads = num_heads
        self.gating = gating
        if self.pack_qkv:
            assert self.c_q == self.c_k and self.c_q == self.c_v, (f"q({self.c_q}), k({self.c_k}),"
                                                                   f" v({self.c_v}) channel must be same")
            dim = self.c_q
            self.linear_qkv = Linear(dim, dim * 3, bias=False)
        else:
            self.linear_q = Linear(self.c_q, self.dim, bias=False, init="glorot")
            self.linear_k = Linear(self.c_k, self.dim, bias=False, init="glorot")
            self.linear_v = Linear(self.c_v, self.dim, bias=False, init="glorot")

        self.linear_o = Linear(self.dim, self.c_q, init="final")
        self.linear_g = None
        if self.gating:
            self.linear_g = Linear(self.c_q, self.dim, init="gating")
        self.use_evo_attn = get_config().model.attention_mode == "evo_attn"

    def _project_qkv(self,
                     q: torch.Tensor,
                     k: torch.Tensor = None,
                     v: torch.Tensor = None
                     ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            q_x: [*, Lq, C] query data, is also sequence feature s
        """
        k = q if k is None else k
        v = k if v is None else v
        # [*, Lq/k/v, num_heads * c]
        if self.pack_qkv:
            assert k is q and v is q
            q, k, v = self.linear_qkv(q).chunk(3, dim=-1)
        else:
            q = self.linear_q(q)
            k = self.linear_k(k)
            v = self.linear_v(v)
        # [*, Lq/k, num_heads, c]
        q = q.view(q.shape[:-1] + (self.num_heads, -1))
        k = k.view(k.shape[:-1] + (self.num_heads, -1))
        v = v.view(v.shape[:-1] + (self.num_heads, -1))

        return q, k, v

    def _deepspeed_evo_attn(
            self,
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
            biases: List[torch.Tensor],
    ):
        """""
        Compute attention using the DeepSpeed DS4Sci_EvoformerAttention kernel.

        Args:
            q: [*, Q, H, C_hidden] query data
            k: [*, K, H, C_hidden] key data
            v: [*, V, H, C_hidden] value data
            biases: List of biases that broadcast to [*, H, Q, K]
        """

        def reshape_dims(x):
            no_batch_dims = len(x.shape[:-3])
            if no_batch_dims < 2:
                return x.reshape(*((1,) * (2 - no_batch_dims) + x.shape))
            if no_batch_dims > 2:
                return x.reshape(*((x.shape[0], -1) + x.shape[-3:]))
            return x

        # Reshape tensors to match expected input shape [B, N, Q/K, H, C_hidden]
        # for DS4Sci_EvoformerAttention() by adding or flattening batch dims as needed.
        orig_shape = q.shape
        if len(orig_shape[:-3]) != 2:
            q = reshape_dims(q)
            k = reshape_dims(k)
            v = reshape_dims(v)
            biases = [reshape_dims(b) for b in biases]
        # requires inputs to be type bf16 or fp16
        orig_dtype = q.dtype
        if orig_dtype not in [torch.bfloat16, torch.float16]:
            o = DS4Sci_EvoformerAttention(q.to(dtype=torch.bfloat16),
                                          k.to(dtype=torch.bfloat16),
                                          v.to(dtype=torch.bfloat16),
                                          [b.to(dtype=torch.bfloat16) for b in biases])
            o = o.to(dtype=orig_dtype)
        else:
            o = DS4Sci_EvoformerAttention(q, k, v, biases)

        o = o.reshape(orig_shape)
        return o

    def _scale_dot_product_attention(
            self,
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            attn_biases: Optional[List[torch.Tensor]] = None,
            attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if attn_mask is not None:
            attn_mask = attn_mask.bool()

        if attn_biases:
            attn_biases = [attn_biases, ] if isinstance(attn_biases, torch.Tensor) else attn_biases
            attn_bias = attn_biases[0]
            for b in attn_biases[1:]:
                attn_bias = attn_bias + b
            attn_bias = attn_bias.type_as(query)
            if attn_mask is not None:
                attn_bias.masked_fill_(~attn_mask, float("-inf"))

            attn_mask = attn_bias

        query = query.transpose(-2, -3)  # [*, num_heads, seq_len, dim]
        key = key.transpose(-2, -3)
        value = value.transpose(-2, -3)
        if hasattr(F, "scaled_dot_product_attention"):
            y = F.scaled_dot_product_attention(query, key, value, attn_mask=attn_mask)
        else:
            head_dim = query.shape[-1]
            scale_factor = head_dim ** -0.5
            query = scale_factor * query
            # [*, H, C_hidden, K]
            key = permute_final_dims(key, (1, 0))
            # [*, H, Q, K]
            scores = query @ key
            if attn_mask.dtype == torch.bool:
                scores = scores.masked_fill(~attn_mask.bool(), float('-inf'))
            else:
                scores += attn_mask
            scores = F.softmax(scores.float(), dim=-1).type_as(query)
            # [*, H, Q, C_hidden]
            y = scores @ value

        y = y.transpose(-2, -3)  # [*, seq_len, H, C_hidden]

        return y

    def forward(
            self,
            query: torch.Tensor,
            key: torch.Tensor = None,
            value: torch.Tensor = None,
            biases: Optional[List[torch.Tensor]] = None,
            attn_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            query: [*, Lq, Cq] query data, sequence feature[bs, seq_len, c] or msa feature[bs, num_seqs, seq_len, c]
            key: [*, Lk, Ck] key data
            value: [*, Lk, Ck] value data
            biases: [*, num_heads, Lq, Lk] List of biases that broadcast to the shape
            attn_mask: [*, num_heads, Lq, Lk] attention mask

        Returns:
            [*, Lq, Cq] attention update
        """
        q, k, v = self._project_qkv(query, key, value)  # [*, Lq/Lk/Lv, num_heads, C]
        if biases is None:
            biases = []
        # remove none value
        biases = [b for b in biases if b is not None]
        # when two more biases
        use_ds_attn = DS4Sci_EvoformerAttention is not None and len(biases) <= 2 and attn_mask is None

        if use_ds_attn and self.use_evo_attn:
            y = self._deepspeed_evo_attn(q, k, v, biases)
        else:
            y = self._scale_dot_product_attention(q, k, v,
                                                  attn_biases=biases,
                                                  attn_mask=attn_mask)
        if self.gating:
            g = self.linear_g(query).sigmoid()
            g = g.view(g.shape[:-1] + (self.num_heads, -1))  # [*, Q, H, C_hidden]
            y = y * g

        y = flatten_final_dims(y, 2)  # [*, seq_len, num_heads * c_hidden]
        # [*, seq_len, c]
        y = self.linear_o(y)

        return y
