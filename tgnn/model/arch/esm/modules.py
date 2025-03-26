# Copyright (c) 2025, Tencent Inc. All rights reserved.
# Data: 2025/1/9 15:58
# Author: chenchenqin
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from tgnn.model.layer import GELU, Linear, LayerNorm
from .multihead_attention import MultiheadAttention

try:
    from apex.normalization import FusedLayerNorm as _FusedLayerNorm

    class ESM1bLayerNorm(_FusedLayerNorm):
        @torch.jit.unused
        def forward(self, x):
            if not x.is_cuda:
                return super().forward(x)
            else:
                with torch.cuda.device(x.device):
                    return super().forward(x)
except ImportError:
    from torch.nn import LayerNorm as ESM1bLayerNorm


class RobertaLMHead(nn.Module):
    """Head for masked language modeling."""

    def __init__(self, dim, output_dim, weight):
        super().__init__()
        self.dense = nn.Linear(dim, dim)
        self.layer_norm = ESM1bLayerNorm(dim)
        self.weight = weight
        self.bias = nn.Parameter(torch.zeros(output_dim))
        self.gelu = GELU('erf')

    def forward(self, features):
        x = self.dense(features)
        x = self.gelu(x)
        x = self.layer_norm(x)
        # project back to size of vocabulary with bias
        x = F.linear(x, self.weight) + self.bias

        return x


def apc(x):
    "Perform average product correct, used for contact prediction."
    a1 = x.sum(-1, keepdims=True)
    a2 = x.sum(-2, keepdims=True)
    a12 = x.sum((-1, -2), keepdims=True)

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
            in_features: int,
            prepend_bos: bool,
            append_eos: bool,
            bias=True,
            eos_idx: Optional[int] = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.prepend_bos = prepend_bos
        self.append_eos = append_eos
        if append_eos and eos_idx is None:
            raise ValueError("Using an alphabet with eos token, but no eos token was passed in.")
        self.eos_idx = eos_idx
        self.regression = nn.Linear(in_features, 1, bias)
        self.activation = nn.Sigmoid()

    def forward(self, tokens, attentions):
        # remove eos token attentions
        if self.append_eos:
            eos_mask = tokens.ne(self.eos_idx).to(attentions)
            eos_mask = eos_mask.unsqueeze(1) * eos_mask.unsqueeze(2)
            attentions = attentions * eos_mask[:, None, None, :, :]
            attentions = attentions[..., :-1, :-1]
        # remove cls token attentions
        if self.prepend_bos:
            attentions = attentions[..., 1:, 1:]
        batch_size, layers, heads, seqlen, _ = attentions.size()
        attentions = attentions.view(batch_size, layers * heads, seqlen, seqlen)

        # features: B x C x T x T
        attentions = attentions.to(
            self.regression.weight.device
        )  # attentions always float32, may need to convert to float16
        attentions = apc(symmetrize(attentions))
        attentions = attentions.permute(0, 2, 3, 1)
        return self.activation(self.regression(attentions).squeeze(3))


class TransformerLayer(nn.Module):
    """Transformer layer block."""

    def __init__(
            self,
            dim,
            num_heads,
            ffn_dim=None,
            use_rotary_embeddings=True
    ):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim or 4 * dim
        self.num_heads = num_heads
        self.self_attn = MultiheadAttention(
            self.dim,
            self.num_heads,
            use_rotary_embeddings=use_rotary_embeddings
        )
        self.self_attn_layer_norm = LayerNorm(self.dim)
        self.fc1 = Linear(self.dim, self.ffn_dim)
        self.fc2 = Linear(self.ffn_dim, self.dim, init="final")
        self.gelu = GELU('erf')
        self.final_layer_norm = LayerNorm(self.dim)

    def forward(
            self,
            x,
            attn_mask=None,
            key_padding_mask=None,
            need_weights=False
    ):
        residual = x
        x = self.self_attn_layer_norm(x)
        x, attn = self.self_attn(x,
                                 attn_mask=attn_mask,
                                 key_padding_mask=key_padding_mask,
                                 need_weights=need_weights
                                 )
        x = residual + x

        residual = x
        x = self.final_layer_norm(x)
        x = self.gelu(self.fc1(x))
        x = self.fc2(x)
        x = residual + x

        return x, attn
