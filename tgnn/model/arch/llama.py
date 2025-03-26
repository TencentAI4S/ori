# Copyright (c) 2025, Tencent Inc. All rights reserved.
# Data: 2025/1/9 15:58
# Author: chenchenqin
from typing import Optional, Tuple

import deepspeed
import math
import torch
from torch import nn

from tgnn.config import configurable
from tgnn.tokenizer import build_tokenizer
from ..build import MODEL_REGISTRY
from ..generation import GenerationMixin
from ..layer import RMSNorm, precompute_freqs_cis
from ..module import CausalMultiheadAttention, SwiGLU


class Block(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 max_len,
                 num_kv_heads=None,
                 multiple_of=256,
                 ffn_dim_multiplier: Optional[float] = None,
                 bias=False,
                 eps=1e-6):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads
        self.attn = CausalMultiheadAttention(dim, num_heads, max_len,
                                       num_kv_heads=num_kv_heads,
                                       bias=bias)
        self.mlp = SwiGLU(dim,
                       hidden_dim=4 * dim,
                       multiple_of=multiple_of,
                       ffn_dim_multiplier=ffn_dim_multiplier,
                       )
        self.rms_1 = RMSNorm(dim, eps=eps)
        self.rms_2 = RMSNorm(dim, eps=eps)

    def forward(self,
                x: torch.Tensor,
                freqs_cis: torch.Tensor,
                attn_mask: Optional[torch.Tensor] = None,
                start_pos: Optional[torch.Tensor] = None,
                return_attn_weight=False,
                ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        hidden, attn_weight = self.attn(self.rms_1(x),
                                        freqs_cis=freqs_cis,
                                        attn_mask=attn_mask,
                                        start_pos=start_pos,
                                        return_attn_weight=return_attn_weight)
        x = x + hidden
        x = x + self.mlp(self.rms_2(x))

        return x, attn_weight


@MODEL_REGISTRY.register("llama")
@MODEL_REGISTRY.register()
class LLaMA(nn.Module, GenerationMixin):
    CONFIG = {
        "llama-0.1b": dict(num_layers=6, num_heads=12, embedding_dim=768, bias=False),
        "llama-0.5b": dict(num_layers=12, num_heads=12, embedding_dim=1536, bias=False),
        "llama-1b": dict(num_layers=12, num_heads=24, embedding_dim=2304, bias=False),
        "llama-1.5b": dict(num_layers=20, num_heads=24, embedding_dim=2304, bias=False),
        "llama-3b": dict(num_layers=24, num_heads=24, embedding_dim=3072, bias=False),
        "llama-7b": dict(num_layers=32, num_heads=32, embedding_dim=4096, bias=False),
        "llama-13b": dict(num_layers=40, num_heads=40, embedding_dim=5120, bias=False),
        "llama2-13b": dict(num_layers=40, num_heads=40, embedding_dim=5120, bias=False),
        "llama-30b": dict(num_layers=60, num_heads=52, embedding_dim=6656, bias=False),
        "llama-65b": dict(num_layers=80, num_heads=64, embedding_dim=8192, bias=False)
    }

    @classmethod
    def from_config(cls, cfg):
        model_type = cfg.model.type
        vocab_size = build_tokenizer(cfg).vocab_size
        seq_len = cfg.dataset.seq_len
        if model_type:
            mcfg = cls.CONFIG[model_type]
            num_layers, num_heads, embedding_dim = mcfg['num_layers'], mcfg['num_heads'], mcfg['embedding_dim']
            num_kv_heads = mcfg.get("num_kv_heads", None)
            bias = mcfg.get("bias", False)
        else:
            num_layers = cfg.model.num_layers
            num_heads = cfg.model.num_heads
            embedding_dim = cfg.model.num_hiddens
            num_kv_heads = cfg.model.num_kv_heads
            bias = cfg.model.bias

        return {
            "vocab_size": vocab_size,
            "max_len": seq_len,
            "embedding_dim": embedding_dim,
            "num_layers": num_layers,
            "num_heads": num_heads,
            "num_kv_heads": num_kv_heads,
            "bias": bias,
            "eps": cfg.model.eps
        }

    @configurable
    def __init__(self,
                 vocab_size: int,
                 max_len: int = 2048,
                 embedding_dim=512,
                 num_layers: int = 8,
                 num_heads: int = 8,
                 num_kv_heads: int = None,
                 multiple_of: int = 256,
                 ffn_dim_multiplier: Optional[float] = None,
                 bias=False,
                 eps=1e-5,
                 include_head=True):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.embedding_dim = embedding_dim
        self.head_dim = self.embedding_dim // self.num_heads
        self.eps = eps
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(self.vocab_size, self.embedding_dim),
                h=nn.ModuleList([
                    Block(
                        self.embedding_dim,
                        self.num_heads,
                        self.max_len,
                        num_kv_heads=num_kv_heads,
                        multiple_of=multiple_of,
                        ffn_dim_multiplier=ffn_dim_multiplier,
                        eps=self.eps) for _ in
                    range(self.num_layers)
                ]),
                ln_f=RMSNorm(self.embedding_dim, eps=self.eps),
            )
        )
        self.lm_head = nn.Linear(embedding_dim, vocab_size, bias=bias) if include_head else None
        self.freqs_cis = None
        self.apply(self._init_weights)
        self.activation_checkpoint = False
        self.activation_checkpoint_func = deepspeed.checkpointing.checkpoint
        n_params = sum(p.numel() for p in self.parameters())
        print("number of parameters: %.2fM" % (n_params / 1e6,))

    def enable_activation_checkpoint(self, enabled=True):
        self.activation_checkpoint = enabled

    @classmethod
    def checkpoint_layers(cls):
        return (Block,)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02 / math.sqrt(2 * self.num_layers))
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02 / math.sqrt(2 * self.num_layers))

    def _make_casual_mask(self, device):
        """
        Args:
            input_ids: [1, 1, seq_len, seq_len]
        """
        ones = torch.ones((self.max_len, self.max_len), dtype=torch.bool, device=device)
        return torch.tril(ones)[None, None]  # [1, 1, seq_len, seq_len]

    def _make_rope_mask(self, device, dtype=torch.float32):
        return precompute_freqs_cis(
            seq_len=self.max_len,
            rotary_dim=self.head_dim,
            dtype=dtype,
            device=device
        )

    def _forward_embedding_impl(self, input_ids):
        return self.transformer.wte(input_ids)

    def _forward_transformer_imp(self,
                                 x,
                                 attn_mask: Optional[torch.Tensor] = None,
                                 start_pos: Optional[torch.Tensor] = None,
                                 return_attn_weight: bool = False,
                                 return_representation: bool = False):
        device = x.device
        dtype = x.dtype
        bs, seq_len = x.shape[:2]

        if self.freqs_cis is None:
            self.freqs_cis = self._make_rope_mask(device=device, dtype=dtype)  # [seq_len, head_dim // 2, 2]

        if start_pos is not None:
            freqs_cis = self.freqs_cis.index_select(0, start_pos)
            if attn_mask is None:
                attn_mask = self._make_casual_mask(device)
            attn_mask = attn_mask.index_select(2, start_pos)
        else:
            freqs_cis = self.freqs_cis[:seq_len]
            if attn_mask is not None:
                attn_mask = attn_mask[:, :, :seq_len, :seq_len]
        attn_weights = []
        representations = []
        for block in self.transformer.h:
            if self.activation_checkpoint:
                x, attn_weight = self.activation_checkpoint_func(block, x, freqs_cis, attn_mask, start_pos,
                                                                 return_attn_weight)
            else:
                x, attn_weight = block(x, freqs_cis=freqs_cis, attn_mask=attn_mask, start_pos=start_pos,
                                       return_attn_weight=return_attn_weight)

            if return_attn_weight:
                attn_weights.append(attn_weight)
            if return_representation:
                representations.append(x)

        return x, representations, attn_weights

    def _forward_head_impl(self, x):
        x = self.transformer.ln_f(x)
        if self.lm_head is not None:
            x = self.lm_head(x)  # (b, t, vocab_size)

        return x

    def forward(self,
                input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                pos_ids: Optional[torch.Tensor] = None,
                return_attn_weight: bool = False,
                return_representation: bool = False):
        """
        Args:
            input_ids: [bs, seq_len], input token indics
            attention_mask: [bs, 1, seq_len, seq_len], attention mask, when it's none,
                default casual mask
            pos_ids: [seq_len or 1], use it when inference generating new token id or
                keep it none when training.
        """
        bs, seq_len = input_ids.shape
        assert (
                seq_len <= self.max_len
        ), f"Cannot forward sequence of length {seq_len}, max length is only {self.max_len}"
        x = self._forward_embedding_impl(input_ids)  # [bs, seq_len, hidden_dim]
        x, representations, attentions = self._forward_transformer_imp(x,
                                                                       attn_mask=attention_mask,
                                                                       start_pos=pos_ids,
                                                                       return_attn_weight=return_attn_weight,
                                                                       return_representation=return_representation
                                                                       )
        outputs = self._forward_head_impl(x)
        if return_representation or return_attn_weight:
            return outputs, representations, attentions

        return outputs

    def reset_cache(self):
        for layer in self.transformer.h:
            layer.attn.kv_cache = None

    def update_cache(self, batch_ids):
        for layer in self.transformer.h:
            k, v = layer.attn.kv_cache
            layer.attn.kv_cache = (
                k[batch_ids],
                v[batch_ids]
            )

    @classmethod
    def from_name(cls, name, max_len=2048, vocab_size=32000) -> "LLaMA":
        return cls(**cls.CONFIG[name], max_len=max_len, vocab_size=vocab_size)


@MODEL_REGISTRY.register()
class LLaMAClassifier(LLaMA):

    @classmethod
    def from_config(cls, cfg):
        num_classes = cfg.model.num_classes
        pad_id = build_tokenizer(cfg).pad_id
        return {
            **super().from_config(cfg),
            "num_classes": num_classes,
            "pad_id": pad_id
        }

    @configurable
    def __init__(self,
                 num_classes,
                 vocab_size: int,
                 pad_id=None,
                 max_len: int = 2048,
                 embedding_dim=512,
                 num_layers: int = 8,
                 num_heads: int = 8,
                 num_kv_heads: int = None,
                 multiple_of: int = 256,
                 ffn_dim_multiplier: Optional[float] = None,
                 bias=False,
                 eps=1e-5):
        super().__init__(vocab_size, max_len, embedding_dim,
                         num_layers=num_layers,
                         num_heads=num_heads,
                         num_kv_heads=num_kv_heads,
                         multiple_of=multiple_of,
                         ffn_dim_multiplier=ffn_dim_multiplier,
                         include_head=False,
                         bias=bias,
                         eps=eps)
        self.pad_id = pad_id
        self.num_classes = num_classes
        self.cls_head = nn.Linear(self.embedding_dim, self.num_classes, bias=False)

    def forward(self, token_ids, pos_ids=None, attention_mask=None):
        hiddens = super().forward(token_ids, pos_ids=pos_ids, attention_mask=attention_mask)
        batch_size = hiddens.shape[0]
        if self.pad_id is None:
            sequence_lengths = -1  # last token for classification or regression
        else:
            sequence_lengths = torch.ne(token_ids, self.pad_id).sum(dim=-1) - 1

        hiddens = hiddens[torch.arange(batch_size, device=hiddens.device), sequence_lengths]

        return self.cls_head(hiddens)


class LLaMAFeaturizer(LLaMA):

    def forward(self,
                input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                pos_ids: Optional[torch.Tensor] = None,
                return_representation=False,
                return_attn_weight=False):
        bs, seq_len = input_ids.shape
        assert (
                seq_len <= self.max_len
        ), f"Cannot forward sequence of length {seq_len}, max length is only {self.max_len}"
        embeddings = self._forward_embedding_impl(input_ids)  # [bs, seq_len, hidden_dim]
        hiddens, representations, attentions = self._forward_transformer_imp(embeddings,
                                                                             attn_mask=attention_mask,
                                                                             start_pos=pos_ids,
                                                                             return_representation=return_representation,
                                                                             return_attn_weight=return_attn_weight)
        logits = self._forward_head_impl(hiddens)
        outputs = {"output": logits}
        if return_representation:
            representations.insert(0, embeddings)
            outputs["representations"] = torch.stack(representations, dim=1)

        if return_attn_weight:
            outputs["attentions"] = torch.stack(attentions, dim=1)

        return outputs
