# Copyright (c) 2025, Tencent Inc. All rights reserved.
# Data: 2025/1/9 15:58
# Author: chenchenqin
import re
from typing import Optional, Tuple

import deepspeed
import torch
import torch.nn as nn

from tgnn.config import configurable
from tgnn.model.layer import LayerNorm
from tgnn.model.module.head.contact_head import ContactPredictionHead
from tgnn.model.module.head.lm_head import RobertaLMHead
from tgnn.protein import residue_constants as rc
from tgnn.tokenizer import Alphabet, build_tokenizer
from .modules import TransformerLayer
from ...build import MODEL_REGISTRY


@MODEL_REGISTRY.register()
class ESM2(nn.Module):

    CONFIG = {
        "esm2-8m": dict(num_layers=6, num_heads=20, embedding_dim=320),
        "esm2-35m": dict(num_layers=12, num_heads=20, embedding_dim=480),
        "esm2-150m": dict(num_layers=30, num_heads=20, embedding_dim=640),
        "esm2-650m": dict(num_layers=33, num_heads=20, embedding_dim=1280),
        "esm2-3b": dict(num_layers=36, num_heads=40, embedding_dim=2560),
        "esm2-15b": dict(num_layers=48, num_heads=40, embedding_dim=2560)
    }

    @classmethod
    def from_config(cls, cfg):
        num_layers = cfg.model.num_layers
        num_heads = cfg.model.num_heads
        embedding_dim = cfg.model.num_hiddens
        if cfg.model.type:
            mcfg = cls.CONFIG[cfg.model.type]
            num_layers = mcfg["num_layers"]
            num_heads = mcfg["num_heads"]
            embedding_dim = mcfg["embedding_dim"]
        tokenizer = build_tokenizer(cfg)
        return {
            "vocab_size": len(tokenizer),
            "num_layers": num_layers,
            "num_heads": num_heads,
            "embedding_dim": embedding_dim,
            "pad_id": tokenizer.pad,
            "mask_id": tokenizer.mask,
            "include_contact_head": cfg.model.get("include_contact_head", False),
        }

    @configurable
    def __init__(
            self,
            vocab_size=33,
            num_layers: int = 33,
            num_heads: int = 20,
            embedding_dim: int = 1280,
            token_dropout: bool = True,
            pad_id: int = 1,
            mask_id: int = 32,
            include_head=True,
            include_contact_head=False
    ):
        super().__init__()
        self.pad_id = pad_id
        self.mask_id = mask_id
        self.vocab_size = vocab_size
        self.include_head = include_head
        self.include_contact_head = include_contact_head
        self.num_layers = num_layers
        self.embed_dim = embedding_dim
        self.num_heads = num_heads
        self.token_dropout = token_dropout
        self.embed_tokens = nn.Embedding(self.vocab_size, self.embed_dim, padding_idx=self.pad_id)
        self.layers = nn.ModuleList(
            [
                TransformerLayer(
                    self.embed_dim,
                    num_heads=self.num_heads
                )
                for _ in range(self.num_layers)
            ]
        )
        self.emb_layer_norm_after = LayerNorm(self.embed_dim)
        if self.include_head:
            if self.include_contact_head:
                self.contact_head = ContactPredictionHead(self.num_layers * self.num_heads)
            self.lm_head = RobertaLMHead(
                dim=self.embed_dim,
                output_dim=self.vocab_size,
                weight=self.embed_tokens.weight
            )
        self.activation_checkpoint = False
        self.activation_checkpoint_fn = deepspeed.checkpointing.checkpoint

    def enable_activation_checkpoint(self, enabled=True):
        self.activation_checkpoint = enabled

    def forward(self,
                tokens,
                seq_mask=None,
                repr_layers=(),
                need_weights=False,
                return_contacts=False):
        if return_contacts:
            need_weights = True

        assert tokens.ndim == 2
        padding_mask = tokens.eq(self.pad_id)  # B, T
        x = self.embed_tokens(tokens)
        # scale sequence embeddings based on the ratio of masked tokens
        if self.token_dropout:
            x.masked_fill_((tokens == self.mask_id).unsqueeze(-1), 0.0)
            # x: B x T x C
            mask_ratio_train = 0.15 * 0.8
            src_lengths = (~padding_mask).sum(-1)
            mask_ratio_observed = (tokens == self.mask_id).sum(-1).to(x.dtype) / src_lengths
            x = x * (1 - mask_ratio_train) / (1 - mask_ratio_observed)[:, None, None]

        # reset padding token embedding to zeros
        if padding_mask is not None:
            x.masked_fill_(padding_mask.unsqueeze(-1), 0.0)

        repr_layers = set(repr_layers)
        # return dict
        hidden_representations = {}
        if 0 in repr_layers:
            hidden_representations[0] = x

        attn_weights = []
        # (B, T, E) => (T, B, E)
        x = x.transpose(0, 1)

        if not padding_mask.any():
            padding_mask = None

        for layer_idx, layer in enumerate(self.layers):
            if self.activation_checkpoint:
                x, attn = self.activation_checkpoint_fn(
                    layer,
                    x,
                    None,
                    padding_mask,
                    need_weights
                )
            else:
                x, attn = layer(
                    x,
                    key_padding_mask=padding_mask,
                    need_weights=need_weights
                )

            if (layer_idx + 1) in repr_layers:
                hidden_representations[layer_idx + 1] = x.transpose(0, 1)

            if need_weights:
                # (H, B, T, T) => (B, H, T, T)
                attn_weights.append(attn.transpose(0, 1))

        x = self.emb_layer_norm_after(x)
        x = x.transpose(0, 1)  # (T, B, E) => (B, T, E)

        # last hidden representation should have layer norm applied
        if (layer_idx + 1) in repr_layers:
            hidden_representations[layer_idx + 1] = x

        if self.include_head:
            x = self.lm_head(x)

        result = {"output": x}

        if len(repr_layers) > 0:
            result["representations"] = hidden_representations

        if need_weights:
            # attentions: bs x L x H x T x T
            attentions = torch.stack(attn_weights, 1)
            if seq_mask is None:
                seq_mask = padding_mask

            result["attentions"] = attentions
            if return_contacts and self.include_contact_head:
                contacts = self.contact_head(attentions, seq_mask=seq_mask)
                result["contacts"] = contacts

        return result


class ESM2ClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self,
                 dim,
                 num_classes=2,
                 dropout=0.0):
        super().__init__()
        self.dense = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.out_proj = nn.Linear(dim, num_classes)

    def forward(self, features):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)

        return x


@MODEL_REGISTRY.register()
class ESM2SeqClassifier(ESM2):

    @classmethod
    def from_config(cls, cfg):
        params = super().from_config(cfg)
        params.pop("include_contact_head", None)
        return {
            "num_classes": cfg.model.num_classes,
            **params
        }

    @configurable
    def __init__(self,
                 vocab_size,
                 num_layers: int = 33,
                 num_heads: int = 20,
                 embedding_dim: int = 1280,
                 num_classes=2,
                 pad_id: int = 1,
                 mask_id: int = 32
                 ):
        super().__init__(vocab_size,
                         num_layers,
                         num_heads,
                         embedding_dim,
                         token_dropout=False,
                         pad_id=pad_id,
                         mask_id=mask_id,
                         include_head=False,
                         include_contact_head=False)
        self.classifier = ESM2ClassificationHead(embedding_dim, num_classes)

    def forward(self, tokens):
        x = super().forward(tokens)["output"]

        return self.classifier(x)


MODEL_REGISTRY.register("ESM2SeqRegressor", ESM2SeqClassifier)


class ESMFeaturizer(ESM2):

    def __init__(self,
                 model_name="esm2_t36_3B_UR50D",
                 use_attn_map=False):
        super().__init__()
        alias_names = {
            "esm2-15b": "esm2_t48_15B_UR50D",
            "esm2-3b": "esm2_t36_3B_UR50D",
            "esm2-650m": "esm2_t33_650M_UR50D",
            "esm2-150m": "esm2_t30_150M_UR50D",
            "esm2-35m": "esm2_t12_35M_UR50D",
            "esm2-8m": "esm2_t6_8M_UR50D"
        }
        if model_name in alias_names:
            model_name = alias_names[model_name]

        model_url = f"https://dl.fbaipublicfiles.com/fair-esm/models/{model_name}.pt"
        regression_url = f"https://dl.fbaipublicfiles.com/fair-esm/regression/{model_name}-contact-regression.pt"
        model_data = torch.hub.load_state_dict_from_url(model_url, progress=False, map_location="cpu")
        regression_data = torch.hub.load_state_dict_from_url(regression_url, progress=False, map_location="cpu")
        model_data["model"].update(regression_data["model"])

        def upgrade_state_dict(state_dict):
            """Removes prefixes 'model.encoder.sentence_encoder.' and 'model.encoder.'."""
            prefixes = ["encoder.sentence_encoder.", "encoder."]
            pattern = re.compile("^" + "|".join(prefixes))
            state_dict = {pattern.sub("", name): param for name, param in state_dict.items()}
            return state_dict

        self.alphabet = Alphabet.from_architecture(model_name)
        cfg = model_data["cfg"]["model"]
        self.use_attn_map = use_attn_map
        super().__init__(vocab_size=len(self.alphabet),
                         num_layers=cfg.encoder_layers,
                         embedding_dim=cfg.encoder_embed_dim,
                         num_heads=cfg.encoder_attention_heads,
                         token_dropout=cfg.token_dropout,
                         pad_id=self.alphabet.pad_id,
                         mask_id=self.alphabet.mask_id,
                         include_head=False)
        state_dict = model_data["model"]
        state_dict = upgrade_state_dict(state_dict)
        self.load_state_dict(state_dict, strict=False)
        self.register_buffer("af2_to_esm", self._af2_to_esm(self.alphabet))

    @staticmethod
    def _af2_to_esm(d: Alphabet):
        # Remember that t is shifted from residue_constants by 1 (0 is padding).
        esm_reorder = [d.pad_id] + [
            d.get_idx(v) for v in rc.restypes_with_x
        ]
        return torch.tensor(esm_reorder)

    def _af2_idx_to_esm_idx(self, aa, mask=None):
        aa = aa + 1
        if mask is not None:
            aa.masked_fill(~mask.bool(), 0)

        return self.af2_to_esm[aa]

    def forward(
            self,
            aatype: torch.Tensor,
            mask: Optional[torch.Tensor] = None,
            masking_pattern: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        esmaa = self._af2_idx_to_esm_idx(aatype, mask)
        if masking_pattern is not None:
            esmaa[masking_pattern.bool()] = self.alphabet.mask_id

        """Adds bos/eos tokens for the language model, since the structure module doesn't use these."""
        bs = esmaa.size(0)
        bosi, eosi = self.alphabet.cls_id, self.alphabet.eos_id
        bos = esmaa.new_full((bs, 1), bosi)
        eos = esmaa.new_full((bs, 1), self.alphabet.pad_id)
        esmaa = torch.cat([bos, esmaa, eos], dim=1)
        # Use the first padding index as eos during inference.
        esmaa[range(bs), (esmaa != 1).sum(1)] = eosi
        res = super().forward(
            esmaa,
            repr_layers=range(self.num_layers + 1),
            need_weights=self.use_attn_map
        )
        esm_s = torch.stack(
            [v for _, v in sorted(res["representations"].items())], dim=2
        )
        # remove prepend and apend token
        esm_s = esm_s[:, 1:-1]  # bs, seq_len, num_layers, C

        if self.use_attn_map:
            esm_z = res["attentions"].permute(0, 4, 3, 1, 2).flatten(3, 4)[:, 1:-1, 1:-1, :]
        else:
            esm_z = None

        return esm_s, esm_z
