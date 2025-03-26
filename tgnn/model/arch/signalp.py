# Copyright (c) 2025, Tencent Inc. All rights reserved.
# Data: 2025/1/9 15:58
# Author: chenchenqin
from typing import Optional

import torch
import torch.nn as nn

from tgnn.config import configurable
from tgnn.utils.tensor import isin, masked_mean
from .usm import USM, USMClassificationHead
from ..build import MODEL_REGISTRY


@MODEL_REGISTRY.register()
class SignalP(USM):
    SIGNALP6_CLASS_LABEL_MAP = {
        'NO_SP': [0, 1, 2],
        'SP': [3, 4, 5, 6, 7, 8],
        'LIPO': [9, 10, 11, 12, 13, 14, 15],
        'TAT': [16, 17, 18, 19, 20, 21, 22],
        'TATLIPO': [23, 24, 25, 26, 27, 28, 29, 30],
        'PILIN': [31, 32, 33, 34, 35, 36]
    }

    N_TAG_IDS = [3, 9, 16, 17, 23, 24, 31]
    H_TAG_IDS = [4, 10, 18, 25]
    C_TAG_IDS = [5, 11, 19, 26]
    MATURE_TAG_IDS = [12, 27, 32, 33]
    OTHER_TAG_IDS = [0, 1, 2, 6, 7, 8, 13, 14, 15, 20, 21, 22, 28, 29, 30, 34, 35, 36]
    CS_TAG_IDS = [5, 11, 19, 26, 31]

    @classmethod
    def from_config(cls, cfg):
        configs = super().from_config(cfg)
        tag_to_global = getattr(cfg.model, "tag_to_global", True)
        return {
            "num_classes": cfg.model.num_classes,
            "tag_to_global": tag_to_global,
            "num_kindom_classes": getattr(cfg.model, "num_kindom_classes", 4),
            **configs
        }

    @configurable
    def __init__(self,
                 num_classes: int,
                 vocab_size: int,
                 embedding_dim=512,
                 num_layers: int = 8,
                 num_heads: int = 8,
                 num_row_kv_heads: int = None,
                 num_col_kv_heads: int = None,
                 ffn_dim_multiplier: Optional[float] = None,
                 bias=False,
                 eps=1e-5,
                 tag_to_global=True,
                 padding_idx=None,
                 num_kindom_classes=4):
        super(SignalP, self).__init__(vocab_size,
                                      embedding_dim=embedding_dim,
                                      num_layers=num_layers,
                                      num_heads=num_heads,
                                      num_row_kv_heads=num_row_kv_heads,
                                      num_col_kv_heads=num_col_kv_heads,
                                      ffn_dim_multiplier=ffn_dim_multiplier,
                                      bias=bias,
                                      eps=eps,
                                      include_head=False,
                                      padding_idx=padding_idx)
        self.classifier = USMClassificationHead(self.embedding_dim, num_classes)
        self.num_kindom_classes = num_kindom_classes
        if self.num_kindom_classes > 1:
            self.kingdom_embedding = nn.Embedding(self.num_kindom_classes, self.embedding_dim)
            nn.init.zeros_(self.kingdom_embedding.weight)
        self.tag_to_global = tag_to_global
        if not self.tag_to_global:
            self.type_classifier = USMClassificationHead(self.embedding_dim, 6)

    def forward(self,
                token_ids: torch.Tensor,
                kindom_ids: torch.Tensor = None):
        """
        Args:
            token_ids: [bs, num_seq, seq_len], input mas token ids
            kindom_ids: [bs, ]
        """
        is_single_seq = token_ids.dim() == 2
        if is_single_seq:
            token_ids = token_ids[:, None]

        padding_mask = None
        if self.padding_idx is not None:
            padding_mask = token_ids.eq(self.padding_idx)  # B, row, col
            if not padding_mask.any():
                padding_mask = None  # [bs, row, col]

        mask = None if padding_mask is None else ~padding_mask
        emb = self.embedding(token_ids)  # [bs, seq_len, hidden_dim]
        if self.num_kindom_classes > 1 and kindom_ids is not None:
            emb = emb + self.kingdom_embedding(kindom_ids)[..., None, None, :]

        x, *_ = self._forward_transformer_imp(emb, mask=mask)
        x = x.mean(dim=1)  # [ba, seq_len, dim]
        x = self.norm_final(x)
        logits = self.classifier(x)
        if mask is not None:
            mask = mask[..., 0, :][..., None]

        if self.tag_to_global:
            global_logits = self.compute_global_labels(logits.float(), mask)
        else:
            global_logits = self.type_classifier(masked_mean(mask, x, dim=-2))

        return {"tag": logits, 'global': global_logits}

    def compute_global_labels(self, logits, mask=None):
        """Aggregates probabilities for region-tagging
        Args:
            logits: [bs, seq_len, 37]
            mask: [bs, seq_len, 1]

        Returns:
            global: [bs, 6]
        """
        global_probs = masked_mean(mask, logits, dim=-2)  # [bs, 37]
        probs = []
        for name in ['NO_SP', 'SP', 'LIPO', 'TAT', 'TATLIPO', 'PILIN']:
            indices = self.SIGNALP6_CLASS_LABEL_MAP[name]
            probs.append(global_probs[:, indices].sum(dim=1))

        return torch.stack(probs, dim=1)

    def compute_post_probabilities(self, all_probs: torch.Tensor, kingdom_id: str):
        """Process the marginal probabilities to get cleaner outputs.
        Eukarya: sum all other classes into Sec/SPI
        Sec/SPII: sum h region and lipobox
        Sum all I,M,O probabilities into 0 (for plotting)
        """
        all_probs_out = all_probs.clone()
        # set 0 in out, then fill in again.
        if kingdom_id.lower() == 'eukarya':
            all_probs_out[..., self.N_TAG_IDS + self.H_TAG_IDS + self.C_TAG_IDS + self.MATURE_TAG_IDS] = 0
            all_probs_out[..., 3] = all_probs[..., self.N_TAG_IDS].sum(dim=-1)
            all_probs_out[..., 4] = all_probs[..., self.H_TAG_IDS].sum(dim=-1)
            all_probs_out[..., 5] = all_probs[..., self.C_TAG_IDS].sum(dim=-1)
            all_probs_out[..., 6] = all_probs[..., self.MATURE_TAG_IDS].sum(dim=-1)
        else:
            # no c region for SPII
            h_sec_probs = all_probs[..., [10, 11]].sum(dim=-1)  # merge Sec/SPII h + c probs
            h_tat_probs = all_probs[..., [25, 26]].sum(dim=-1)  # merge Tat/SPII h + c probs
            all_probs_out[..., [11, 26]] = 0
            all_probs_out[..., 10] = h_sec_probs
            all_probs_out[..., 25] = h_tat_probs

        # merge all OTHER probs into 0
        all_probs_out[..., 0] = all_probs[..., self.OTHER_TAG_IDS].sum(dim=-1)

        return all_probs_out

    def get_cleavage_sites(self, logits: torch.Tensor):
        """Convert sequences of tokens to the indices of the cleavage site.

        Args:
            logits: [seq_len, 37], position-wise logits , with "C" tags for cleavage sites

        Returns:
            cs_sites: integer of position that is a CS. -1 if no SP present in sequence."""
        tags = logits.argmax(dim=-1)
        cs_mask = isin(tags, self.CS_TAG_IDS)
        sp_idx = torch.nonzero(cs_mask)
        if sp_idx.numel() == 0:
            max_idx = -1
        else:
            max_idx = sp_idx.max().item() + 1

        return max_idx

    def get_cleavage_probility(self, logits: torch.Tensor):
        tags = logits.argmax(dim=-1)
        cs_mask = isin(tags, self.CS_TAG_IDS)
        sp_idx = torch.nonzero(cs_mask)
        if sp_idx.numel() == 0:
            return None

        max_idx = sp_idx.max().item()
        tag_id = tags[max_idx]
        prob = torch.sigmoid(logits[max_idx, tag_id])
        return prob

    def plot_sequence(self,
                      tag_probs,
                      aa_seq,
                      cleavage_site=-1,
                      hide_threshold=0.01,
                      max_len=70,
                      figsize=(12, 4.5),
                      title=None):
        import matplotlib.pyplot as plt
        IDS_TO_PLOT_LABEL = {
            0: 'O',
            3: 'N',  # SP
            4: 'H',
            5: 'C',
            9: 'N',  # LIPO
            10: 'H',
            12: 'c',
            16: 'N',  # TAT
            17: 'R',
            18: 'H',
            19: 'C',
            23: 'N',  # TATLIPO
            24: 'R',
            25: 'H',
            27: 'c',
            31: 'P',  # PILIN
            32: 'c',
            33: 'H'
        }
        # colors for letters, depending on region
        REGION_PLOT_COLORS = {'N': 'red',
                              'H': 'orange',
                              'C': 'gold',
                              'I': 'gray',
                              'M': 'gray',
                              'O': 'gray',
                              'c': 'cyan',
                              'R': 'lime',
                              'P': 'red'
                              }
        LABEL_IDS_TO_PLOT = [3, 4, 5, 9, 10, 12, 16, 17, 18, 19, 23, 24, 25, 27, 31, 32, 33]
        PROB_PLOT_COLORS = {3: 'red',
                            4: 'orange',
                            5: 'gold',
                            9: 'red',
                            10: 'orange',
                            12: 'cyan',
                            16: 'red',
                            17: 'lime',
                            18: 'orange',
                            19: 'gold',
                            23: 'red',
                            24: 'lime',
                            25: 'orange',
                            27: 'cyan',
                            31: 'red',
                            32: 'gold',
                            33: 'orange'}

        # labels to write to legend for probability channels
        IDX_LABEL_MAP = {
            3: 'Sec/SPI n',
            4: 'Sec/SPI h',
            5: 'Sec/SPI c',
            6: 'Sec/SPI I',
            7: 'Sec/SPI M',
            8: 'Sec/SPI O',
            9: 'Sec/SPII n',
            10: 'Sec/SPII h',
            12: 'Sec/SPII cys',
            13: 'Sec/SPII I',
            14: 'Sec/SPII M',
            15: 'Sec/SPII O',
            16: 'Tat/SPI n',
            17: 'Tat/SPI RR',
            18: 'Tat/SPI h',
            19: 'Tat/SPI c',
            20: 'Tat/SPI I',
            21: 'Tat/SPI M',
            22: 'Tat/SPI O',
            23: 'Tat/SPII n',
            24: 'Tat/SPII RR',
            25: 'Tat/SPII h',
            27: 'Tat/SPII cys',
            28: 'Tat/SPII I',
            29: 'Tat/SPII M',
            30: 'Tat/SPII O',
            31: 'Sec/SPIII P',  # PILIN
            32: 'Sec/SPIII cons.',
            33: 'Sec/SPIII h',
            34: 'Sec/SPIII I',
            35: 'Sec/SPIII M',
            36: 'Sec/SPIII O',
        }
        seq_length = min(len(aa_seq), max_len)
        tag_probs = tag_probs[:seq_length]
        tag_ids = tag_probs.argmax(dim=-1).tolist()
        pos_labels_to_plot = [IDS_TO_PLOT_LABEL[x] for x in tag_ids]
        fig = plt.figure(figsize=figsize)
        for x in torch.arange(1, seq_length + 1):
            plt.axvline(x, c='whitesmoke')  # Gridline at each pos
            tag = pos_labels_to_plot[x - 1]  # Predicted label at each pos
            plt.text(x, -0.05, tag, c=REGION_PLOT_COLORS[tag], ha='center', va='center')
            aa = aa_seq[x - 1]  # AA at each pos
            plt.text(x, -0.1, aa, ha='center', va='center')

        # post-processed marginals: all other states summed in 0
        other_probs = tag_probs[:, 0]
        plt.plot(torch.arange(1, seq_length + 1), other_probs, label='OTHER', c='lightcoral', linestyle='--')
        colors_used = []  # keep track of colors used - when already used, switch linestyle
        # NOTE this is still not optimal, only works when colors not used more than two times
        for idx in LABEL_IDS_TO_PLOT:
            probs = tag_probs[:, idx]
            # skip low
            if (probs > hide_threshold).any():
                linestyle = '--' if PROB_PLOT_COLORS[idx] in colors_used else '-'
                plt.plot(torch.arange(1, seq_length + 1), probs, label=IDX_LABEL_MAP[idx], c=PROB_PLOT_COLORS[idx],
                         linestyle=linestyle)
                colors_used.append(PROB_PLOT_COLORS[idx])

        if cleavage_site > 0:
            plt.plot((cleavage_site, cleavage_site), (0, 1.25), c='darkgreen', linestyle='--', label='CS')

        # adjust lim to fit pos labels
        plt.ylim((-0.15, 1.05))
        plt.xlim((0, seq_length + 1))

        plt.ylabel('Probability')
        plt.xlabel('Protein sequence')
        if title is not None:
            plt.title(title)

        plt.legend(loc='upper right')
        plt.tight_layout()

        return fig
