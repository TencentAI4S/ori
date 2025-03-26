# Copyright (c) 2025, Tencent Inc. All rights reserved.
# Data: 2025/1/9 15:58
# Author: chenchenqin
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import typing as T
from dataclasses import dataclass

import deepspeed
import torch
import torch.nn as nn

from tgnn.model.layer import Linear, LayerNorm, RelativePositionEmbeddingSS
from tgnn.model.module import StructureModule, EvoformerBlockSS
from tgnn.model.module.head import DistogramHead


@dataclass
class StructureModuleConfig:
    c_s: int = 384
    c_z: int = 128
    c_ipa_hidden: int = 16
    c_angle_hidden: int = 128
    num_heads_ipa: int = 12
    num_qk_points: int = 4
    num_v_points: int = 8
    dropout_rate: float = 0.1
    num_blocks: int = 8
    num_transition_blocks: int = 1
    num_resnet_blocks: int = 2
    trans_scale_factor: float = 10
    inf: float = 1e5
    eps: float = 1e-8


class FoldingTrunk(nn.Module):
    def __init__(self,
                 c_s=1024,
                 c_z=128,
                 num_heads_seq=32,
                 num_heads_pair=4,
                 num_relative_positions=32,
                 num_evoformer_blocks=48,
                 num_recycle_bins=15,
                 max_recycles=4,
                 dropout=0.,
                 structure_module=StructureModuleConfig(),
                 activation_checkpoint=False,
                 chunk_size=None,
                 checkpoint_fn=deepspeed.checkpointing.checkpoint
                 ):
        super().__init__()
        assert c_s % num_heads_seq == 0
        assert c_z % num_heads_pair == 0
        self.max_recycles = max_recycles
        self.c_s = c_s
        self.c_z = c_z
        self.dropout = dropout
        self.num_evoformer_blocks = num_evoformer_blocks
        self.num_relative_positions = num_relative_positions
        self.pairwise_positional_embedding = RelativePositionEmbeddingSS(self.c_z, num_bins=self.num_relative_positions)
        self.blocks = nn.ModuleList(
            [
                EvoformerBlockSS(
                    c_s=self.c_s,
                    c_z=self.c_z,
                    num_heads_seq=num_heads_seq,
                    num_heads_pair=num_heads_pair,
                    dropout=self.dropout,
                )
                for _ in range(self.num_evoformer_blocks)
            ]
        )
        self.num_recycle_bins = num_recycle_bins
        self.recycle_s_norm = LayerNorm(self.c_s)
        self.recycle_z_norm = LayerNorm(self.c_z)
        self.recycle_disto = nn.Embedding(self.num_recycle_bins, self.c_z)
        # zero first input weight, it's empty
        torch.nn.init.zeros_(self.recycle_disto.weight[0])
        self.structure_module = StructureModule(**vars(structure_module))
        self.trunk2sm_s = Linear(self.c_s, structure_module.c_s)
        self.trunk2sm_z = Linear(self.c_z, structure_module.c_z)
        self.activation_checkpoint = activation_checkpoint
        self.checkpoint_fn = checkpoint_fn
        self.chunk_size = chunk_size

    def set_chunk_size(self, chunk_size):
        self.chunk_size = chunk_size

    def enable_activation_checkpoint(self, enabled=True):
        self.activation_checkpoint = enabled

    def trunk_iter(self,
                   s, z,
                   residue_index: T.Optional[torch.Tensor] = None,
                   mask: T.Optional[torch.Tensor] = None,
                   chunk_size: T.Optional[int] = None):
        """
        Args:
            s: [bs, seq_len, c_s], single feature
            z: [bs, seq_len, seq_len, c_z], pair feature
        """
        if residue_index is None:
            seq_len = s.shape[1]
            residue_index = torch.arange(seq_len, device=s.device)[None]

        z = z + self.pairwise_positional_embedding(residue_index, mask=mask)
        # forloop evoformer blocks
        for block in self.blocks:
            s, z = block(s, z, mask=mask, chunk_size=self.chunk_size)

        return s, z

    def forward(self,
                seq_feats,
                pair_feats,
                aatype,
                residue_index: T.Optional[torch.Tensor] = None,
                mask: T.Optional[torch.Tensor] = None,
                chunk_size: T.Optional[int] = None,
                num_recycles: T.Optional[int] = None):
        """
        Args:
            seq_feats: [bs, seq_len, c_s], tensor of sequence features
            pair_feats: [bs, seq_len, seq_len, c_z], tensor of pair features
            aatype: [bs, seq_len], residue ids
            residue_index: [B, L], long tensor giving the position in the sequence
            mask: [B, L], boolean tensor indicating valid residues, seq_mask
        Outputs:
            predicted_structure: [B, L, num_atoms * 3] tensor wrapped in a Coordinates object
        """
        device = seq_feats.device
        s_s_0 = seq_feats
        s_z_0 = pair_feats
        if num_recycles is None:
            num_recycles = self.max_recycles
        else:
            assert num_recycles >= 0, "Number of recycles must not be negative."
            num_recycles += 1  # First 'recycle' is just the standard forward pass through the model.
        chunk_size = chunk_size or self.chunk_size
        grad_enabled = torch.is_grad_enabled()
        s_s = s_s_0
        s_z = s_z_0  # [bs, seq_len, seq_len, c_z]
        recycle_s = torch.zeros_like(s_s)
        recycle_z = torch.zeros_like(s_z)
        # distogram bins range[0, 14]
        recycle_bins = torch.zeros(*s_z.shape[:-1], device=device, dtype=torch.int64)  # [bs, seq_len, seq_len]
        for recycle_idx in range(num_recycles):
            # last time enabled grad
            with torch.set_grad_enabled((recycle_idx == num_recycles - 1) and grad_enabled):
                recycle_s = self.recycle_s_norm(recycle_s.detach())
                recycle_z = self.recycle_z_norm(recycle_z.detach())
                recycle_z += self.recycle_disto(recycle_bins.detach())  # [bs, seq_len, seq_len, c_z]
                s_s = s_s_0 + recycle_s
                s_z = s_z_0 + recycle_z
                s_s, s_z = self.trunk_iter(s_s, s_z,
                                           residue_index=residue_index,
                                           mask=mask, chunk_size=chunk_size)
                structure = self.structure_module(
                    self.trunk2sm_s(s_s),
                    aatype=aatype,
                    z=self.trunk2sm_z(s_z),
                    mask=mask
                )
                recycle_s = s_s
                recycle_z = s_z
                # only need structure_module last time ouput position
                positions = structure["positions"][-1]  # [bs, seq_len, 14, 3]
                # Distogram needs the N, CA, C coordinates, and bin constants same as alphafold.
                recycle_bins = DistogramHead.compute_distogram(positions[:, :, :3],
                                                               num_bins=self.num_recycle_bins)

        structure["s_s"] = s_s
        structure["s_z"] = s_z
        return structure
