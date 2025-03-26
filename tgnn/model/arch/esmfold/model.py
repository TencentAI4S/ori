# Copyright (c) 2025, Tencent Inc. All rights reserved.
# Data: 2025/1/9 15:58
# Author: chenchenqin

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import typing as T
from dataclasses import dataclass

import torch
from torch import nn

from tgnn.config import configurable
from tgnn.model.arch.esm import ESMFeaturizer
from tgnn.model.build import MODEL_REGISTRY
from tgnn.model.layer import LayerNorm, Linear
from tgnn.model.module.head import LDDTHead, TMScoreHead
from tgnn.protein import residue_constants as rc, data_transform
from tgnn.protein.utils import output_to_pdb
from tgnn.utils.tensor import masked_mean, collate_dense_tensors
from .fold_trunk import FoldingTrunk, StructureModuleConfig
from .utils import batch_encode_sequences


@dataclass
class FoldingTrunkConfig:
    num_blocks: int = 48
    sequence_state_dim: int = 1024  # c_s
    pairwise_state_dim: int = 128  # pairwise state dim
    sequence_head_width: int = 32
    pairwise_head_width: int = 32
    position_bins: int = 32
    dropout: float = 0
    layer_drop: float = 0
    max_recycles: int = 4
    chunk_size: T.Optional[int] = None
    structure_module = None


@MODEL_REGISTRY.register()
class ESMFold(nn.Module):
    CONFIG = {
        "esm2-8m": "esm2_t6_8M_UR50D",
        "esm2-35m": "esm2_t12_35M_UR50D",
        "esm2-150m": "esm2_t30_150M_UR50D",
        "esm2-650m": "esm2_t33_650M_UR50D",
        "esm2-3b": "esm2_t36_3B_UR50D",
        "esm2-15b": "esm2_t48_15B_UR50D"
    }

    @classmethod
    def from_config(cls, cfg):
        esm_name = cls.CONFIG[cfg.model.type]
        return {
            "trunk_config": cfg.model.trunk,
            "lddt_hidden_dim": cfg.model.head.lddt.hidden_dim,
            "lddt_bins": cfg.model.head.lddt.num_bins,
            "distogram_bins": cfg.model.head.distogram.num_bins,
            "esm_name": esm_name
        }

    @configurable
    def __init__(self,
                 trunk_config: FoldingTrunkConfig,
                 use_esm_attn_map=False,
                 lddt_hidden_dim=128,
                 lddt_bins=50,
                 distogram_bins=64,
                 esm_name="esm2_t36_3B_UR50D"):
        super().__init__()
        self.tcfg = trunk_config
        self.scfg = self.tcfg.structure_module
        self.use_esm_attn_map = use_esm_attn_map
        self.esm = ESMFeaturizer(esm_name, use_attn_map=self.use_esm_attn_map).half()
        self.esm.requires_grad_(False)
        self.esm_s_combine = nn.Parameter(torch.zeros(self.esm.num_layers + 1))
        self.c_s = self.tcfg.sequence_state_dim
        self.c_z = self.tcfg.pairwise_state_dim
        self.esm_s_mlp = nn.Sequential(
            LayerNorm(self.esm.embed_dim),
            Linear(self.esm.embed_dim, self.c_s),
            nn.ReLU(),
            Linear(self.c_s, self.c_s)
        )
        if self.use_esm_attn_map:
            self.esm_z_mlp = nn.Sequential(
                LayerNorm(self.esm_attns),
                Linear(self.esm_attns, self.c_z),
                nn.ReLU(),
                Linear(self.c_z, self.c_z),
            )
        # 0 is padding, N is 'x' unknown residues, N + 1 is mask. 23
        self.vocab_size = rc.num_residue_types + 3
        self.embedding = nn.Embedding(self.vocab_size, self.c_s, padding_idx=0)
        self.trunk = FoldingTrunk(
            c_s=self.c_s,
            c_z=self.c_z,
            num_evoformer_blocks=self.tcfg.num_blocks,
            num_heads_seq=self.c_s // self.tcfg.sequence_head_width,
            num_heads_pair=self.c_z // self.tcfg.pairwise_head_width,
            num_relative_positions=self.tcfg.position_bins,
            max_recycles=self.tcfg.max_recycles,
            structure_module=StructureModuleConfig(
                c_s=self.scfg.c_s,
                c_z=self.scfg.c_z,
                c_ipa_hidden=self.scfg.c_ipa,
                c_angle_hidden=self.scfg.c_resnet,
                num_heads_ipa=self.scfg.no_heads_ipa,
                num_qk_points=self.scfg.no_qk_points,
                num_v_points=self.scfg.no_v_points,
                dropout_rate=self.scfg.dropout_rate,
                num_blocks=self.scfg.no_blocks,
                num_transition_blocks=self.scfg.no_transition_layers,
                num_resnet_blocks=self.scfg.no_resnet_blocks,
                inf=self.scfg.inf
            ),
            dropout=self.tcfg.dropout
        )
        # auxliary head
        self.lddt_bins = lddt_bins
        self.distogram_bins = distogram_bins
        self.distogram_head = Linear(self.c_z, self.distogram_bins)
        # TM-score prediction
        self.ptm_head = Linear(self.c_z, self.distogram_bins)
        self.lm_head = Linear(self.c_s, self.vocab_size)
        # LDDT prediction
        self.lddt_hidden_dim = lddt_hidden_dim
        self.lddt_head = nn.Sequential(
            LayerNorm(self.scfg.c_s),
            Linear(self.scfg.c_s, self.lddt_hidden_dim),
            Linear(self.lddt_hidden_dim, self.lddt_hidden_dim),
            Linear(self.lddt_hidden_dim, 37 * self.lddt_bins)
        )

    def enable_activation_checkpoint(self, enabled=True):
        self.trunk.enable_activation_checkpoint(enabled)

    def _input_embedding_imp(self, aatype, mask, masking_pattern):
        # [bs, seq_len, num_layers + 1, dim]
        esm_s, esm_z = self.esm(aatype, mask=mask, masking_pattern=masking_pattern)
        esm_s = esm_s.detach().to(self.esm_s_combine.dtype)
        # to sequence feature
        esm_s = (self.esm_s_combine.softmax(0).unsqueeze(0) @ esm_s).squeeze(2)
        s_s_0 = self.esm_s_mlp(esm_s)
        # keep seq embedding information with esm seq feature
        s_s_0 += self.embedding(aatype)  # [bs, seq_len, c_s]
        if self.use_esm_attn_map:
            esm_z = esm_z.to(self.esm_s_combine.dtype)
            esm_z = esm_z.detach()
            s_z_0 = self.esm_z_mlp(esm_z)
        else:
            B, L = aatype.shape[:2]
            s_z_0 = s_s_0.new_zeros(B, L, L, self.c_z)
        return s_s_0, s_z_0

    def _auxliary_head_impl(self, structure):
        sz = structure.pop("s_z")
        ss = structure.pop("s_s")
        states = structure.pop("states")
        structure["lm_logits"] = self.lm_head(ss)  # [bs, seq_len, vocab_size(23)]
        disto_logits = self.distogram_head(sz)  # [bs, seq_len, seq_len, 64]
        disto_logits = (disto_logits + disto_logits.transpose(-2, -3)) / 2
        structure["distogram_logits"] = disto_logits
        lddt_logits = self.lddt_head(states)  # [num_layers, bs, seq_len, 37*50]
        # [num_layers, bs, seq_len, 37, num_lddt_bins]
        structure["lddt_logits"] = lddt_logits.reshape(*lddt_logits.shape[:-1], -1, self.lddt_bins)
        structure["tm_logits"] = self.ptm_head(sz)  # [bs, seq_len, seq_len, 64]

        return structure

    def forward(
            self,
            aatype: torch.Tensor,
            mask: T.Optional[torch.Tensor] = None,
            residx: T.Optional[torch.Tensor] = None,
            masking_pattern: T.Optional[torch.Tensor] = None,
            num_recycles: T.Optional[int] = None,
            need_head=True,
            chunk_size: T.Optional[int] = None
    ):
        """
        Args:
            aatype: [bs, seq_len], containing indices corresponding to amino acids, restype_order_with_x
            mask: [bs, seq_len], residue mask, binary tensor with 1 meaning position is unmasked
                and 0 meaning position is masked. Actually it is padding mask.
            residx: [bs, seq_len], Residue indices of amino acids. Will assume contiguous if not provided.
            masking_pattern: Optional masking to pass to the input. Positions with 1 will be masked.
                ESMFold sometimes produces different samples when different masks are provided.
            num_recycles: How many recycle iterations to perform. If None, defaults to training max
                recycles, which is 3.
        """
        device = aatype.device
        if residx is None:
            residx = torch.arange(aatype.shape[1], device=device).expand_as(aatype)
        s, z = self._input_embedding_imp(aatype, mask=mask, masking_pattern=masking_pattern)
        structure: dict = self.trunk(s, z, aatype, residx, mask, num_recycles=num_recycles, chunk_size=chunk_size)
        structure = {
            k: v
            for k, v in structure.items()
            if k
               in [
                   "s_z",
                   "s_s",
                   "frames",
                   "sidechain_frames",
                   "unnormalized_angles",
                   "angles",
                   "positions",
                   "states"
               ]
        }
        structure["final_atom_positions"] = data_transform.atom14_to_atom37_positions(aatype,
                                                                                      structure["positions"][-1])
        if not need_head:
            return {
                "final_atom_positions": structure["final_atom_positions"]
            }

        structure = self._auxliary_head_impl(structure)
        return structure

    @torch.no_grad()
    def infer(
            self,
            sequences: T.Union[str, T.List[str]],
            residx=None,
            masking_pattern: T.Optional[torch.Tensor] = None,
            num_recycles: T.Optional[int] = None,
            residue_index_offset: T.Optional[int] = 512,
            chain_linker: T.Optional[str] = "G" * 25,
            need_head: bool = False,
            chunk_size: T.Optional[int] = None
    ):
        """
        Args:
            sequences: A list of sequences to make predictions for. Multimers can also be passed in,
                each chain should be separated by a ':' token (e.g. "<chain1>:<chain2>:<chain3>").
            residx: Residue indices of amino acids. Will assume contiguous if not provided.
            masking_pattern: Optional masking to pass to the input. Binary tensor of the same size
                as `aa`. Positions with 1 will be masked. ESMFold sometimes produces different samples when
                different masks are provided.
            num_recycles: How many recycle iterations to perform. If None, defaults to training max
                recycles (cfg.trunk.max_recycles), which is 4.
            residue_index_offset: Residue index separation between chains if predicting a multimer. Has no effect on
                single chain predictions. Default: 512.
            chain_linker: Linker to use between chains if predicting a multimer. Has no effect on single chain
                predictions. Default: length-25 poly-G ("G" * 25).
        """
        if isinstance(sequences, str):
            sequences = [sequences, ]

        aatype, mask, _residx, linker_mask, chain_index = batch_encode_sequences(
            sequences, residue_index_offset, chain_linker
        )

        if residx is None:
            residx = _residx

        elif not isinstance(residx, torch.Tensor):
            residx = collate_dense_tensors(residx)

        aatype, mask, residx, linker_mask = map(
            lambda x: x.to(self.device), (aatype, mask, residx, linker_mask)
        )
        outputs = self.forward(
            aatype,
            mask=mask,
            residx=residx,
            masking_pattern=masking_pattern,
            num_recycles=num_recycles,
            need_head=need_head,
            chunk_size=chunk_size
        )
        atom37_exists = data_transform.make_atom37_mask(aatype) * mask[..., None]
        outputs["atom37_mask"] = atom37_exists * linker_mask.unsqueeze(2)
        # for write pdb
        outputs["chain_index"] = chain_index
        outputs["residue_index"] = residx
        outputs["aatype"] = aatype
        if need_head:
            bs, seq_len = aatype.shape
            # we predict plDDT between 0 and 1, scale to be between 0 and 100.
            outputs["plddt"] = LDDTHead.compute_plddt(outputs["lddt_logits"][-1]).view(bs, seq_len, -1)
            outputs["mean_plddt"] = masked_mean(atom37_exists, outputs["plddt"], dim=(1, 2))
            seq_lens = mask.type(torch.int64).sum(1)
            outputs["ptm"] = torch.stack(
                [
                    TMScoreHead.compute_tm_score(
                        batch_ptm_logits[None, :sl, :sl],
                        max_bin=31,
                        num_bins=self.distogram_bins,
                    )
                    for batch_ptm_logits, sl in zip(outputs["tm_logits"], seq_lens)
                ]
            )
        return outputs

    def output_to_pdb(self, output: T.Dict) -> T.List[str]:
        """Returns the pbd (file) string from the model given the model output.

        Returns:
            list of pad string
        """
        return output_to_pdb(output["aatype"],
                             output["final_atom_positions"],
                             output["atom37_mask"],
                             plddt=output.get("plddt", None),
                             residue_index=output.get("residue_index", None),
                             chain_index=output.get("chain_index", None))

    def infer_pdbs(self, seqs: T.List[str], *args, **kwargs) -> T.List[str]:
        """Returns list of pdb (files) strings from the model given a list of input sequences."""
        output = self.infer(seqs, *args, **kwargs)
        return self.output_to_pdb(output)

    def infer_pdb(self, sequence: str, *args, **kwargs) -> str:
        """Returns the pdb (file) string from the model given an input sequence."""
        return self.infer_pdbs([sequence], *args, **kwargs)[0]

    def set_chunk_size(self, chunk_size: T.Optional[int]):
        # This parameter means the axial attention will be computed
        # in a chunked manner. This should make the memory used more or less O(L) instead of O(L^2).
        # It's equivalent to running a for loop over chunks of the dimension we're iterative over,
        # where the chunk_size is the size of the chunks, so 128 would mean to parse 128-lengthed chunks.
        # Setting the value to None will return to default behavior, disable chunking.
        self.trunk.set_chunk_size(chunk_size)

    @property
    def device(self):
        return self.esm_s_combine.device
