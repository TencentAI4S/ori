# Copyright (c) 2025, Tencent Inc. All rights reserved.
# Data: 2025/1/9 15:58
# Author: chenchenqin
from typing import Optional

import deepspeed
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from tgnn.model.layer import Linear, apply_rotary_emb, RMSNorm, precompute_freqs_cis
from tgnn.model.module.attention import InvariantPointAttention
from tgnn.model.module.head import AngleHead, FrameHead
from tgnn.model.module.mlp import SwiGLU
from tgnn.model.module.structure_module.protein_mapper import ProteinMapper
from tgnn.transform.affine import Rigid
from tgnn.utils.tensor import dict_multimap, permute_final_dims


class SequenceAttention(nn.Module):

    def __init__(self,
                 c_s: int,
                 c_z: int,
                 num_heads: int = 8,
                 rotary_embedding: bool = True,
                 bias: bool = False):
        super(SequenceAttention, self).__init__()
        assert c_s % num_heads == 0, f"number of heads must devide dim"
        self.c_s = c_s
        self.c_z = c_z
        self.dim = self.c_z
        self.num_heads = num_heads
        self.head_dim = self.dim // num_heads
        assert self.head_dim % 2 == 0
        self.bias = bias
        self.linear_qk = Linear(self.dim, 2 * self.dim, bias=self.bias)
        self.rotary_embedding = rotary_embedding
        self.freqs_cis = None
        self.linear_s = Linear(self.c_s, self.dim, bias=bias)
        self.norm_s = RMSNorm(self.dim)
        self.linear_o = Linear(self.num_heads, self.dim, bias=bias)
        self.norm_z = RMSNorm(self.dim)

    def update_freqs_cis(self, seq_len, dtype=None, device=None):
        if self.freqs_cis is None or seq_len > self.freqs_cis.shape[0]:
            self.freqs_cis = precompute_freqs_cis(seq_len,
                                                  rotary_dim=self.head_dim,
                                                  dtype=dtype,
                                                  device=device)

    def _project_qk(self,
                    x: torch.Tensor,
                    freqs_cis: torch.Tensor = None):
        """
        Args:
            x: tensor[*, seq_len, dim]

        Returns:
            q, k: tensor[*, seq_len, num_heads, head_dim]
        """
        q, k = self.linear_qk(x).split([self.dim, self.dim], dim=-1)
        q = q.view(q.shape[:-1] + (-1, self.head_dim))  # [*, seq_len, num_heads, head_dim]
        k = k.view(k.shape[:-1] + (-1, self.head_dim))  # [*, seq_len, num_heads, head_dim]
        if freqs_cis is not None:
            q = apply_rotary_emb(q, freqs_cis)  # [*, seq_len, num_heads, head_dim]
            k = apply_rotary_emb(k, freqs_cis)  # [*, seq_len, num_heads, head_dim]

        return q, k

    def forward(self, s, mask=None):
        """
        Args:
            s: [*, seq_len, c_s], sequence feature
            mask: [*, seq_len], sequence mask

        Returns:
            z: [*, seq_len, seq_len, c_s], pair feature
        """
        s = self.linear_s(s)
        s = self.norm_s(s)

        if self.rotary_embedding:
            seq_len = s.shape[-2]
            self.update_freqs_cis(seq_len, dtype=s.dtype, device=s.device)

        q, k = self._project_qk(s, freqs_cis=self.freqs_cis)  # [*, seq_len, num_heads, head_dim]

        # [*, num_heads, seq_len, seq_len]
        z = permute_final_dims(q, (1, 0, 2)) @ permute_final_dims(k, (1, 2, 0)) / math.sqrt(self.head_dim)
        if mask is not None:
            mask = mask[..., None] * mask[..., None, :]
            mask = mask.unsqueeze(-3).bool()
            z = z.masked_fill(~mask, torch.finfo(z.dtype).min)
        z = F.softmax(z.float(), dim=-1).type_as(z)  # [*, num_heads, seq_len, seq_len]
        z = permute_final_dims(z, (1, 2, 0))
        z = self.linear_o(z)
        z = self.norm_z(z)

        return z


class SequenceToPair(nn.Module):
    """like Outer-produce mean in alphafold"""

    def __init__(self, c_s, c_z, c_h=None):
        super().__init__()
        self.c_h = c_h or c_z // 2
        self.c_z = c_z
        self.c_s = c_s
        self.proj = Linear(self.c_s, self.c_h * 2)
        self.o_proj = Linear(self.c_h * 2, self.c_z, init="final")
        self.norm = RMSNorm(self.c_z)

    def forward(self, s, mask=None):
        """
        Args:
            s: [bs, seq_len, c_s], sequence feature

        Returns:
            z: [bs, seq_len, seq_len, c_s], pair feature
        """
        assert len(s.shape) == 3, f"expect 3D tensor, got {s.shape}"
        s = self.proj(s)
        q, k = s.chunk(2, dim=-1)
        prod = q[:, None, :, :] * k[:, :, None, :]
        diff = q[:, None, :, :] - k[:, :, None, :]
        x = torch.cat([prod, diff], dim=-1)
        if mask is not None:
            mask = mask.unsqueeze(-1) * mask.unsqueeze(-2)
            mask = mask.unsqueeze(-3).bool()
            x = x.masked_fill(~mask, 0)

        x = self.o_proj(x)
        x = self.norm(x)

        return x


class StructureTransformer(nn.Module):
    def __init__(self,
                 c_s,
                 c_z=None,
                 num_heads=1,
                 num_qk_points: int = 4,
                 num_v_points: int = 8,
                 rotary_embedding=True,
                 pack_qkv=True,
                 bias=False,
                 dof=3):
        super(StructureTransformer, self).__init__()
        self.c_s = c_s
        self.c_z = c_z
        if self.c_z is not None:
            self.norm_z = RMSNorm(self.c_z)

        self.num_heads = num_heads
        self.num_qk_points = num_qk_points
        self.num_v_points = num_v_points
        self.ipa = InvariantPointAttention(
            self.c_s,
            c_z=self.c_z,
            num_heads=self.num_heads,
            num_qk_points=self.num_qk_points,
            num_v_points=self.num_v_points,
            rotary_embedding=rotary_embedding,
            pack_qkv=pack_qkv,
            bias=bias
        )
        self.mlp = SwiGLU(c_s, bias=bias, _pack_weights=True, xformer=False)
        self.norm_1 = RMSNorm(c_s)
        self.norm_2 = RMSNorm(c_s)
        self.norm_3 = RMSNorm(c_s)
        self.dof = dof
        self.bb_update = FrameHead(self.c_s, dof=self.dof)
        self.activation_checkpoint = False
        self.activation_checkpoint_func = deepspeed.checkpointing.checkpoint

    def enable_activation_checkpoint(self, enabled=True):
        self.activation_checkpoint = enabled

    def forward(self,
                s: torch.Tensor,
                r: Rigid = None,
                z: torch.Tensor = None,
                mask: torch.Tensor = None):
        if z is not None:
            z = self.norm_z(z)

        s0 = self.norm_1(s)
        if self.activation_checkpoint:
            hidden = self.activation_checkpoint_func(self.ipa, s0, z, r, mask)
        else:
            hidden = self.ipa(s0, z=z, r=r, mask=mask)

        s = s + hidden
        s = s + self.mlp(self.norm_2(s))
        r = r @ self.bb_update(self.norm_3(s))  # [*, seq_len]

        return s, r


class StructureModule(ProteinMapper):
    """

    Args:
        c_s: Single representation channel dimension
        c_angle_hidden: Angle resnet (Alg. 23 lines 11-14) hidden channel dimension
        num_heads_ipa: Number of IPA heads
        num_qk_points: Number of query/key points to generate during IPA
        num_v_points: Number of value points to generate during IPA
        num_blocks: Number of structure module blocks
        num_transition_blocks: Number of layers in the single representation transition
        num_resnet_blocks: Number of blocks in the angle resnet
        num_angles: Number of angles to generate in the angle resnet
        trans_scale_factor: Scale of single representation transition hidden dimension
    """

    def __init__(
            self,
            c_s: int = 384,
            c_z: int = None,
            c_angle_hidden: int = 128,
            num_heads_ipa: int = 12,
            num_qk_points: int = 4,
            num_v_points: int = 8,
            num_blocks: int = 8,
            num_resnet_blocks: int = 2,
            num_angles: int = 7,
            trans_scale_factor: int = 10,
            rotary_embedding=False,
            num_heads_z: int = 8,
            attn_map_to_pair: bool = False,
            bias: bool = False,
            dof: int = 3,
            eps: float = 1e-8,
    ):
        super(StructureModule, self).__init__()
        self.c_s = c_s
        self.c_z = c_z
        self.dof = dof
        self.eps = eps
        self.num_blocks = num_blocks
        self.trans_scale_factor = trans_scale_factor
        if self.c_z is not None:
            if attn_map_to_pair:
                self.seq_to_pair = SequenceAttention(self.c_s,
                                                     self.c_z,
                                                     num_heads=num_heads_z,
                                                     rotary_embedding=rotary_embedding,
                                                     bias=bias)
            else:
                self.seq_to_pair = SequenceToPair(self.c_s, self.c_z)
        else:
            self.seq_to_pair = None

        self.norm_s = RMSNorm(self.c_s)
        self.linear_in = Linear(self.c_s, self.c_s)
        self.layer = StructureTransformer(
            self.c_s,
            num_heads=num_heads_ipa,
            num_qk_points=num_qk_points,
            num_v_points=num_v_points,
            rotary_embedding=rotary_embedding,
            bias=bias,
            dof=dof
        )
        self.c_angle_hidden = c_angle_hidden
        self.num_angles = num_angles
        self.num_resnet_blocks = num_resnet_blocks
        self.angle_resnet = AngleHead(
            self.c_s,
            self.c_angle_hidden,
            self.num_resnet_blocks,
            self.num_angles,
            eps=self.eps
        )
        self.activation_checkpoint = False
        self.activation_checkpoint_func = deepspeed.checkpointing.checkpoint

    def enable_activation_checkpoint(self, enabled=True):
        self.activation_checkpoint = enabled

    def forward(self,
                s: torch.Tensor,
                aatype: Optional[torch.Tensor],
                mask: Optional[torch.Tensor] = None,
                rigids: Optional[Rigid] = None):
        """
        Args:
            s: [bs, seq_len, c_s] single representation
            aatype: [bs, seq_len] amino acid indices
            mask: Optional [bs, seq_len] sequence mask
            rigids: [bs, seq_len], initial backbone frame
        """
        s = self.norm_s(s)
        s_initial = s
        z = None
        if self.seq_to_pair is not None:
            if self.activation_checkpoint:
                z = self.activation_checkpoint_func(self.seq_to_pair, s, mask)
            else:
                z = self.seq_to_pair(s, mask=mask)

        s = self.linear_in(s)
        if rigids is None:
            rigids = Rigid.identity(
                s.shape[:-1],
                dtype=torch.float32,
                device=s.device,
                requires_grad=self.training,
                fmt="quat" if self.dof in (3, 4) else "rot"
            )

        outputs = []
        for i in range(self.num_blocks):
            s, rigids = self.layer(s, r=rigids, mask=mask)  # [*, seq_len, c_s]
            backb_to_global = rigids.scale_translation(self.trans_scale_factor)
            unnormalized_angles = self.angle_resnet(s, s_initial)
            angles = F.normalize(unnormalized_angles, dim=-1, eps=self.eps)
            if i == self.num_blocks - 1:
                sidechain_frames = self.torsion_angles_to_frames(
                    backb_to_global,
                    angles,
                    aatype
                )
                pred_xyz = self.frames_and_literature_positions_to_atom14_pos(
                    sidechain_frames,
                    aatype
                )

            preds = {
                "frames": backb_to_global.to_tensor(self.dof),
                "unnormalized_angles": unnormalized_angles,
                "angles": angles
            }
            outputs.append(preds)
            rigids = rigids.stop_rot_gradient()

        outputs = dict_multimap(torch.stack, outputs)
        outputs["frames"] = Rigid.from_tensor(outputs["frames"])
        outputs["final_atom14_positions"] = pred_xyz
        outputs["final_sidechain_frames"] = sidechain_frames
        outputs["single"] = s
        outputs["pair"] = z

        return outputs
