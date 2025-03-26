# -*- coding: utf-8 -*-
# Copyright (c) 2023, Tencent Inc. All rights reserved.
# Author: chenchenqin
# Data: 2023/11/2 15:07
import torch

from tgnn.utils.tensor import batched_gather
from .. import residue_constants as rc


def make_aatype(aaseq, device=None, table=rc.restype_order_with_x):
    if isinstance(aaseq, (list, tuple)):
        return torch.stack([make_aatype(seq, device, table=table) for seq in aaseq], dim=0)

    seq_len = len(aaseq)
    aatype = torch.zeros((seq_len,), dtype=torch.long, device=device)
    for i, aa in enumerate(aaseq):
        aatype[i] = table[aa]

    return aatype


def make_atom37_mask(aatype, dtype=None):
    """make atom37 exist mask"""
    device = aatype.device
    dtype = torch.float32 if dtype is None else dtype
    restype_atom37_mask = torch.zeros(
        [21, 37], dtype=dtype, device=device
    )
    for restype, restype_letter in enumerate(rc.restypes):
        restype_name = rc.restype_1to3[restype_letter]
        atom_names = rc.residue_atoms[restype_name]
        for atom_name in atom_names:
            atom_idx = rc.atom_order[atom_name]
            restype_atom37_mask[restype, atom_idx] = 1

    return restype_atom37_mask[aatype]


def atom14_to_atom37_positions(aatype, atom14_positions, atom14_mask=None):
    assert atom14_positions.shape[-2:] == (
        14, 3), f"expect atom14 positions shape but get shape: {atom14_positions.shape}"
    if atom14_mask is not None:
        assert atom14_positions.shape[:2] == atom14_mask.shape

    residx_atom37_to_atom14 = []
    for rt in rc.restypes:
        atom_names = rc.restype_name_to_atom14_names[rc.restype_1to3[rt]]
        atom_name_to_idx14 = {name: i for i, name in enumerate(atom_names)}
        residx_atom37_to_atom14.append(
            [
                (atom_name_to_idx14[name] if name in atom_name_to_idx14 else 0)
                for name in rc.atom_types
            ]
        )
    residx_atom37_to_atom14.append([0] * 37)
    residx_atom37_to_atom14 = torch.tensor(
        residx_atom37_to_atom14,
        dtype=torch.int32,
        device=aatype.device,
    )[aatype]

    atom37_atom_exists = make_atom37_mask(aatype)
    atom37_positions = atom37_atom_exists[..., None] * batched_gather(
        atom14_positions,
        residx_atom37_to_atom14,
        dim=-2,
        no_batch_dims=len(atom14_positions.shape[:-2]),
    )

    # validness masks for specified residue(s) & atom(s)
    if atom14_mask is not None:
        atom37_mask = atom37_atom_exists * batched_gather(
            atom14_mask,
            residx_atom37_to_atom14,
            dim=-1,
            no_batch_dims=len(atom14_mask.shape[:-1]))
        return atom37_positions, atom37_mask

    return atom37_positions


def aatype_to_seq(aatype, unkown=None):
    assert len(aatype.shape) in (1, 2)
    if len(aatype.shape) == 2:
        return [aatype_to_seq(at, unkown=unkown) for at in aatype]

    aaseq = []
    for sid in aatype:
        if sid < 0 or sid > 20:
            sid = 20  # X, unkown residue

        if sid == 20 and unkown is not None:
            sid = unkown
        else:
            sid = rc.restypes_with_x[sid]

        aaseq.append(sid)
    return "".join(aaseq)
