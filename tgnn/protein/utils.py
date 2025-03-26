# Copyright (c) 2025, Tencent Inc. All rights reserved.
# Data: 2025/1/9 15:58
# Author: chenchenqin
import typing as T
import numpy as np
import torch

from . import residue_constants as rc
from .protein import Protein as OFProtein, to_pdb


def is_valid_aaseq(seq: str):
    assert isinstance(seq, str)
    aatypes = set(rc.restypes)
    return set(list(seq)).issubset(aatypes)


def output_to_pdb(aatype,
                  atom37_positions,
                  atom37_mask,
                  plddt=None,
                  residue_index=None,
                  chain_index=None) -> T.List[str]:
    """Returns the pbd (file) string from the model given the model output."""
    if isinstance(aatype, torch.Tensor):
        aatype = aatype.cpu().numpy()

    if isinstance(atom37_positions, torch.Tensor):
        atom37_positions = atom37_positions.cpu().float().numpy()

    if isinstance(atom37_mask, torch.Tensor):
        atom37_mask = atom37_mask.cpu().numpy()

    if residue_index is not None:
        residue_index = residue_index.cpu().numpy()

    if plddt is not None:
        if isinstance(plddt, torch.Tensor):
            # bfloat16 cannot convert numpy
            plddt = plddt.cpu().float().numpy()

        if len(plddt.shape) == 2:
            plddt = plddt[..., None]

        if plddt.shape[-1] == 1:
            plddt = np.repeat(plddt, 37, axis=-1)

    if chain_index is not None and isinstance(chain_index, torch.Tensor):
        chain_index = chain_index.cpu().numpy()

    bs = aatype.shape[0]
    pdbs = []
    for i in range(bs):
        aa = aatype[i]
        pred_pos = atom37_positions[i]
        mask = atom37_mask[i]
        if residue_index is not None:
            resid = residue_index[i] + 1
        else:
            resid = None

        pred = OFProtein(
            aatype=aa,
            atom_positions=pred_pos,
            atom_mask=mask,
            residue_index=resid,
            b_factors=plddt[i] if plddt is not None else None,
            chain_index=chain_index[i] if chain_index is not None else None
        )
        pdbs.append(to_pdb(pred))

    return pdbs
