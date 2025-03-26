# Copyright (c) 2025, Tencent Inc. All rights reserved.
# Data: 2025/1/9 15:58
# Author: chenchenqin
import io
import string
from typing import Optional

import numpy as np
from Bio.PDB import PDBParser

from .. import residue_constants as rc

def parse_pdb_string(pdb_str: str, chain_id: Optional[str] = None):
    """parse a PDB string

    WARNING: All non-standard residue types will be converted into UNK. All
      non-standard atoms will be ignored.

    Args:
      pdb_str: The contents of the pdb file
      chain_id: If None, then the whole pdb file is parsed. If chain_id is specified (e.g. A), then only that chain
        is parsed.

    Returns:
      A new `Protein` parsed from the pdb contents.
    """
    pdb_fh = io.StringIO(pdb_str)
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("none", pdb_fh)
    models = list(structure.get_models())
    if len(models) != 1:
        raise ValueError(
            f"Only single model PDBs are supported. Found {len(models)} models."
        )
    model = models[0]
    atom_positions = []
    aaseq = []
    atom_mask = []
    residue_index = []
    chain_ids = []
    b_factors = []

    for chain in model:
        if (chain_id is not None and chain.id != chain_id):
            continue
        for res in chain:
            if res.id[2] != " ":
                raise ValueError(
                    f"PDB contains an insertion code at chain {chain.id} and residue "
                    f"index {res.id[1]}. These are not supported."
                )
            res_name = rc.restype_3to1.get(res.resname, "X")
            pos = np.zeros((rc.num_atom_types, 3))
            mask = np.zeros((rc.num_atom_types,))
            res_b_factors = np.zeros((rc.num_atom_types,))
            for atom in res:
                if atom.name not in rc.atom_types:
                    continue
                pos[rc.atom_order[atom.name]] = atom.coord
                mask[rc.atom_order[atom.name]] = 1.0
                res_b_factors[
                    rc.atom_order[atom.name]
                ] = atom.bfactor
            if np.sum(mask) < 0.5:
                # If no known atom positions are reported for the residue then skip it.
                continue
            aaseq.append(res_name)
            atom_positions.append(pos)
            atom_mask.append(mask)
            residue_index.append(res.id[1])
            chain_ids.append(chain.id)
            b_factors.append(res_b_factors)

    parents = None
    parents_chain_index = None
    if ("PARENT" in pdb_str):
        parents = []
        parents_chain_index = []
        chain_id = 0
        for l in pdb_str.split("\n"):
            if ("PARENT" in l):
                if (not "N/A" in l):
                    parent_names = l.split()[1:]
                    parents.extend(parent_names)
                    parents_chain_index.extend([
                        chain_id for _ in parent_names
                    ])
                chain_id += 1

    chain_id_mapping = {cid: n for n, cid in enumerate(string.ascii_uppercase)}
    chain_index = np.array([chain_id_mapping[cid] for cid in chain_ids])

    return {
        "aaseq": "".join(aaseq),
        "atom_positions": np.array(atom_positions),
        "atom_mask": np.array(atom_mask),
        "chain_index": chain_index,
        "residue_index": np.array(residue_index),
        "b_factors": np.array(b_factors),
        "parents": parents,
        "parents_chain_index": parents_chain_index,
    }
