# Copyright (c) 2025, Tencent Inc. All rights reserved.
# Data: 2025/1/9 15:58
# Author: chenchenqin

# Copyright 2021 AlQuraishi Laboratory
# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import dataclasses
import io
import string
from typing import Any, Sequence, Mapping, Optional

import modelcif
import modelcif.alignment
import modelcif.dumper
import modelcif.model
import modelcif.protocol
import modelcif.qa_metric
import modelcif.reference
import numpy as np

from . import residue_constants
from .parser.pdbx_parsing import parse_pdb_string

FeatureDict = Mapping[str, np.ndarray]
ModelOutput = Mapping[str, Any]  # Is a nested dict.
PICO_TO_ANGSTROM = 0.01


@dataclasses.dataclass(frozen=True)
class Protein:
    """Protein structure representation."""

    # Cartesian coordinates of atoms in angstroms. The atom types correspond to
    # residue_constants.atom_types, i.e. the first three are N, CA, CB.
    atom_positions: np.ndarray  # [seq_len, num_atoms, 3]

    # Amino-acid type for each residue represented as an integer between 0 and
    # 20, where 20 is 'X'.
    aatype: np.ndarray  # [seq_len]

    # Binary float mask to indicate presence of a particular atom. 1.0 if an atom
    # is present and 0.0 if not. This should be used for loss masking.
    atom_mask: np.ndarray  # [seq_len, num_atoms]

    # Residue index as used in PDB. It is not necessarily continuous or 0-indexed.
    residue_index: Optional[np.ndarray] = None  # [seq_len]

    # B-factors, or temperature factors of each residue (in sq. angstroms units),
    # representing the displacement of the residue from its ground truth mean
    # value.
    b_factors: Optional[np.ndarray] = None  # [seq_len, num_atoms]

    # Chain indices for multi-chain predictions
    chain_index: Optional[np.ndarray] = None  # [seq_len, ]

    # Optional remark or description about the protein. Included as a comment in output PDB
    # files
    remark: Optional[str] = None

    # TODO: read and write release date
    release_date: Optional[str] = None

    # Templates used to generate this protein (prediction-only)
    parents: Optional[Sequence[str]] = None

    # Chain corresponding to each parent
    parents_chain_index: Optional[Sequence[int]] = None


def from_pdb_string(pdb_str: str, chain_id: Optional[str] = None) -> Protein:
    """Takes a PDB string and constructs a Protein object.

    WARNING: All non-standard residue types will be converted into UNK. All
      non-standard atoms will be ignored.

    Args:
      pdb_str: The contents of the pdb file
      chain_id: If None, then the whole pdb file is parsed. If chain_id is specified (e.g. A), then only that chain
        is parsed.

    Returns:
      A new `Protein` parsed from the pdb contents.
    """
    data = parse_pdb_string(pdb_str, chain_id)
    aaseq = data.pop("aaseq")
    aatype = [residue_constants.restype_order_with_x[restype] for restype in aaseq]
    return Protein(
        aatype=np.array(aatype),
        **data
    )


def from_pdb(filename):
    with open(filename, "r") as fp:
        pdb_string = fp.read()
        return from_pdb_string(pdb_string, None)


def get_pdb_headers(prot: Protein, chain_id: int = 0) -> Sequence[str]:
    pdb_headers = []

    remark = prot.remark
    if (remark is not None):
        pdb_headers.append(f"REMARK {remark}")

    parents = prot.parents
    parents_chain_index = prot.parents_chain_index
    if (parents_chain_index is not None):
        parents = [
            p for i, p in zip(parents_chain_index, parents) if i == chain_id
        ]

    if (parents is None or len(parents) == 0):
        parents = ["N/A"]

    pdb_headers.append(f"PARENT {' '.join(parents)}")

    return pdb_headers


def add_pdb_headers(prot: Protein, pdb_str: str) -> str:
    """ Add pdb headers to an existing PDB string. Useful during multi-chain
        recycling
    """
    out_pdb_lines = []
    lines = pdb_str.split('\n')

    remark = prot.remark
    if (remark is not None):
        out_pdb_lines.append(f"REMARK {remark}")

    if (prot.parents is not None and len(prot.parents) > 0):
        parents_per_chain = []
        if (prot.parents_chain_index is not None):
            parent_dict = {}
            for p, i in zip(prot.parents, prot.parents_chain_index):
                parent_dict.setdefault(str(i), [])
                parent_dict[str(i)].append(p)

            max_idx = max([int(chain_idx) for chain_idx in parent_dict])
            for i in range(max_idx + 1):
                chain_parents = parent_dict.get(str(i), ["N/A"])
                parents_per_chain.append(chain_parents)
        else:
            parents_per_chain.append(prot.parents)
    else:
        parents_per_chain = [["N/A"]]

    make_parent_line = lambda p: f"PARENT {' '.join(p)}"

    out_pdb_lines.append(make_parent_line(parents_per_chain[0]))

    chain_counter = 0
    for i, l in enumerate(lines):
        if ("PARENT" not in l and "REMARK" not in l):
            out_pdb_lines.append(l)
        if ("TER" in l and not "END" in lines[i + 1]):
            chain_counter += 1
            if (not chain_counter >= len(parents_per_chain)):
                chain_parents = parents_per_chain[chain_counter]
            else:
                chain_parents = ["N/A"]

            out_pdb_lines.append(make_parent_line(chain_parents))

    return '\n'.join(out_pdb_lines)


def to_pdb(prot: Protein) -> str:
    """Converts a `Protein` instance to a PDB string.

    Args:
      prot: The protein to convert to PDB.

    Returns:
      PDB string.
    """
    restypes = residue_constants.restypes + ["X"]
    res_1to3 = lambda r: residue_constants.restype_1to3.get(restypes[r], "UNK")
    atom_types = residue_constants.atom_types

    pdb_lines = []

    atom_mask = prot.atom_mask
    aatype = prot.aatype
    seq_len = aatype.shape[0]
    atom_positions = prot.atom_positions
    residue_index = prot.residue_index
    if residue_index is None:
        residue_index = np.arange(seq_len)
    residue_index = residue_index.astype(np.int32)
    b_factors = prot.b_factors
    if b_factors is None:
        b_factors = np.ones((seq_len, 37), dtype=np.float32)

    chain_index = prot.chain_index
    if np.any(aatype > residue_constants.num_residue_types):
        raise ValueError(f"Invalid aatypes max({np.max(aatype)} > {residue_constants.num_residue_types}).")

    headers = get_pdb_headers(prot)
    if len(headers) > 0:
        pdb_lines.extend(headers)

    atom_index = 1
    prev_chain_index = 0
    chain_tags = string.ascii_uppercase
    # Add all atom sites.
    for i in range(seq_len):
        res_name_3 = res_1to3(aatype[i])
        for atom_name, pos, mask, b_factor in zip(atom_types, atom_positions[i], atom_mask[i], b_factors[i]):
            # drop masked atom
            if mask < 0.5:
                continue

            record_type = "ATOM"
            name = atom_name if len(atom_name) == 4 else f" {atom_name}"
            alt_loc = ""
            insertion_code = ""
            occupancy = 1.00
            element = atom_name[
                0
            ]  # Protein supports only C, N, O, S, this works.
            charge = ""

            chain_tag = "A"
            if (chain_index is not None):
                chain_tag = chain_tags[chain_index[i]]

            # PDB is a columnar format, every space matters here!
            atom_line = (
                f"{record_type:<6}{atom_index:>5} {name:<4}{alt_loc:>1}"
                f"{res_name_3:>3} {chain_tag:>1}"
                f"{residue_index[i]:>4}{insertion_code:>1}   "
                f"{pos[0]:>8.3f}{pos[1]:>8.3f}{pos[2]:>8.3f}"
                f"{occupancy:>6.2f}{b_factor:>6.2f}          "
                f"{element:>2}{charge:>2}"
            )
            pdb_lines.append(atom_line)
            atom_index += 1

        should_terminate = (i == seq_len - 1)
        if (chain_index is not None):
            if (i != seq_len - 1 and chain_index[i + 1] != prev_chain_index):
                should_terminate = True
                prev_chain_index = chain_index[i + 1]

        if (should_terminate):
            # Close the chain.
            chain_end = "TER"
            chain_termination_line = (
                f"{chain_end:<6}{atom_index:>5}      "
                f"{res_1to3(aatype[i]):>3} "
                f"{chain_tag:>1}{residue_index[i]:>4}"
            )
            pdb_lines.append(chain_termination_line)
            atom_index += 1

            if (i != seq_len - 1):
                # "prev" is a misnomer here. This happens at the beginning of
                # each new chain.
                pdb_lines.extend(get_pdb_headers(prot, prev_chain_index))

    pdb_lines.append("END")
    pdb_lines.append("")
    return "\n".join(pdb_lines)


def to_modelcif(prot: Protein) -> str:
    """
    Converts a `Protein` instance to a ModelCIF string. Chains with identical modelled coordinates
    will be treated as the same polymer entity. But note that if chains differ in modelled regions,
    no attempt is made at identifying them as a single polymer entity.

    Args:
        prot: The protein to convert to PDB.

    Returns:
        ModelCIF string.
    """
    restypes = residue_constants.restypes_with_x
    atom_types = residue_constants.atom_types
    atom_mask = prot.atom_mask
    aatype = prot.aatype
    seq_len = aatype.shape[0]
    atom_positions = prot.atom_positions

    residue_index = prot.residue_index
    if residue_index is None:
        residue_index = np.arange(seq_len)
    residue_index = residue_index.astype(np.int32)

    b_factors = prot.b_factors
    if b_factors is None:
        b_factors = np.ones((seq_len,), dtype=np.float32)

    chain_index = prot.chain_index
    if chain_index is None:
        chain_index = [0 for i in range(seq_len)]

    system = modelcif.System(title='Model prediction')

    # Finding chains and creating entities
    seqs = {}
    seq = []
    last_chain_idx = None
    for i in range(seq_len):
        if last_chain_idx is not None and last_chain_idx != chain_index[i]:
            seqs[last_chain_idx] = seq
            seq = []
        seq.append(restypes[aatype[i]])
        last_chain_idx = chain_index[i]
    # finally add the last chain
    seqs[last_chain_idx] = seq

    # now reduce sequences to unique ones (note this won't work if different asyms have different unmodelled regions)
    unique_seqs = {}
    for chain_idx, seq_list in seqs.items():
        seq = "".join(seq_list)
        if seq in unique_seqs:
            unique_seqs[seq].append(chain_idx)
        else:
            unique_seqs[seq] = [chain_idx]

    # adding 1 entity per unique sequence
    entities_map = {}
    for key, value in unique_seqs.items():
        model_e = modelcif.Entity(key, description='Model subunit')
        for chain_idx in value:
            entities_map[chain_idx] = model_e

    chain_tags = string.ascii_uppercase
    asym_unit_map = {}
    for chain_idx in set(chain_index):
        # Define the model assembly
        chain_id = chain_tags[chain_idx]
        asym = modelcif.AsymUnit(entities_map[chain_idx], details='Model subunit %s' % chain_id, id=chain_id)
        asym_unit_map[chain_idx] = asym
    modeled_assembly = modelcif.Assembly(asym_unit_map.values(), name='Modeled assembly')

    class _LocalPLDDT(modelcif.qa_metric.Local, modelcif.qa_metric.PLDDT):
        name = "pLDDT"
        software = None
        description = "Predicted lddt"

    class _GlobalPLDDT(modelcif.qa_metric.Global, modelcif.qa_metric.PLDDT):
        name = "pLDDT"
        software = None
        description = "Global pLDDT, mean of per-residue pLDDTs"

    class _MyModel(modelcif.model.AbInitioModel):
        def get_atoms(self):
            # Add all atom sites.
            for i in range(seq_len):
                for atom_name, pos, mask, b_factor in zip(
                        atom_types, atom_positions[i], atom_mask[i], b_factors[i]
                ):
                    if mask < 0.5:
                        continue
                    element = atom_name[0]  # Protein supports only C, N, O, S, this works.
                    yield modelcif.model.Atom(
                        asym_unit=asym_unit_map[chain_index[i]], type_symbol=element,
                        seq_id=residue_index[i], atom_id=atom_name,
                        x=pos[0], y=pos[1], z=pos[2],
                        het=False, biso=b_factor, occupancy=1.00)

        def add_scores(self):
            # local scores
            plddt_per_residue = {}
            for i in range(seq_len):
                for mask, b_factor in zip(atom_mask[i], b_factors[i]):
                    if mask < 0.5:
                        continue
                    # add 1 per residue, not 1 per atom
                    if chain_index[i] not in plddt_per_residue:
                        # first time a chain index is seen: add the key and start the residue dict
                        plddt_per_residue[chain_index[i]] = {residue_index[i]: b_factor}
                    if residue_index[i] not in plddt_per_residue[chain_index[i]]:
                        plddt_per_residue[chain_index[i]][residue_index[i]] = b_factor
            plddts = []
            for chain_idx in plddt_per_residue:
                for residue_idx in plddt_per_residue[chain_idx]:
                    plddt = plddt_per_residue[chain_idx][residue_idx]
                    plddts.append(plddt)
                    self.qa_metrics.append(
                        _LocalPLDDT(asym_unit_map[chain_idx].residue(residue_idx), plddt))
            # global score
            self.qa_metrics.append((_GlobalPLDDT(np.mean(plddts))))

    # Add the model and modeling protocol to the file and write them out:
    model = _MyModel(assembly=modeled_assembly, name='Best scoring model')
    model.add_scores()

    model_group = modelcif.model.ModelGroup([model], name='All models')
    system.model_groups.append(model_group)

    fh = io.StringIO()
    modelcif.dumper.write(fh, [system])
    return fh.getvalue()
