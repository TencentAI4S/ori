# Copyright (c) 2025, Tencent Inc. All rights reserved.
# Data: 2025/1/9 15:58
# Author: chenchenqin
import typing as T

import torch

from tgnn.protein import residue_constants
from tgnn.utils.tensor import collate_dense_tensors


def encode_sequence(
        seq: str,
        residue_index_offset: T.Optional[int] = 512,
        chain_linker: T.Optional[str] = "G" * 25,
) -> T.Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Args:
        seq: aa seq replace ":" by chain linker

    Returns:
        seq_ids: chained seq index tensor, index in range[0, 20]
        residx: residues indices, note that muliter index with offset,
        linker_mask: chain link mask, 1 is real aa seq, 0 is G linker
        chain_index: chain indices, like [0,0, 0, 0, 1, 1, 1, 1], include chain linker seq
    """
    if chain_linker is None:
        chain_linker = ""

    if residue_index_offset is None:
        residue_index_offset = 0

    # replace : by chain linker
    chains = seq.split(":")
    seq = chain_linker.join(chains)

    # replace non aa char by "x"
    unk_idx = residue_constants.restype_order_with_x["X"]
    encoded = torch.tensor(
        [residue_constants.restype_order_with_x.get(aa, unk_idx) for aa in seq]
    )
    seq_len = len(encoded)
    residx = torch.arange(seq_len)

    if residue_index_offset > 0:
        start = 0
        for i, chain in enumerate(chains):
            residx[start: start + len(chain) + len(chain_linker)] += (
                    i * residue_index_offset
            )
            start += len(chain) + len(chain_linker)

    linker_mask = torch.ones_like(encoded, dtype=torch.float32)
    chain_index = []
    offset = 0
    for i, chain in enumerate(chains):
        if i > 0:
            chain_index.extend([i - 1] * len(chain_linker))

        chain_index.extend([i] * len(chain))
        offset += len(chain)
        linker_mask[offset: offset + len(chain_linker)] = 0
        offset += len(chain_linker)

    chain_index = torch.tensor(chain_index, dtype=torch.int64)

    return encoded, residx, linker_mask, chain_index


def batch_encode_sequences(
        sequences: T.Sequence[str],
        residue_index_offset: T.Optional[int] = 512,
        chain_linker: T.Optional[str] = "G" * 25,
) -> T.Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    aatype_list = []
    residx_list = []
    linker_mask_list = []
    chain_index_list = []
    for seq in sequences:
        aatype_seq, residx_seq, linker_mask_seq, chain_index_seq = encode_sequence(
            seq,
            residue_index_offset=residue_index_offset,
            chain_linker=chain_linker,
        )
        aatype_list.append(aatype_seq)
        residx_list.append(residx_seq)
        linker_mask_list.append(linker_mask_seq)
        chain_index_list.append(chain_index_seq)

    aatype = collate_dense_tensors(aatype_list)

    # batch padding mask
    mask = collate_dense_tensors(
        [aatype.new_ones(len(aatype_seq)) for aatype_seq in aatype_list]
    )
    residx = collate_dense_tensors(residx_list)
    linker_mask = collate_dense_tensors(linker_mask_list)
    chain_index_list = collate_dense_tensors(chain_index_list, -1)

    return aatype, mask, residx, linker_mask, chain_index_list
