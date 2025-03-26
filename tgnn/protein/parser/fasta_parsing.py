# Copyright (c) 2025, Tencent Inc. All rights reserved.
# Data: 2025/1/9 15:58
# Author: chenchenqin
import gzip
from collections import OrderedDict
from functools import partial


def parse_fasta(path, to_dict=False):
    """Parse the FASTA file.

    Args:
        path: path to the FASTA file (could be GZIP-compressed)

    Returns:
        prot_id: protein ID (as in the commentary line)
        aa_seq: amino-acid sequence
    """
    open_fn = partial(gzip.open, mode='rt') if path.endswith('.gz') else partial(open, mode='r')
    with open_fn(path) as f:
        fasta_string = f.read()

    return parse_fasta_string(fasta_string, to_dict=to_dict)


def parse_fasta_string(fasta_string, to_dict=False):
    """Parses FASTA string and returns list of strings with amino-acid sequences.

    Args:
        fasta_string: The string contents of a FASTA file.

    Returns:
        A tuple of two lists:
        * A list of sequences.
        * A list of sequence ids
        * A list of sequence descriptions taken from the comment lines. In the
            same order as the sequences.
    """
    sequences = []
    ids = []
    descriptions = []
    index = -1
    for line in fasta_string.splitlines():
        line = line.strip()
        if line.startswith('>'):
            index += 1
            seq_id, *description = line[1:].split(None, 1)  # Remove the '>' at the beginning.
            ids.append(seq_id)
            if len(description) > 0:
                descriptions.append(description)
            else:
                descriptions.append("")
            sequences.append('')
            continue
        elif line.startswith('#'):
            continue
        elif not line:
            continue  # Skip blank lines.
        sequences[index] += line

    assert len(sequences) == len(ids), f"unvalid fasta file"

    if to_dict:
        return OrderedDict(zip(ids, sequences))

    return sequences, ids, descriptions
