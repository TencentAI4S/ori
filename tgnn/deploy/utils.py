# Copyright (c) 2025, Tencent Inc. All rights reserved.
# Data: 2025/1/13 19:09
# Author: chenchenqin
import os

from tgnn.data import MMapIndexedJsonlDataset
from tgnn.utils import jload
from tgnn.protein.parser import parse_fasta

def load_seqs(filename):
    if os.path.exists(filename):
        if filename.endswith(".txt"):
            seqs = open(filename).read().split("\n")
        elif filename.endswith((".fa", ".fasta")):
            seqs, _, _ = parse_fasta(filename)
        elif filename.endswith(".json"):
            seqs = jload(filename)
        elif filename.endswith("jsonl"):
            seqs = MMapIndexedJsonlDataset(filename)
            seqs = [s["seq"] for s in seqs]
        else:
            raise Exception(f"input prompt file format error, only support txt, json or jsonl format file")
    else:
        seqs = [filename, ]

    print(f"number of seqs: {len(seqs)}")
    return seqs


def load_prompts(prompt):
    if os.path.exists(prompt):
        if prompt.endswith(".txt"):
            prompt = open(prompt).read().split("\n")
            prompt = [{"name": f"protein_{i}", "prompt": p} for i, p in enumerate(prompt)]
        elif prompt.endswith(".json"):
            prompt = jload(prompt)
        elif prompt.endswith("jsonl"):
            prompt = MMapIndexedJsonlDataset(prompt)
        else:
            raise Exception(f"input prompt file format error, only support txt, json or jsonl format file")
    else:
        prompt = [{"name": "protein_0", "prompt": prompt}]
    print(f"number of prompts: {len(prompt)}")
    return prompt
