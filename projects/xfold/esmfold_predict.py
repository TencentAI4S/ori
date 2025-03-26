# Copyright (c) 2025, Tencent Inc. All rights reserved.
# Data: 2025/1/9 21:16
# Author: chenchenqin
import argparse
import os
import sys

sys.path.append(".")

import torch
from tgnn.data import MMapIndexedJsonlDataset
from tgnn.config import get_config
from tgnn.protein.parser import parse_fasta
from tgnn.engine.default import setup
from tgnn.model.arch.esmfold import load_esmfold_model


def parse_args():
    parser = argparse.ArgumentParser(description="Protein Structure Predication")
    parser.add_argument("--input", "-i",
                        default="data/test.fasta",
                        help="path to input fasta file")
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="directory of output",
    )
    parser.add_argument(
        '--chunk_size', '-cs',
        type=int,
        default=None,
        help='chunk size for long chain inference',
    )
    args = parser.parse_args()
    return args


def load_seqs(filename):
    if os.path.exists(filename):
        save_dir, basename = os.path.split(filename)
        if filename.endswith(".fasta"):
            sequences, ids, _ = parse_fasta(filename)
            batches = []
            for seq, seq_id in zip(sequences, ids):
                output = args.output or f"{save_dir}/{seq_id}.pdb"
                batches.append(
                    {
                        "id": seq_id,
                        "seq": seq,
                        "output": output
                    })
        elif filename.endswith("jsonl"):
            batches = MMapIndexedJsonlDataset(filename)
            for data in batches:
                if "output" not in data:
                    seq_id = data["id"]
                    data["output"] = f"{save_dir}/{seq_id}.pdb"
        else:
            raise Exception(f"input prompt file format error, only support txt, json or jsonl format file")
    else:
        output = args.output or "protein.pdb"
        batches = [{"seq": filename, "id": "protein", "output": output}, ]

    print(f"number of seqs: {len(batches)}")
    return batches


def main(args):
    cfg = get_config()
    setup(cfg)
    model = load_esmfold_model("esmfold_3B_v1").eval()
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float16
    model.to(dtype)
    if torch.cuda.is_available():
        model.cuda()

    # model = torch.compile(model)
    chunk_size = args.chunk_size
    for data in load_seqs(args.input):
        seq_id = data["id"]
        aaseq = data["seq"]
        if chunk_size is None:
            # adjust chunk size if OOM
            if len(aaseq) > 1024:
                model.set_chunk_size(32)
            else:
                model.set_chunk_size(None)
        else:
            model.set_chunk_size(chunk_size)
        outputs = model.infer([aaseq, ], need_head=True)
        plddt = outputs["mean_plddt"][0].item()
        print(f"{seq_id} mean plddt: {plddt:.3f}")
        output = data["output"]
        with open(output, 'w', encoding='utf-8') as f:
            pdb_string = model.output_to_pdb(outputs)[0]
            f.write(pdb_string)


if __name__ == "__main__":
    args = parse_args()
    main(args)
