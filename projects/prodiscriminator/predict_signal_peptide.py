# Copyright (c) 2025, Tencent Inc. All rights reserved.
# Data: 2025/1/9 21:16
# Author: chenchenqin
import argparse
import os

import sys

sys.path.append(".")
import torch
from tgnn.config import get_config
from tgnn.data import MMapIndexedJsonlDataset
from tgnn.utils import jload
from tgnn.deploy import SignalPeptidePredictor
from tgnn.engine.default import setup
from tgnn.protein.parser import parse_fasta


def load_seqs(filename):
    if os.path.exists(filename):
        if filename.endswith(".txt"):
            seqs = open(filename).read().split("\n")
            seqs = [{"seq": s} for s in seqs]
        elif filename.endswith(".json"):
            seqs = jload(filename)
        elif filename.endswith((".fa", ".fasta")):
            seqs, _, _ = parse_fasta(filename)
            seqs = [{"seq": s} for s in seqs]
        elif filename.endswith("jsonl"):
            seqs = MMapIndexedJsonlDataset(filename)
        else:
            raise Exception(f"input prompt file format error, only support txt, fasta, json or jsonl format file")
    else:
        seqs = [{"seq": filename}]

    print(f"number of seqs: {len(seqs)}")
    return seqs


def parse_args():
    seq = ("MHSRLKFLAYLHFICASSIFWPEFSSAQQQQQTVSLTEKIPLGAIFEQGTDDVQSAFKYAMLNHNLNVSSRRFELQAYVDVINTADAFKLSRLICNQFSRGVYSM"
           "LGAVSPDSFDTL")
    parser = argparse.ArgumentParser(description="Protein Peptide Predication")
    parser.add_argument("--input", "-i",
                        default=seq,
                        type=str,
                        help="path to input prompt files or prompt string")
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="directory of output fasta file",
    )
    parser.add_argument(
        "--plot", "-p",
        action="store_true",
        help="plot sequence",
    )
    args = parser.parse_args()
    return args


def main(args):
    cfg = get_config()
    setup(cfg)
    seqs = load_seqs(args.input)
    predictor = SignalPeptidePredictor()
    predictor.load_state(model="usm_100m_signalp")
    if torch.cuda.is_available():
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float16
    else:
        dtype = torch.float32

    predictor.to(dtype)
    predictor.predict(seqs, plot_seq=args.plot)


if __name__ == "__main__":
    args = parse_args()
    main(args)
