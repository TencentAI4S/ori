# Copyright (c) 2025, Tencent Inc. All rights reserved.
# Data: 2025/1/9 21:16
# Author: chenchenqin
import argparse
import sys

sys.path.append(".")
import torch
from tgnn.deploy.utils import load_seqs
from tgnn.engine.default import setup
from tgnn.config import get_config
from tgnn.deploy import SolubilityPredictor


def parse_args():
    parser = argparse.ArgumentParser(description="Protein Thermostability Predication")
    parser.add_argument("--input", "-i",
                        default="data/solubility_demo.fasta",
                        type=str,
                        help="path to input prompt files or prompt string")
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="directory of output fasta file",
    )
    args = parser.parse_args()
    return args


def main(args):
    cfg = get_config()
    setup(cfg)
    classifier = SolubilityPredictor()
    classifier.load_state(model="usm_100m_solubility")
    if torch.cuda.is_available():
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float16
    else:
        dtype = torch.float32
    classifier.to(dtype)
    seqs = load_seqs(args.input)
    classifier.predict(seqs)


if __name__ == "__main__":
    args = parse_args()
    main(args)
