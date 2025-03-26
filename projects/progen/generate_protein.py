# Copyright (c) 2025, Tencent Inc. All rights reserved.
# Data: 2025/1/9 21:16
# Author: chenchenqin
import argparse
import os
import sys

sys.path.append(".")

import torch
from tgnn.config import cfg
from tgnn.engine.default import setup
from tgnn.deploy import ProteinGenerator
from tgnn.deploy.utils import load_prompts


def parse_args():
    parser = argparse.ArgumentParser(description="Enzyme Generation")
    parser.add_argument("--model", "-m",
                        type=str,
                        default="pgm_3b",
                        help="model name or model path of generation model")
    parser.add_argument("--input", "-i", "-p",
                        type=str,
                        default="<EC:3.2.1.17><temperature90><:>",
                        help="path to input prompt files or prompt string")
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="directory of output fasta file",
    )
    parser.add_argument(
        "--num_samples", "-ns",
        type=int,
        default=10,
        help="number of samples generated per prompt",
    )
    parser.add_argument(
        "--temperature", "-t",
        type=float,
        default=1.0,
        help="temperature of generation",
    )
    parser.add_argument(
        "--top_k", "-tk",
        type=int,
        default=None,
        help="top k of generation",
    )
    parser.add_argument(
        "--top_p", "-tp",
        type=float,
        default=None,
        help="top p of generation",
    )
    args = parser.parse_args()

    return args


def main(args):
    setup(cfg)
    progen = ProteinGenerator()
    model_name = None
    model_path = None
    if os.path.exists(args.model):
        model_path = args.model
    else:
        model_name = args.model

    progen.load_state(model_name=model_name, model_path=model_path)
    if args.temperature is not None:
        cfg.model.generation.temperature = args.temperature
    if args.top_k is not None:
        cfg.model.generation.top_k = args.top_k
    if args.top_p is not None:
        cfg.model.generation.top_p = args.top_p
    print(cfg.model.generation)

    if torch.cuda.is_available():
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float16
    else:
        dtype = torch.float32

    progen.to(dtype)
    prompts = load_prompts(args.input)
    num_samples = args.num_samples
    progen.generate(prompts, num_samples=num_samples, output=args.output)


if __name__ == "__main__":
    args = parse_args()
    main(args)
