# Copyright (c) 2025, Tencent Inc. All rights reserved.
# Data: 2025/1/13 11:37
# Author: chenchenqin
import os.path

import torch
import torch.nn as nn
import tqdm

from tgnn.config import get_config, CN
from tgnn.data import MMapIndexedJsonlDatasetBuilder
from tgnn.model import build_model
from tgnn.model.utils import get_model_path
from tgnn.tokenizer import build_tokenizer

MODEL_URL = "https://zenodo.org/records/14639034/files"


class ProteinDiscriminator(nn.Module):

    def __init__(self, device=None):
        super(ProteinDiscriminator, self).__init__()
        self.cfg = get_config()
        self.model = None
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

    def load_state(self, model="usm_100m_thermostability"):
        if os.path.exists(model):
            model_path = model
        else:
            model_dir = f"{torch.hub.get_dir()}/checkpoints"
            model_path = f"{model_dir}/{model.lower()}.pt"
            if not os.path.exists(model_path):
                get_model_path(f"{MODEL_URL}/{model.lower()}.pt")

        state = torch.load(model_path, map_location="cpu")
        model_states = state.get("model", state)
        model_states = model_states.get("module", model_states)
        self.cfg.update(CN(state["config"]))
        with torch.device(self.device):
            self.model = build_model(self.cfg).eval()
            self.model.load_state_dict(model_states)
        self.tokenizer = build_tokenizer(self.cfg)

    def forward(self, seq, softmax=False):
        max_len = self.cfg.dataset.seq_len
        input_ids = self.tokenizer.encode(seq, bos=True, eos=True, device=self.device)
        if len(input_ids) > max_len:
            input_ids = input_ids[:max_len]

        output = self.model(input_ids[None])[0]
        if softmax:
            output = torch.softmax(output, dim=-1)
        return output

    @torch.no_grad()
    def predict(self, seqs, output=None):
        if output is not None:
            if output.endswith(".jsonl"):
                builder = MMapIndexedJsonlDatasetBuilder(output)
            else:
                raise f"format error, only support fasta or jsonl format of output file"
        else:
            builder = None

        for i, seq in tqdm.tqdm(enumerate(seqs)):
            output = self(seq).tolist()
            if builder is None:
                print(f"seq:{seq}\noutput:{output}")
            else:
                builder.add_item({
                    "name": f"seq_{i}",
                    "sequence": seq,
                    "output": output
                })
                builder.end_document()

        if builder is not None:
            builder.finalize()


class SolubilityPredictor(ProteinDiscriminator):

    @torch.no_grad()
    def predict(self, seqs, output=None):
        if output is not None:
            if output.endswith(".jsonl"):
                builder = MMapIndexedJsonlDatasetBuilder(output)
            else:
                raise f"format error, only support fasta or jsonl format of output file"
        else:
            builder = None

        for i, seq in tqdm.tqdm(enumerate(seqs)):
            output = self(seq, softmax=True).tolist()[1]
            if builder is None:
                print(f"seq:{seq}\nsolubility probability:{output:.3f}")
            else:
                builder.add_item({
                    "name": f"seq_{i}",
                    "sequence": seq,
                    "probability": output
                })
                builder.end_document()

        if builder is not None:
            builder.finalize()

    @torch.no_grad()
    def predict(self, seqs, output=None):
        if output is not None:
            if output.endswith(".jsonl"):
                builder = MMapIndexedJsonlDatasetBuilder(output)
            else:
                raise f"format error, only support fasta or jsonl format of output file"
        else:
            builder = None

        for i, seq in tqdm.tqdm(enumerate(seqs)):
            output = self(seq, softmax=True).tolist()[1]
            if builder is None:
                print(f"seq:{seq}\nsolubility probability:{output:.3f}")
            else:
                builder.add_item({
                    "name": f"seq_{i}",
                    "sequence": seq,
                    "probability": output
                })
                builder.end_document()

        if builder is not None:
            builder.finalize()


class ThermostabilityPredictor(ProteinDiscriminator):

    @torch.no_grad()
    def predict(self, seqs, output=None):
        if output is not None:
            if output.endswith(".jsonl"):
                builder = MMapIndexedJsonlDatasetBuilder(output)
            else:
                raise f"format error, only support fasta or jsonl format of output file"
        else:
            builder = None

        for i, seq in tqdm.tqdm(enumerate(seqs)):
            output = self(seq, softmax=True).tolist()[1]
            if builder is None:
                print(f"seq:{seq}\nsolubility probability:{output:.3f}")
            else:
                builder.add_item({
                    "name": f"seq_{i}",
                    "sequence": seq,
                    "probability": output
                })
                builder.end_document()

        if builder is not None:
            builder.finalize()

    @torch.no_grad()
    def predict(self, seqs, output=None):
        if output is not None:
            if output.endswith(".jsonl"):
                builder = MMapIndexedJsonlDatasetBuilder(output)
            else:
                raise f"format error, only support fasta or jsonl format of output file"
        else:
            builder = None

        for i, seq in tqdm.tqdm(enumerate(seqs)):
            output = self(seq).item()
            if builder is None:
                print(f"seq:{seq}\nthermostability:{output:.3f}")
            else:
                builder.add_item({
                    "name": f"seq_{i}",
                    "sequence": seq,
                    "thermostability": output
                })
                builder.end_document()

        if builder is not None:
            builder.finalize()