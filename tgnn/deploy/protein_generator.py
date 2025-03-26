# Copyright (c) 2025, Tencent Inc. All rights reserved.
# Data: 2025/1/13 11:37
# Author: chenchenqin
import os

import torch
import torch.nn as nn
import tqdm

from tgnn.config import get_config, CN
from tgnn.data import MMapIndexedJsonlDatasetBuilder, MMapIndexedFastaDatasetBuilder
from tgnn.model import build_model
from tgnn.model.utils import get_model_path, get_tokenizer_path
from tgnn.tokenizer import build_tokenizer

MODEL_URL = "https://zenodo.org/records/14639034/files"
TOKENIZER_URL = f"{MODEL_URL}/pgm_tokenizer.model"
TOKENIZER_VOCAB_URL = f"{MODEL_URL}/pgm_tokenizer.vocab"


class ProteinGenerator(nn.Module):

    def __init__(self, device=None):
        super(ProteinGenerator, self).__init__()
        self.cfg = get_config()
        self.model = None
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

    @property
    def top_k(self):
        return self.cfg.model.generation.top_k

    @property
    def top_p(self):
        return self.cfg.model.generation.top_p

    @property
    def temperature(self):
        return self.cfg.model.generation.temperature

    def load_state(self, model_name="pgm_3b", model_path=None, tokenizer_path=None):
        if tokenizer_path is None:
            tokenizer_dir = f"{torch.hub.get_dir()}/tokenizers"
            tokenizer_path = f"{tokenizer_dir}/pgm_tokenizer.model"
            if not os.path.exists(tokenizer_path):
                get_tokenizer_path(TOKENIZER_URL)

        if model_path is None:
            model_dir = f"{torch.hub.get_dir()}/checkpoints"
            model_path = f"{model_dir}/{model_name.lower()}.pt"
            if not os.path.exists(model_path):
                get_model_path(f"{MODEL_URL}/{model_name.lower()}.pt")

        assert os.path.exists(model_path), f"{model_path} does not exist"
        state = torch.load(model_path, map_location="cpu")
        model_states = state.get("model", state)
        model_states = model_states.get("module", model_states)
        self.cfg.update(CN(state["config"]))
        # overrite default tokenizer
        self.cfg.tokenizer.path = tokenizer_path
        with torch.device(self.device):
            self.model = build_model(self.cfg).eval()
            self.model.load_state_dict(model_states)
            self.model.to(self.device)

        self.tokenizer = build_tokenizer(self.cfg)

    @torch.no_grad()
    def generate(self, prompts, num_samples=1, output=None):
        if output is not None:
            if output.endswith((".fasta", ".fa")):
                builder = MMapIndexedFastaDatasetBuilder(output)
            elif output.endswith(".jsonl"):
                builder = MMapIndexedJsonlDatasetBuilder(output)
            else:
                raise f"format error, only support fasta or jsonl format of output file"
        else:
            builder = None

        stop_ids = (self.tokenizer.bos, self.tokenizer.eos, self.tokenizer.pad)

        for p in tqdm.tqdm(prompts):
            prompt = p["prompt"]
            name = p["name"]
            # use custom temperature or default temperaturex
            temp = p.get("temperature", self.temperature)
            ns = p.get("num_samples", num_samples)
            token_ids = self.tokenizer.encode(prompt, bos=True).to(self.device)
            token_ids = token_ids[None].repeat(ns, 1)
            outputs = self.model.generate(token_ids,
                                          max_new_tokens=512,
                                          temperature=temp,
                                          top_k=self.top_k,
                                          top_p=self.top_p,
                                          output_score=True,
                                          stop_ids=stop_ids)
            token_len = token_ids.shape[1]
            for i, out in enumerate(outputs):
                output_ids = out.sequences[token_len:]
                output_probs = torch.tensor(out.scores)
                confidence = output_probs.mean().item()
                output_seq = self.tokenizer.decode(output_ids)
                if builder is None:
                    print(prompt, output_seq)
                    continue

                if isinstance(builder, MMapIndexedFastaDatasetBuilder):
                    builder.add_item(output_seq, f"{name}_{i}", {"prompt": prompt, "score": confidence})
                else:
                    builder.add_item({
                        "name": f"{name}_{i}",
                        "prompt": prompt,
                        "sequence": output_seq,
                        "score": confidence
                    })
                builder.end_document()

        if builder is not None:
            builder.finalize()
