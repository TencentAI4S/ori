# Copyright (c) 2025, Tencent Inc. All rights reserved.
# Data: 2025/1/14 10:45
# Author: chenchenqin
import os

import torch
import torch.nn as nn

from tgnn.config import get_config, CN
from tgnn.model import build_model
from tgnn.model.utils import get_model_path
from tgnn.protein.data_transform import make_aatype
from tgnn.tokenizer import build_tokenizer

MODEL_URL = "https://zenodo.org/records/14639034/files"


class ProteinStructurePredictor(nn.Module):

    def __init__(self, device=None):
        super(ProteinStructurePredictor, self).__init__()
        self.cfg = get_config()
        self.model = None
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

    def load_state(self, model="usmfold_100m"):
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

    @torch.no_grad()
    def infer(self, seq, output, num_cycles=None, verbose=False):
        aatype = make_aatype(seq, device=self.device)[None]
        token_ids = self.tokenizer.encode(seq, device=self.device)
        token_ids = token_ids[None]  # [1, seq_len]
        outputs = self.model.infer(token_ids, aatype, num_cycles=num_cycles, verbose=verbose)
        print(outputs["mean_plddt"][0].item())
        with open(output, mode="w") as f:
            pdb_string = self.model.output_to_pdb(outputs)[0]
            f.write(pdb_string)
