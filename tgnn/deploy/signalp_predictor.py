# Copyright (c) 2025, Tencent Inc. All rights reserved.
# Data: 2025/1/13 11:37
# Author: chenchenqin
import os

import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from tgnn.config import get_config, CN
from tgnn.model import build_model
from tgnn.model.utils import get_model_path
from tgnn.tokenizer import build_tokenizer

MODEL_URL = "https://zenodo.org/records/14639034/files"


class SignalPeptidePredictor(nn.Module):

    def __init__(self, device=None):
        super(SignalPeptidePredictor, self).__init__()
        self.cfg = get_config()
        self.model = None
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.max_len = 70

    def load_state(self, model="usm_100m_thermostablility"):
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
    def predict(self, seqs, plot_seq=False):
        max_len = self.max_len
        num_kindom_classes = getattr(self.cfg.model, "num_kindom_classes", 4)
        if num_kindom_classes == 4:
            kingdom_dcit = {"EUKARYA": 0, "POSITIVE": 1, "NEGATIVE": 2, "ARCHAEA": 3}
        elif num_kindom_classes == 5:
            kingdom_dcit = {"EUKARYA": 0, "POSITIVE": 1, "NEGATIVE": 2, "ARCHAEA": 3, "OTHER": 4}
        else:
            kingdom_dcit = {"EUKARYA": 0, "POSITIVE": 1, "NEGATIVE": 1, "ARCHAEA": 1, "OTHER": 1}
        for data in seqs:
            seq = data["seq"]
            kingdom = data.get("kingdom", "EUKARYA")
            input_ids = self.tokenizer.encode(seq)
            if len(input_ids) > max_len:
                input_ids = input_ids[:max_len]
            input_ids = input_ids.to(dtype=torch.long, device=self.device)[None]
            kindom_ids = torch.tensor([kingdom_dcit[kingdom.upper()], ], dtype=torch.long,
                                      device=self.device)
            outputs = self.model(input_ids, kindom_ids)
            logits = outputs['tag'][0]
            global_probs = outputs['global'][0].softmax(dim=-1).tolist()
            print(f"probilities of "
                  f"NO_SP: {global_probs[0]:.3f}, "
                  f"SP: {global_probs[1]:.3f}, "
                  f"LIPO: {global_probs[2]:.3f}, "
                  f"TAT: {global_probs[3]:.3f}, "
                  f"TATLIPO: {global_probs[4]:.3f}, "
                  f"PILIN: {global_probs[5]:.3f}")
            # 1. Infer the cleavage sites
            cleavage_site = self.model.get_cleavage_sites(logits)  # 1-based
            print(f"cleavage_site: {cleavage_site}")
            if cleavage_site != -1:
                print(f"cleavage_site probility: {self.model.get_cleavage_probility(logits)}")

            if plot_seq:
                # 2. Simplify the marginal probabilities and merge classes into Sec/SPI for eukarya.
                probs = logits.softmax(dim=-1)
                marginal_probs = self.model.compute_post_probabilities(probs, kingdom)
                self.model.plot_sequence(marginal_probs.cpu(), seq, cleavage_site)
                plt.show()
