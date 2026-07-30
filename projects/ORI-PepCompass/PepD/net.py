"""Model definition: only PTNet for peptide property fine-tuning."""

import torch
import torch.nn as nn


class PTNet(nn.Module):
    """Peptide property prediction network.

    Three modes:
      - "full"   : concat AMNet residue-graph emb (256) + SMNet SMILES emb (256)
                   → 512-d → MLP. This is the original two-scale model.
      - "residue": only the residue-graph branch (AMNet.pep_model). 256-d → MLP.
                   Acts as a residue-only baseline.
      - "smiles" : only the SMILES branch (SMNet.drug_model). 256-d → MLP.
                   Acts as a SMILES-only baseline.

    The unused branch is set to None so it consumes no GPU memory and so any
    accidental forward call into it surfaces as a clear error.
    """

    VALID_MODES = ("full", "residue", "smiles")

    def __init__(self, amnet=None, smnet=None, mode="full",
                 mlp_dims=[1024, 512], mlp_dropout=0.25):
        super().__init__()
        if mode not in self.VALID_MODES:
            raise ValueError(f"mode must be one of {self.VALID_MODES}, got {mode!r}")
        self.mode = mode

        # Keep only the encoders this mode actually uses.
        self.amnet = amnet if mode in ("full", "residue") else None
        self.smnet = smnet if mode in ("full", "smiles") else None

        # Each branch outputs 128-dim embedding from its graph encoder; the
        # original Pep2DGraphModel / Drug2DGraphModel actually emit 256-dim
        # (last d_gcn). Concatenated → 512. Single branch → 256.
        if mode == "full":
            in_dim = 128 * 4   # original behavior: pep(256) + smi(256)
        else:
            in_dim = 128 * 2   # single 256-dim branch

        self.mlp = self._build_mlp([in_dim] + mlp_dims + [1], dropout=mlp_dropout)

    def _build_mlp(self, hidden_sizes, dropout=0):
        act_fn = nn.LeakyReLU()
        layers = []
        for i, (in_dim, out_dim) in enumerate(zip(hidden_sizes[:-1], hidden_sizes[1:])):
            layers.append(nn.Linear(in_dim, out_dim))
            is_last = i == len(hidden_sizes) - 2
            if not is_last:
                layers.append(act_fn)
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
        return nn.Sequential(*layers)

    def forward(self, pept_data, smiles_data):
        if self.mode == "residue":
            pep_emb = self.amnet.pep_model(pept_data)
            return self.mlp(pep_emb)
        if self.mode == "smiles":
            smi_emb = self.smnet.drug_model(smiles_data)
            return self.mlp(smi_emb)
        # full
        smi_emb = self.smnet.drug_model(smiles_data)
        pep_emb = self.amnet.pep_model(pept_data)
        combined = torch.cat([smi_emb, pep_emb], dim=1)
        return self.mlp(combined)
