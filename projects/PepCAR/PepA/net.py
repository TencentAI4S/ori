from amnet.net import AMNet
from smnet.net import SMNet
import torch
import torch.nn as nn
import torch.nn.functional as F

class PTNet(nn.Module):
    def __init__(self, amnet=None, smnet=None, mlp_dims=[1024, 512], mlp_dropout=0.25):
        super().__init__()
        self.amnet = amnet
        self.smnet = smnet

        self.mlp = self.get_fc_layers(
            [128 * 4] + mlp_dims + [1],
            dropout=mlp_dropout, batchnorm=False,
            no_last_dropout=True, no_last_activation=True
        )
    def get_fc_layers(self, hidden_sizes,
                      dropout=0, batchnorm=False,
                      no_last_dropout=True, no_last_activation=True):
        act_fn = torch.nn.LeakyReLU()
        layers = []
        for i, (in_dim, out_dim) in enumerate(zip(hidden_sizes[:-1], hidden_sizes[1:])):
            layers.append(nn.Linear(in_dim, out_dim))
            if not no_last_activation or i != len(hidden_sizes) - 2:
                layers.append(act_fn)
            if dropout > 0:
                if not no_last_dropout or i != len(hidden_sizes) - 2:
                    layers.append(nn.Dropout(dropout))
            if batchnorm and i != len(hidden_sizes) - 2:
                layers.append(nn.BatchNorm1d(out_dim))
        return nn.Sequential(*layers)

    def forward(self, pept_data, smiles_data):
        smi_emb = self.smnet.drug_model(smiles_data)
        pep_emb = self.amnet.pep_model(pept_data)

        combined = torch.cat([smi_emb, pep_emb], dim=1)

        pred= self.mlp(combined)
        return pred


class ADMETNet(nn.Module):
    def __init__(self, smnet=None, mlp_dims=[512, 256], mlp_dropout=0.25):
        super().__init__()
        self.smnet = smnet

        self.mlp = self.get_fc_layers(
            [128 * 2] + mlp_dims + [1],
            dropout=mlp_dropout, batchnorm=False,
            no_last_dropout=True, no_last_activation=True
        )
    def get_fc_layers(self, hidden_sizes,
                      dropout=0, batchnorm=False,
                      no_last_dropout=True, no_last_activation=True):
        act_fn = torch.nn.LeakyReLU()
        layers = []
        for i, (in_dim, out_dim) in enumerate(zip(hidden_sizes[:-1], hidden_sizes[1:])):
            layers.append(nn.Linear(in_dim, out_dim))
            if not no_last_activation or i != len(hidden_sizes) - 2:
                layers.append(act_fn)
            if dropout > 0:
                if not no_last_dropout or i != len(hidden_sizes) - 2:
                    layers.append(nn.Dropout(dropout))
            if batchnorm and i != len(hidden_sizes) - 2:
                layers.append(nn.BatchNorm1d(out_dim))
        return nn.Sequential(*layers)

    def forward(self, smiles_data, is_reg='True'):
        smi_emb = self.smnet.drug_model(smiles_data)

        pred = self.mlp(smi_emb)

        if is_reg:
            return pred
        else:
            return F.sigmoid(pred)