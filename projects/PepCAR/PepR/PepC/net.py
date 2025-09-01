from .amnet.net import Prot3DGraphModel, Pep2DGraphModel, AMNet
from .smnet.net import Drug2DGraphModel, SMNet
import torch.nn as nn
import torch
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, input_dim):
        super(Attention, self).__init__()
        self.linear1 = nn.Linear(input_dim, 128)
        self.linear2 = nn.Linear(128, 1)

    def forward(self, x):
        scores = self.linear2(F.relu(self.linear1(x)))
        weights = F.softmax(scores, dim=1)
        return weights * x

class ASMNet(nn.Module):
    def __init__(self,
                amnet=None,
                smnet=None,
                mlp_dims=[1024, 512],
                mlp_dropout=0.25
                ):
        super(ASMNet, self).__init__()

        self.amnet = amnet
        self.smnet = smnet

        self.att1 = Attention(128*2)
        self.att2 = Attention(128*2)
        self.att3 = Attention(128*2)
        self.att4 = Attention(128*2)
        self.att5 = Attention(128*2)

        self.mlp_three = self.get_fc_layers(
            [128 * 6] + mlp_dims + [1],
            dropout=mlp_dropout, batchnorm=False,
            no_last_dropout=True, no_last_activation=True
        )

        self.mlp_two = self.get_fc_layers(
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

    def forward(self, prot_data, pept_data, drug_data, train_mode='both'):
        if train_mode == 'both':
            
            x_fea_three_smnet, x_fea_two_smnet = self.smnet(drug_data, prot_data)
            x_fea_three_amnet, x_fea_two_amnet = self.amnet(pept_data, prot_data)

            x_three_pep = self.att1(torch.cat([x_fea_three_smnet[:, 0:128], x_fea_three_amnet[:, 0:128]], dim=1))
            x_three_poc = self.att2(torch.cat([x_fea_three_smnet[:, 128:128*2], x_fea_three_amnet[:, 128:128*2]], dim=1))
            x_three_pro = self.att3(torch.cat([x_fea_three_smnet[:, 128*2:128*3], x_fea_three_amnet[:, 128*2:128*3]], dim=1))

            x_two_pep = self.att4(torch.cat([x_fea_two_smnet[:, 0:128], x_fea_two_amnet[:, 0:128]], dim=1))
            x_two_pro = self.att5(torch.cat([x_fea_two_smnet[:, 128:128*2], x_fea_two_amnet[:, 128:128*2]], dim=1))

            combined_three = torch.cat([x_three_pep, x_three_poc, x_three_pro], dim=1)
            combined_two = torch.cat([x_two_pep, x_two_pro], dim=1)

            x_three = self.mlp_three(combined_three)
            x_two = self.mlp_two(combined_two)

            return x_three