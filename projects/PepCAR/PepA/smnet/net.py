import torch
import torch.nn as nn
import torch_geometric
from torch_geometric.nn import GCNConv, GINConv, GATConv
import torch.nn.functional as F
class Prot3DGraphModel(nn.Module):
    def __init__(self,
        d_vocab=25, d_embed=20,
        d_dihedrals=6, d_pretrained_emb=1280, d_edge=39,
        d_gcn=[128, 256, 256],
    ):
        super(Prot3DGraphModel, self).__init__()
        d_gcn_in = d_gcn[0]
        self.embed = nn.Embedding(d_vocab, d_embed)
        self.proj_node = nn.Linear(d_embed + d_dihedrals + d_pretrained_emb, d_gcn_in)
        self.proj_edge = nn.Linear(d_edge, d_gcn_in)
        gcn_layer_sizes = [d_gcn_in] + d_gcn
        layers_prot = []
        for i in range(len(gcn_layer_sizes) - 1):            
            layers_prot.append((
                torch_geometric.nn.TransformerConv(
                    gcn_layer_sizes[i], gcn_layer_sizes[i + 1], edge_dim=d_gcn_in),
                'x, edge_index, edge_attr -> x'
            ))            
            layers_prot.append(nn.LeakyReLU())

        layers_pock = []
        for i in range(len(gcn_layer_sizes) - 1):            
            layers_pock.append((
                torch_geometric.nn.TransformerConv(
                    gcn_layer_sizes[i], gcn_layer_sizes[i + 1], edge_dim=d_gcn_in),
                'x, edge_index, edge_attr -> x'
            ))            
            layers_pock.append(nn.LeakyReLU())              
        
        self.gcn_prot = torch_geometric.nn.Sequential(
            'x, edge_index, edge_attr', layers_prot)
        self.gcn_pock = torch_geometric.nn.Sequential(
            'x, edge_index, edge_attr', layers_pock)         
        self.pool = torch_geometric.nn.global_mean_pool
        
    def forward(self, data):
        x, edge_index = data.seq, data.edge_index
        batch = data.batch

        x = self.embed(x)
        s = data.node_s
        emb = data.seq_emb

        x = torch.cat([x, s, emb], dim=-1)

        epitope = data.importance
        x_epitope = x[epitope==1]

        epitope_indices = torch.nonzero(epitope).squeeze(-1)
        mask = (edge_index[0].unsqueeze(-1) == epitope_indices).any(dim=-1) & \
            (edge_index[1].unsqueeze(-1) == epitope_indices).any(dim=-1)
        edge_index_epitope = edge_index[:, mask]

        node_idx_mapping = {idx.item(): i for i, idx in enumerate(epitope_indices)}
        edge_index_epitope = torch.tensor(
            [[node_idx_mapping[edge_idx.item()] for edge_idx in edge_pair] for edge_pair in edge_index_epitope.t()],
            dtype=torch.long).t().to(edge_index.device)
        
        edge_attr = data.edge_s
        edge_attr_epitope = edge_attr[mask]

        x = self.proj_node(x)
        edge_attr = self.proj_edge(edge_attr)
        x = self.gcn_prot(x, edge_index, edge_attr)

        x_epitope = self.proj_node(x_epitope)
        edge_attr_epitope = self.proj_edge(edge_attr_epitope)
        x_epitope = self.gcn_pock(x_epitope, edge_index_epitope, edge_attr_epitope)

        x_epitope_padded = torch.zeros_like(x)
        x_epitope_padded[epitope == 1] = x_epitope

        x = torch_geometric.nn.global_mean_pool(x, batch)
        x_epitope = torch_geometric.nn.global_mean_pool(x_epitope_padded, batch)
        return x, x_epitope

class Drug2DGraphModel(nn.Module):
    def __init__(self, d_vocab=78, d_embed=78, d_gcn=[128, 256, 256]):
        super(Drug2DGraphModel, self).__init__()
        d_gcn_in = d_gcn[0]
        self.proj_node = nn.Linear(d_embed, d_gcn_in)
        gcn_layer_sizes = [d_gcn_in] + d_gcn
        layers = []
        for i in range(len(gcn_layer_sizes) - 1):            
            layers.append((
                torch_geometric.nn.TransformerConv(
                    gcn_layer_sizes[i], gcn_layer_sizes[i + 1]),
                'x, edge_index -> x'
            ))            
            layers.append(nn.LeakyReLU())            
        
        self.gcn = torch_geometric.nn.Sequential(
            'x, edge_index', layers)

        self.pool = torch_geometric.nn.global_mean_pool

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.proj_node(x)


        x = self.gcn(x, edge_index)
        x = self.pool(x, data.batch)
        return x
    

class SMNet(nn.Module):
    def __init__(self,
                 drug_gcn_dims=[128, 256, 256],
                 drug_fc_dims=[1024, 128],
                 prot_emb_dim=1280,
                 prot_gcn_dims=[128, 256, 256],
                 prot_fc_dims=[1024, 128],
                 mlp_dims=[1024, 512],
                 mlp_dropout=0.25):
        super(SMNet, self).__init__()

        self.drug_model = Drug2DGraphModel(d_gcn=drug_gcn_dims)
        drug_emb_dim = drug_gcn_dims[-1]

        self.prot_model = Prot3DGraphModel(d_pretrained_emb=prot_emb_dim, d_gcn=prot_gcn_dims)
        prot_emb_dim = prot_gcn_dims[-1]

        self.drug_fc = self.get_fc_layers(
            [drug_emb_dim] + drug_fc_dims,
            dropout=mlp_dropout, batchnorm=False,
            no_last_dropout=True, no_last_activation=True)

        self.prot_fc = self.get_fc_layers(
            [prot_emb_dim] + prot_fc_dims,
            dropout=mlp_dropout, batchnorm=False,
            no_last_dropout=True, no_last_activation=True)
        
        self.pock_fc = self.get_fc_layers(
            [prot_emb_dim] + prot_fc_dims,
            dropout=mlp_dropout, batchnorm=False,
            no_last_dropout=True, no_last_activation=True)

        self.top_fc_three = self.get_fc_layers(
            [drug_fc_dims[-1] + 2 * prot_fc_dims[-1]] + mlp_dims + [1],
            dropout=mlp_dropout, batchnorm=False,
            no_last_dropout=True, no_last_activation=True)
        
        self.top_fc_two = self.get_fc_layers(
            [drug_fc_dims[-1] + 1 * prot_fc_dims[-1]] + mlp_dims + [1],
            dropout=mlp_dropout, batchnorm=False,
            no_last_dropout=True, no_last_activation=True)

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

    def forward(self, drug_data, prot_data):
        drug_emb = self.drug_model(drug_data)
        protein_emb, pocket_emb = self.prot_model(prot_data)

        drug_emb = self.drug_fc(drug_emb)
        protein_emb = self.prot_fc(protein_emb)
        pocket_emb = self.pock_fc(pocket_emb)

        x_fea_three = torch.cat([drug_emb, pocket_emb, protein_emb], dim=1)
        x_fea_two = torch.cat([drug_emb, protein_emb], dim=1)
        
        x_three = self.top_fc_three(x_fea_three)
        x_two = self.top_fc_two(x_fea_two)


        return x_fea_three, x_fea_two