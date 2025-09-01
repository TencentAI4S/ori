from amnet.constants import *
import torch
from torch_geometric.data import Data
def featurize_peptide_graph(peptide, name=None):

    with torch.no_grad():
        seq = torch.as_tensor([LETTER_TO_NUM[a] for a in peptide['seq']], dtype=torch.long)
        ids = torch.as_tensor(peptide['ids'], dtype=torch.float)
        emb = torch.load(peptide['embed'])

        num_nodes = len(peptide['seq'])
        edge_index = []
        for i in range(num_nodes - 1):
            edge_index.append([i, i + 1])
            edge_index.append([i + 1, i])
        edge_index = torch.tensor(edge_index, dtype=torch.long).t()

        data = Data(x=seq, edge_index=edge_index, 
                    ids=ids, emb=emb, name=name)
    return data