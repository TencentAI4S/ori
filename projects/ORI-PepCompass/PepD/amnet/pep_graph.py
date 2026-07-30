from amnet.constants import *
import torch
from torch_geometric.data import Data
def featurize_peptide_graph(peptide, name=None):

    with torch.no_grad():
        seq_str = peptide['seq'].upper()  # normalize D-amino acid lowercase to uppercase
        seq = torch.as_tensor([LETTER_TO_NUM[a] for a in seq_str], dtype=torch.long)
        ids = torch.as_tensor(peptide['ids'], dtype=torch.float)
        emb = torch.load(peptide['embed'])

        # Ensure ids has shape (L, 3). A missing/failed iupred2a run leaves an
        # empty list [], producing tensor([]) shape (0,) — which breaks PyG's
        # collate when batched with tensors of shape (L, 3).
        if ids.numel() == 0:
            ids = ids.view(0, 3)

        num_nodes = len(peptide['seq'])

        # ── Defensive: align ids/emb lengths with seq length ──────────────
        # If iupred2a or ESM produced fewer/more rows than residues, pad with
        # zeros or truncate so torch.cat([x, ids, emb], dim=-1) doesn't fail.
        if ids.shape[0] != num_nodes:
            print(f"WARNING: [{name}] ids length {ids.shape[0]} != seq length "
                  f"{num_nodes}, padding/truncating ids to match")
            if ids.numel() == 0:
                ids = torch.zeros(num_nodes, 3, dtype=torch.float)
            elif ids.shape[0] < num_nodes:
                pad = torch.zeros(num_nodes - ids.shape[0], ids.shape[1],
                                  dtype=ids.dtype)
                ids = torch.cat([ids, pad], dim=0)
            else:
                ids = ids[:num_nodes]

        if emb.shape[0] != num_nodes:
            print(f"WARNING: [{name}] emb length {emb.shape[0]} != seq length "
                  f"{num_nodes}, padding/truncating emb to match")
            if emb.shape[0] < num_nodes:
                pad = torch.zeros(num_nodes - emb.shape[0], emb.shape[1],
                                  dtype=emb.dtype)
                emb = torch.cat([emb, pad], dim=0)
            else:
                emb = emb[:num_nodes]
        edge_index = []
        for i in range(num_nodes - 1):
            edge_index.append([i, i + 1])
            edge_index.append([i + 1, i])
        # Ensure edge_index always has shape (2, E) even when E=0
        # (single-residue peptide produces (0,) otherwise, breaking PyG collate).
        if len(edge_index) == 0:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
        else:
            edge_index = torch.tensor(edge_index, dtype=torch.long).t()

        data = Data(x=seq, edge_index=edge_index, 
                    ids=ids, emb=emb, name=name)
    return data