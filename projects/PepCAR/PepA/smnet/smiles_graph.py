
import torch
from rdkit import Chem
from torch_geometric.data import Data
import networkx as nx
from rdkit import Chem
from rdkit.Chem import MolFromSmiles
import numpy as np
def atom_features(atom):
    return np.array(one_of_k_encoding_unk(atom.GetSymbol(),['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na','Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb','Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H','Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr','Cr', 'Pt', 'Hg', 'Pb', 'Unknown']) +
                    one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6,7,8,9,10]) +
                    one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6,7,8,9,10]) +
                    one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6,7,8,9,10]) +
                    [atom.GetIsAromatic()])

def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))

def one_of_k_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))

def smile_to_graph(smile):
    mol = Chem.MolFromSmiles(smile)
    
    c_size = mol.GetNumAtoms()
    
    features = []
    for atom in mol.GetAtoms():
        feature = atom_features(atom)
        features.append( feature / sum(feature) )

    edges = []
    for bond in mol.GetBonds():
        edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
    g = nx.Graph(edges).to_directed()
    edge_index = []
    for e1, e2 in g.edges:
        edge_index.append([e1, e2])
        
    return c_size, features, edge_index

def featurize_smiles_graph(
    smiles,
    name=None
    ):
    smiles = smiles['seq']
    c_size, features, edge_index = smile_to_graph(smiles)

    x = torch.tensor(np.array(features), dtype=torch.float)
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

    data = Data(x=x, edge_index=edge_index)
    return data

if __name__=="__main__":

    import argparse
    from rdkit import Chem, RDLogger
    from rdkit.Chem import MolStandardize

    class MolClean(object):
        def __init__(self):
            self.normizer = MolStandardize.normalize.Normalizer()
            self.lfc = MolStandardize.fragment.LargestFragmentChooser()
            self.uc = MolStandardize.charge.Uncharger()

        def clean(self, smi):
            mol = Chem.MolFromSmiles(smi)
            if mol:
                mol = self.normizer.normalize(mol)
                mol = self.lfc.choose(mol)
                mol = self.uc.uncharge(mol)
                smi = Chem.MolToSmiles(mol,  isomericSmiles=False, canonical=True)
                return smi
            else:
                return None


    def pep_to_smile(pep):
        peptide_smiles = Chem.MolToSmiles(Chem.MolFromFASTA(pep))
        mc = MolClean()
        clean_smiles = mc.clean(peptide_smiles)
        return clean_smiles

    seqs = 'KWTLERLKRKYRN'
    num = 0

    smile = pep_to_smile(seqs)

    c, f, e = smile_to_graph(smile)
    mol = Chem.MolFromSmiles(smile)
    total_num = len(mol.GetAtoms())

    for i in range(len(seqs)):
        fragment = seqs[:i]+seqs[i+1:]
        smile = pep_to_smile(fragment)
        c, f, e = smile_to_graph(smile)
        mol = Chem.MolFromSmiles(smile)
        fragment_num = len(mol.GetAtoms())
