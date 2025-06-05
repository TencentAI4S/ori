import math
import yaml
import json
import os
from pathlib import Path
from functools import partial
from typing import Optional, Dict, Any, List
import numpy as np
import pandas as pd
import torch
import torch.utils.data as data
from tqdm import tqdm
from Bio import SeqIO
from transformers import EsmModel, AutoTokenizer

from amnet.pdb_graph import pdb_to_graphs, featurize_protein_graph
from amnet.pep_graph import featurize_peptide_graph
from smnet.smiles_graph import featurize_smiles_graph
from smnet.pep_to_smiles import pep_to_smiles
from utils.iupred.calculate_matrix_single import fasta_2_matrix

class FPA(data.Dataset):
    def __init__(self, df=None, data_list=None, onthefly=False,
                 prot_featurize_fn=None, pep_featurize_fn=None, smiles_featurize_fn=None):
        super().__init__()
        self.data_df = df
        self.data_list = data_list
        self.onthefly = onthefly
        self.prot_featurize_fn = prot_featurize_fn if onthefly else None
        self.pep_featurize_fn = pep_featurize_fn if onthefly else None
        self.smiles_featurize_fn = smiles_featurize_fn if onthefly else None

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data_entry = self.data_list[idx]
        if self.onthefly:
            pep = self.pep_featurize_fn(data_entry['pept'])
            prot = self.prot_featurize_fn(data_entry['prot'], name=data_entry['prot_name'])
            smiles = self.smiles_featurize_fn(data_entry['smiles'], name=data_entry['pept_name'])
        else:
            pep, prot, smiles = None, data_entry['protein'], None

        return {'peptide': pep, 'protein': prot, 'smiles': smiles}

def create_test_fold(df):
    return {'test': df}

class SPATask:
    def __init__(self, task_name: Optional[str] = None, protein_id: Optional[str] = None,
                 peptide_seq: Optional[str] = None, esm_pro_path: Optional[str] = None,
                 esm_pep_path: Optional[str] = None, coord_data: Optional[Dict[str, Any]] = None,
                 rec_seq: Optional[Dict[str, Any]] = None, epitope_data: Optional[Dict[str, Any]] = None,
                 num_pos_emb: int = 16, num_rbf: int = 16, contact_cutoff: float = 8.0,
                 split_method: str = 'test'):
        self.task_name = task_name
        self.protein_id = protein_id
        self.peptide_seq = peptide_seq
        self.esm_pro_path = esm_pro_path
        self.esm_pep_path = esm_pep_path
        self.coord_data = coord_data
        self.rec_seq = rec_seq
        self.epitope_data = epitope_data

    def format_protein(self):
        coords = list(zip(self.coord_data[self.protein_id]["N"],
                          self.coord_data[self.protein_id]["CA"],
                          self.coord_data[self.protein_id]["C"],
                          self.coord_data[self.protein_id]["O"]))
        seq = self.rec_seq[self.protein_id]
        embed = Path(self.esm_pro_path) / f"{self.protein_id}.pt"

        if not embed.exists():
            self.extract_protein_embed(seq, embed)

        epitope = self.epitope_data[self.protein_id]
        protein = {"name": self.protein_id, "seq": seq, "coords": coords, "embed": str(embed), "epitope": epitope}
        return featurize_protein_graph(protein)

    def format_peptide(self):
        pep = self.peptide_seq
        ids = fasta_2_matrix(pep)
        embed = Path(self.esm_pep_path) / f"{pep}.pt"

        if not embed.exists():
            self.extract_peptide_embed(pep, embed)

        peptide = {'seq': pep, 'ids': ids, 'embed': str(embed)}
        return featurize_peptide_graph(peptide)

    def format_smiles(self):
        smiles_seq = pep_to_smiles(self.peptide_seq)
        smiles = {'seq': smiles_seq}
        return featurize_smiles_graph(smiles)

    def get_data(self):
        return self.format_protein(), self.format_peptide(), self.format_smiles()

    def extract_protein_embed(self, seq: str, output_path: Path):
        device = 'cuda:0'
        esm_tokenizer = AutoTokenizer.from_pretrained('../model_weights/ESM-2/models--facebook--esm2_t33_650M_UR50D')
        esm2 = EsmModel.from_pretrained("../model_weights/ESM-2/models--facebook--esm2_t33_650M_UR50D").to(device)
        encoded_input = esm_tokenizer(seq, return_tensors='pt', padding=False, truncation=False, max_length=5120).to(device)

        with torch.no_grad():
            output = esm2(**encoded_input)
            embeddings = output.last_hidden_state.squeeze(dim=0)[1:-1]

        torch.save(embeddings.cpu(), output_path)

    def extract_peptide_embed(self, seq: str, output_path: Path):
        device = 'cuda:0'
        esm_tokenizer = AutoTokenizer.from_pretrained('../model_weights/ESM-Pep/models--facebook--esm2_t33_650M_UR50D')
        esm_pep = EsmModel.from_pretrained("../model_weights/ESM-Pep/models--facebook--esm2_t33_650M_UR50D").to(device)
        encoded_input = esm_tokenizer(seq, return_tensors='pt', padding=False, truncation=False, max_length=5120).to(device)

        with torch.no_grad():
            output = esm_pep(**encoded_input)
            embeddings = output.last_hidden_state.squeeze(dim=0)[1:-1]

        torch.save(embeddings.cpu(), output_path)

class FPATask:
    def __init__(self, task_name: Optional[str] = None, df: Optional[Dict[str, Any]] = None,
                 esm_dir: Optional[str] = None, coord_data: Optional[Dict[str, Any]] = None,
                 epitope_data: Optional[Dict[str, Any]] = None, pep_seq: Optional[Dict[str, Any]] = None,
                 rec_seq: Optional[Dict[str, Any]] = None, ids_data: Optional[Dict[str, Any]] = None,
                 pep_esm: Optional[str] = None, num_pos_emb: int = 16, num_rbf: int = 16,
                 contact_cutoff: float = 8.0, split_method: str = 'test', onthefly: bool = False):
        self.task_name = task_name
        self.df = df
        self.coord_data = coord_data
        self.epitope_data = epitope_data
        self.pep_seq = pep_seq
        self.rec_seq = rec_seq
        self.ids_data = ids_data

        self.prot_featurize_params = {
            'num_pos_emb': num_pos_emb,
            'num_rbf': num_rbf,
            'contact_cutoff': contact_cutoff
        }
        self.split_method = split_method
        self.onthefly = onthefly
        self.esm_dir = esm_dir
        self.pep_esm = pep_esm
        self.records = self.df.to_dict('records')

        self._prot2pdb = None
        self._pdb_graph_db = None
        self._pep2emb = None
        self._smiles2smile = None

    def _format_pdb_entry(self, pdb_id: str) -> Dict[str, Any]:
        coords = list(zip(self.coord_data[pdb_id]["N"],
                          self.coord_data[pdb_id]["CA"],
                          self.coord_data[pdb_id]["C"],
                          self.coord_data[pdb_id]["O"]))
        seq = self.rec_seq[pdb_id]
        embed = Path(self.esm_dir) / f"{pdb_id}.pt"
        epitope = self.epitope_data.get(pdb_id, np.ones(len(seq)))

        return {"name": pdb_id, "seq": seq, "coords": coords, "embed": str(embed), "epitope": epitope}

    def _format_pep_entry(self, pep_id: str) -> Dict[str, Any]:
        seq = self.pep_seq[pep_id]['aa']
        ids = self.ids_data[pep_id]
        embed = Path(self.pep_esm) / f"{pep_id}.pt" if self.pep_esm else None

        return {"name": pep_id, "seq": seq, "ids": ids, 'embed': str(embed) if embed else None}

    def _format_smiles_entry(self, data: Dict[str, Any], use_given_smile: bool = True) -> Dict[str, Any]:
        name = f"{data['PDB']}_{data['chain_pep']}"
        seq = data.get("given_smiles") if use_given_smile else self.pep_seq[name]['smile']

        return {"name": name, "seq": seq}

    @property
    def prot2pdb(self) -> Dict[str, Any]:
        if self._prot2pdb is None:
            self._prot2pdb = {
                f"{entry['PDB']}_{entry['chain_pep']}_{entry['chain_pro']}":
                self._format_pdb_entry(f"{entry['PDB']}_{entry['chain_pro']}")
                for entry in self.records
            }
        return self._prot2pdb

    @property
    def pep2emb(self) -> Dict[str, Any]:
        if self._pep2emb is None:
            self._pep2emb = {
                f"{entry['PDB']}_{entry['chain_pep']}_{entry['chain_pro']}":
                self._format_pep_entry(f"{entry['PDB']}_{entry['chain_pep']}")
                for entry in self.records
            }
        return self._pep2emb

    @property
    def smiles2smile(self) -> Dict[str, Any]:
        if self._smiles2smile is None:
            self._smiles2smile = {
                f"{entry['PDB']}_{entry['chain_pep']}_{entry['chain_pro']}":
                self._format_smiles_entry(entry, use_given_smile=False)
                for entry in self.records
            }
        return self._smiles2smile

    @property
    def pdb_graph_db(self) -> Any:
        if self._pdb_graph_db is None:
            self._pdb_graph_db = pdb_to_graphs(self.prot2pdb, self.prot_featurize_params)
        return self._pdb_graph_db

    def build_data(self, df: Any, onthefly: bool = False) -> Any:
        data_list = []
        for entry in tqdm(df.to_dict('records'), desc="Building data"):
            both_id = f"{entry['PDB']}_{entry['chain_pep']}_{entry['chain_pro']}"
            prof = self.prot2pdb[both_id] if onthefly else self.pdb_graph_db[entry['chain_pro']]
            pepf = self.pep2emb[both_id] if onthefly else None
            druf = self.smiles2smile[both_id] if onthefly else None
            data_list.append({
                'prot': prof,
                'pept': pepf,
                'smiles': druf,
                'prot_name': f"{entry['PDB']}_{entry['chain_pro']}",
                'pept_name': f"{entry['PDB']}_{entry['chain_pep']}",
            })

        prot_featurize_fn = partial(featurize_protein_graph, **self.prot_featurize_params) if onthefly else None
        pep_featurize_fn = partial(featurize_peptide_graph) if onthefly else None
        smiles_featurize_fn = partial(featurize_smiles_graph) if onthefly else None

        return FPA(df=df, data_list=data_list, onthefly=onthefly,
                   prot_featurize_fn=prot_featurize_fn,
                   pep_featurize_fn=pep_featurize_fn,
                   smiles_featurize_fn=smiles_featurize_fn)

    def get_split(self, df: Optional[Any] = None,
                  split_method: Optional[str] = None,
                  onthefly: Optional[bool] = None,
                  return_df: bool = False) -> Any:
        df = df or self.df
        split_method = split_method or self.split_method
        onthefly = onthefly or self.onthefly
        if split_method == 'test':
            split_df = create_test_fold(self.df)
            split_data = {split: self.build_data(df_split, onthefly=onthefly) for split, df_split in split_df.items()}
            return (split_data, split_df) if return_df else split_data
        else:
            raise ValueError(f"Unknown split method: {split_method}")

class Batch_Prediction(FPATask):
    def __init__(self, data_path='./utils/preprocess/all_data.tsv',
                 esm_path='./receptor_data/esm',
                 coord_json='./receptor_data/coordinates.json',
                 epitope_path='./receptor_data/rec_interface.json',
                 pep_path='./utils/preprocess/pep_seq_smiles.json',
                 rec_path='./receptor_data/mod_rec_seq.json',
                 ids_json='./utils/preprocess/ids.json',
                 pep_esm='./utils/preprocess/esm_pep',
                 num_pos_emb=16, num_rbf=16,
                 contact_cutoff=8.0,
                 split_method='test',
                 onthefly=True):
        df = pd.read_table(data_path)
        coord_data = json.load(open(coord_json, 'r'))
        epitope_data = json.load(open(epitope_path, 'r'))
        pep_data = json.load(open(pep_path, 'r'))
        rec_data = json.load(open(rec_path, 'r'))
        ids_data = json.load(open(ids_json, 'r'))

        os.makedirs(esm_path, exist_ok=True)
        os.makedirs(pep_esm, exist_ok=True)

        super().__init__(task_name='Batch_Prediction',
                         df=df,
                         esm_dir=esm_path,
                         coord_data=coord_data,
                         epitope_data=epitope_data,
                         pep_seq=pep_data,
                         rec_seq=rec_data,
                         ids_data=ids_data,
                         pep_esm=pep_esm,
                         num_pos_emb=num_pos_emb, num_rbf=num_rbf,
                         contact_cutoff=contact_cutoff,
                         split_method=split_method,
                         onthefly=onthefly)

class Single_Prediction(SPATask):
    def __init__(self, protein_id='7lll_R',
                 peptide_seq='AELAELAEL',
                 esm_pro_path='./receptor_data/esm',
                 esm_pep_path='./peptide_data/esm',
                 coord_json='./receptor_data/coordinates.json',
                 epitope_path='./receptor_data/rec_interface.json',
                 rec_path='./receptor_data/mod_rec_seq.json',
                 num_pos_emb=16, num_rbf=16,
                 contact_cutoff=8.0,
                 split_method='test'):
        coord_data = json.load(open(coord_json, 'r'))
        epitope_data = json.load(open(epitope_path, 'r'))
        rec_data = json.load(open(rec_path, 'r'))

        Path(esm_pep_path).mkdir(parents=True, exist_ok=True)
        Path(esm_pro_path).mkdir(parents=True, exist_ok=True)

        super().__init__(task_name='Single_Prediction',
                         protein_id=protein_id,
                         peptide_seq=peptide_seq,
                         esm_pro_path=esm_pro_path,
                         esm_pep_path=esm_pep_path,
                         coord_data=coord_data,
                         epitope_data=epitope_data,
                         rec_seq=rec_data,
                         num_pos_emb=num_pos_emb, num_rbf=num_rbf,
                         contact_cutoff=contact_cutoff,
                         split_method=split_method)