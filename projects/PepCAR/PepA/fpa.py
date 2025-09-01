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

from amnet.pdb_graph import pdb_to_graphs, featurize_protein_graph, featurize_hla_graph
from amnet.pep_graph import featurize_peptide_graph
from smnet.smiles_graph import featurize_smiles_graph
from smnet.pep_to_smiles import pep_to_smiles
from utils.iupred.calculate_matrix_single import fasta_2_matrix

class SPATask:
    def __init__(self, task_name: Optional[str] = None,
                 peptide_seq: Optional[str] = None, 
                 esm_pep_path: Optional[str] = None,
                 split_method: str = 'test'):
        self.task_name = task_name
        self.peptide_seq = peptide_seq
        self.esm_pep_path = esm_pep_path

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
        return self.format_peptide(), self.format_smiles()

    def extract_peptide_embed(self, seq: str, output_path: Path):
        device = 'cuda:0'
        esm_tokenizer = AutoTokenizer.from_pretrained('../model_weights/ESM-Pep/models--facebook--esm2_t33_650M_UR50D')
        esm_pep = EsmModel.from_pretrained("../model_weights/ESM-Pep/models--facebook--esm2_t33_650M_UR50D").to(device)
        encoded_input = esm_tokenizer(seq, return_tensors='pt', padding=False, truncation=False, max_length=5120).to(device)

        with torch.no_grad():
            output = esm_pep(**encoded_input)
            embeddings = output.last_hidden_state.squeeze(dim=0)[1:-1]

        torch.save(embeddings.cpu(), output_path)

class Single_Prediction(SPATask):
    def __init__(self, 
                 peptide_seq='AELAELAEL',
                 esm_pep_path='./peptide_data/esm',
                 split_method='test'):

        Path(esm_pep_path).mkdir(parents=True, exist_ok=True)

        super().__init__(task_name='Single_Prediction',
                         peptide_seq=peptide_seq,
                         esm_pep_path=esm_pep_path,
                         split_method=split_method)