import json
import os
import time
import torch
from Bio import SeqIO
from transformers import EsmModel
from transformers import AutoTokenizer
from tqdm import tqdm
device = 'cuda:0'
esm_tokenizer = AutoTokenizer.from_pretrained('../../../model_weights/ESM-Pep/models--facebook--esm2_t33_650M_UR50D')
esm2 = EsmModel.from_pretrained("../../../model_weights/ESM-Pep/models--facebook--esm2_t33_650M_UR50D").to(device)

def process_sequences(input_file, output_folder, start_line=0, end_line=None):
    with open(input_file, "r") as f:
        lines = f.readlines()[start_line:end_line]

    for line in tqdm(lines, desc="Processing sequences"):
        id, sequence = line.strip().split("\t")
        pt_file = os.path.join(output_folder, id + ".pt")

        encoded_input = esm_tokenizer(sequence, return_tensors='pt', padding=False, truncation=False, max_length=5120).to(device)

        with torch.no_grad():
            output = esm2(**encoded_input)
            embeddings = output.last_hidden_state

        embeddings = embeddings.squeeze(dim=0)

        embeddings = embeddings[1:-1]

        torch.save(embeddings.cpu(), pt_file)

def main():
    input_file = "pep_seq.tsv"
    output_folder = "./esm_pep"
    os.makedirs(output_folder, exist_ok=True)
    process_sequences(input_file, output_folder)
if __name__ == "__main__":
    main()