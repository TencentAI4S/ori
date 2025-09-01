import json
import os
import torch
from transformers import EsmModel, AutoTokenizer
from tqdm import tqdm

device = 'cuda:0'
esm_tokenizer = AutoTokenizer.from_pretrained('../../../model_weights/ESM-2/models--facebook--esm2_t33_650M_UR50D')
esm2 = EsmModel.from_pretrained("../../../model_weights/ESM-2/models--facebook--esm2_t33_650M_UR50D").to(device)

def load_sequences(json_file):
    """Load amino acid sequences from a JSON file."""
    with open(json_file, 'r') as f:
        sequences = json.load(f)
    return sequences

def process_sequences(input_file, sequences, output_folder):
    """Process sequences and save embeddings."""
    with open(input_file, "r") as f:
        ids = [line.strip() for line in f.readlines()]

    for id in tqdm(ids, desc="Processing sequences"):
        if id in sequences:
            sequence = sequences[id]
            pt_file = os.path.join(output_folder, f"{id}.pt")

            encoded_input = esm_tokenizer(sequence, return_tensors='pt', padding=False, truncation=False, max_length=5120).to(device)

            with torch.no_grad():
                output = esm2(**encoded_input)
                embeddings = output.last_hidden_state

            embeddings = embeddings.squeeze(dim=0)[1:-1]  # Remove padding tokens

            torch.save(embeddings.cpu(), pt_file)
        else:
            print(f"Warning: ID {id} not found in sequences.")

def main():
    input_file = "target.txt"  # Input file containing IDs
    json_file = "../../receptor_data/mod_rec_seq.json"  # JSON file containing sequences
    output_folder = "../../receptor_data/esm"
    os.makedirs(output_folder, exist_ok=True)

    sequences = load_sequences(json_file)  # Load sequences from JSON
    process_sequences(input_file, sequences, output_folder)

if __name__ == "__main__":
    main()