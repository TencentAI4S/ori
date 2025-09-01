import torch
from transformers import EsmModel
from transformers import AutoTokenizer


device = 'cuda:0'

esm_tokenizer = AutoTokenizer.from_pretrained('../model_weights/ESM-2/models--facebook--esm2_t33_650M_UR50D')
esm2 = EsmModel.from_pretrained("../model_weights/ESM-2/models--facebook--esm2_t33_650M_UR50D").to(device)

def get_protein_embedding(sequence):
    encoded_input = esm_tokenizer(sequence, return_tensors='pt', padding=False, truncation=False, max_length=5120).to(device)

    with torch.no_grad():
        output = esm2(**encoded_input)
        embeddings = output.last_hidden_state

    embeddings = embeddings.squeeze(dim=0)

    embeddings = embeddings[1:-1]

    return embeddings.cpu()

def main():
    sequence = "YOUR_SEQUENCE_HERE"
    embeddings = get_protein_embedding(sequence)
    print(embeddings)

if __name__ == "__main__":
    main()