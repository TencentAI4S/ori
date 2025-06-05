import os

os.makedirs('fasta', exist_ok=True)
with open('pep_seq.tsv', 'r') as f:
    pep_seq_lines = f.read().splitlines()

for line in pep_seq_lines:
    file_name, sequence = line.split('\t')
    with open(f'./fasta/{file_name}.fasta', 'w') as fasta_file:
        fasta_file.write(f'>{file_name}\n{sequence}\n')