import os
import numpy as np
import tempfile

def parse_iupred_output(output):
    scores = []
    for line in output.split('\n'):
        if line.startswith('#') or not line.strip():
            continue
        parts = line.split()
        scores.append(float(parts[2]))
    return np.array(scores)

def parse_anchor2_output(output):
    scores = []
    for line in output.split('\n'):
        if line.startswith('#') or not line.strip():
            continue
        parts = line.split()
        scores.append(float(parts[3]))
    return np.array(scores)

def fasta_2_matrix(sequence):
    with tempfile.NamedTemporaryFile(suffix=".fasta", delete=True) as temp:
        temp.write(f">temp\n{sequence}".encode())
        temp.flush()

        long_output = os.popen(f'python utils/iupred/iupred2a.py  {temp.name} long').read()
        short_output = os.popen(f'python utils/iupred/iupred2a.py  {temp.name} short').read()
        anchor_output = os.popen(f'python utils/iupred/iupred2a.py -a {temp.name}  long').read()

        long_scores = parse_iupred_output(long_output)
        short_scores = parse_iupred_output(short_output)
        anchor_scores = parse_anchor2_output(anchor_output)

        scores_matrix = np.column_stack((long_scores, short_scores, anchor_scores))

    return scores_matrix

if __name__=="__main__":
    sequence = "ATCGATCGATCGATCGATCG"
    matrix = fasta_2_matrix(sequence)
    print(matrix.shape)
