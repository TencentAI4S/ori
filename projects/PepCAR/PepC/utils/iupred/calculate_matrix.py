import os
import numpy as np
import tempfile
import argparse

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

def fasta_2_matrix(iupred_data_dir):

    long_output = os.popen(f'python iupred2a.py  {iupred_data_dir} long').read()
    short_output = os.popen(f'python iupred2a.py  {iupred_data_dir} short').read()
    anchor_output = os.popen(f'python iupred2a.py -a {iupred_data_dir}  long').read()

    long_scores = parse_iupred_output(long_output)
    short_scores = parse_iupred_output(short_output)
    anchor_scores = parse_anchor2_output(anchor_output)

    scores_matrix = np.column_stack((long_scores, short_scores, anchor_scores))

    return scores_matrix

def batch_process(input_folder, output_folder):
    fasta_files = [f for f in os.listdir(input_folder) if f.endswith('.fasta')]

    for fasta_file in fasta_files:
        matrix = fasta_2_matrix(os.path.join(input_folder, fasta_file))

        output_file = os.path.join(output_folder, os.path.splitext(fasta_file)[0] + '.txt')
        np.savetxt(output_file, matrix)

def main(input_folder, output_folder):
    try:
        os.mkdir(output_folder)
    except:
        pass
    batch_process(input_folder, output_folder)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process fasta files with IUPred2A and ANCHOR2")
    parser.add_argument("input_folder", help="Path to the input folder containing fasta files")
    parser.add_argument("output_folder", help="Path to the output folder where results will be saved")

    args = parser.parse_args()

    main(args.input_folder, args.output_folder)