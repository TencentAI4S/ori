with open('target.txt', 'r') as f:
    target_lines = f.read().splitlines()

with open('peptide.txt', 'r') as f:
    peptide_lines = f.read().splitlines()

with open('all_data.tsv', 'w') as all_data, open('pep_seq.tsv', 'w') as pep_seq:
    all_data.write('PDB\tchain_pep\tchain_pro\ty_ddg\tpro_name\tpep_name\tgiven_smiles\ty\tpep_seq\n')
    for target_line in target_lines:
        PDB, chainpro = target_line.split('_')
        for i, peptide_line in enumerate(peptide_lines):
            all_data.write(f'{PDB}\t{i}\t{chainpro}\t0\t{PDB}_{chainpro}\t{PDB}_{i}\t\t0\t{peptide_line}\n')
            pep_seq.write(f'{PDB}_{i}\t{peptide_line}\n')