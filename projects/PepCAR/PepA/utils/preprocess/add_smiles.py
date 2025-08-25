import csv
import concurrent.futures

from rdkit import Chem
from rdkit.Chem import Draw
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

def process_row(row):
    sequence = row[1].replace('X', '')
    smiles = pep_to_smile(sequence)
    return row + [smiles]

# Read pep_seq.tsv
with open('pep_seq.tsv', 'r') as input_file:
    reader = csv.reader(input_file, delimiter='\t')
    data = list(reader)

# Convert sequences to SMILES and write to a new tsv file
with open('pep_seq_smiles.tsv', 'w') as output_file:
    writer = csv.writer(output_file, delimiter='\t')
    with concurrent.futures.ProcessPoolExecutor() as executor:
        for row in executor.map(process_row, data):
            writer.writerow(row)