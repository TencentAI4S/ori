from rdkit import Chem
from rdkit.Chem import Draw

import argparse
from rdkit import Chem, RDLogger
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

#IPythonConsole.ipython_useSVG = True

def pep_to_smiles(pep):
    peptide_smiles = Chem.MolToSmiles(Chem.MolFromFASTA(pep))
    mc = MolClean()
    clean_smiles = mc.clean(peptide_smiles)
    return clean_smiles