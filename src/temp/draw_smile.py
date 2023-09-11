from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
from PIL import Image

import pubchempy as pcp

smiles = ["Cn1c(=O)c(=O)n(C)c2cc(Oc3ccccc3)c(NC(=O)c3ccccc3F)cc21"]
print(smiles)

def main(smile):
    m = Chem.MolFromSmiles(smile)
    Draw.MolToFile(m, f'skeletons\{smile}.png')

if __name__ == "__main__":
    for smile in smiles:
        main(smile)