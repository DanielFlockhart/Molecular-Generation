from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
from PIL import Image

import pubchempy as pcp

smiles = ['NC' * c for c in range(1,100)]
print(smiles)

def main(smile):
    m = Chem.MolFromSmiles(smile)
    Draw.MolToFile(m, f'skeletons\{smile}.png')

if __name__ == "__main__":
    for smile in smiles:
        main(smile)