from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
from PIL import Image

import pubchempy as pcp

smiles = ["CC(CC1=CC2=C(C=C1)OCO2)NC"]
print(smiles)

def main(smile):
    m = Chem.MolFromSmiles(smile)
    Draw.MolToFile(m, f'skeletons\{smile}.png')
    # Show image
    img = Image.open(f'skeletons\{smile}.png')
    img.show()

if __name__ == "__main__":
    for smile in smiles:
        main(smile)