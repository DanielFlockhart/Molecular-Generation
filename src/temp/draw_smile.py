from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
from PIL import Image

import pubchempy as pcp
def main(smile):
    m = Chem.MolFromSmiles(smile)
    Draw.MolToFile(m, f'{smile}.png')
    img = Image.open(f'{smile}.png')
    img.show()
    
if __name__ == "__main__":
   main("CC(=O)Oc1ccccc1C(=O)O")