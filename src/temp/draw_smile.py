from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
from PIL import Image

import pubchempy as pcp

smiles = ["CC(C)(C)c1ccc2occ(CC(=O)Nc3ccccc3F)c2c1",
"Cc1ccc(CNC(=O)c2ccccc2NC(=O)[C@@H]2CC(=O)N(c3ccc(C)cc3)C2)cc1",
"CCc1ccc(-c2nc(C(=O)N3CCO[C@H](CC)C3)cs2)cc1",]
print(smiles)

def main(smile):
    m = Chem.MolFromSmiles(smile)
    Draw.MolToFile(m, f'skeletons\{smile}.png')

if __name__ == "__main__":
    for smile in smiles:
        main(smile)