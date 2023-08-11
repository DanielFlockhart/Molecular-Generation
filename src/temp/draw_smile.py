from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
from PIL import Image

import pubchempy as pcp

smiles = ['O=N(=O)c1ccc(cc1)C1=Nc2c3ccccc3c3ccccc3c2N1c1ccccc1',
 'O=S1(=O)N=C(c2ccc3ccccc3c2)C(=N1)c1ccc2ccccc2c1',
 'O=N(=O)c1ccc(C=Cc2ccc(cc2)N2c3ccccc3c3ccccc23)cc1',
 'O=N(=O)c1ccc(cc1)C(=Cc1ccc2N(c3ccccc3)c3ccccc3c2c1)C#N',
 'O=N(=O)c1ccc(cc1)N=Nc1ccc(OCCCCN2c3ccccc3c3cc(ccc23)C#C)cc1',
 'CC(=O)c1ccc(Sc2ccc(cc2)N=Cc2c(O)ccc3ccccc23)cc1',
 'O=N(=O)c1ccc(cc1)C1=NOC(CN2c3ccccc3C=Cc3ccccc23)C1',
 'COc1ccc(CN2C(=O)C(=NP(=S)(c3ccccc3)c3ccccc3)c3cc(Cl)ccc23)cc1',
 'O=N(=O)c1ccc(NC(=S)N2N=C(CC2c2c3ccccc3nc3ccccc23)c2ccccc2)cc1']

def main(smile):
    m = Chem.MolFromSmiles(smile)
    Draw.MolToFile(m, f'skeletons\{smile}.png')

if __name__ == "__main__":
    for smile in smiles:
        main(smile)