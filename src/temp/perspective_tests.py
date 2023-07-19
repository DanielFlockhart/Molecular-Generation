from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import rdMolDraw2D

# Define the SMILES string
smiles = 'CSC1=CC2=C(S1)C1=C(C=C(S1)C1=CC3=C(S1)C1=C(C=C(SC)S1)[Si]3(c1ccccc1)c1ccccc1)[Si]2(c1ccccc1)c1ccccc1'

# Convert SMILES to a molecule object
mol = Chem.MolFromSmiles(smiles)

# Generate conformers
num_conformers = 10
AllChem.EmbedMultipleConfs(mol, numConfs=num_conformers)

# Access the conformers
conformers = mol.GetConformers()

# Iterate over the conformers and draw them
for idx, conf in enumerate(conformers):
    print(f"Conformer {idx+1}")
    drawer = rdMolDraw2D.MolDraw2DCairo(400, 400)  # Create a new drawing object for each conformer
    drawer.DrawMolecule(mol, confId=idx)
    drawer.FinishDrawing()
    drawer.WriteDrawingText(f"conformer_{idx+1}.png")








