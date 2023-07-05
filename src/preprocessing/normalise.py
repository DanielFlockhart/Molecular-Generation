# import numpy as np
# from rdkit import Chem
# from rdkit.Chem import Draw
# from PIL import Image
# import random
# def generate_molecule_image(smiles):
#     mol = Chem.MolFromSmiles(smiles)

#     # Set the desired bond length
#     bond_length = random.randint(10,50) # Adjust this value as needed

#     # Modify the bond lengths in the molecule
#     for bond in mol.GetBonds():
#         bond.SetBondType(Chem.rdchem.BondType.SINGLE)
#         bond.SetBondDir(Chem.rdchem.BondDir.NONE)  # Clear bond direction

#     # Generate the 2D depiction of the molecule
#     image = Draw.MolToImage(mol, size=(128, 128), fitImage=False, wedgeBonds=False, useSVG=False)

#     # Calculate the scaling factor based on bond length
#     scaling_factor = bond_length / max(image.size)

#     # Resize the image while maintaining the aspect ratio
#     #new_size = tuple(int(dim * scaling_factor) for dim in image.size)
#     #image = image.resize(new_size)

#     # Create a blank white background image of 128x128 pixels
#     background = Image.new('RGB', (128, 128), (255, 255, 255))

#     # Paste the molecule image onto the center of the background
#     offset = tuple((bg_dim - img_dim) // 2 for bg_dim, img_dim in zip(background.size, image.size))
#     background.paste(image, offset)

#     # Return the image object
#     return background

# # Example usage
# smiles = ["CC(=O)OC1=CC=CC=C1C(=O)O", "c1ccccc1"]
# max_size = 128
# for smile in smiles:
#     image = generate_molecule_image(smile)
#     image.save(f"{smile}.png")


import numpy as np
from rdkit import Chem
from rdkit.Chem import Draw
from PIL import Image

def color_specific_atoms(mol, atom_indices, color):
    for atom_idx in atom_indices:
        atom = mol.GetAtomWithIdx(atom_idx)
        atom.SetProp("atomColor", color)

def generate_molecule_image(smiles, highlight_atoms, highlight_color):
    mol = Chem.MolFromSmiles(smiles)

    # Set the desired bond length and highlight color
    bond_length = 30  # Adjust this value as needed

    # Modify the bond lengths in the molecule
    for bond in mol.GetBonds():
        bond.SetBondType(Chem.rdchem.BondType.SINGLE)
        bond.SetBondDir(Chem.rdchem.BondDir.NONE)  # Clear bond direction

    # Color the specific atoms
    color_specific_atoms(mol, highlight_atoms, highlight_color)

    # Generate the 2D depiction of the molecule
    image = Draw.MolToImage(mol, size=(128, 128), fitImage=False, wedgeBonds=False, useSVG=False)

    # Calculate the scaling factor based on bond length
    scaling_factor = bond_length / max(image.size)

    # Resize the image while maintaining the aspect ratio
    new_size = tuple(int(dim * scaling_factor) for dim in image.size)
    image = image.resize(new_size)

    # Create a blank white background image of 128x128 pixels
    background = Image.new('RGB', (128, 128), (255, 255, 255))

    # Paste the molecule image onto the center of the background
    offset = tuple((bg_dim - img_dim) // 2 for bg_dim, img_dim in zip(background.size, image.size))
    background.paste(image, offset)

    # Return the image object
    return background

# Example usage
smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"
highlight_atoms = [2, 4, 6]  # Indices of atoms to highlight
highlight_color = (255, 0, 0)  # RGB color for highlighting (red in this case)

image = generate_molecule_image(smiles, highlight_atoms, highlight_color)
image.save("highlighted_molecule.png")








'''
Current Ideas:
- Keep looking for library built-in methods

- If not, set scaling factor to smallest scale to fit largest molecule in. 
    - If that scale is too small, then increase the size of the image or just don't include larger molecules.
    - Then, resize all molecules to that scale by getting the length of carbon bonds in the molecule and approximating the bond length to the scale.
'''