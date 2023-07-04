from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
from PIL import Image

import pubchempy as pcp
import os
import sys
import io
sys.path.insert(0, os.path.abspath('..'))
from utils import *

# Given a canonicle smile, a 2D structure can be generated using the rdkit library.

class Structure:
    def __init__(self,smile):
        self.smile = smile
        self.molecule = self.build_structure()
        self.name = self.get_name()
        self.data_folder = os.path.join(get_directory(),"data")
        self.structures = os.path.join(self.data_folder,"2D-Structures")
        self.structure_3D = os.path.join(self.data_folder,"3D-Structures")
        
    
    def build_structure(self):
        """
        Builds a 2D structure from a smile
        """
        m = Chem.MolFromSmiles(self.smile)
        return m
    
    def save_structure(self):
        """
        Saves the 2D structure as a png to the data folder
        """
        if self.molecule is not None:
            Draw.MolToFile(self.molecule, self.structures+fr"\{self.name}.png")

    def get_name(self):
        """
         Gets the name of the chemical from the smile
        """
        try:
            choice = pcp.get_compounds(self.smile, 'smiles')[0].synonyms[0]
        except:
            choice = "Unknown"
        return choice
        

    def display_structure(self):
        """
        Displays the 2D structure using PIL
        """
        img = Image.open(self.structures+fr"\{self.name}.png")
        img.show()

    def to_3D_structure(self):
        """
        Converts the 2D structure to a 3D structure using the rdkit library
        """

        raise NotImplementedError("This function is not yet implemented")
        # mol = Chem.AddHs(self.molecule)
        # Chem.AllChem.EmbedMolecule(mol)
        # # Create a 2D image of the 3D structure
        # drawer = Draw.rdMolDraw2D.MolDraw2DCairo(400, 400)  # Adjust the image size as needed
        # drawer.DrawMolecule(mol)
        # drawer.FinishDrawing()

        # # Get the image as a PIL object
        # image_data = drawer.GetDrawingText()
        # image = Image.open(io.BytesIO(image_data))

        # # Display or save the image
        # image.show()
    
if __name__ == "__main__":
    b= Structure("CC(=O)Oc1ccccc1C(=O)O")
    b.save_structure()
    b.display_structure()
    b.to_3D_structure()