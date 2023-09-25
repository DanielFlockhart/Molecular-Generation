import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
from PIL import Image
import math

from rdkit.Chem import Draw
from rdkit import Chem

class Report:
    def __init__(self, starting_mol, generated_mols):
        self.starting_mol = self.img_from_SMILE(starting_mol)
        self.generated_mols = generated_mols

    def img_from_SMILE(self,smile):
        '''
        Shows a molecule from a SMILE string.
        '''
        m = Chem.MolFromSmiles(smile)
        Draw.MolToFile(m, f'start_molecule.png')
        return Image.open(f'start_molecule.png')

    def build_report(self):
        # Calculate the number of generated molecules
        num_generated = len(self.generated_mols)

        # Calculate the number of rows and columns for the grid
        num_rows = int(math.sqrt(num_generated))
        num_cols = int(math.sqrt(num_generated))

        # Create a subplot grid
        fig, axes = plt.subplots(num_rows+1, num_cols, figsize=(24, 16))
        fig.set_facecolor("white")  # Set the figure background color to white

        axes[0, 0].imshow(self.starting_mol)
        axes[0, 0].set_title("Starting Molecule")
        axes[0, 0].axis('off')
        

        # Flatten the axes array if necessary
        #axes = axes.flatten()

        # Display generated molecules in the rest of the subplots
        #axes[].set_title("Generated Molecules")
        for img in self.generated_mols:
            img.show()
        
        for i in range(num_rows):
            for j in range(1,num_cols+1):
                axes[j,i].imshow(self.generated_mols[(j-1)*num_rows+i], cmap='gray')
                
            
        axes = axes.flatten()
        for ax in axes:
            ax.axis('off')

        plt.tight_layout()

        # Display the plot
        plt.show()
