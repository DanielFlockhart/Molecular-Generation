
from PIL import Image
import numpy as np
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import IPythonConsole, rdDepictor, rdMolDraw2D
from PIL import Image
from io import BytesIO
import random,glob,os
from database import *



def black_and_white(folder,threshold=245):
    # Iterate over each file in the folder
    for file_name in os.listdir(folder):
        if file_name.endswith('.png'):
            # Load the image
            image_path = os.path.join(folder, file_name)
            image = Image.open(image_path)

            # Convert the image to grayscale
            image = image.convert('L')

            # Convert the grayscale image to binary black and white
            image = image.point(lambda x: 0 if x < threshold else 255, '1')

            # Save the converted image (overwrite the original)
            image.save(image_path)

            # Close the image file
            image.close()


def generate_normalised_image(mol):
    # make sure the mol coordinates are normalized to the scale RDKit uses internally
    rdDepictor.NormalizeDepiction(mol)
    # -1, -1 means flexicanvas: the canvas will be as large as needed to display the molecule (no scaling)
    drawer = rdMolDraw2D.MolDraw2DCairo(-1, -1)
    opts = rdMolDraw2D.MolDrawOptions()
    drawer.SetDrawOptions(opts)
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    with BytesIO(drawer.GetDrawingText()) as hnd:
        with Image.open(hnd) as image:
            image.load()
    return image

def scale(folder,scale_factor,size):

    # Iterate through every image in the folder and scale

    # Use the glob module to get a list of image files in the folder
    image_files = glob.glob(os.path.join(folder, '*.png'))  # Change the file extension to match your image types

    for skeleton in image_files:
        smile = os.path.splitext(os.path.basename(skeleton))[0]
        # Process each image file
        with Image.open(skeleton) as img:
            # Get new scaled image dimensions
            new_width = img.width * scale_factor
            new_height = img.height * scale_factor
            img = img.resize((new_width, new_height))  # Resize the image to the new dimensions
        # Create a blank white background
        background = Image.new('RGB', (size,size), (255, 255, 255))

        # Paste the molecule image onto the center of the background - Still working out if this is necessary
        #offset = tuple((bg_dim - img_dim) // 2 for bg_dim, img_dim in zip(background.size, img.size))
        background.paste(img)
        background.show()
        background.save(fr'{folder}\{smile}.png')


def normalise_images(smiles,size,folder):
    max_width = 0
    max_height = 0
    for smile in smiles:
        mol = Chem.MolFromSmiles(smile)
        rdDepictor.Compute2DCoords(mol)
        img = generate_normalised_image(mol)
        if img.width > max_width: max_width = img.width
        if img.height > max_height: max_height = img.height
        img.save(fr'{folder}\{smile}.png')
        # Get the maximum value out of the two maximum measurements, as only that value needs to be scaled to.

    max_length = max(max_height,max_width)
    print(max_length)
    scale_factor = size / max_length
    scale(folder,scale_factor,size)
    
# Now that scaling works
# Get the largst image required, set that as default scaling, update all the others.
data_folder = r'C:\Users\0xdan\Documents\CS\WorkCareer\Chemistry Internship\Project-Code\data\dataset'
if __name__ == "__main__":
    db = Database(fr'{data_folder}\CSD_EES_DB.csv')
    molecules = fr'{data_folder}\test-data'
    size = 400
    normalise_images(molecules,size,data_folder)


# Find a way to simplify the graphics down.