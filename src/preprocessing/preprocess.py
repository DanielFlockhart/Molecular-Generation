import rdkit
from rdkit import RDLogger

# Disable the RDKit logger
RDLogger.DisableLog('rdApp.error')

from PIL import Image
import numpy as np
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import IPythonConsole, rdDepictor, rdMolDraw2D
from PIL import Image
from io import BytesIO
import random,glob,os
from preprocessing.database import *



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

    drawer.DrawMolecule(mol) # Error Is Here - Causing the program to crash on some smiles without output error

    

    drawer.FinishDrawing()
    with BytesIO(drawer.GetDrawingText()) as hnd:
        with Image.open(hnd) as image:
            image.load()
    return image
def scale(folder,scale_factor,size):

    # Iterate through every image in the folder and scale

    # Use the glob module to get a list of image files in the folder
    image_files = glob.glob(os.path.join(folder, '*.png'))  # Change the file extension to match your image types

    for (i,skeleton) in enumerate(image_files):
        smile = os.path.splitext(os.path.basename(skeleton))[0]
        # Process each image file
        with Image.open(skeleton) as img:
            # Get new scaled image dimensions
            new_width = int(img.width * scale_factor)
            new_height = int(img.height * scale_factor)
            img = img.resize((new_width, new_height))  # Resize the image to the new dimensions

        # Create a blank white background
        background = Image.new('RGB', (size,size), (255, 255, 255))

        # Paste the molecule image onto the center of the background
        offset = tuple((bg_dim - img_dim) // 2 for bg_dim, img_dim in zip(background.size, img.size))
        background.paste(img,offset)
        #background.show()
        background.save(fr'{folder}\{smile}.png')
        if (i%1000) == 0:
            print(f"{i}/{len(image_files)} Processed")

def truncate_smile(passed_smile,chars):
    chars = min(chars,250) 
    # Truncate the smile to the specified number of characters as windows 11 has a file length limit of 255 characters
    if len(passed_smile) > chars:
        return passed_smile[:chars]
    else:
        return passed_smile

def normalise_images(smiles,size,folder):
    sizes = []
    uncounted = 0
    for (i,smile) in enumerate(smiles):

        try:
            # Check if the smile is valid by removing any % characters
            #smile = smile.replace("%","") Temp Fix - next try -> remove any non ascii
            mol = Chem.MolFromSmiles(smile)

            rdDepictor.Compute2DCoords(mol)
            img = generate_normalised_image(mol)
            sizes.append(img.width)
            sizes.append(img.height)
            
            
            smile = truncate_smile(smile,200)
            img.save(fr'{folder}\{smile}.png')
        except Exception as e:
            uncounted += 1
            #print(f"Error processing smile: {smile}")
            #print(f"Exception message: {str(e)}")
            pass
        if (i%1000) == 0:
            print(f"{i}/{len(smiles)} Processed For Scaling")
        # Get the maximum value out of the two maximum measurements, as only that value needs to be scaled to.
    # Possible way of angling molecules to decrease max_length?
    print("Uncounted: ",uncounted)
    # Calculate the upper bound of the sizes
    sizes = np.sort(sizes)
    bound = calculate_upper_bounds(sizes)
    print(bound)
    plot_size_distribution(sizes)
    
    scale_factor = size / bound
    scale(folder,scale_factor,size)
    #black_and_white(folder) # -> this effects the clarity

def calculate_upper_bounds(numbers,std_dev=1.6):
    '''
    Gets the standard deviation of the numbers and returns the upper bound of the numbers
    '''
    # Calculate the standard deviation of the numbers
    std = np.std(numbers)
    # Calculate the mean of the numbers
    mean = np.mean(numbers)
    # Calculate the upper bound of the numbers
    upper_bound = mean + (std * std_dev)
    # This gets the upper bound of the numbers which is the percentile of the number
    return upper_bound

def plot_size_distribution(sizes):
    # Using matplot lib to plot the distribution of the sizes
    import matplotlib.pyplot as plt
    plt.hist(sizes, bins=100)
    plt.gca().set(title='Size Distribution of Molecules', ylabel='Frequency')
    plt.show()

def clear_folder(folder):
    image_files = glob.glob(os.path.join(folder, '*.png'))  # Change the file extension to match your image types
    for image_file in image_files:
        os.remove(image_file)



# Find a way to simplify the graphics down.
# Add more error handling for the ones that can't be processed



'''

Preprocessing Pipeline:
- Load the smiles from the database
- Normalise the smiles to the same length
- Get the maximum dimensions of the smiles
    - Calculate the average dimensions of the smiles
    - Work out an upper bound for the dimensions so the 98% of smiles will fit into the images. This accounts for outliers.
    - Choose a max dimension size to include the majority of smiles
    - Add some checks for the outliers
    - Calculate the scale factor
- Scale the smiles to the same size
- Convert the smiles to black and white
- Save the smiles to the folder


'''