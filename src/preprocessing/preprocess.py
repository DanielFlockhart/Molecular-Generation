from tqdm import tqdm
from rdkit import RDLogger
import json

# Disable the RDKit logger
RDLogger.DisableLog('rdApp.error')

import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from rdkit import Chem
from rdkit.Chem.Draw import rdDepictor, rdMolDraw2D
from PIL import Image
from io import BytesIO
import random,glob,os,sys,string
from preprocessing.database import *

sys.path.insert(0, os.path.abspath('..'))
from CONSTANTS import *
from ui.terminal_ui import *




class Preprocessor:
    def __init__(self,data_folder,database):
        '''
        Preprocessor class for preprocessing the smiles into preprocessed images

        Parameters
        ----------
        data_folder : str
            The path to the data folder

        database : Database
            The database object containing the smiles
        '''
        self.database = database
        self.dataset_name = ''.join(random.choices(string.ascii_uppercase + string.digits, k=10))
        self.final_folder = fr"{data_folder}\data\{self.dataset_name}"
        self.unscaled_folder = fr"{data_folder}\resized"
        self.smiles = self.database.get_smiles()

    def process(self,download=True):
        '''
        Preprocesses the smiles into preprocessed images
        
        -> Load the smiles from the database if requested
        -> Normalise the molecules so bonds and atoms are the same length
        -> Get the maximum dimensions of the normalised smiles
            -> Calculate the average dimensions of the smiles
            -> Work out an upper bound for the dimensions so most of smiles will fit into the images. This accounts for outliers.
            -> Calculate the scale factor
        -> Scale the smiles to the same size
        -> Convert the smiles to black and white
        -> Save the smiles to the folder

        '''

        # --- Redownload the images if Requested ---
        if download:
            print(format_title("Downloading Images"))
            self.clear_folder(self.unscaled_folder)
            self.create_skeletons()

        # --- Scale the images ---
        print(format_title("Scaling Images"))
        sizes = np.sort(self.get_skeleton_sizes())
        upper_bound = self.calculate_upper_bounds(sizes,STD_DEV)
        scale_factor = IMG_SIZE / upper_bound
        

        # Create the final folder
        os.mkdir(self.final_folder)

        # Rescale the images with new scale factor
        self.rescale_skeletons(scale_factor)

        print(f"Preprocessing Complete : {len(self.get_images(self.final_folder))}/{len(self.smiles)} smiles processed successfully")


        # --- Additional Processing ---
        #self.recolour() - Leave this for now
        self.save_dataset_info(sizes)

        # --- Visualise the dataset ---
        print(f"Upper Bound of Scaled Images : {upper_bound}")
        self.plot_size_distribution(sizes)
        

    def truncate_smile(self,smile):
        '''
        Truncate the smile to  fit into file name length limit

        Windows 11 has a file length limit of 255 characters
        '''
        if len(smile) > MAX_CHARS:
            return smile[:MAX_CHARS]
        else:
            return smile
    
    def plot_size_distribution(self, sizes):
        '''
        Plots the size distribution of the molecules for visualisation
        '''
        plt.hist(sizes, bins=100)
        plt.gca().set(title='Size Distribution of Molecules', ylabel='Frequency')
        plt.show()  
    
    def save_dataset_info(self,sizes):
        '''
        Saves dataset information to a json file

        Information Saved:
            -> Dataset Name
            -> Dataset Size
            -> Dataset Target Image Size
            -> Dataset Standard Deviation
            -> Dataset Unscaled Sizes
        '''
        dataset_info = {
            "dataset_NAME": self.dataset_name,
            "dataset_SIZE": len(self.smiles),
            "dataset_TARGET_IMG_SIZE": IMG_SIZE,
            "dataset_STD_DEV": STD_DEV,
            "dataset_UNCSCALED_SIZES": sizes.tolist()
        }
        with open(fr'{self.final_folder}\dataset_info.json', 'w') as outfile:
            json.dump(dataset_info, outfile)

    
    def clear_folder(self,folder):
        ''' 
        Clears Contents of a Specified folder
        '''
        image_files = self.get_images(folder)
        for image_file in image_files:
            os.remove(image_file)

    def get_images(self,folder):
        '''
        Returns a list of images in a folder
        '''
        return glob.glob(os.path.join(folder, '*.png'))
    
    def calculate_upper_bounds(self,array,std_dev):
        '''
        Gets the standard deviation of the numbers and returns the upper bound of the numbers
        '''

        # Calculate the standard deviation of the numbers
        std = np.std(array)
        # Calculate the mean of the numbers
        mean = np.mean(array)
        # Calculate the upper bound of the numbers
        upper_bound = mean + (std * std_dev)

        return upper_bound
    


    def create_skeletons(self):
        '''
        Creates the skeleton images from smiles
        '''

        for (i,smile) in tqdm(enumerate(self.smiles),total=len(self.smiles), bar_format=LOADING_BAR, ncols=80, colour='green'):
            try:
                # Create the skeleton image
                mol = Chem.MolFromSmiles(smile)
                # Calculate the 2D coordinates of the molecule
                rdDepictor.Compute2DCoords(mol)
                # Scale the skeleton image
                img = self.scale_skeleton(mol)
                # Truncate the smile to fit into file name length limit
                smile = self.truncate_smile(smile)
                img.save(fr'{self.unscaled_folder}\{smile}.png')

            except Exception as e:
                pass

    def scale_skeleton(self,mol):
        '''
        Normalises the molecule so bonds and atoms are the same length
        '''
    
        # Molecule cordinates are normalised so bonds and atoms are the same length
        rdDepictor.NormalizeDepiction(mol)

        # -1, -1 flexicanvas: the canvas will be as large as needed to display the molecule (no scaling)
        drawer = rdMolDraw2D.MolDraw2DCairo(-1, -1)
        opts = rdMolDraw2D.MolDrawOptions()
        drawer.SetDrawOptions(opts)
        drawer.DrawMolecule(mol) 
        drawer.FinishDrawing()

        with BytesIO(drawer.GetDrawingText()) as hnd:
            with Image.open(hnd) as image:
                image.load()

        return image
    
    def get_skeleton_sizes(self):
        '''
        Iterates through every image in the folder and returns a list of the skeleton sizes
        '''
        img_sizes = []
        for image in os.listdir(self.unscaled_folder):
            with Image.open(fr"{self.unscaled_folder}\{image}") as img:
                img_sizes.append(img.width)
                img_sizes.append(img.height)
        return img_sizes

    def rescale_skeletons(self,scale_factor):
        '''
        Rescales the skeleton images to the target image size without distorting the image

        This function standardises the data for the model
        '''
        image_files = self.get_images(self.unscaled_folder)

        for (i,skeleton) in tqdm(enumerate(image_files),total=len(image_files),bar_format=LOADING_BAR, ncols=80, colour='green'):
            smile = os.path.splitext(os.path.basename(skeleton))[0]
            # Process each image file
            with Image.open(skeleton) as img:
                # Get new scaled image dimensions
                new_width = int(img.width * scale_factor)
                new_height = int(img.height * scale_factor)
                
                # Resize the image to the new dimensions
                img = img.resize((new_width, new_height)) 

            # Create a blank white background
            background = Image.new('RGB', (IMG_SIZE,IMG_SIZE), (255, 255, 255))

            # Paste the molecule image onto the center of the background
            offset = tuple((bg_dim - img_dim) // 2 for bg_dim, img_dim in zip(background.size, img.size))
            background.paste(img,offset)
            background.save(fr'{self.final_folder}\{smile}.png')
    
    
    def recolour(self,threshold=245):
        '''
        Recolours the images to black and white with a threshold value

        Not yet decided whether to use this function or maintain continuous values
        '''
        # Iterate over each file in the folder
        for file_name in os.listdir(self.unscaled_folder):
            if file_name.endswith('.png'):
                # Load the image
                image_path = os.path.join(self.unscaled_folder, file_name)
                image = Image.open(image_path)

                # Convert the image to grayscale
                image = image.convert('L')

                # Convert the grayscale image to binary black and white
                image = image.point(lambda x: 0 if x < threshold else 255, '1')

                # Save the converted image (overwrite the original)
                image.save(f"{self.final_folder}\{file_name}.png")

                # Close the image file
                image.close()


    