from tqdm import tqdm
from rdkit import RDLogger

# Disable the RDKit logger
RDLogger.DisableLog('rdApp.error')

import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from rdkit import Chem
from rdkit.Chem.Draw import rdDepictor, rdMolDraw2D
from PIL import Image
from io import BytesIO
import random,glob,os,sys,string,math,json
from preprocessing.database import *

sys.path.insert(0, os.path.abspath('..'))
from CONSTANTS import *
from ui.terminal_ui import *
from utils import *




class Preprocessor:
    def __init__(self,data_folder,database,name):
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
        self.dataset_name = name
        self.processed_folder = fr"{data_folder}\processed\{self.dataset_name}"
        self.smiles = self.database.get_smiles()[:1000]
    def process(self):
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
        os.mkdir(self.processed_folder)
        # --- Redownload the images if Requested ---
        print(format_title("Downloading Images"))
        self.clear_folder(self.processed_folder)
        self.create_skeletons()

        # --- Scale the images ---
        print(format_title("Scaling Images"))
        sizes = np.sort(self.get_skeleton_sizes())
        upper_bound = get_upper_bound(sizes,BOUND_PERCENTILE) # Get Upper Bound from Percentile
        #upper_bound = self.calculate_upper_bounds(sizes,STD_DEV)
        upper_bound = min(upper_bound,800)
        scale_factor = IMG_SIZE / upper_bound

        

        # Create the processed data folder
        

        # Rescale the images with new scale factor
        self.rescale_skeletons(scale_factor,upper_bound)
        # --- Additional Processing ---
        #self.recolour() - Leave this for now
        self.save_dataset_info(sizes)

        # --- Visualise the dataset ---
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
            "dataset_UNSCALED_SIZES": sizes.tolist()
        }
        with open(fr'{self.processed_folder}\dataset_info.json', 'w') as outfile:
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
                img.save(fr'{self.processed_folder}\{smile}.png')

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
        for image in os.listdir(self.processed_folder):
            with Image.open(fr"{self.processed_folder}\{image}") as img:
                img_sizes.append(img.width)
                img_sizes.append(img.height)
        return img_sizes


    def get_new_dims(self,angle,width,height):
        '''
        Unused Function
        '''

        # Convert the angle to radians
        angle_rad = math.radians(angle)

        # Calculate the sine and cosine of the angle
        cos_theta = math.cos(angle_rad)
        sin_theta = math.sin(angle_rad)

        # Get the original width and height

        # Calculate the new width and height after rotation
        new_width = int(math.fabs(width * cos_theta) + math.fabs(height * sin_theta))
        new_height = int(math.fabs(width * sin_theta) + math.fabs(height * cos_theta))
        return new_width,new_height
    
    def dimension_loss(self,rotated_width,rotated_height):
        '''
        Unused Function
        '''
        # Calculate the dimensions of the cropped rectangle
        cropped_width = min(rotated_width, IMG_SIZE)
        cropped_height = min(rotated_height, IMG_SIZE)
        
        # Calculate the area of the cropped rectangle
        cropped_area = cropped_width * cropped_height
        
        # Calculate the area loss (difference between the rotated rectangle area and the cropped rectangle area)
        area_loss = rotated_width * rotated_height - cropped_area
        
        return area_loss
    
    def calculate_best_rotation(self,img_width,img_height,sf):
        '''
        Unused Function
        '''
        best_loss = float("inf")
        best_angle = 0
        for x in range(0,180):
            new_width,new_height = self.get_new_dims(x,img_width*sf,img_height*sf)
            loss = self.dimension_loss(new_width,new_height) # Calculates the image loss with rotation
            if loss < best_loss:
                best_angle = x
                best_loss = loss
                if loss == 0:
                    break
        return best_angle
    def rescale_skeletons(self,scale_factor,scale_size):
        '''
        Rescales the skeleton images to the target image size without distorting the image

        This function standardises the data for the model

        If one of the dimensions of an image is greater than the target scale factor it is rotated 45 degrees
        '''
        image_files = self.get_images(self.processed_folder)
        rotated = 0
        for (i, skeleton) in tqdm(enumerate(image_files), total=len(image_files), bar_format=LOADING_BAR, ncols=80, colour='green'):
            smile = os.path.splitext(os.path.basename(skeleton))[0]
            rotation = None
            
            # Process each image file
            with Image.open(skeleton) as img:
                # Calculates what the new dimensions would be with scaling
                if img.width > scale_size or img.height > scale_size:
                    #rotation = self.calculate_best_rotation(img.width,img.height,scale_factor)
                    rotated += 1
                    rotation = 45
                
                if rotation is not None:
                    img = img.rotate(rotation, expand=True, fillcolor='white')
            
                new_width = int(img.width * scale_factor)
                new_height = int(img.height * scale_factor)
                
                # Resize the image to the new dimensions
                img = img.resize((new_width, new_height))
                
                # Create a blank white background
                background = Image.new('RGB', (IMG_SIZE, IMG_SIZE), (255, 255, 255))
                
                # Paste the molecule image onto the center of the background
                offset = tuple((bg_dim - img_dim) // 2 for bg_dim, img_dim in zip(background.size, img.size))
                
                background.paste(img, offset)
                

            background.save(fr'{self.processed_folder}\{smile}.png')

        print(format_title("Preprocessing Summary"))
        strings = ["Generated From Smiles","Bound Used","Amount of Images Above bound"]
        outputs = [f"{(len(image_files)*100/len(self.smiles))}%",scale_size,f"{rotated}/{len(image_files)}"]
        print(console_grid(strings,outputs))
    
    
    def recolour(self,threshold=245):
        '''
        Recolours the images to black and white with a threshold value

        Not yet decided whether to use this function or maintain continuous values
        '''
        # Iterate over each file in the folder
        for file_name in os.listdir(self.processed_folder):
            if file_name.endswith('.png'):
                # Load the image
                image_path = os.path.join(self.processed_folder, file_name)
                image = Image.open(image_path)

                # Convert the image to grayscale
                image = image.convert('L')

                # Convert the grayscale image to binary black and white
                image = image.point(lambda x: 0 if x < threshold else 255, '1')

                # Save the converted image (overwrite the original)
                image.save(f"{self.processed_folder}\{file_name}.png")

                # Close the image file
                image.close()