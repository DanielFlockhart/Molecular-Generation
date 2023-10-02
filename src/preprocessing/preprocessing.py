
from preprocessing.smiles_to_vec import *

from preprocessing.target_generation import *

import pandas as pd

from preprocessing.database import *



import sys, os
from tqdm import tqdm

sys.path.insert(0, os.path.abspath('..'))
from Constants import file_constants,preprop_constants,ui_constants
from ui.terminal_ui import *

# Model seyonec/ChemBERTa-zinc-base-v1

class Preprocessor:
    def __init__(self,embedding_model,df_name):
        self.csv = f"{file_constants.DATA_FOLDER}/{df_name}"
        self.database = Database(self.csv)
        self.target_generator = TargetGenerator(self.database)
        self.embedding_model = embedding_model

    
    def process(self,subset=False):
        '''
        Main Preprocessing function
        - Generates the target images
        - Generates the input vectors

        - Target Images have the right structures and IDs
        
        
        '''
        self.get_targets(preprop_constants.SUBSET_COUNT if subset else None)
        self.normalise_targets()
        #self.target_generator.save_dataset_info()
        #self.generate_data_file()

    def generate_data_file(self):
        '''
        Generates CSV with inputs for the neural network
        '''
        print(format_title("Getting Vector Representations of Smiles and Conditions"))
        self.smiles = self.database.get_smiles_from_ids()

        # Clear Inputs Folder
        file_utils.clear_csv(file_constants.INPUTS_FOLDER)

        # Add Headers
        df = pd.DataFrame(columns=['ID', 'SMILES', 'conditions', 'vector'])
        df.to_csv(file_constants.INPUTS_FOLDER, mode='a', header=True, index=False)
        print("Warning: This may take a while")
        print("Dataset must be sorted by ID alphabetically, will make it more robust later - I need to review if this is still the case")

        #target_vecs,labels = img_utils.load_images() # This line causes 30Gb of RAM to be used
        (mins,maxs) = self.preprocess_conditions()
        for (i,smile) in tqdm(enumerate(self.smiles),total=len(self.smiles), bar_format=ui_constants.LOADING_BAR, ncols=80, colour='green'):
            id = self.database.get_id(smile)
            #print("Warning, do not load target from inputs.csv as it is not the full image")
            
            (smiles_vec, condition_vec) = self.get_input(smile,mins,maxs)
            self.add_entry(id,smile,condition_vec,smiles_vec)
            
            
    def add_entry(self,id,smile,conditions,smiles_vec):
        '''
        Add data to inputs.csv file for storage for input to neural network
        '''
        # Create an empty DataFrame
        
        df = pd.DataFrame(columns=['ID', 'SMILES', 'conditions', 'vector'])
        # Create a new row as a list
        new_row = [id, smile, conditions, smiles_vec]
        # Append the new row to the DataFrame
        df.loc[len(df)] = new_row

        # Write the DataFrame to a CSV file
        df.to_csv(file_constants.INPUTS_FOLDER, mode='a', header=False, index=False)

    def get_input(self,smile,mins,maxs):
        '''
        Produces vector from concatenation of the smiles vector representation and its conditions for input to network
        '''
        
        condition_vec = self.get_conditions(smile,mins,maxs)
        
        smile_vec = smile_to_vector_ChemBERTa(self.embedding_model,smile)
        
        return (smile_vec,condition_vec)
    
    def get_conditions(self,smile,mins,maxs):
        '''
        Search throught dataset.csv to find the SMILE id, it then returns the other columns which represent the conditons
        '''
        
        conditions = []
        row = self.database.file[self.database.file['SMILES'].replace("\n", "") == smile]
        
        # Check if a matching row was found
        if not row.empty:
            # Access the values of other columns for the matching row
            # Making this iterable instead of manual later
            for key in preprop_constants.keys:
                condition = self.normalise_condition(row[key].values[0],mins[key],maxs[key])
                conditions.append(condition)
        else:
            print("Error: No matching row found for smile: ", smile)
        
        return conditions
    def normalise_condition(self,condition,mins,maxs):
        '''
        Normalise the conditions to be between 0 and 1
        '''
        res = (condition - mins) / (maxs - mins)
        return res 

    def preprocess_conditions(self):

        min_values = self.database.file[preprop_constants.keys].min()
        max_values = self.database.file[preprop_constants.keys].max()
        return (min_values,max_values)

    def get_targets(self,count):
        '''
        Using RDKit to generate molecule skeletons to use as targets for network from the dataset of smiles
        ''' 
        file_utils.clear_folder(file_constants.PROCESSED_DATA)
        self.target_generator.generate_skeletons(count)
        

    def normalise_targets(self):
        '''
        Normalise the target images to standardise structural proportions and fit into target dimensions
        '''
        self.target_generator.normalise_targets()



