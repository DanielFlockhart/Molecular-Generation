
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
        self.csv = f"{file_constants.DATA_FOLDER}/{df_name}.csv"
        self.database = Database(self.csv)
        self.target_generator = TargetGenerator(self.database)
        self.embedding_model = embedding_model

    
    def process(self):
        '''
        Main Preprocessing function
        - Generates the target images
        - Generates the input vectors
        
        '''
        self.get_targets()
        self.normalise_targets()
        self.target_generator.save_dataset_info()
        self.generate_data_file()

    def generate_data_file(self):
        '''
        Generates CSV with inputs for the neural network
        '''
        
        
        print(format_title("Getting Vector Representations of Smiles and Conditions"))
        self.smiles = self.database.get_smiles()
        # Clear Inputs Folder
        file_utils.clear_csv(file_constants.INPUTS_FOLDER)
        # Add Headers
        df = pd.DataFrame(columns=['ID', 'SMILES', 'conditions', 'vector'])
        df.to_csv(file_constants.INPUTS_FOLDER, mode='a', header=True, index=False)
        
        for smile in tqdm(self.smiles, bar_format=ui_constants.LOADING_BAR, ncols=80, colour='green'):
            
            id = self.database.get_id(smile)
            (smiles_vec, condition_vec) = self.get_input(smile)
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

    def get_input(self,smile):
        '''
        Produces vector from concatenation of the smiles vector representation and its conditions for input to network
        '''
        condition_vec = self.get_conditions(smile)
        smile_vec = smile_to_vector_ChemBERTa(self.embedding_model,smile)
        return (smile_vec,condition_vec)
    
    def get_conditions(self,smile):
        '''
        Search throught dataset.csv to find the SMILE id, it then returns the other columns which represent the conditons
        '''
        
        conditions = []
        row = self.database.file[self.database.file['SMILES'] == smile]

        # Check if a matching row was found
        if not row.empty:
            # Access the values of other columns for the matching row
            # Making this iterable instead of manual later
            ID = row['ID'].values[0]
            NAts = row['NAts'].values[0]
            HOMO = row['HOMO'].values[0]
            LUMO = row['LUMO'].values[0]
            es1 = row['E(S1)'].values[0]
            fs1 = row['f(S1)'].values[0]
            es2 = row['E(S2)'].values[0]
            fs2 = row['f(S2)'].values[0]
            es3 = row['E(S3)'].values[0]
            fs3 = row['f(S3)'].values[0]
            et1 = row['E(T1)'].values[0]
            et2 = row['E(T2)'].values[0] 
            et3 = row['E(T3)'].values[0]
            conditions = [NAts, HOMO,LUMO,es1,fs1,es2,fs2,es3,fs3,et1,et2,et3]
        else:
            conditions = [0 for x in range(10)]
        
        return utils.normalise_vector(conditions)
    

    def get_targets(self):
        '''
        Using RDKit to generate molecule skeletons to use as targets for network from the dataset of smiles
        '''
        self.target_generator.generate_skeletons()
        

    def normalise_targets(self):
        '''
        Normalise the target images to standardise structural proportions and fit into target dimensions
        '''
        self.target_generator.normalise_targets()



