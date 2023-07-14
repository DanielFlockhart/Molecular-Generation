
from smiles_to_vec import *

from target_generation import *

import pandas as pd

from database import *

# Import generic wrappers
from transformers import AutoModel, AutoTokenizer 


# Model seyonec/ChemBERTa-zinc-base-v1

class Preprocessor:
    def __init__(self,data_file,embedding_model,df_name):
        self.database = Database(data_file)
        self.target_generator = TargetGenerator(data_file,self.database,df_name)
        self.embedding_model = embedding_model
        self.datafile = self.database.get_file()

    
    def process(self):
        '''
        Main Preprocessing function
        - Generates the target images
        - Generates the input vectors
        
        '''
        self.get_targets()
        self.normalise_targets()
        self.target_generator.save_dataset_info()
        self.generate_vectors()

    def generate_vectors(self):
        self.smiles = self.database.get_smiles()
        for smile in self.smiles:
            self.get_input(smile)

    def get_input(self,smile):
        '''
        Produces vector from concatenation of the smiles vector representation and its conditions for input to network
        '''
        condition_vec = self.get_conditions(smile)
        smile_vec = smile_to_vector_ChemBERTa(self.embedding_model,smile)
        return np.concatenate((condition_vec,smile_vec), axis=0)
    
    def get_conditions(self,smile):
        '''
        Search throught dataset.csv to find the SMILE id, it then returns the other columns which represent the conditons
        '''
        
        conditions = []
        row = self.datafile[self.datafile['SMILES'] == smile]

        # Check if a matching row was found
        if not row.empty:
            # Access the values of other columns for the matching row
            # Making this iterable instead of manual later
            ID = row['ID'].values[0]
            NAts = row['NAts'].values[0]
            HOMO = row['HOMO'].values[0]
            LUMO = row['LUMO'].values[0]
            es1 = row['E(S1)'].values[0]
            fs1 = row['F(S1)'].values[0]
            es2 = row['E(S2)'].values[0]
            fs2 = row['F(S2)'].values[0]
            es3 = row['E(S3)'].values[0]
            fs3 = row['F(S3)'].values[0]
            et1 = row['E(T1)'].values[0]
            et2 = row['E(T2)'].values[0] 
            et3 = row['E(T3)'].values[0]
            conditions = [NAts, HOMO,LUMO,es1,fs1,es2,fs2,es3,fs3,et1,et2,et3]
        else:
            conditions = [0 for x in range(10)]
        

        return conditions
    
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



