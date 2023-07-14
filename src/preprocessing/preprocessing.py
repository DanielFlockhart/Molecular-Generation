from condition_to_vec import *

from smiles_to_mol import *
from smiles_to_vec import *

from target_to_vec import *
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
        

        return conditions_to_vector(conditions)
    
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

    

    