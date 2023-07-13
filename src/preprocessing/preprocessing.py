from condition_to_vec import *

from smiles_to_mol import *
from smiles_to_vec import *

from target_to_vec import *
from target_scaling import *

import pandas as pd

from database import *

class Preprocessor:
    def __init__(self,data_file):
        self.database = Database(data_file)

        # Assuming you have the pre-trained ChemBERTa model and tokenizer
        self.embedding_model = ChemBERTaModel.from_pretrained('path_to_pretrained_model')
        self.embedding_tokenizer = ChemTokenizer.from_pretrained('path_to_tokenizer')

    def produce_input(self,smile):
        '''
        Produces vector from concatenation of the smiles vector representation and its conditions for input to network
        '''
        condition_vec = self.get_conditions(smile)
        smile_vec = smile_to_vector_ChemBERTa(self.embedding_model,self.embedding_tokenizer,smile)
        return np.concatenate((condition_vec,smile_vec), axis=0)
    
    def get_conditions(self,smile):
        '''
        Search throught dataset.csv to find the SMILE id, it then returns the other columns which represent the conditons
        '''
        conditions = []
        

        return conditions_to_vector(conditions)
    
    def generate_targets(self):
        '''
        Using RDKit to generate molecule skeletons to use as targets for network from the dataset of smiles
        '''
        pass

    def normalise_targets(self):
        '''
        Normalise the target images to standardise structural proportions and fit into target dimensions
        '''
        pass
    

    