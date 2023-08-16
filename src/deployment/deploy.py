# Module for deployment of the application.
import os,sys
from PIL import *
import numpy as np
from Constants import file_constants,preprop_constants
sys.path.insert(0, os.path.abspath('..'))
from deployment import generation
import pandas as pd
import ast

from preprocessing.smiles_to_vec import *
class App:
    '''
    Main class for deployment of the application.
    '''

    def __init__(self,model):
        self.model = model
        self.gen = generation.Generator(self.model)

    def generate_molecule(self,vector,condition,name="mol"):
        '''
        Generates a molecule from the model with a starting vector and condition.
        '''
        img = self.gen.generate_molecule(np.array(vector),np.array(condition))
        img.save(fr"{file_constants.GENERATED_FOLDER}\new_molecules\mol{name}.png")

    def get_mols(self):
        '''
        Gets input from the user.
        '''
        SMILE = input("Enter a starting Molecule SMILE: ")
        SMILE = smile_to_vector_ChemBERTa(preprop_constants.EMBEDDING_MODEL,SMILE)
        condition = self.gen.create_condition()
        
        return self.gen.generate_molecule(SMILE,condition)
    
    def get_test_molecules(self,num_molecules=3):
        '''
        Gets a test molecule from the user.
        '''
        df = pd.read_csv(file_constants.INPUTS_FOLDER)
        mols = []
        # Iterate through the csv and get the 1st (num_molecules) molecules
        for i in range(num_molecules):
            row = df.iloc[i]
            mols.append([ast.literal_eval(row['vector']),ast.literal_eval(row['conditions'])])
        return mols