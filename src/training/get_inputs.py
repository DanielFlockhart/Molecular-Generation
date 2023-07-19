import sys,os,ast
import numpy as np
import pandas as pd
from PIL import Image
sys.path.insert(0, os.path.abspath('..'))
from Constants import file_constants,ui_constants
from utilities import img_utils
from ui.terminal_ui import *
from tqdm import tqdm
# Gets the inputs for the training data

def get_inputs(ID,df):
    '''
    Gets the inputs for the training data
    smiles variable is extracted from target images
    make sure the target images are in the same order as the inputs
    '''
    
    # Find the row with the specific smile
    row = df[df['ID'] == ID]
    
    conditions = []
    vectors = []

    if len(row) > 0:
        # Retrieve the conditions value
        conditions = ast.literal_eval(row['conditions'].values[0])
        vectors = ast.literal_eval(row['vector'].values[0])

    # Concatenate the conditions and vectors into a single list for input
    return np.array(conditions), np.array(vectors)

def get_targets():
    '''
    Gets the targets images for the training data and converts them to numpy arrays
    '''
    targets,labels = img_utils.load_images()
    return targets,labels


def concat_vectors(v1,v2):
    ''' 
    Concatenates two vectors together
    '''
    return np.concatenate((v1.flatten(), v2.flatten()))


def get_training_data():
    '''
    Gets the inputs and targets for the training data
    '''
    # Get the targets
    targets,labels = get_targets()
    conditions = []
    vectors = []
    # Read the CSV file into a DataFrame
    df = pd.read_csv(file_constants.INPUTS_FOLDER)

    print(format_title("Loading inputs"))
    for label in labels:
        (condition, vector) = get_inputs(label,df)
        conditions.append(condition)
        vectors.append(vector)

    # Iterate through conditions and vectors and concatenate them
    inputs = [concat_vectors(conditions[i],vectors[i]) for i in range(len(conditions))]
    # Return the inputs and targets
    return inputs,conditions,targets
