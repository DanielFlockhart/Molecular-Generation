import sys,os,ast
import numpy as np
import pandas as pd
from PIL import Image
sys.path.insert(0, os.path.abspath('..'))
from Constants import file_constants,ui_constants
from utilities import img_utils,utils
from ui.terminal_ui import *
from tqdm import tqdm
# Gets the inputs for the training data
def numpy_array_converter(s):
    return np.fromstring(s[1:-1], sep=' ')
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
    target = []
    if len(row) > 0:
        # Retrieve the conditions value
        conditions = ast.literal_eval(row['conditions'].values[0])
        vectors = ast.literal_eval(row['vector'].values[0])
        target = img_utils.load_image(ID)
    else:
        return False

    # Concatenate the conditions and vectors into a single list for input
    return np.array(conditions), np.array(vectors), np.array(target[0])



def concat_vectors(v1,v2):
    ''' 
    Concatenates two vectors together
    '''
    return np.concatenate((v1.flatten(), v2.flatten()))


def get_training_data(count=None):
    '''
    Gets the inputs and targets for the training data
    '''
    # Get the targets
    labels = []
    conditions = []
    vectors = []
    targets = []
    # Read the CSV file into a DataFrame
    df = pd.read_csv(file_constants.INPUTS_FOLDER)
    # Get all the IDs 
    labels = df['ID'].values[:count if count is not None else len(df)]
    print(format_title("Loading inputs"))
    for (i,label) in enumerate(labels):
        if count is not None and i >= count:
            break
        (condition, vector,target) = get_inputs(label,df)
        conditions.append(condition)
        vectors.append(vector)
        # Not loading from csv yet
        targets.append(target)

    # Iterate through conditions and vectors and concatenate them
    #inputs = [concat_vectors(conditions[i],vectors[i]) for i in range(count if count is not None else len(conditions) )]
    # Return the inputs and targets
    return labels,vectors,conditions,targets
