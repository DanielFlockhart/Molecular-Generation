import sys,os
import numpy as np
import pandas as pd
from PIL import Image
import tqdm from tqdm
sys.path.insert(0, os.path.abspath('..'))
from Constants import file_constants
from utilities import img_utils
# Gets the inputs for the training data

def get_inputs(smiles):
    '''
    Gets the inputs for the training data
    smiles variable is extracted from target images
    make sure the target images are in the same order as the inputs
    '''
    # Read the CSV file into a DataFrame
    df = pd.read_csv(file_constants.INPUTS_FOLDER)

    # Find the row with the specific smile
    row = df[df['SMILES'] == smiles]

    if len(row) > 0:
        # Retrieve the conditions value
        conditions = row['conditions'].iloc[0]
        vectors = row['vector'].iloc[0]
    else:
        # Return None if the smile is not found
        return None

    # Concatenate the conditions and vectors into a single list for input
    return np.array(conditions), np.array(vectors)

def get_targets():
    '''
    Gets the targets images for the training data and converts them to numpy arrays
    '''
    targets = img_utils.load_images(file_constants.PROCESSED_DATA)
    return targets


def concat_vectors(v1,v2):
    ''' 
    Concatenates two vectors together
    '''
    return np.concatenate((v1,v2),axis=1)


def get_training_data():
    '''
    Gets the inputs and targets for the training data
    '''
    # Get the targets
    targets = get_targets()

    # Create empty lists for the inputs and targets
    inputs = []

    # Iterate over each row in the CSV file
    for index, row in tqdm(df.iterrows(), total=len(df)):
        # Get the inputs for the current row
        conditions, vectors = get_inputs(row['SMILES'])

    # Convert the lists to NumPy arrays
    inputs = np.array(inputs)
    targets = np.array(targets)

    # Return the inputs and targets
    return inputs,conditions, targets

if __name__ == "__main__":
    pass