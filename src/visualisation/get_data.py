import csv
import ast
import numpy as np
import pandas as pd
from tqdm import tqdm  # Import tqdm
import sys,os
sys.path.insert(0, os.path.abspath('..'))
from Constants import ui_constants
dataset = r"C:\Users\0xdan\Documents\CS\WorkCareer\Chemistry Internship\Ai-Chem-Intership\data\datasets\db1\inputs.csv"

def get_dataset():
    '''
    Get every vector from csv file in column "vector"
    '''
    vectors = []
    labels = []
    
    # Count the total number of rows in the CSV file
    with open(dataset, newline='') as csvfile:
        num_rows = sum(1 for line in csvfile)
    
    # Reopen the CSV file to reset the iterator
    with open(dataset, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        
        # Create a tqdm object to display progress bar
        tqdm_instance = tqdm(total=num_rows, desc="Reading CSV data",bar_format=ui_constants.LOADING_BAR, ncols=80, colour='green')
        
        for row in reader:
            vectors.append(np.array(ast.literal_eval(row['vector'])))
            labels.append(row['SMILES'])
            tqdm_instance.update(1)  # Update progress bar
            
    tqdm_instance.close()  # Close tqdm after the loop

    return np.array(vectors), labels
