import csv,ast
import numpy as np
import pandas as pd
dataset = r"C:\Users\0xdan\Documents\CS\WorkCareer\Chemistry Internship\Ai-Chem-Intership\data\datasets\db1\inputs.csv"
def get_dataset():
    '''
    Get every vector from csv file in column "vector"
    '''
    vectors = []
    labels = []
    with open(dataset, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            vectors.append(np.array(ast.literal_eval(row['vector'])))
            labels.append(row['SMILES'])

    return np.array(vectors),labels