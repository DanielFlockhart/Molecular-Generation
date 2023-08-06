'''
Used to Clean up an old dataset file that was in a different format
'''

import ast
import pandas as pd
path = r"C:\Users\0xdan\Documents\CS\WorkCareer\Chemistry Internship\Project-Code\data\datasets\db2-pas\dataset"
path2 = r"C:\Users\0xdan\Documents\CS\WorkCareer\Chemistry Internship\Project-Code\data\datasets\db2-pas\names_and_smiles.csv"

# Data is in form [[drug name, smiles],...]
# Extract to CSV file with columns [drug name,smiles]
with open(f'{path}\smiles.txt',"r") as f:
    data = ast.literal_eval(f.read())

# Write Drug Name and SMIles to CSV with column header
df = pd.DataFrame(data,columns=["ID","SMILES"])
df.to_csv(f"{path}\inputs.csv",index=False)




