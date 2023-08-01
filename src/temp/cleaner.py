'''
Used to Clean up an old dataset file that was in a different format
'''

import ast
import pandas as pd
path = r"C:\Users\0xdan\Documents\CS\WorkCareer\Chemistry Internship\Ai-Chem-Intership\data\datasets\db2-pas"

# Data is in form [[drug name, smiles],...]
# Extract to CSV file with columns [drug name,smiles]
with open(f'{path}\smiles.txt',"r") as f:
    data = ast.literal_eval(f.read())

# Write Drug Name and SMIles to CSV with column header
df = pd.DataFrame(data,columns=["Drug Name","SMILES"])
df.to_csv(f"{path}\inputs.csv",index=False)




