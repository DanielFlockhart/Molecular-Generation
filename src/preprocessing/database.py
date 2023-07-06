import pandas as pd
import random,os
from structure_builder import *



# Get the top 30 longest SMILES
top_30_longest = sorted_data[:100]

# Print the top 30 longest SMILES
#for item in top_30_longest:
#    print(len(item[1]))


# Calculate the longest carbon chain for each SMILES
# longest_chain_lengths = []
# for item in top_30_longest:
#     smiles = item[1]
#     carbon_atoms = [atom for atom in smiles if atom == 'C']
#     longest_chain_length = max([len(chain) for chain in ''.join(carbon_atoms).split('H')])
#     longest_chain_lengths.append(longest_chain_length)

# Create a dictionary with SMILES as keys and chain lengths as values
smiles_chain_lengths = dict(zip([item[1] for item in top_30_longest], longest_chain_lengths))

# Sort the SMILES by chain length in descending order
sorted_smiles = sorted(smiles_chain_lengths.items(), key=lambda x: x[1], reverse=True)

print(sorted_smiles[0])

# Print the longest carbon chain lengths
#for i, length in enumerate(longest_chain_lengths):
#    print(f"SMILES: {top_30_longest[i][1]}, Longest Carbon Chain Length: {length}")


class Database:
    def __init__(self,file):
        self.file = pd.read_csv(file)
    
    def load_data(self):
        # Extract ID and SMILE columns
        self.data = self.file[['ID', 'SMILES']].values.tolist()

    def sort_data(self,reversed=True):
        # Sort the data by the string length of SMILE in descending order
        self.data = sorted(self.data, key=lambda x: len(str(x[1])), reverse=reversed)
    
    def retrieve_longest_smiles(self,x):
        # Retrieves the longest x smiles
        self.sort_data()
        return self.data[:x]


