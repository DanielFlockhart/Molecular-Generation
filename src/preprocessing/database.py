import pandas as pd
import random,os

# Print the longest carbon chain lengths
#for i, length in enumerate(longest_chain_lengths):
#    print(f"SMILES: {top_30_longest[i][1]}, Longest Carbon Chain Length: {length}")

# Get a test set of the data that my computer will be able to process
# Will proceed to Train data on computer clusters at a later date
# Why spend 20 Mins doing something when you could spend double the time automating it huh?

class Database:
    def __init__(self,file):
        self.file = pd.read_csv(file)
        

    def load_data(self):
        # Extract ID and SMILE columns
        self.data = self.file[['ID', 'SMILES']].values.tolist()
        random.seed(random.randint(10,1000)) # Temporary
        random.shuffle(self.data)

    def sort_data(self,reversed=True):
        # Sort the data by the string length of SMILE in descending order
        self.data = sorted(self.data, key=lambda x: len(str(x[1])), reverse=reversed)
    
    def retrieve_longest_smiles(self,x):
        # Retrieves the longest x smiles
        self.sort_data()
        return self.data[:x]

    def slice(self,slice_size):
        # Get a portion of the dataset

        # Check the Current Number in the test data folder to save time, Change the size to how many more is required
        # size = self.check_current_data(slice_size) - Might implement in future -> Mean's that you don't have to keep downloading and processing new slices

        return [molecule[1] for molecule in self.data[:slice_size]]
    
    def get_smiles(self):
        # Return Full list of Smiles
        return [molecule[1] for molecule in self.data]

    # def check_current_data(self,required):
    #     # Checks the current data in the test data folder, if it already has amount of images required then return 0 else return the amount of images required
    #     current_data = len(os.listdir(self.data_folder + r"\test-data"))
    #     if (required - current_data) > 0:
    #         print("Current Data in Test Data Folder : ",current_data)
    #         print("Adding ",required - current_data," more images to the test data folder")
    #         return required - current_data
        
    #     print("data in test data folder is already greater than required, no need to add more images")
    #     return 0 
