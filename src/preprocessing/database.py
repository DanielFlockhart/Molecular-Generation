import pandas as pd
import random,os

class Database:
    def __init__(self,file):
        self.file = pd.read_csv(file)
        

    def load_data(self):
        '''
        Extract SMILE columns from the dataset
        '''
        self.data = self.file[['SMILES']].values.tolist()
        #self.clean_data()
    
    def clean_data(self):
        '''
        Remove any invalid smiles
        '''
        for (i,smile) in enumerate(self.data):
            if smile[0] == "" or all(ord(c) < 128 for c in smile[0]) or ("%" not in smile[0]):
                self.data.pop(i)

    def shuffle_data(self):
        '''
        Shuffle the dataset
        '''
        random.seed(random.randint(10,1000)) # Temporary
        random.shuffle(self.data)

    def sort_data(self,reversed=True):
        '''
        Sort the data by the string length of the smiles
        '''
        self.data = sorted(self.data, key=lambda x: len(str(x[0])), reverse=reversed)
    
    def retrieve_longest_smiles(self,x):
        '''
        Retrieves the longest x smiles
        '''
        self.sort_data()
        return self.data[:x]


    def get_slice(self,amount,shuffle=True,start_index=0):
        '''
        Returns a slice of the data
        '''
        if shuffle:
            self.shuffle_data()
        return [molecule[0] for molecule in self.data[start_index:amount+start_index]]
    
    def get_smiles(self):
        '''
        Returns Full List of Smiles
        '''
        return [molecule[0] for molecule in self.data]


# if __name__ == "__main__":
#     db = Database(r'C:\Users\0xdan\Documents\CS\WorkCareer\Chemistry Internship\Project-Code\data\dataset\CSD_EES_DB.csv')
#     db.load_data()
#     # Return the 30 longest smiles
#     top_30_longest = db.retrieve_longest_smiles(30)
    
#     # Iterate through and print the longest smiles
#     for i, smile in enumerate(top_30_longest):
#         print(smile[0][:50])