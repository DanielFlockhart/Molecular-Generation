import pandas as pd
import random,os

class Database:
    def __init__(self,file):
        '''
        Database class to handle the loading of the dataset from dataset file
        
        Parameters
        ----------
        file : str
            The path to the dataset file
        
        '''
        self.file = pd.read_csv(file)
        self.load_smiles()

    def get_file(self):
        return self.file
        
    def load_smiles(self):
        '''
        Extract SMILE columns from the dataset
        '''
        self.data = self.file[['SMILES']].values.tolist()
    
    def clean_data(self):
        '''
        Remove any invalid smiles

        Not Used as of yet (08/07/2023) - Dataset is already clean
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
        return [molecule[0] for molecule in self.data][:10] # Temporary
    
    def get_id(self,smile):
        '''
        Gets the ID of a smile from dataset by searching through smiles
        '''
        row = self.file[self.file['SMILES'] == smile]

        # Check if a matching row was found
        if not row.empty:
            return row['ID'].values[0]
    
    


        