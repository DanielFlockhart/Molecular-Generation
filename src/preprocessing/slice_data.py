import random,os
import pandas as pd
from structure_builder import *
# Get a test set of the data that my computer will be able to process
# Will proceed to Train data on computer clusters at a later date
# Why spend 20 Mins doing something when you could spend double the time automating it huh?

class Slicer:
    def __init__(self,data_folder):
        self.data_folder = data_folder
        # Read the CSV file
        self.data_file = pd.read_csv(fr'{data_folder}\CSD_EES_DB.csv')

        # Extract the data from the folder
        self.data = self.extract()

        # Reseed Randomizer and Shuffle the data
        random.seed(random.randint(10,1000))
        random.shuffle(self.data)

    def extract(self):
        # Extract ID and SMILE columns
        return self.data_file[['ID', 'SMILES']].values.tolist()

    def slice(self,size):
        # Check the Current Number in the test data folder to save time, Change the size to how many more is required
        size = self.check_current_data(size)
        # Get's a Slice of Dataset
        return self.data[:size] # If it passes some due to exceptions in the code then, the number in the test data folder < size
        
    def save(self):
        # Saves Slice of Dataset of Molecules to test data folder
        for molecule in self.data:
            # Create a Structure Object and save it to image of size 128x128
            mol = Structure(molecule[1],400)
            try:
                mol.save_structure()
            except Exception as e:
                print("Error occured : " ,e)
                continue
    def check_current_data(self,required):
        # Checks the current data in the test data folder, if it already has amount of images required then return 0 else return the amount of images required
        current_data = len(os.listdir(self.data_folder + r"\test-data"))
        if (required - current_data) > 0:
            print("Current Data in Test Data Folder : ",current_data)
            print("Adding ",required - current_data," more images to the test data folder")
            return required - current_data
        
        print("data in test data folder is already greater than required, no need to add more images")
        return 0 


if __name__ == "__main__":
    data_folder = r"C:\Users\0xdan\Documents\CS\WorkCareer\Chemistry Internship\Project-Code\data\dataset"

    # Get instance of Slicer Class
    slicer = Slicer(data_folder)
    # Randomly select 500 compounds from the dataset
    slicer.data = slicer.slice(500)
    slicer.save()