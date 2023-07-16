
import numpy as np
import os,sys,glob
from matplotlib import pyplot as plt

sys.path.insert(0, os.path.abspath('..'))
from Constants import ui_constants

def colour_smile(smile):
    '''
    Colours a smile string based on the colours dictionary
    '''
    # Dictionary of colours for smile
    coloured_smile = ""

    # Strip any whitespace
    smile = smile.replace(" ","")

    # Used for elements with 2 letters
    doubled = False

    # Iterate through letters in smile
    for i, letter in enumerate(smile):

        # Check if letter is doubled, if so, skip
        if doubled:
            doubled = False
            continue

        
        string = ""

        # Get colour for letter
        if letter in ui_constants.colours.keys():
            colour = ui_constants.colours.get(letter)
            string = letter

        # Check if elements with 2 letters
        if i+1 < len(smile):
            if (letter + smile[i+1]) in ui_constants.colours.keys():
                colour = ui_constants.colours.get(letter + smile[i+1])
                string = letter + smile[i+1]
                doubled = True

        # If no colour found, set to white
        if string == "":
            string = letter
            colour = ui_constants.colours.get("Other")

        coloured_smile += f"{colour}{string}"
        coloured_smile += "\033[0m"
    return coloured_smile


def perform_checks(folder):
    '''
    Check if there is a preprocessed dataset.
    '''
    if len(os.listdir(folder)) == 0:
        print("No dataset found, downloading and processing...")
        return True # Need to preprocess
    
    # If there is a preprocessed dataset, tell the user how many images there are in the folder inside the data folder
    print(f"Found preprocessed dataset, skipping preprocessing...")
    print("To manually preprocess, delete the contents of the data folder and run the program again") # Temporary
    
    return False # No need to preprocess


def get_percentile_upper_bound(array,percentile):
    '''
    Gets the upper bound of an array of numbers from a percentile
    '''
    array = np.array(array)
    upper_bound = np.percentile(array, percentile)
    return upper_bound


def get_distribution_upper_bound(array,std_dev):
    '''
    Gets the standard deviation of the numbers and returns the upper bound of the numbers
    '''
    # Calculate the standard deviation of the numbers
    std = np.std(array)
    # Calculate the mean of the numbers
    mean = np.mean(array)
    # Calculate the upper bound of the numbers
    upper_bound = mean + (std * std_dev)
    return upper_bound

def normalise_vector(vector):
    '''
    Normalises a vector
    '''
    vector = np.array(vector)
    vector = vector / np.linalg.norm(vector)
    return vector.tolist()


def plot_distribution(sizes,xlabel,ylabel):
    '''
    Plots the size distribution of the molecules for visualisation
    '''
    plt.hist(sizes, bins=200)
    plt.gca().set(title=xlabel, ylabel=ylabel)
    plt.show()  