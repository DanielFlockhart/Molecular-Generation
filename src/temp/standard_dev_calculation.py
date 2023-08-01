'''
Used for calculating the standard deviation of the dataset sizes to decide on a cutoff point at which to ignore outliers scale.

I have implemented an automatic version of this taking the 99th percentile of the dataset sizes as the cutoff point for outliers.

'''

import json
import numpy as np
import matplotlib.pyplot as plt
distributions = []

file = r"C:\Users\0xdan\Documents\CS\WorkCareer\Chemistry Internship\Ai-Chem-Intership\data\datasets\processed\CSD_EES_DB\dataset_info.json"
def get_distributions(file):
    with open(file) as json_file:
        data = json.load(json_file)
    return data["dataset_UNCSCALED_SIZES"]
    
def calculate_upper_bounds(array,std_dev):
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


def plot_frequency_graph(numbers, upper_bound):
    '''
    Plots a graph of the frequency of the sizes
    '''
    
    # Convert numbers to a NumPy array
    numbers = np.array(numbers)

    upper_bound = np.percentile(numbers, percentile)
    print(upper_bound)
    # Calculate the frequencies
    unique_numbers, counts = np.unique(numbers, return_counts=True)

    # Get the count of numbers above the upper bound
    count_above_upper_bound = np.sum(numbers > upper_bound)

    # Set the width for each bar
    bar_width = 0.8

    # Plot the frequencies
    plt.bar(unique_numbers, counts, width=bar_width)

    # Highlight the upper bound
    plt.axhline(0, color='red', linestyle='--', linewidth=2)
    plt.axvline(upper_bound, color='red', linestyle='--', linewidth=2, label='Upper Bound')

    # Set plot labels and title
    plt.xlabel('Numbers')
    plt.ylabel('Frequency')
    plt.title('Frequency of Numbers')

    # Show the count of numbers above the upper bound
    plt.text(upper_bound, np.max(counts), f'Count above upper bound: {count_above_upper_bound}',
             horizontalalignment='right', verticalalignment='bottom', color='red')

    # Display the plot
    plt.legend()
    plt.show()

dists = get_distributions(file)
percentile = 99
plot_frequency_graph(dists,percentile)
