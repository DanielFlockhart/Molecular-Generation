import os,sys,random
from PIL import *
import numpy as np
import tensorflow as tf

from preprocessing import preprocess
from training import vae
from postprocessing import *
from deployment import *
from CONSTANTS import *
data_folder = r'C:\Users\0xdan\Documents\CS\WorkCareer\Chemistry Internship\Ai-Chem-Intership\data\datasets'


def initialise(processor,download=True):
    processor.process(download=download)

    

def main():
    print("Main Executing")

if __name__ == "__main__":
    database = preprocess.Database(fr'{data_folder}\CSD_EES_DB.csv')
    processor = preprocess.Preprocessor(data_folder,database)
    initialise(processor,download=True)
    main()