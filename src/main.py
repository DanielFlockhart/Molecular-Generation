import os,sys,random
from PIL import *
import numpy as np
import tensorflow as tf

from preprocessing import preprocess
from training import vae
from postprocessing import *
from deployment import *
from CONSTANTS import *
from ui.terminal_ui import *
data_folder = r'C:\Users\0xdan\Documents\CS\WorkCareer\Chemistry Internship\Ai-Chem-Intership\data\datasets'


def initialise(processor,download=False):
    processor.process(download=download)

    

def main():
    print(format_title("Training Model"))

if __name__ == "__main__":
    mode = "terminal"
    database = preprocess.Database(fr'{data_folder}\CSD_EES_DB.csv')
    processor = preprocess.Preprocessor(data_folder,database)
    initialise(processor,download=False)
    main()