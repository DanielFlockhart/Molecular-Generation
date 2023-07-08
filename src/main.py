import os,sys,random
from PIL import *
import numpy as np
import tensorflow as tf

from preprocessing import preprocess,database,image_manipulation
from training import vae
from postprocessing import *
from deployment import *
from CONSTANTS import *
data_folder = r'C:\Users\0xdan\Documents\CS\WorkCareer\Chemistry Internship\Project-Code\data\dataset'


def initialise():
    preprocess.clear_folder(data_folder+r"\test-data")
    db = preprocess.Database(fr'{data_folder}\CSD_EES_DB.csv')
    db.load_data()
    molecules_storage = fr'{data_folder}\test-data'
    
    smiles = db.get_smiles()
    preprocess.normalise_images(smiles,IMG_SIZE,molecules_storage,STD_DEV)
    print("Pre Processing Ended")

def main():
    network = vae.VAE()
    pass

if __name__ == "__main__":
    initialise() #  ! WARNING ! This will clear the test-data folder
    #main()

db = preprocess.Database(fr'{data_folder}\CSD_EES_DB.csv')
preprocess.Preprocessor("dataset1",data_folder,db)