import os,sys,random
from PIL import *
import numpy as np
import tensorflow as tf
from preprocessing import preprocess
from training import train,vae,gan
from postprocessing import *
from deployment import *
from CONSTANTS import *
from ui.terminal_ui import *
from preprocessing import inputify as im


def preprocess_data(download=False):
    '''
    Initialises the program and get it ready for training

    Parameters
    ----------
    download : bool, optional
        Whether to redownload the data from the database, by default Falses
    '''
    print(format_title("Preprocessing Data"))
    database = preprocess.Database(fr'{DATA_FOLDER}\CSD_EES_DB.csv')
    processor = preprocess.Preprocessor(DATA_FOLDER,database)
    processor.process(download=download)

    

def train_model(model,name,imgs):
    '''
    Main training loop
    '''
    print(format_title(f"Training Model {name}"))

    # Create Default Optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=LRN_RATE)

    # Train Model
    trained_model = train.train_model(model,imgs,optimizer)

    # Save Model
    train.save_model(trained_model,name)


def main():
    '''
    Check whether user wants to :
    1. Train Model
    2. Generate Images
    3. Exit
    '''
    print(format_title("Main Menu"))
    print("1. Train Model")
    print("2. Generate Images")
    print("3. Exit")
    choice = input("Enter Choice: ")
    if choice == "1":
        train_model()
    elif choice == "2":
        #generate_images()
        pass
    elif choice == "3":
        sys.exit(0)

if __name__ == "__main__":

    imgs = im.load_images(PROCESSED_DATA,IMG_SIZE)[:TRAIN_SUBSET_COUNT] # Train on smaller subset of data
    vae_model = vae.VAE(LATENT_DIM)
    gan_model = gan.GAN(gan.Generator(),gan.Discriminator())
    
    
    #preprocess_data(download=False)
    train_model(vae_model,"vae",imgs)
    train_model(gan_model,"gan",imgs)