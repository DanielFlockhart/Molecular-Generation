import os,sys,random
from PIL import *
import numpy as np
import tensorflow as tf
from preprocessing import preprocess
from training import train,vae,gan
from postprocessing import *
from deployment import *
from utils import *
from CONSTANTS import *
from ui.terminal_ui import *
from ui.dialogue import *
from preprocessing import inputify as im

from deployment import generation
def initialise():
    '''
    Initialises the program and get it ready for training/generation
    
    If there is already a dataset in \resized\, then it will skip the download step
    If there is already a preprocessed dataset in \data\, then it will skip the preprocessing step
    '''

    print(format_title("Initialising"))
    if perform_checks(PROCESSED_DATA):
        preprocess_data()
    
    
    

def preprocess_data():
    '''
    Initialises the program and get it ready for training

    Parameters
    ----------
    download : bool, optional
        Whether to redownload the data from the database, by default Falses
    '''
    print(format_title("Preprocessing Data"))
    database = preprocess.Database(fr'{DATA_FOLDER}\CSD_EES_DB.csv')
    processor = preprocess.Preprocessor(DATA_FOLDER,database,"CSD_EES_DB")
    processor.clear_folder(PROCESSED_DATA)
    processor.process()

    

def train_model(model,name,use_subset=False):
    '''
    Main training loop

    Parameters
    ----------
    model : tf.keras.Model
        The model to train
    name : str
        The name of the model
    imgs : list
        A list of images to train on
    use_subset : bool, optional
        Whether to use a subset of the data, by default False
    '''
    imgs = im.load_images(PROCESSED_DATA,IMG_SIZE)
    if use_subset:
        print("You have selected to use subset of data for training process.")
        imgs = imgs[:TRAIN_SUBSET_COUNT]
    
    print(format_title(f"Training Model {name}"))

    # Create Default Optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=LRN_RATE)

    # Train Model
    trained_model = train.train_model(model,imgs,optimizer)

    # Save Model
    train.save_model(trained_model,name)

def generate_molecule():
    '''
    Generates a new molecule with a previously trained model of either VAE or GAN
    '''
    print(format_title("Generating Molecule"))
    gen = generation.Generator()
    gen.generate_image_vae()
    gen.generate_image_gan()


def main(models):
    '''
    Main function for the program

    Parameters
    ----------
    models : list
        A list of possible models to train
    '''

    user_choice = get_user_choice()
    confirmed = confirm_choice(user_choice)

    if confirmed:
        if user_choice == '1':
            # Will add in option to choose model type later
            train_model(models[0],"vae",use_subset=True)
        elif user_choice == '2':
            # Will add in option to choose model type later
            #generate_molecule()
            raise NotImplementedError
    else:
        print("Choice not confirmed. Exiting...")

if __name__ == "__main__":
    
    #initialise()

    vae_model = vae.VAE(LATENT_DIM)
    gan_model = gan.GAN(gan.Generator(),gan.Discriminator())
    main(models=[vae_model,gan_model])




