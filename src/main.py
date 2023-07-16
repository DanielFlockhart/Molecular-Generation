import os,sys,random
from PIL import *
import numpy as np
import tensorflow as tf
from preprocessing import preprocessing
from training import train,vae,gan
from postprocessing import *
from deployment import *
from utilities import utils, file_utils, img_utils
from Constants import ui_constants, file_constants, preprop_constants, ml_constants
from ui.terminal_ui import *
from ui.dialogue import *

from deployment import generation
def initialise():
    '''
    Initialises the program and get it ready for training/generation
    
    If there is already a dataset in \resized\, then it will skip the download step
    If there is already a preprocessed dataset in \data\, then it will skip the preprocessing step
    '''

    print(format_title("Initialising"))
    #if utils.perform_checks(file_constants.PROCESSED_DATA): # Rework Check
    preprocess_data()
    
    
    

def preprocess_data():
    '''
    Initialises the program and get it ready for training

    Parameters
    ----------
    download : bool, optional
        Whether to redownload the data from the database, by default Falses
    '''
    embedding_model ="seyonec/ChemBERTa-zinc-base-v1"
    print(format_title("Preprocessing Data"))
    processor = preprocessing.Preprocessor(embedding_model,"CSD_EES_DB") # Working Here
    file_utils.clear_folder(file_constants.PROCESSED_DATA)
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
    # imgs = img_utils.load_images()
    # if use_subset:
    #     print("You have selected to use subset of data for training process.")
    #     imgs = imgs[:ml_constants.TRAIN_SUBSET_COUNT]
    
    print(format_title(f"Training Model {name}"))

    # Create Default Optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=ml_constants.LRN_RATE)
    # Train Model
    trained_model = train.train_model(model,optimizer)

    # Save Model
    train.save_model(trained_model,name)

def generate_molecule():
    '''
    Generates a new molecule with a previously trained model of either VAE or GAN
    '''
    print(format_title("Generating Molecule"))
    gen = generation.Generator(fr"{file_constants.MODELS_FOLDER}\vae")
    for x in range(100):
        gen.generate_image_vae(gen.generate_noise()).save(fr"{file_constants.GENERATED_FOLDER}\vae\{x}.png")
    #gen.generate_image_gan()

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
            train_model(models[0],"vae",use_subset=False)
        elif user_choice == '2':
            generate_molecule()
    else:
        print("Choice not confirmed. Exiting...")

if __name__ == "__main__":
    # Preprocess the data
    print("This Program is currently a work in progress - Limited functionality to just generating dataset")
    #initialise()

    vae_model = vae.VariationalAutoencoder(ml_constants.INPUT_SIZE,ml_constants.LATENT_DIM + ml_constants.CONDITIONS_SIZE)
    
    main(models=[vae_model])




