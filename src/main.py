import os,sys,random
from PIL import *
import numpy as np
import tensorflow as tf
from preprocessing import preprocessing
from training import train,vae
from postprocessing import *
from deployment import *
from Constants import file_constants, ml_constants
from ui.terminal_ui import *
from ui.dialogue import *
sys.path.insert(0, os.path.abspath('..'))

sys.path.insert(0, os.path.abspath('../training'))
from training.vae import *
from deployment import generation,deploy

from visualisation import results

    
def preprocess_data(name="CSD_EES_DB.csv"):
    '''
    Initialises the program and get it ready for training
    '''
    print(format_title("Preprocessing Data"))
    processor = preprocessing.Preprocessor(preprop_constants.EMBEDDING_MODEL,name)
    processor.process(subset=False)

def train_model(model,name):
    '''
    Main training loop
    '''

    print(format_title(f"Training Model"))

    # Define the initial learning rate and decay rate
    initial_learning_rate = ml_constants.LRN_RATE
    
    # Initialize the optimizer with the learning rate schedule
    optimizer = tf.keras.optimizers.Adam(learning_rate=initial_learning_rate)
    # Train Model
    model.training = True  # Set training to True before training
    trained_model = train.train_model(model,optimizer,use_subset=False)

    # Save Model
    train.save_model(trained_model,name)

def generate_molecule(model):
    app = deploy.App(fr"{file_constants.MODELS_FOLDER}\vae\model.h5")
    start_mol,new_mols = app.get_mols()
    report = results.Report(start_mol,new_mols)
    report.build_report()



def main(model):
    '''
    Main Entrance for the program
    '''

    user_choice = get_user_choice()
    confirmed = confirm_choice(user_choice)


    if confirmed:
        if user_choice == '1':
            model.training = True
            train_model(model,"vae")

        elif user_choice == '2':
            print(format_title("Generating Molecule"))
            model.training = False
            generate_molecule(model)
            
    else:
        print("Choice not confirmed. Exiting...")

if __name__ == "__main__":
    
    #preprocess_data(fr"{file_constants.DATASET}\dataset.csv")
    


    vae_model = vae.VariationalAutoencoder(ml_constants.INPUT_SIZE,ml_constants.LATENT_DIM,ml_constants.OUTPUT_DIM,ml_constants.CONDITIONS_SIZE)

    main(vae_model)

