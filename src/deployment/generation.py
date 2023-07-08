'''
Developer Note - Daniel Flockhart

This file is unfinished and is not used in the project. It is a work in progress.

'''



import tensorflow as tf
from PIL import Image
import numpy as np
import os
import sys

sys.path.insert(0, os.path.abspath('..'))
from preprocessing import inputify as im

sys.path.insert(0, os.path.abspath('../training'))
from vae import *

from CONSTANTS import *

# Load the model
vae = VAE(LATENT_DIM)
optimizer = tf.keras.optimizers.Adam(learning_rate=LRN_RATE)
vae.compile(optimizer=optimizer)

# Call the model once to create its variables
vae(np.zeros((1, IMG_SIZE, IMG_SIZE, 3)))

# Load the weights
vae.load_weights(r'C:\Users\0xdan\Documents\CS\WorkCareer\Chemistry Internship\Project-Code\data\models\vae\model.h5')


class Generator:
    def __init__(self,model_path):
        '''
        Generator class for generating images from a trained model

        Parameters
        ----------
        model_path : str
            The path to the model
        '''
        self.model = tf.keras.models.load_model(model_path)

    def build_model(self):
        '''
        Build the model from previously trained weights
        '''
        pass

    def generate_noise(self):
        '''
        Generate a random noise vector for input to network.
        '''
        return np.random.normal(size=(1, LATENT_DIM))
    
    def generate_image_vae(self,noise_vector):
        '''
        Generate an image from a noise vector
        '''
        image = self.model.decode(noise_vector)
        image = im.tensor_to_image(image)
        return image
    
    def generate_image_gan(self,noise_vector):
        pass


    
    def save_image(self,image,path):
        '''
        Saves an image to a path
        '''
        image.save(path)
        image.show()