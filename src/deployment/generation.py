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

sys.path.insert(0, os.path.abspath('../training'))
from training.vae import *


from Constants import ml_constants
from utilities import img_utils

class Generator:
    def __init__(self,model_path):
        '''
        Generator class for generating images from a trained model

        Parameters
        ----------
        model_path : str
            The path to the model
        '''
        self.model = tf.saved_model.load(model_path)
        self.model.training = False
    def build_model(self):
        '''
        Build the model from previously trained weights
        '''
        pass

    def generate_noise(self):
        '''
        Generate a random noise vector for input to network.
        '''
        np.random.seed(random.randint(0,1000))
        return np.random.uniform(size=(ml_constants.LATENT_DIM,)).astype(np.float32)  
    
    def generate_image_vae(self,noise):
        '''
        Generate an image from a noise vector
        '''
        noise = np.expand_dims(self.generate_noise(), axis=0)
        
        image = self.model.decoder(noise)
        #noise = np.expand_dims(np.array([-1]).astype(np.float32), axis=0)

        #image = self.model.decoder(noise)
        # Reshape the image to 100x100
        image = tf.reshape(image,ml_constants.OUTPUT_DIM)
        image = np.squeeze(image)

        # Convert the image to a NumPy array and cast to uint8
        image = (image * 255).astype(np.uint8)
        # Create a PIL Image from the NumPy array
        # Image shape is (100,100,1)
        pil_image = Image.fromarray(image, mode='L')

        return pil_image

    def generate_through_vae(self,vector):
        #ector = np.expand_dims(vector.astype(np.float32), axis=0)
        vector = np.expand_dims(vector, axis=0)

        # Convert the vector to a tensor
        vector = tf.convert_to_tensor(vector, dtype=tf.float32)

        # Call the model with the reshaped vector
        image = self.model(vector)
        image = tf.reshape(image, ml_constants.OUTPUT_DIM)
        image = np.squeeze(image)

        # Convert the image to a NumPy array and cast to uint8
        image = (image * 255).astype(np.uint8)

        # Create a PIL Image from the NumPy array
        # Image shape is (100, 100, 1)
        pil_image = Image.fromarray(image, mode='L')

        return pil_image
  


    
    def save_image(self,image,path):
        '''
        Saves an image to a path
        '''
        image.save(path)
        image.show()