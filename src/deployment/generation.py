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
        with tf.keras.utils.custom_object_scope({'VariationalAutoencoder': VariationalAutoencoder}):
            self.model = tf.keras.models.load_model(model_path)

        self.model.training = False
        self.model.compile()
    def build_model(self):
        '''
        Build the model from previously trained weights
        '''
        pass

    def generate_noise(self,x=ml_constants.LATENT_DIM+12):
        '''
        Generate a random noise vector for input to network.
        '''
        np.random.seed(random.randint(0,1000))
        return np.random.uniform(size=(x)).astype(np.float32)  
    
    def generate_image_vae(self,conditions=None):
        '''
        Generate an image from a noise vector
        '''
        noise = np.expand_dims(self.generate_noise(), axis=0)
        if conditions is not None:
            conditions = np.expand_dims(conditions, axis=0)
            noise = tf.concat([noise, conditions], axis=1)

        # Use the model's layers to access the decoder
        decoder = self.model.get_layer('variational_autoencoder_1')
        image = decoder([noise, conditions])

        image = tf.reshape(image,ml_constants.OUTPUT_DIM)
        image = np.squeeze(image)

        image = (image * 255).astype(np.uint8)
        pil_image = Image.fromarray(image, mode='L')

        return pil_image

    def generate_through_vae(self, vector, condition):
        vector = np.expand_dims(vector, axis=0)
        condition = np.expand_dims(condition, axis=0)
        vector = tf.concat([vector, tf.cast(condition, tf.float32)], axis=1)

        # Call the decoder to generate the image
        image = self.model.predict([vector,condition])


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