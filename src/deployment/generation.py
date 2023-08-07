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
        '''
        with tf.keras.utils.custom_object_scope({'VariationalAutoencoder': VariationalAutoencoder}):
            self.model = tf.keras.models.load_model(model_path)

        self.model.training = False
        self.model.compile()

    def generate_noise(self,x=ml_constants.LATENT_DIM+ml_constants.CONDITIONS_SIZE):
        '''
        Generate a random noise vector for input to network.
        '''
        np.random.seed(random.randint(0,1000))
        return np.random.uniform(size=(x)).astype(np.float32)  


    def generate_molecule(self, vector, condition):
        '''
        Generate a molecule from a vector and condition using the trained model.
        '''
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
        pil_image = Image.fromarray(image, mode='L')

        return pil_image
    
    def save_image(self,image,path):
        '''
        Saves an image to a path
        '''
        image.save(path)
        image.show()