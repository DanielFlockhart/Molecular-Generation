import tensorflow as tf
from PIL import Image
import numpy as np
import os
import sys
import pandas as pd
sys.path.insert(0, os.path.abspath('..'))

sys.path.insert(0, os.path.abspath('../training'))
from architectures import vae_im_to_sm, vae_sm_to_im

from Constants import ml_constants,file_constants,preprop_constants
from utilities import img_utils,utils


class Generator:
    def __init__(self,model_path):
        '''
        Generator class for generating images from a trained model
        '''
        with tf.keras.utils.custom_object_scope({'VariationalAutoencoder': vae_sm_to_im.VariationalAutoencoder}):
            self.model = tf.keras.models.load_model(model_path)

        self.model.training = False
        self.model.compile()
        self.database = pd.read_csv(f"{file_constants.DATA_FOLDER + file_constants.DATASET}/dataset.csv")

    def generate_multiple_molecules(self, vector, condition,amount=64):
        '''
        Generate multiple molecules from randomly sampled vectors and conditions.
        '''
        images = []
        for i in range(amount):
            generated_molecule = self.generate_molecule(vector, condition)
            generated_molecule.save(f"{file_constants.GENERATED_FOLDER}/generated_molecule_{i}.png")
            images.append(generated_molecule)
        return images

    def generate_molecule(self, vector, condition):
        '''
        Generate a molecule from a vector and condition using the trained model.
        '''
        vector = np.expand_dims(vector, axis=0)
        condition = np.expand_dims(condition, axis=0)
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

    def create_condition(self):
        (min_values,max_values) = self.preprocess_conditions()
        condition = self.get_conditions(min_values,max_values)
        return condition


    def get_conditions(self,mins,maxs):
        
        conditions = []
        for key in preprop_constants.keys:
            condition = self.normalise_condition(float(input("Enter a value for " + key + ": ")),mins[key],maxs[key])
            conditions.append(condition)
        

        return utils.normalise_vector(conditions) # Test With Removing This
    
    def normalise_condition(self,condition,mins,maxs):
        '''
        Normalise the conditions to be between 0 and 1
        '''
        return  (condition - mins) / (maxs - mins)

    def preprocess_conditions(self):

        min_values = self.database[preprop_constants.keys].min()
        max_values = self.database[preprop_constants.keys].max()
        return (min_values,max_values)
