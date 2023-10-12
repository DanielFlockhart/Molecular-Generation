import tensorflow as tf
import numpy as np
import sys,os,random,time
from tensorflow.keras import layers,Model,regularizers
sys.path.insert(0, os.path.abspath('..'))
from keras import backend as K
from ml.training.get_inputs import get_training_data
from Constants import preprop_constants,ml_constants
from PIL import Image

import matplotlib.pyplot as plt
class VariationalAutoencoder(tf.keras.Model):
    def __init__(self, input_dim, latent_dim, output_dim,conditions_size,temperature=1):
        super(VariationalAutoencoder, self).__init__()
        self.output_dim = output_dim
        self.temperature = temperature
        self.condition_size = conditions_size
        self.encoder = self.build_encoder(input_dim, latent_dim)
        self.decoder = self.build_decoder(latent_dim+self.condition_size, self.output_dim)
        self.training = True  # Set the training attribute to True initially
        self.latent_stores = []
    def build_encoder(self, input_dim, latent_dim):
        '''
        Building the encoder part of the model
        '''
        encoder_inputs = tf.keras.Input(shape=input_dim)

        # Add batch normalization after each Dense layer in the encoder
        x = layers.Dense(512, activation="relu")(encoder_inputs)
        x = layers.BatchNormalization()(x)  # Add batch normalization
        x = layers.Dense(256, activation="relu")(x)
        x = layers.BatchNormalization()(x)  # Add batch normalization
        x = layers.Dense(128, activation="relu")(x)
        x = layers.BatchNormalization()(x)  # Add batch normalization
        z_mean = layers.Dense(latent_dim, activation="relu")(encoder_inputs)
        z_log_var = layers.Dense(latent_dim, activation="relu")(encoder_inputs)
        return tf.keras.Model(encoder_inputs,[z_mean,z_log_var], name='encoder')#[z_mean, z_log_var], name='encoder')

    def build_decoder(self, latent_dim, output_dim):
        '''
        Build the decoder part of the model
        '''
        # Define the input layer for the latent vector
        latent_inputs = tf.keras.Input(shape=(latent_dim,), name='latent_inputs')

        # Reshape the latent vector to match the input shape for the convolutional layers
        x = layers.Dense(units=output_dim[0]*output_dim[1]*output_dim[2], activation='relu')(latent_inputs)
        x = layers.Reshape(target_shape=(output_dim[0], output_dim[1], output_dim[2]))(x)
        x = layers.Conv2DTranspose(128, kernel_size=(3, 3), padding='same',activation="relu")(x)
        x = layers.BatchNormalization()(x)  # Add batch normalization
        x = layers.Conv2DTranspose(64, kernel_size=(3, 3), padding='same',activation="relu")(x)
        x = layers.BatchNormalization()(x)  # Add batch normalization
        x = layers.Conv2DTranspose(32, kernel_size=(3, 3), padding='same',activation="relu")(x)
        x = layers.BatchNormalization()(x)  # Add batch normalization
        x = layers.Conv2DTranspose(16, kernel_size=(3, 3), padding='same', activation="relu")(x)
        x = layers.BatchNormalization()(x)  # Add batch normalization
        # Output layer with tanh activation instead of relu
        outputs = layers.Conv2DTranspose(filters=output_dim[2], kernel_size=(3,3), padding='same',activation="sigmoid")(x)
        outputs = layers.Flatten()(outputs)

        # Define the decoder model
        decoder = tf.keras.Model(latent_inputs, outputs, name='decoder')
        return decoder
