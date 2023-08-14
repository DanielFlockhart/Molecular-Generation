import tensorflow as tf
import numpy as np
import sys,os,random,time
from tensorflow.keras import layers,Model,regularizers
sys.path.insert(0, os.path.abspath('..'))
from keras import backend as K
from training.get_inputs import get_training_data
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
        x = layers.Conv2DTranspose(128, kernel_size=(4, 4), padding='same',activation="relu")(x)
        x = layers.Conv2DTranspose(64, kernel_size=(4, 4), padding='same',activation="relu")(x)
        x = layers.Conv2DTranspose(32, kernel_size=(4, 4), padding='same',activation="relu")(x)
        x = layers.Conv2DTranspose(16, kernel_size=(3, 3), padding='same',activation="relu")(x)
        # Output layer with tanh activation instead of relu
        outputs = layers.Conv2DTranspose(filters=1, kernel_size=(3, 3), padding='same',activation="relu")(x)
        outputs = layers.Flatten()(outputs)
        outputs = layers.Dense(units=output_dim[0] * output_dim[1] * output_dim[2],activation="sigmoid")(latent_inputs)
        # Define the decoder model
        decoder = tf.keras.Model(latent_inputs, outputs, name='decoder')
        return decoder

    
    def sampling(self, args):
        '''
        Samples from the latent space using the reparameterisation
        '''
        z_mean, z_log_var = args
        epsilon = tf.random.normal(shape=tf.shape(z_mean))
        scaled_epsilon = epsilon * self.temperature
        res = z_mean + tf.exp(0.5 * z_log_var) * scaled_epsilon
        return res

    def call(self, inputs,condition_vector):
        '''
        Runs the model

        if it is converging on just averaging the output then the inputs are having no effect.
        '''
        
        z_mean, z_log_var = self.encoder(inputs)
        if self.training:
            z = self.sampling([z_mean, z_log_var])
        else:
            z = z_mean
        
        z_condition = tf.concat([z, condition_vector], axis=1)
        reconstructed = self.decoder(z_condition)



        return reconstructed

    def compute_loss(self, inputs, targets, reconstructed, sample_weight=None, training=False):
        '''
        Calculates the loss of the model
        '''

        # Calculate the SSIM loss between the targets and the reconstructed images
        #ssim_loss = tf.reduce_mean(1 - tf.image.ssim(targets, reconstructed, max_val=1.0))

        bce_loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(targets, reconstructed))

        mse_loss = tf.reduce_mean(tf.keras.losses.mean_squared_error(targets, reconstructed))

        z_mean, z_log_var = self.encoder(inputs)

        
        kl_loss = -0.5 * tf.reduce_mean(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))

        reconstruction_loss =  mse_loss+ kl_loss #BCE or MSE

        # Changes 07/07/2023
        # Re added VAE PART - changed loss to mse and kl


        #tf.print(" - total_loss:", reconstruction_loss)
        return reconstruction_loss
    
    def show_image(self, tensor,name):
        """
        Displays an image from the given tensor.
        """
        # Assuming the tensor shape is (batch_size, height, width, channels)
        if len(tensor.shape) == 4:
            tensor = tensor[0]  # Take the first image from the batch
        elif len(tensor.shape) == 3:
            tensor = tensor  # Use the tensor as is

        def show_image_from_np(tensor_np):
            tensor_reshaped = tf.reshape(tensor_np, (preprop_constants.IMG_SIZE, (preprop_constants.IMG_SIZE)))
            # Convert the tensor to a NumPy array
            tensor_rescaled = tf.cast(tensor_reshaped * 255, tf.uint8)

            # Create and display the image using PIL
            image = Image.fromarray(tensor_rescaled.numpy())
            image.save(f"{name}.png")

        # Use tf.py_function to call the Python function within the graph function
        tf.py_function(show_image_from_np, [tensor], [])