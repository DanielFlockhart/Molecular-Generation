import tensorflow as tf
import numpy as np
import sys,os
from tensorflow.keras import layers,Model
sys.path.insert(0, os.path.abspath('..'))

from training.get_inputs import get_training_data
from Constants import preprop_constants

class VariationalAutoencoder(tf.keras.Model):
    def __init__(self, input_dim, latent_dim, output_dim,temperature=1.0):
        super(VariationalAutoencoder, self).__init__()
        self.output_dim = output_dim
        self.temperature = temperature
        self.encoder = self.build_encoder(input_dim, latent_dim)
        self.decoder = self.build_decoder(latent_dim, self.output_dim)

    def build_encoder(self, input_dim, latent_dim):
        '''
        Building the encoder part of the model
        '''
        encoder_inputs = tf.keras.Input(shape=input_dim)
        x = layers.Dense(512, activation='relu')(encoder_inputs)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dense(128, activation='relu')(x)
        z_mean = layers.Dense(latent_dim)(x)
        z_log_var = layers.Dense(latent_dim)(x)
        return tf.keras.Model(encoder_inputs, [z_mean, z_log_var], name='encoder')

    def build_decoder(self, latent_dim, output_dim):
        '''
        Build the decoder part of the model
        '''
        # Define the input layer for the latent vector
        latent_inputs = tf.keras.Input(shape=(latent_dim,), name='latent_inputs')
        
        # Reshape the latent vector to match the input shape for the convolutional layers
        x = layers.Dense(units=output_dim[0]* output_dim[1] * output_dim[2], activation='relu')(latent_inputs)
        x = layers.Reshape(target_shape=(output_dim[0],output_dim[1],output_dim[2]))(x)

        x = layers.Conv2DTranspose(filters=64, kernel_size=4, strides=2, padding='same', activation='relu')(x)
        x = layers.Conv2DTranspose(filters=32, kernel_size=4, strides=2, padding='same', activation='relu')(x)
        x = layers.Conv2DTranspose(filters=16, kernel_size=3, strides=2, padding='same', activation='relu')(x)
        x = layers.Conv2DTranspose(filters=8, kernel_size=3, strides=2, padding='same', activation='relu')(x)

        
        # # Output layer with relu activation instead of sigmoid
        outputs = layers.Conv2DTranspose(filters=1, kernel_size=4, strides=1, padding='same', activation='relu')(x)
        outputs = layers.Flatten()(outputs)
        outputs = layers.Dense(units=output_dim[0]* output_dim[1] * output_dim[2], activation='sigmoid')(outputs)

        # Define the decoder model
        decoder = tf.keras.Model(latent_inputs, outputs, name='decoder')
        
        return decoder

    def sampling(self, args):
        '''
        Samples from the latent space using the reparameterisation
        '''
        z_mean, z_log_var = args
        epsilon = tf.random.normal(shape=tf.shape(z_mean), mean=0.0, stddev=1.0)
        scaled_epsilon = epsilon * self.temperature
        return z_mean + tf.exp(0.5 * z_log_var) * scaled_epsilon

    def call(self, inputs):
        '''
        Runs the model
        '''
        z_mean, z_log_var = self.encoder(inputs)
        z = self.sampling([z_mean, z_log_var])
        reconstructed = self.decoder(z)
        return reconstructed

    def compute_loss(self, inputs, targets, reconstructed, sample_weight=None, training=False):
        '''
        Calculates the loss of the model
        '''
        z_mean, z_log_var = self.encoder(inputs)
        reconstruction_loss = tf.reduce_mean(tf.square(targets - reconstructed))  # Use MSE for reconstruction loss
        kl_loss = -0.5 * tf.reduce_mean(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
        return reconstruction_loss + kl_loss