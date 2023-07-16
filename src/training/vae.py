import sys, os
import numpy as np

import tensorflow as tf
from tensorflow.keras import layers

from tensorflow.keras import regularizers

class VariationalAutoencoder(tf.keras.Model):
    def __init__(self, input_dim, latent_dim, condition_dim, output_dim):
        super(VariationalAutoencoder, self).__init__()
        self.condition_dim = condition_dim
        self.output_dim= output_dim
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()

    def build_encoder(self):
        '''
        Building the encoder part of the model
        '''
        encoder_inputs = tf.keras.Input(shape=self.input_dim)
        x = layers.Dense(512, activation='relu')(encoder_inputs)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dense(128, activation='relu')(x)
        z_mean = layers.Dense(self.latent_dim)(x)
        z_log_var = layers.Dense(self.latent_dim)(x)
        return tf.keras.Model(encoder_inputs, [z_mean, z_log_var], name='encoder')

    def build_decoder(self):
        '''
        Build the decoder part of the model
        '''
        decoder_inputs = tf.keras.Input(shape=(self.latent_dim +self.condition_dim,))
        x = layers.Dense(128, activation='relu')(decoder_inputs)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dense(512, activation='relu')(x)
        x = layers.Dense(self.output_dim[0] * self.output_dim[1] * self.output_dim[2], activation='relu')(x)
        outputs = layers.Reshape(self.output_dim)(x)
        return tf.keras.Model(decoder_inputs, outputs, name='decoder')




       
    def sampling(self, args):
        '''
        Samples from the latent space using the reparameterisation
        '''
        z_mean, z_log_var = args
        epsilon = tf.random.normal(shape=(tf.shape(z_mean)[0], self.latent_dim), mean=0.0, stddev=1.0)
        return z_mean + tf.exp(z_log_var / 2) * epsilon

    def call(self, inputs, condition_vector):
        '''
        Runs the model
        '''
        z_mean, z_log_var = self.encoder(inputs)
        z = self.sampling([z_mean, z_log_var])
        # Concatenate the condition vector with the latent variable z
        z_condition = tf.concat([z, condition_vector], axis=1)
        
        reconstructed = self.decoder(z_condition)
        return reconstructed

    def compute_loss(self, inputs, reconstructed):
        '''
        Calculates the loss of the model
        
        Parameters
        ----------
        inputs : tensor
            The input images
        reconstructed : tensor
            The reconstructed images
            
        Returns
        -------
        loss : tensor
            The calculated loss value
        '''
        z_mean, z_log_var = self.encoder(inputs)
        reconstruction_loss = tf.keras.losses.binary_crossentropy(inputs, reconstructed)
        reconstruction_loss = tf.reduce_mean(reconstruction_loss)
        kl_loss = -0.5 * tf.reduce_mean(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
        return reconstruction_loss + kl_loss





