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
        self.encoder = self.build_encoder(input_dim, latent_dim)
        self.decoder = self.build_decoder(latent_dim, self.condition_dim, self.output_dim)

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

    def build_decoder(self, latent_dim, condition_dim, output_dim):
        '''
        Build the decoder part of the model
        '''
        decoder_inputs = tf.keras.Input(shape=(latent_dim + condition_dim,))
        x = layers.Dense(128, activation='relu')(decoder_inputs)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dense(512, activation='relu')(x)
        x = layers.Dense(output_dim[0] * output_dim[1] * output_dim[2], activation='relu')(x)
        outputs = layers.Reshape(output_dim)(x)
        return tf.keras.Model(decoder_inputs, outputs, name='decoder')

    def sampling(self, args):
        '''
        Samples from the latent space using the reparameterisation
        '''
        z_mean, z_log_var = args
        epsilon = tf.random.normal(shape=tf.shape(z_mean), mean=0.0, stddev=1.0)
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

    def call(self, inputs):
        '''
        Runs the model
        '''
        encoder_inputs, condition_vector = inputs
        z_mean, z_log_var = self.encoder(encoder_inputs)
        z = self.sampling([z_mean, z_log_var])
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
        encoder_inputs, _ = inputs
        z_mean, z_log_var = self.encoder(encoder_inputs)
        reconstruction_loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(encoder_inputs, reconstructed))
        kl_loss = -0.5 * tf.reduce_mean(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
        return reconstruction_loss + kl_loss