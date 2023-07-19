import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import sys,os
import numpy as np
sys.path.insert(0, os.path.abspath('..'))

from training import get_inputs

latent_dim = 32  # Dimensionality of the latent space
output_dim = 400 * 400  # Output dimension for the image vector

# Encoder
encoder_inputs = keras.Input(shape=(780,))  # Input for the vector representation
condition_inputs = keras.Input(shape=(12,))  # Input for the condition vector

# Concatenate the inputs
encoder_concat = layers.concatenate([encoder_inputs, condition_inputs])

# Encoder layers
x = layers.Dense(128, activation="relu")(encoder_concat)
x = layers.Dense(64, activation="relu")(x)

# Latent space mean and variance outputs
latent_mean = layers.Dense(latent_dim)(x)
latent_log_var = layers.Dense(latent_dim)(x)

# Sampling from the latent space
latent_sample = keras.Input(shape=(latent_dim,))
z = layers.concatenate([latent_sample, condition_inputs])

# Decoder layers
x = layers.Dense(64, activation="relu")(z)
x = layers.Dense(128, activation="relu")(x)
decoder_outputs = layers.Dense(output_dim, activation="sigmoid")(x)  # Output for the image vector

# Define the encoder and decoder models
encoder = keras.Model([encoder_inputs, condition_inputs], [latent_mean, latent_log_var], name="encoder")
decoder = keras.Model([latent_sample, condition_inputs], decoder_outputs, name="decoder")

# Define the VAE model
outputs = decoder([encoder([encoder_inputs, condition_inputs])[0], condition_inputs])
vae = keras.Model([encoder_inputs, condition_inputs], outputs, name="vae")

# Define the loss function
def vae_loss(inputs, outputs):
    reconstruction_loss = keras.losses.binary_crossentropy(inputs, outputs)
    reconstruction_loss *= output_dim
    kl_loss = 1 + latent_log_var - tf.square(latent_mean) - tf.exp(latent_log_var)
    kl_loss = tf.reduce_mean(kl_loss, axis=-1)
    kl_loss *= -0.5
    return tf.reduce_mean(reconstruction_loss + kl_loss)

# Compile the VAE model
vae.compile(optimizer=keras.optimizers.Adam(), loss=vae_loss)

# Train the VAE model

(x_train,conditions,y_train) = get_inputs.get_training_data()
print(np.array(x_train).shape)
print(np.array(conditions).shape)
print(np.array(y_train).shape)
vae.fit([np.array(x_train), np.array(conditions)], y_train, batch_size=32, epochs=10)