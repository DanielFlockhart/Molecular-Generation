import tensorflow as tf
import numpy as np
from tensorflow.keras import layers
from training.get_inputs import get_training_data

import sys,os




class CVAE(tf.keras.Model):
    """Convolutional variational autoencoder."""
    def __init__(self, latent_dim):
        super(CVAE, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(780, 1)),
                tf.keras.layers.Dense(512),
                tf.keras.layers.Dense(256),
                tf.keras.layers.Dense(latent_dim + latent_dim),
            ]
        )

        self.decoder = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
                tf.keras.layers.Dense(units=400*400*3, activation=tf.nn.relu),
                tf.keras.layers.Reshape(target_shape=(400, 400, 3)),
                tf.keras.layers.Conv2DTranspose(
                    filters=64, kernel_size=3, strides=2, padding='same',
                    activation='relu'),
                tf.keras.layers.Conv2DTranspose(
                    filters=32, kernel_size=3, strides=2, padding='same',
                    activation='relu'),
                # No activation
                tf.keras.layers.Conv2DTranspose(
                    filters=1, kernel_size=3, strides=1, padding='same'),
            ]
        )
    def encode(self, x):
        # Encode the input
        hidden = self.encoder(x)
        mean, log_var = tf.split(hidden, num_or_size_splits=2, axis=-1)  # Split into mean and log_var
        return mean, log_var
    
    def reparameterize(self, mean, log_var):
        std = tf.exp(0.5 * log_var)
        eps = tf.random.normal(shape=tf.shape(std))
        return mean + eps * std
    
    def decode(self, z, condition):
        # Concatenate latent representation and condition
        z = tf.concat([z, condition], axis=-1)
        return self.decoder(z)
    
    def call(self, x, condition):
        mean, log_var = self.encode(x)
        z = self.reparameterize(mean, log_var)
        return self.decode(z, condition)






# Define hyperparameters
input_size = 780
hidden_size = 512
latent_size = 128
condition_size = 12
learning_rate = 0.001
batch_size = 128
num_epochs = 50
count=1000

labels, input_data,condition_data,target_data = get_training_data(count)
# Create the VAE model
vae = CVAE(latent_size)

# Define loss function and optimizer
criterion = tf.keras.losses.MeanSquaredError()
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

# Assuming you have your preprocessed data ready as input_data, condition_data, and target_data
dataset = tf.data.Dataset.from_tensor_slices((input_data, condition_data, target_data))
dataset = dataset.batch(batch_size).shuffle(buffer_size=len(input_data))

# Training loop
for epoch in range(num_epochs):
    total_loss = 0
    for inputs, conditions, targets in dataset:
        with tf.GradientTape() as tape:
            outputs = vae(inputs, conditions)
            loss_recon = criterion(targets, outputs)
            mean, log_var = vae.encode(inputs)
            kl_divergence = -0.5 * tf.reduce_sum(1 + log_var - tf.square(mean) - tf.exp(log_var))
            loss = loss_recon + kl_divergence
        
        gradients = tape.gradient(loss, vae.trainable_variables)
        optimizer.apply_gradients(zip(gradients, vae.trainable_variables))
        
        total_loss += loss.numpy()
    
    # Print the average loss for this epoch
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss / len(dataset)}")

# Training is done!

# Save the entire model (architecture + weights)
vae.save('tensorflow_vae_model')