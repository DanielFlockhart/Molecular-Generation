import tensorflow as tf
import numpy as np
import sys,os
from tensorflow.keras import layers,Model
sys.path.insert(0, os.path.abspath('..'))

from training.get_inputs import get_training_data


from Constants import preprop_constants



# Define the sizes of input and output vectors
input_size = 768
output_size = 100 * 100  # 100x100 pixels

class Encoder(Model):
    def __init__(self, input_shape, latent_dim):
        super(Encoder, self).__init__()
        self.input_layer = layers.Input(shape=input_shape)
        # Add your custom encoder layers here
        # ...
        self.mean = layers.Dense(latent_dim)
        self.log_variance = layers.Dense(latent_dim)

    def call(self, x):
        x = self.input_layer(x)
        mean = self.mean(x)
        log_variance = self.log_variance(x)
        return mean, log_variance

class Decoder(Model):
    def __init__(self, latent_dim, output_shape):
        super(Decoder, self).__init__()
        self.input_layer = layers.Input(shape=(latent_dim,))
        # Add your custom decoder layers here
        # ...
        self.outputs = layers.Dense(output_size, activation='sigmoid')
        self.reshape = layers.Reshape((100, 100))

    def call(self, z):
        x = self.input_layer(z)
        outputs = self.outputs(x)
        outputs = self.reshape(outputs)
        return outputs

class Sampling(layers.Layer):
    def call(self, inputs):
        mean, log_variance = inputs
        epsilon = tf.random.normal(shape=tf.shape(mean))
        return mean + tf.exp(0.5 * log_variance) * epsilon

class VariationalAutoencoder(Model):
    def __init__(self, encoder, decoder):
        super(VariationalAutoencoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def call(self, x):
        mean, log_variance = self.encoder(x)
        z = Sampling()([mean, log_variance])
        output_layer = self.decoder(z)
        return output_layer

def vae_loss(targets, predictions, mean, log_variance):
    # Calculate reconstruction loss
    reconstruction_loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(targets, predictions))
    # Calculate KL divergence loss
    kl_loss = -0.5 * tf.reduce_mean(1 + log_variance - tf.square(mean) - tf.exp(log_variance))
    # Combine the losses
    return reconstruction_loss + kl_loss

labels,vectors,conditions,targets = get_training_data(1000)
X_train = vectors
Y_train = targets



latent_dim = 64  # Adjust this based on your needs
encoder = Encoder(input_shape=(input_size,), latent_dim=latent_dim)
decoder = Decoder(latent_dim=latent_dim, output_shape=(100, 100))
vae = VariationalAutoencoder(encoder, decoder)
vae.compile(optimizer='adam', loss=lambda targets, predictions: vae_loss(targets, predictions, encoder.output[0], encoder.output[1]))

print(len(X_train))
print(len(Y_train))
# Train the VAE model
vae.fit(X_train, Y_train, batch_size=32, epochs=50)

# Generate images using the trained VAE
# Sample from the latent space and decode the samples to get images
num_samples = 10
latent_samples = np.random.normal(size=(num_samples, latent_dim))
generated_images = decoder.predict(latent_samples)


