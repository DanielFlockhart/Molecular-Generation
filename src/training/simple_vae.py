import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import os


# Define the VAE
class VAE(keras.Model):
    def __init__(self, latent_dim):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()

    def build_encoder(self):
        inputs = keras.Input(shape=(128, 128, 3))
        x = layers.Conv2D(32, kernel_size=3, strides=2, padding="same", activation="relu")(inputs)
        x = layers.Conv2D(64, kernel_size=3, strides=2, padding="same", activation="relu")(x)
        x = layers.Flatten()(x)
        z_mean = layers.Dense(self.latent_dim)(x)
        z_log_var = layers.Dense(self.latent_dim)(x)
        return keras.Model(inputs, [z_mean, z_log_var], name="encoder")

    def build_decoder(self):
        latent_inputs = keras.Input(shape=(self.latent_dim,))
        x = layers.Dense(16 * 16 * 64, activation="relu")(latent_inputs)
        x = layers.Reshape((16, 16, 64))(x)
        x = layers.Conv2DTranspose(64, kernel_size=3, strides=2, padding="same", activation="relu")(x)
        x = layers.Conv2DTranspose(32, kernel_size=3, strides=2, padding="same", activation="relu")(x)
        outputs = layers.Conv2DTranspose(3, kernel_size=3, padding="same", activation="sigmoid")(x)
        return keras.Model(latent_inputs, outputs, name="decoder")

    def encode(self, x):
        z_mean, z_log_var = self.encoder(x)
        return z_mean, z_log_var

    def reparameterize(self, z_mean, z_log_var):
        batch_size = tf.shape(z_mean)[0]
        epsilon = tf.random.normal(shape=(batch_size, self.latent_dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

    def decode(self, z):
        return self.decoder(z)

    def call(self, inputs):
        z_mean, z_log_var = self.encode(inputs)
        z = self.reparameterize(z_mean, z_log_var)
        reconstructed = self.decode(z)
        return reconstructed


# Load images from directory
def load_images(directory, img_size):
    images = []
    for filename in os.listdir(directory):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            img_path = os.path.join(directory, filename)
            image = keras.preprocessing.image.load_img(img_path, target_size=(img_size, img_size))
            image = keras.preprocessing.image.img_to_array(image)
            image = image.astype(np.float32) / 255.0
            images.append(image)
    return np.array(images)


def vae_loss(inputs, reconstructed):
    resized_reconstructed = tf.image.resize(reconstructed, tf.shape(inputs)[1:3])
    reconstruction_loss = keras.backend.mean(keras.backend.square(inputs - resized_reconstructed))
    kl_loss = -0.5 * keras.backend.mean(1 + vae.encoder.outputs[1] - keras.backend.square(vae.encoder.outputs[0]) - keras.backend.exp(vae.encoder.outputs[1]))
    return reconstruction_loss + kl_loss


# Set the hyperparameters
latent_dim = 64
epochs = 10
batch_size = 64


# Create the VAE model
vae = VAE(latent_dim)

# Load and preprocess images
training_images_dir = r"C:\Users\0xdan\Documents\CS\WorkCareer\Chemistry Internship\Project-Code\data\dataset\test-data"
IMG_SIZE = 128
training_images = load_images(training_images_dir, IMG_SIZE)

# Enable eager execution for TensorFlow data functions
tf.data.experimental.enable_debug_mode()

# Compile the model
vae.compile(optimizer=keras.optimizers.Adam(), loss=vae_loss, run_eagerly=True)

# Train the VAE
vae.fit(training_images, training_images, epochs=epochs, batch_size=batch_size)

# Save the model weights
vae.save_weights(r'C:\Users\0xdan\Documents\CS\WorkCareer\Chemistry Internship\Project-Code\data\models\vae\model.h5')
