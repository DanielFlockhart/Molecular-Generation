import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import os
from PIL import Image

# Instead of using a variational autoencoder, we can use a simple autoencoder
# to generate images. This is a simple autoencoder that takes in an image and
# outputs an image. The encoder and decoder are both convolutional neural
# networks. The encoder takes in an image and outputs a vector of size

class SAE(tf.keras.Model):
    

class VAE(tf.keras.Model):
    def __init__(self, latent_dim):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()

    def build_encoder(self):
        encoder_input = tf.keras.Input(shape=(128, 128, 3))
        x = layers.Conv2D(32, 3, activation='relu', strides=2, padding='same')(encoder_input)
        x = layers.Conv2D(64, 3, activation='relu', strides=2, padding='same')(x)
        x = layers.Flatten()(x)
        x = layers.Dense(16, activation='relu')(x)
        z_mean = layers.Dense(self.latent_dim,name="z_mean")(x)
        z_log_var = layers.Dense(self.latent_dim,name="z_log_var")(x)
        encoder_output = z_mean, z_log_var
        return tf.keras.Model(encoder_input, encoder_output,name="encoder")

    def build_decoder(self):
        decoder_input = tf.keras.Input(shape=(self.latent_dim,))
        x = layers.Dense(8 * 8 * 256, activation='relu')(decoder_input)
        x = layers.Reshape((8, 8, 256))(x)
        x = layers.Conv2DTranspose(128, 3, activation='relu', strides=4, padding='same')(x)
        x = layers.Conv2DTranspose(32, 3, activation='relu', strides=4, padding='same')(x)
        decoder_output = layers.Conv2DTranspose(3, 3, activation='sigmoid', padding='same')(x)
        return tf.keras.Model(decoder_input, decoder_output,name="decoder")

    def encode(self, x):
        mean, log_var = self.encoder(x)
        epsilon = tf.random.normal(shape=tf.shape(mean))
        z = mean + tf.exp(0.5 * log_var) * epsilon
        return z, mean, log_var

    def decode(self, z):
        reconstructed = self.decoder(z)
        return reconstructed

    def call(self, x):
        z, mean, log_var = self.encode(x)
        reconstructed = self.decode(z)

        reconstruction_loss = tf.reduce_mean(tf.square(x - reconstructed))
        kl_loss = -0.5 * tf.reduce_mean(1 + log_var - tf.square(mean) - tf.exp(log_var))
        total_loss = reconstruction_loss + kl_loss
        self.add_loss(total_loss)

        return reconstructed, mean, log_var


import cv2


def load_images(directory, img_size):
    images = []
    for filename in os.listdir(directory):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            img_path = os.path.join(directory, filename)
            image = Image.open(img_path)
            image = image.resize((img_size, img_size))
            image = np.array(image) / 255.0
            images.append(image)
    return np.array(images)


def tensor_to_image(tensor):
    # Remove extra dimensions
    tensor = tf.squeeze(tensor, axis=0)

    # Convert tensor values to the correct range and data type
    tensor = tf.cast(tensor * 255, tf.uint8)

    # Convert tensor to NumPy array
    array = np.array(tensor)

    # Create PIL Image
    image = Image.fromarray(array)

    # Resize the image
    image = image.resize((128, 128))

    return image
def image_to_tensor(image):
    image = tf.expand_dims(image, axis=0)
    return image



# Define the VAE model
latent_dim = 64
epochs = 10
IMG_SIZE = 128
batch_size = 64
training_images = r"C:\Users\0xdan\Documents\CS\WorkCareer\Chemistry Internship\Project-Code\data\dataset\test-data"

# Load and preprocess training images
train_images = load_images(training_images, IMG_SIZE)

# Define the Model
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
vae = VAE(latent_dim)
vae.compile(optimizer=optimizer)
vae.fit(train_images, epochs=epochs, batch_size=batch_size)

# Save model weights
# vae.save_weights(r'C:\Users\0xdan\Documents\CS\WorkCareer\Chemistry Internship\Project-Code\data\models\vae\model.h5')

# Generate black and white noise
noise_vector = np.random.normal(size=(1, latent_dim))
# Pass the latent vector through the decoder
reconstructed_image = vae.decode(noise_vector)
reconstructed_image = tensor_to_image(reconstructed_image)
reconstructed_image.show()