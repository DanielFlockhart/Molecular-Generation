import tensorflow as tf
from PIL import Image
import numpy as np
import os
import sys

sys.path.insert(0, os.path.abspath('..'))
from preprocessing import image_manipulation as im
from noise import *
sys.path.insert(0, os.path.abspath('../training'))
from vae import *

# Load the model
latent_dim = 64
vae = VAE(latent_dim)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
vae.compile(optimizer=optimizer)

# Call the model once to create its variables
vae(np.zeros((1, 128, 128, 3)))

# Load the weights
vae.load_weights(r'C:\Users\0xdan\Documents\CS\WorkCareer\Chemistry Internship\Project-Code\data\models\vae\model.h5')

# Generate black and white noise
noise_vector = np.random.normal(size=(1, latent_dim))
# Pass the latent vector through the decoder
reconstructed_image = vae.decode(noise_vector)
reconstructed_image = im.tensor_to_image(reconstructed_image)
reconstructed_image.show()

# Save the image
reconstructed_image.save(r'C:\Users\0xdan\Documents\CS\WorkCareer\Chemistry Internship\Project-Code\data\generated\vae\test.png')