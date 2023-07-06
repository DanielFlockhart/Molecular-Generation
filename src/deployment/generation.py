import tensorflow as tf
from PIL import Image
import numpy as np
import os,sys

sys.path.insert(0, os.path.abspath('..'))
from preprocessing import image_manipulation as im
from noise import *
sys.path.insert(0, os.path.abspath('../training'))
from vae import *
# Load the model
loaded_vae = VAE(latent_dim=64)
loaded_vae.compile(optimizer=tf.keras.optimizers.Adam())
# Call the model once to create the variables
dummy_input = tf.zeros((1, 128, 128, 3))
_ = loaded_vae(dummy_input)

# Load the weights
loaded_vae.load_weights(r'C:\Users\0xdan\Documents\CS\WorkCareer\Chemistry Internship\Project-Code\data\models\vae\model.h5')

# Generate black and white noise 
noise_vector = np.random.normal(size=(1, 128, 128, 3))
# Pass the input tensor through the VAE
reconstructed_image, mean, log_var = loaded_vae(noise_vector)

# Convert tensor to image
reconstructed_image = im.tensor_to_image(reconstructed_image)

# Print the output
reconstructed_image.show()