import tensorflow as tf
from tensorflow.keras import layers
from vae import *
from gan import * 
import numpy as np
import sys
import os
sys.path.insert(0, os.path.abspath('..'))
from preprocessing import image_manipulation as im
from PIL import Image
from CONSTANTS import *


import random

def train_VAE(train_images):
    
    # Generate Temporary Data for Testing (1000 image 128x128 of noise)
    train_images = np.random.normal(size=(1000, IMG_SIZE, IMG_SIZE, 3))
    # Define the Model
    optimizer = tf.keras.optimizers.Adam(learning_rate=LRN_RATE)
    vae = VAE(LATENT_DIM)
    vae.compile(optimizer=optimizer)
    vae.fit(train_images, epochs=EPOCHS, batch_size=BATCH_SIZE)
    
    # # Save model weights
    vae.save_weights(r'C:\Users\0xdan\Documents\CS\WorkCareer\Chemistry Internship\Project-Code\data\models\vae\model.h5')
     
    # Generate black and white noise
    noise_vector = np.random.normal(size=(1, LATENT_DIM))
    # Pass the latent vector through the decoder


    reconstructed_image = vae.decode(noise_vector)
    reconstructed_image = im.tensor_to_image(reconstructed_image)
    reconstructed_image.show()


def train_GAN(training_images):

    # Create an instance of the GAN model
    training_images = np.random.normal(size=(1000, IMG_SIZE, IMG_SIZE, 3))
    # Create an instance of the GAN model
    gan = GAN(Generator(), Discriminator())

    # Define the optimizer for the generator and discriminator
    generator_optimizer = tf.keras.optimizers.Adam(1e-4)
    discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

    # Compile the GAN model
    gan.compile(optimizer=generator_optimizer, loss=gan.discriminator.loss)

    # Fit the GAN model
    gan.fit(training_images, epochs=5, batch_size=64)

    # Save the model
    gan.save(r'C:\Users\0xdan\Documents\CS\WorkCareer\Chemistry Internship\Project-Code\data\models\gan')

    # Generate a sample image using the generator
    noise = tf.random.normal(shape=(1, 100))
    generated_image = gan.generator(noise, training=False)

    # Convert the image tensor to a NumPy array and display
    generated_image = im.tensor_to_image(generated_image)
    generated_image.show()




if __name__ == "__main__":
    training_images = r"C:\Users\0xdan\Documents\CS\WorkCareer\Chemistry Internship\Project-Code\data\dataset\test-data"
    imgs = im.load_images(training_images,IMG_SIZE)

    train_VAE(imgs)



#I am currently trying to generate new images of molecular skeletons using unsupervised learning, I have a dataset of images of skeletons for training.
#Unfortunately, the images of the skeletons are all of different scale and position. I want to use a ML network that has positional and size invariance in mind.
#I am currently looking at other options for the network architecture. I am currently looking at potentially using a GAN or Variational auto encoder. 
#Is there a better architecture or a different way of doing this that would work better.