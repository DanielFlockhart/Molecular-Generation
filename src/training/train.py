import tensorflow as tf
from tensorflow.keras import layers
from training.vae import *
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.insert(0, os.path.abspath('..'))
from Constants import ml_constants
from utilities import utils
from ui.terminal_ui import *
from training import get_inputs
from tensorflow.keras.callbacks import Callback
from PIL import Image

class GenerateImageCallback(Callback):
    def __init__(self, vae_model, num_samples=1):
        self.vae_model = vae_model
        self.num_samples = num_samples

    def on_epoch_end(self, epoch, logs=None):
        # Function to generate and visualize images
        def generate_images():
            random_latent_vectors = np.random.normal(size=(self.num_samples, 128))
            generated_images = self.vae_model.decoder(random_latent_vectors)

            return generated_images

        print("\nGenerating Example Image after Epoch", epoch+1)
        generated_images = generate_images()
        for i in range(self.num_samples):
            image = generated_images[i]
            image = tf.reshape(image,ml_constants.OUTPUT_DIM)
            image = np.squeeze(image)
            image = (image * 255).astype(np.uint8)
            pil_image = Image.fromarray(image, mode='L')
            pil_image.show()
            pil_image.save("generated_image.png")


def train_model(model,optimizer):
    ''' 
    Train a model

    Parameters
    ----------
    model : tf.keras.Model
        The model to train
    x_train : np.array
        The training data which is the vector form of a SMILEs string concatenated with the conditions as a Vector
    y_train : np.array
        The target images vectorised to a 1D array to train on.
    optimizer : tf.keras.optimizers
        The optimizer to use
    '''
    
    # Need to Make sure the x_train and y_train are the same length and a labelled correctly
    print(format_title("Compiling Model"))
    labels,vectors,conditions,targets = get_training_data(ml_constants.TRAIN_SUBSET_COUNT)
    model.compile(optimizer=optimizer,loss=model.compute_loss)
    print(format_title("Training Model"))
    generate_image_callback = GenerateImageCallback(model, num_samples=1)

    # Train the model with the callback
    model.fit(
        np.array(vectors),
        np.array(targets),
        batch_size=ml_constants.BATCH_SIZE,
        epochs=ml_constants.EPOCHS,
        #callbacks=[generate_image_callback]
    )

    # num_samples = 10
    # latent_samples = np.random.normal(size=(num_samples, latent_dim))
    # generated_images = decoder.predict(latent_samples)
    # generated_images = generated_images.reshape(num_samples, 100, 100)



    # Test by Printing output of reconstructed image
    # Test by setting latent vector to 1
    # Could be an issue that the training data just isnt enough, and batch size int enough to approximate it just generalises
    return model


def save_model(model,name):
    '''
    Saves a trained model
    '''
    
    tf.saved_model.save(model,fr'C:\Users\0xdan\Documents\CS\WorkCareer\Chemistry Internship\Ai-Chem-Intership\data\models\{name}')



class Training:
    def __init__(self):
        pass
    def create(self):
        pass