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

from PIL import Image

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
    labels,vectors,conditions,targets = get_training_data()
    model.compile(optimizer=optimizer,loss=model.compute_loss)
    print(format_title("Training Model"))
    # # Train the model with the defined callback
    
    # width = 100  # Width of the image
    # height = 100  # Height of the image

    # # Example 1D array of pixel values (grayscale image)
    # pixels = np.array(targets[100])
    # image = pixels.reshape((height, width))
    # plt.imshow(image, cmap='gray')  # If it's a grayscale image
    # plt.axis('off')  # Remove axes ticks and labels
    # plt.show()

    # sys.exit()
    model.fit(np.array(vectors), np.array(targets), batch_size=ml_constants.BATCH_SIZE, 
            epochs=ml_constants.EPOCHS)

    # num_samples = 10
    # latent_samples = np.random.normal(size=(num_samples, latent_dim))
    # generated_images = decoder.predict(latent_samples)
    # generated_images = generated_images.reshape(num_samples, 100, 100)

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