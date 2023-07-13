import tensorflow as tf
from tensorflow.keras import layers
from training.vae import *
from training.gan import * 
import numpy as np
import sys
import os
sys.path.insert(0, os.path.abspath('..'))
from preprocessing import inputify as im
from CONSTANTS import *
from ui.terminal_ui import *



def train_model(model,training_images,optimizer):
    '''
    Train a model

    Parameters
    ----------
    model : tf.keras.Model
        The model to train
    training_images : list
        A list of images to train on
    optimizer : tf.keras.optimizers
        The optimizer to use
    '''
    #training_images = training_images.astype('float32') / 255.0
    print(format_title("Compiling Model"))

    model.compile(optimizer=optimizer,loss=model.compute_loss)
    print(format_title("Training Model"))
    model.fit(training_images,training_images, epochs=EPOCHS, batch_size=BATCH_SIZE)
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