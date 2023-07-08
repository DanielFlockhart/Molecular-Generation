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
    model.compile(optimizer=optimizer)
    model.fit(training_images, epochs=EPOCHS, batch_size=BATCH_SIZE)
    return model
    

def save_model(model,name):
    '''
    Saves a trained model
    '''
    model.save_weights(fr'C:\Users\0xdan\Documents\CS\WorkCareer\Chemistry Internship\Project-Code\data\models\{name}\model.h5')


    