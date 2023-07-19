import tensorflow as tf
from tensorflow.keras import layers
from training.vae import *
from training.gan import * 
import numpy as np
import sys
import os
sys.path.insert(0, os.path.abspath('..'))
from Constants import ml_constants
from ui.terminal_ui import *
from training import get_inputs



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
    (x_train,conditions,y_train) = get_inputs.get_training_data()
    model.compile(optimizer=optimizer,loss=model.compute_loss)
    print(format_title("Training Model"))
    model.fit([np.array(x_train), np.array(conditions)], y_train,batch_size=ml_constants.BATCH_SIZE, epochs=ml_constants.EPOCHS)
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