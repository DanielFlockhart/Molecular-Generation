import tensorflow as tf
from tensorflow.keras import layers
from training.vae import *
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.insert(0, os.path.abspath('..'))
from Constants import ml_constants,ui_constants
from utilities import utils
from ui.terminal_ui import *
from training import get_inputs
from tensorflow.keras.callbacks import Callback
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

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

    # # Train the model with the callback
    # model.fit(
    #     np.array(vectors),
    #     np.array(targets),
    #     batch_size=ml_constants.BATCH_SIZE,
    #     epochs=ml_constants.EPOCHS,
    # )
    @tf.function
    def train_step(inputs_batch, targets_batch):
        with tf.GradientTape() as tape:
            # Reshape the inputs_batch to have shape (batch_size, 768)
            inputs_batch = tf.reshape(inputs_batch, (tf.shape(inputs_batch)[0], 768))

            # Forward pass through the model
            reconstructed = model(inputs_batch, training=True)

            # Compute the loss
            loss = model.compute_loss(inputs_batch, targets_batch, reconstructed)

        # Compute gradients and update weights
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss

    # Training loop
    for epoch in tqdm(range(ml_constants.EPOCHS), bar_format=ui_constants.LOADING_BAR, ncols=80, colour='green'): 
        total_loss = 0.0
        num_batches = len(vectors) // ml_constants.BATCH_SIZE
        for step in range(num_batches):
            start_idx = step * ml_constants.BATCH_SIZE
            end_idx = (step + 1) * ml_constants.BATCH_SIZE
            inputs_batch = vectors[start_idx:end_idx]
            targets_batch = targets[start_idx:end_idx]

            loss = train_step(inputs_batch, targets_batch)
            total_loss += loss
        average_loss = total_loss / num_batches
        print(f"Loss: {average_loss:.4f}")
    return model


def save_model(model,name):
    '''
    Saves a trained model
    '''
    
    tf.saved_model.save(model,fr'C:\Users\0xdan\Documents\CS\WorkCareer\Chemistry Internship\Ai-Chem-Intership\data\models\{name}')
