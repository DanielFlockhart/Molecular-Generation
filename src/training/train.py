import tensorflow as tf
from tensorflow.keras import layers
from training.vae import *
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.insert(0, os.path.abspath('..'))
from Constants import ml_constants,ui_constants,file_constants
from utilities import utils
from ui.terminal_ui import *
from training import get_inputs
from tensorflow.keras.callbacks import Callback,ModelCheckpoint
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
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

    @tf.function
    def train_step(inputs_batch,conditions_batch, targets_batch):
        with tf.GradientTape() as tape:
            # Reshape the inputs_batch to have shape (batch_size, 768)
            inputs_batch = tf.reshape(inputs_batch, (tf.shape(inputs_batch)[0], ml_constants.INPUT_SIZE))

            # Forward pass through the model
            reconstructed = model(inputs_batch,conditions_batch, training=True)

            # Compute the loss
            loss = model.compute_loss(inputs_batch, targets_batch, reconstructed)

        # Compute gradients and update weights
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss
    # Create a ModelCheckpoint callback to save the model weights
    checkpoint_path = os.path.join(file_constants.CHECKPOINTS_FOLDER, "model_checkpoint")

    # Training loop
    epoch_losses = []
    for epoch in tqdm(range(ml_constants.EPOCHS), bar_format=ui_constants.LOADING_BAR, ncols=80, colour='green'): 
        total_loss = 0.0
        num_batches = len(vectors) // ml_constants.BATCH_SIZE
        
        for step in range(num_batches):
            start_idx = step * ml_constants.BATCH_SIZE
            end_idx = (step + 1) * ml_constants.BATCH_SIZE
            inputs_batch = vectors[start_idx:end_idx]
            conditions_batch = conditions[start_idx:end_idx]
            targets_batch = targets[start_idx:end_idx]

            loss = train_step(inputs_batch,conditions_batch, targets_batch)
            total_loss += loss
            
        average_loss = total_loss / num_batches
        epoch_losses.append(average_loss.numpy())
        print(f" - Loss: {average_loss:.4f}")
        if epoch % 10 == 0:
            model.save_weights(checkpoint_path)

    graph_loss(epoch_losses)

    return model

def graph_loss(epoch_losses):
    print(epoch_losses)
    plt.plot(range(1, len(epoch_losses) + 1), epoch_losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')

    loss_plot_path = f"{file_constants.VISUALISATIONS_FOLDER}\Graphs\Loss\loss.png"
    plt.savefig(loss_plot_path)
    plt.show()  

def save_model(model,name):
    '''
    Saves a trained model
    '''
    # Save the subclassed model's weights (this is required for HDF5 format)
    model_weights_path = fr'C:\Users\0xdan\Documents\CS\WorkCareer\Chemistry Internship\Project-Code\data\models\{name}\weights.h5'
    model.save_weights(model_weights_path)

    # Convert the subclassed model to a functional model using the same input tensors
    inputs = model.encoder.inputs[0]
    condition_vector = tf.keras.Input(shape=(ml_constants.CONDITIONS_SIZE,))
    outputs = model(inputs, condition_vector)
    vae_functional_model = tf.keras.Model(inputs=[inputs, condition_vector], outputs=outputs)

    # Save the functional model in the HDF5 format
    vae_functional_model.save(fr'C:\Users\0xdan\Documents\CS\WorkCareer\Chemistry Internship\Project-Code\data\models\{name}\model.h5')