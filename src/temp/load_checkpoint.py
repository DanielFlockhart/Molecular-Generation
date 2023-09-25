import tensorflow as tf
import os,sys
sys.path.insert(0, os.path.abspath('..'))
from Constants import file_constants, ml_constants


sys.path.insert(0, os.path.abspath('../training'))
from ml.architectures import vae_im_to_sm, vae_sm_to_im
uri = os.path.join(file_constants.CHECKPOINTS_FOLDER, "model_checkpoint")
model = vae_sm_to_im.VariationalAutoencoder(ml_constants.INPUT_SIZE,ml_constants.LATENT_DIM,ml_constants.OUTPUT_DIM,ml_constants.CONDITIONS_SIZE)

def load(uri, model):
    '''
    Load a model from a checkpoint
    '''
    # Load the model weights
    status = model.load_weights(uri)
    
    # Check if the loading was successful
    if not status.assert_existing_objects_matched():
        raise Exception("Model loading failed. Checkpoint may not match the model architecture.")
    else :
        print("Model Loaded Successfully")
    return model

def save(model):
    '''
    Saves a trained model
    '''
    # Define the path to save the model
    model_path = fr'C:\Users\0xdan\Documents\CS\Catergories\Healthcare_Medical\Computational Chemistry\Mol-Generation\Project-Code\data\models\resaved_model'

    # Save the model using the functional model you created
    model.save(model_path)
if __name__ == "__main__":
    new_model = load(uri,model)
    save(new_model)
    