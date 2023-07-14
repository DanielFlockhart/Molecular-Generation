from Constants import preprop_constants
# Training/Machine Learning Constants
BATCH_SIZE = 64 # Number of images to train on at once
EPOCHS = 24 # Number of times to train on the entire dataset
LATENT_DIM = 128 # Number of dimensions in the latent space
LRN_RATE = 0.001 # Learning rate for the optimizer
TRAIN_SUBSET_COUNT = 400 # Number of images to test training on
INPUT_SHAPE = (preprop_constants.IMG_SIZE, preprop_constants.IMG_SIZE, 3)