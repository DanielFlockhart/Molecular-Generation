from Constants import preprop_constants
# Training/Machine Learning Constants
BATCH_SIZE = 256 # Number of images to train on at once
EPOCHS = 300# Number of times to train on the entire dataset
# Number of dimensions in the latent space
LRN_RATE = 0.001 # Learning rate for the optimizer
TRAIN_SUBSET_COUNT = 256# Number of images to test training on
CONDITIONS_SIZE = 12
INPUT_SIZE = 768
LATENT_DIM = 64
OUTPUT_DIM = (preprop_constants.IMG_SIZE,preprop_constants.IMG_SIZE,1)