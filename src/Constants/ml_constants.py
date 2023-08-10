from Constants import preprop_constants
# Training/Machine Learning Constants
BATCH_SIZE = 128 # Number of images to train on at once
EPOCHS = 2000# Number of times to train on the entire dataset
# Number of dimensions in the latent space
LRN_RATE = 0.001 # Learning rate for the optimizer
TRAIN_SUBSET_COUNT = 0# Number of images to test training on
CONDITIONS_SIZE = 10
INPUT_SIZE = 768
LATENT_DIM = 128
OUTPUT_DIM = (preprop_constants.IMG_SIZE,preprop_constants.IMG_SIZE,1)