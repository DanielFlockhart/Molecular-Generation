from Constants import preprop_constants
# Training/Machine Learning Constants
BATCH_SIZE = 128 # Number of images to train on at once
EPOCHS = 2400# Number of times to train on the entire dataset
LATENT_DIM = 256# Number of dimensions in the latent space
LRN_RATE = 0.00001 # Learning rate for the optimizer
TRAIN_SUBSET_COUNT = 1200# Number of images to test training on
INPUT_SIZE = 768
CONDITIONS_SIZE = 12
OUTPUT_DIM = (preprop_constants.IMG_SIZE,preprop_constants.IMG_SIZE,1)