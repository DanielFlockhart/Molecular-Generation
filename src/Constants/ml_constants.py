from Constants import preprop_constants
# Training/Machine Learning Constants
BATCH_SIZE = 10 # Number of images to train on at once
EPOCHS = 30 # Number of times to train on the entire dataset
LATENT_DIM = 200 # Number of dimensions in the latent space
LRN_RATE = 0.0001 # Learning rate for the optimizer
TRAIN_SUBSET_COUNT = 100 # Number of images to test training on
INPUT_SIZE = 768
CONDITIONS_SIZE = 12
OUTPUT_DIM = (preprop_constants.IMG_SIZE,preprop_constants.IMG_SIZE,1)