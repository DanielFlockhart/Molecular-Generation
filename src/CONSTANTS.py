colours = {
        'B': '\033[93m',  # purple
        'C': '\033[30m',  # black
        'N': '\033[94m',  # blue
        'O': '\033[91m',  # red
        'P': '\033[93m',  # yellow
        'S': '\033[92m',  # green
        'F': '\033[96m',  # cyan
        'Cl': '\033[37m',  # green
        'Br': '\033[95m',  # yellow
        'I': '\033[94m',  # blue
        '(': '\033[37m',  # white
        ')': '\033[37m',  # white
        'Other': '\033[37m' # White
        # Add more letters and their color codes as needed
    }


# Preprocessing Constants
IMG_SIZE = 400 # Size to save the images in the dataset
STD_DEV = 1.8 # Standard deviation of scaling factor for the images - Higher = Images will have larger range of sizes
MAX_CHARS = 250 # Maximum number of characters of smile used for file name

# Training/Machine Learning Constants
BATCH_SIZE = 64 # Number of images to train on at once
EPOCHS = 10 # Number of times to train on the entire dataset
LATENT_DIM = 32 # Number of dimensions in the latent space
LRN_RATE = 0.001 # Learning rate for the optimizer


# UI Constants
UI_TERMINAL_WIDTH = 25
LOADING_BAR = "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]" # Loading bar format for tqdm