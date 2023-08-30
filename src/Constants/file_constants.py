import sys, os
sys.path.insert(0, os.path.abspath('..'))
from utilities import file_utils
# Path Constants
DATASET = "\db5-Zinc-250k"
DATA_FOLDER = fr"{file_utils.get_directory()}\data\datasets" # Make this relative
PROCESSED_DATA = fr"{DATA_FOLDER}\{DATASET}\Targets"
MODELS_FOLDER = fr"{file_utils.get_directory()}\data\models"
GENERATED_FOLDER = fr"{file_utils.get_directory()}\data\generated"
INPUTS_FOLDER = fr"{DATA_FOLDER}\{DATASET}\dataset.csv"
VISUALISATIONS_FOLDER = fr"{file_utils.get_directory()}\data\visualisations"
TEST_DATA_FOLDER =  fr"{file_utils.get_directory()}\data\test_data\test_inputs.csv"
CHECKPOINTS_FOLDER = fr"{file_utils.get_directory()}\data\models\vae\checkpoints"

# Post Processing
PROFILES_FOLDER = fr"{file_utils.get_directory()}\data\generated\profiles"
#54541