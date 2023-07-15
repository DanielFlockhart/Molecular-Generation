import sys, os
sys.path.insert(0, os.path.abspath('..'))
from utilities import file_utils
# Path Constants
DATA_FOLDER = fr"{file_utils.get_directory()}\data\datasets" # Make this relative
PROCESSED_DATA = fr"{DATA_FOLDER}\db1\Targets"
MODELS_FOLDER = fr"{file_utils.get_directory()}\data\models"
GENERATED_FOLDER = fr"{file_utils.get_directory()}\data\generated"