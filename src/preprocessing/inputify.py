import tensorflow as tf
from PIL import Image
import numpy as np
import os,sys,cv2
from tqdm import tqdm
sys.path.insert(0, os.path.abspath('..'))
from CONSTANTS import *
from ui.terminal_ui import *

def load_images(directory, img_size):
    '''
    Loads images from a directory and returns them as a NumPy array for input to a model
    '''
    images = []
    for (i,filename) in enumerate(os.listdir(directory)):
        if filename.endswith('.png'):
            img_path = os.path.join(directory, filename)
            image = cv2.imread(img_path)
            
            # Resize image to desired dimensions
            image = cv2.resize(image, (img_size, img_size))

            # Normalize pixel values between 0 and 1
            image = image.astype(np.float32) / 255.0  
            images.append(image)

    return np.array(images)


def tensor_to_image(tensor):
    '''
    Converts a tensor to a PIL Image
    '''
    # Remove extra dimensions
    tensor = tf.squeeze(tensor, axis=0)

    # Convert tensor values to the correct range and data type
    tensor = tf.cast(tensor * 255, tf.uint8)

    # Convert tensor to NumPy array
    array = np.array(tensor)

    # Create PIL Image
    image = Image.fromarray(array)

    # Resize the image
    image = image.resize((128, 128))

    return image

def image_to_tensor(image):
    '''
    Converts a PIL Image to a tensor
    '''
    image = tf.expand_dims(image, axis=0)
    return image
