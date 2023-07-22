import numpy as np
import tensorflow as tf
from PIL import Image
import os,sys,cv2
from tqdm import tqdm
sys.path.insert(0, os.path.abspath('..'))
from Constants import ui_constants,preprop_constants,file_constants
from ui.terminal_ui import *


def tensor_to_image(tensor,img_size):
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
    image = image.resize((img_size,img_size))

    return image

def image_to_tensor(image):
    '''
    Converts a PIL Image to a tensor
    '''
    image = tf.expand_dims(image, axis=0)
    return image


def recolour(folder,threshold=245,file_type='png'):
    '''
    Unused Function
    Recolours the images to black and white with a threshold value

    Not yet decided whether to use this function or maintain continuous values
    '''
    # Iterate over each file in the folder
    for file_name in os.listdir(folder):
        if file_name.endswith(f'.{file_type}'):
            # Load the image
            image_path = os.path.join(folder, file_name)
            image = Image.open(image_path)

            # Convert the image to grayscale
            image = image.convert('L')

            # Convert the grayscale image to binary black and white
            image = image.point(lambda x: 0 if x < threshold else 255, '1')

            # Save the converted image (overwrite the original)
            image.save(f"{folder}\{file_name}.png")

            # Close the image file
            image.close()



def load_images(count=None):
    '''
    Loads images from a directory and returns them as a NumPy array for input to a model
    '''
    images = []
    labels = []
    directory = file_constants.PROCESSED_DATA
    print(format_title(f'Loading images from {directory}'))
    for (i,filename) in tqdm(enumerate(os.listdir(directory)), total=len(os.listdir(directory)), bar_format=ui_constants.LOADING_BAR, ncols=80, colour='green'):
        if count is not None and i >= count:
            break
        try:
            if filename.endswith('.png'):
                # Add the image name to the list without file ending
                labels.append(filename[:-4])
                img_path = os.path.join(directory, filename)
                image = cv2.imread(img_path)
                # Resize image to desired dimensions
                #image = cv2.resize(image, (preprop_constants.IMG_SIZE, preprop_constants.IMG_SIZE))
                # Convert image variable to image
                # Convert image to grayscale
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                #_, image = cv2.threshold(image, 245, 1, cv2.THRESH_BINARY)
                

                # Normalize pixel values between 0 and 1
                image = image.astype(np.float32) / 255.0
                images.append(image.reshape(-1))
        except Exception as e:
            print(e)
    
    return np.array(images),labels


def load_image(id):
    '''
    Loads a single image from a directory and returns it as a NumPy array for input to a model
    '''
    directory = file_constants.PROCESSED_DATA
    filename = f"{id}.png"
    img_path = os.path.join(directory, filename)
    image = cv2.imread(img_path)
    # Resize image to desired dimensions
    #image = cv2.resize(image, (preprop_constants.IMG_SIZE, preprop_constants.IMG_SIZE))
    # Convert image variable to image
    # Convert image to grayscale
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    

    # Normalize pixel values between 0 and 1
    image = image.astype(np.float32) / 255.0
    return image.reshape(-1),filename[:-4]