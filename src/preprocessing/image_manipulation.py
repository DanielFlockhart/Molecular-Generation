import tensorflow as tf
from PIL import Image
import os
def folder_to_vector(folder):
    image_paths = [os.path.join(folder, filename) for filename in os.listdir(folder)]
    vectors = []
    for image_path in image_paths:
        image = Image.open(image_path)
        vector = image_to_vector(image)
        vectors.append(vector)
    return vectors

def image_to_vector(image,size):
    '''
    Converts Image to Input for network
    '''
    # Resize Image if Necessary
    resized_image = image.resize((size,size))

    # Convert to tensor
    image_tensor = tf.convert_to_tensor(resized_image)

    # Normalize the image
    normalized_image = image_tensor / 255.0  # Normalize pixel values to range [0, 1]

    # Add batch dimension if required by the model
    normalized_image = tf.expand_dims(normalized_image, axis=0)

    return normalized_image
