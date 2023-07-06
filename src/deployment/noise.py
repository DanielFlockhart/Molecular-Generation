from PIL import Image
import tensorflow as tf
import numpy as np

# Generate image of noise
def noise(x):
    '''
    Generate image of noise
    '''
    # Create a noise array only black and white

    noise = np.random.randint(0, 2, size=(x, x, 3), dtype=np.uint8) * 255

    # Create an image from the noise array
    image = Image.fromarray(noise)
    return image

