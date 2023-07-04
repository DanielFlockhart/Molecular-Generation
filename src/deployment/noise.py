from PIL import Image
import tensorflow as tf
import numpy as np

# Generate image of noise
def noise(x):
    '''
    Generate image of noise
    '''
    noise = np.random.randint(0, 256, (x, x, 3), dtype=np.uint8)

    # Create an image from the noise array
    image = Image.fromarray(noise)

    
    
    # Save the image to a file
    image.save("random_noise.png")  # Specify the desired file name and format
