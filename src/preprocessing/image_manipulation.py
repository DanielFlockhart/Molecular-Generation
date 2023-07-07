import tensorflow as tf
from PIL import Image
import numpy as np
import os
# def folder_to_vector(folder,img_size):
#     image_paths = [os.path.join(folder, filename) for filename in os.listdir(folder)]
#     vectors = []
#     for image_path in image_paths:
#         image = Image.open(image_path)
#         vector = image_to_vector(image)
#         vectors.append(vector)
#     return vectors

# def image_to_vector(image):
#     '''
#     Converts Image to Input for network
#     '''
#     # Resize Image if Necessary
#     #resized_image = image.resize((size,size))

#     # Convert to image to numpy array
#     image_array = np.asarray(image)

#     return image_array

# def tensor_to_image(tensor):
#     tensor = tf.squeeze(tensor, axis=0)  # Remove unnecessary dimensions
#     tensor = tensor.numpy() * 255.0  # Scale the tensor values to the range [0, 255]
#     tensor = tensor.astype(np.uint8)  # Convert the tensor to unsigned 8-bit integers
#     image = Image.fromarray(tensor)  # Create an image from the array
#     return image


import cv2


def load_images(directory, img_size):
    images = []
    size = 0
    for (i,filename) in enumerate(os.listdir(directory)):
        if (i % 1000) == 0:
            print(str((i*100)/len(os.listdir(directory))) + "% Done")
        if filename.endswith('.jpg') or filename.endswith('.png'):  # Add more extensions if needed
        
            img_path = os.path.join(directory, filename)
            image = cv2.imread(img_path)
            # Get the file size of the image
            if os.path.getsize(img_path) > size:
                size = os.path.getsize(img_path)
                print(size)

            image = cv2.resize(image, (img_size, img_size))  # Resize image to desired dimensions
            image = image.astype(np.float32) / 255.0  # Normalize pixel values between 0 and 1
            images.append(image)
    return np.array(images)


def tensor_to_image(tensor):
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
    image = tf.expand_dims(image, axis=0)
    return image



# if __name__ =="__main__":
#     img_path = r"C:\Users\0xdan\Documents\CS\WorkCareer\Chemistry Internship\Project-Code\data\dataset\test-data\[NH3+]C(C(=O)[O-])c1ccccc1F.png"
#     image = cv2.imread(img_path)
#     image = cv2.resize(image, (128, 128))  # Resize image to desired dimensions
#     image = image.astype(np.float32) / 255.0  # Normalize pixel values between 0 and 1
#     image = tf.expand_dims(image, axis=0)
#     image = tensor_to_image(image)
#     image.show()