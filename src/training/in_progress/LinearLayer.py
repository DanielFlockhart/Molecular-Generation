import tensorflow as tf
import numpy as np
from PIL import Image

# Load the input image
input_image_path = r"C:\Users\0xdan\Documents\CS\WorkCareer\Chemistry Internship\Project-Code\data\dataset\test-data\S1N=C2C=CC3=C(C=CC4=NSN=C34)C2=N1.png"
input_image = Image.open(input_image_path)
input_image = np.array(input_image) / 255.0

# Reshape the input image to a flat vector
input_shape = input_image.shape
input_size = input_shape[0] * input_shape[1] * input_shape[2]
input_image_flat = np.reshape(input_image, (1, input_size))

# Define the linear autoencoder model
encoder_dim = 128
input_dim = input_size

inputs = tf.keras.Input(shape=(input_dim,))
encoded = tf.keras.layers.Dense(encoder_dim)(inputs)
decoded = tf.keras.layers.Dense(input_dim)(encoded)

autoencoder = tf.keras.Model(inputs, decoded)

# Compile the model
autoencoder.compile(optimizer='adam', loss='mse')

# Train the autoencoder
autoencoder.fit(input_image_flat, input_image_flat, epochs=1000)

# Reconstruct the image
reconstructed_image_flat = autoencoder.predict(input_image_flat)
reconstructed_image = np.reshape(reconstructed_image_flat, input_shape)

# Scale the reconstructed image back to the range [0, 255]
reconstructed_image = (reconstructed_image * 255.0).astype(np.uint8)

# Save the reconstructed image
output_image_path = r"C:\Users\0xdan\Documents\CS\WorkCareer\Chemistry Internship\Project-Code\data\generated\ae\gen.png"
reconstructed_image = Image.fromarray(reconstructed_image)
reconstructed_image.save(output_image_path)
