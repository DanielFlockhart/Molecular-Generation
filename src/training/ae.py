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

# Define the VAE model
latent_dim = 128

# Encoder
encoder_inputs = tf.keras.Input(shape=(input_size,))
x = tf.keras.layers.Dense(256, activation='relu')(encoder_inputs)
z_mean = tf.keras.layers.Dense(latent_dim)(x)
z_log_var = tf.keras.layers.Dense(latent_dim)(x)

# Reparameterization trick
epsilon = tf.keras.backend.random_normal(shape=(tf.shape(z_mean)[0], latent_dim), mean=0.0, stddev=1.0)
z = z_mean + tf.exp(0.5 * z_log_var) * epsilon

# Decoder
decoder_inputs = tf.keras.Input(shape=(latent_dim,))
x = tf.keras.layers.Dense(256, activation='relu')(decoder_inputs)
decoder_outputs = tf.keras.layers.Dense(input_size, activation='sigmoid')(x)

# Define the VAE model
encoder = tf.keras.Model(encoder_inputs, [z_mean, z_log_var, z], name='encoder')
decoder = tf.keras.Model(decoder_inputs, decoder_outputs, name='decoder')
outputs = decoder(encoder(encoder_inputs)[2])
vae = tf.keras.Model(encoder_inputs, outputs, name='vae')

# Define the VAE loss
reconstruction_loss = tf.keras.losses.binary_crossentropy(encoder_inputs, outputs) * input_size
kl_loss = -0.5 * tf.keras.backend.sum(1 + z_log_var - tf.keras.backend.square(z_mean) - tf.keras.backend.exp(z_log_var), axis=-1)
vae_loss = tf.keras.backend.mean(reconstruction_loss + kl_loss)

# Compile the VAE model
vae.add_loss(vae_loss)
vae.compile(optimizer='adam')

# Train the VAE
vae.fit(input_image_flat, epochs=10, batch_size=1)

# Reconstruct the image
reconstructed_image_flat = vae.predict(input_image_flat)
reconstructed_image = np.reshape(reconstructed_image_flat, input_shape)

# Scale the reconstructed image back to the range [0, 255]
reconstructed_image = (reconstructed_image * 255.0).astype(np.uint8)

# Save the reconstructed image
output_image_path = r"C:\Users\0xdan\Documents\CS\WorkCareer\Chemistry Internship\Project-Code\data\generated\ae\gen.png"
reconstructed_image = Image.fromarray(reconstructed_image)
reconstructed_image.save(output_image_path)
