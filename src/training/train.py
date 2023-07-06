import tensorflow as tf
from tensorflow.keras import layers
from vae import *
from gan import * 
import numpy as np
import sys
import os
sys.path.insert(0, os.path.abspath('..'))
from preprocessing import image_manipulation as im
from PIL import Image


def train_VAE(train_images):
    # Define the VAE model
    latent_dim = 64  # Adjust the desired latent dimension size
    vae = VAE(latent_dim)
    # Issue is with the images dimensions
    
    # Define the optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    vae.compile(optimizer=optimizer,loss=vae.compute_loss)

    # Define the training step function
    @tf.function
    def train_step(inputs):
        print("training bruh " + str(inputs.shape))
        encoded_images, mean, log_var = vae.encode(inputs)  # Encode the input images
        with tf.GradientTape() as tape:
            reconstructed = vae.decode(encoded_images)  # Decode the encoded images
            total_loss = vae.compute_loss(inputs, reconstructed, mean, log_var)
        gradients = tape.gradient(total_loss, vae.trainable_variables)
        optimizer.apply_gradients(zip(gradients, vae.trainable_variables))
        return total_loss

    # Training loop
    epochs = 2
    batch_size = 64

    for epoch in range(epochs):
        epoch_loss = 0.0
        num_batches = 0
        for i in range(0, len(train_images), batch_size):
            batch_images = train_images[i:i+batch_size]
            loss = train_step(batch_images)
            epoch_loss += loss
            num_batches += 1
        print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss/num_batches:.4f}")

    # Save model architecture as JSON
    vae_architecture = vae.to_json()
    with open('vae_architecture.json', 'w') as json_file:
        json_file.write(vae_architecture)

    # Save model weights
    vae.save_weights(r'C:\Users\0xdan\Documents\CS\WorkCareer\Chemistry Internship\Project-Code\data\models\vae\model.h5')
    # Generate black and white noise 
    noise_vector = np.random.normal(size=(1, 128, 128, 3))
    # Pass the input tensor through the VAE
    reconstructed_image, mean, log_var = vae(noise_vector)

    # convert reconstructed image to PIL image, using numpy, PIL and tensorflow
    reconstructed_image = im.tensor_to_image(reconstructed_image)


    # Print the output
    reconstructed_image.show()


def train_GAN(training_images):
    # Create instances of the generator and discriminator
    #generator = Generator()
    #discriminator = Discriminator()

    # Create an instance of the GAN model
    gan = GAN(Generator(), Discriminator())



    # Define the optimizer for the generator and discriminator
    generator_optimizer = tf.keras.optimizers.Adam(1e-4)
    discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

    # Define the training loop
    @tf.function
    def train_step(real_images):

        # Generate random noise as input to the generator
        noise = tf.random.normal([BATCH_SIZE, 100])

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            # Generate fake images using the generator
            generated_images = gan.generator(noise, training=True)


            # Compute discriminator outputs for real and fake images
            real_output = gan.discriminator(real_images, training=True)
            fake_output = gan.discriminator(generated_images, training=True)

            # Compute the generator and discriminator losses
            gen_loss = gan.generator.loss(fake_output)
            disc_loss = gan.discriminator.loss(real_output, fake_output)

        # Compute the gradients and update the generator and discriminator
        gradients_of_generator = gen_tape.gradient(gen_loss, gan.generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, gan.discriminator.trainable_variables)

        generator_optimizer.apply_gradients(zip(gradients_of_generator, gan.generator.trainable_variables))
        discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, gan.discriminator.trainable_variables))

    # Prepare the dataset (Replace this with your actual dataset loading/preprocessing)
    BATCH_SIZE = 64

    train_dataset = tf.data.Dataset.from_tensor_slices(training_images).batch(BATCH_SIZE)

    # Train the GAN model
    EPOCHS = 10
    for epoch in range(EPOCHS):
        for batch in train_dataset:
            train_step(batch)



    # Save the model
    gan.save(r'C:\Users\0xdan\Documents\CS\WorkCareer\Chemistry Internship\Project-Code\data\models\gan')

    # Generate a sample image using the generator 128 pixel image
    noise = tf.random.normal(shape=(1, 128, 128, 3))
    generated_image = gan.generator(noise, training=False)

    # Pass the fake image through the discriminator
    discriminator_output = gan.discriminator(generated_image)

    # Print the shape of the generated image and discriminator output
    print("Shape of the generated image:", generated_image.shape)
    print("Shape of the discriminator output:", discriminator_output.shape)


if __name__ == "__main__":
    IMG_SIZE = 128
    training_images = r"C:\Users\0xdan\Documents\CS\WorkCareer\Chemistry Internship\Project-Code\data\dataset\test-data"
    imgs = im.load_images(training_images,IMG_SIZE)

    train_VAE(imgs)

# List to Experiment With

# CVAE?
# Transformer?
# CNN?


#I am currently trying to generate new images of molecular skeletons using unsupervised learning, I have a dataset of images of skeletons for training.
#Unfortunately, the images of the skeletons are all of different scale and position. I want to use a ML network that has positional and size invariance in mind.
#I am currently looking at other options for the network architecture. I am currently looking at potentially using a GAN or Variational auto encoder. 
#Is there a better architecture or a different way of doing this that would work better.