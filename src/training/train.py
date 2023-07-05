import tensorflow as tf
from tensorflow.keras import layers
from vae import *
from gan import * 
import numpy as np
sys.path.insert(0, os.path.abspath('..'))
from preprocessing import image_manipulation as im


def train_VAE(train_images):
    # Create an instance of the VAE model
    latent_dim = 64  # Adjust the desired latent dimension size
    vae = VAE(latent_dim)


    # Compile the VAE model
    vae.compile(optimizer='adam', loss=vae.loss)

    # Generate training data (Replace this with your actual training data)
    #train_images = tf.random.normal(shape=(1000, 128, 128, 3))

    # Train the VAE model
    vae.fit(train_images, train_images, batch_size=64, epochs=10)




    # Generate an input tensor from a 128x128px image
    input_image = tf.random.normal(shape=(1, 128, 128, 3))

    # Pass the input tensor through the VAE
    reconstructed_image, mean, log_var = vae(input_image)

    # Print the shapes of the outputs
    print("Shape of the reconstructed image:", reconstructed_image.shape)
    print("Shape of the mean:", mean.shape)
    print("Shape of the log variance:", log_var.shape)

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

    # Generate a sample image using the generator 128 pixel image
    noise = tf.random.normal(shape=(1, 128, 128, 3))
    generated_image = gan.generator(noise, training=False)

    # Pass the fake image through the discriminator
    discriminator_output = gan.discriminator(generated_image)

    # Print the shape of the generated image and discriminator output
    print("Shape of the generated image:", generated_image.shape)
    print("Shape of the discriminator output:", discriminator_output.shape)


if __name__ == "__main__":
    training_images = r"C:\Users\0xdan\Documents\CS\WorkCareer\Chemistry Internship\Project-Code\data\dataset"
    training_images = im.folder_to_vector(training_images)
    training_images = np.reshape(training_images, (len(training_images), -1))
    train_VAE(training_images)

# List to Experiment With

# CVAE?
# Transformer?
# CNN?


#I am currently trying to generate new images of molecular skeletons using unsupervised learning, I have a dataset of images of skeletons for training.
#Unfortunately, the images of the skeletons are all of different scale and position. I want to use a ML network that has positional and size invariance in mind.
#I am currently looking at other options for the network architecture. I am currently looking at potentially using a GAN or Variational auto encoder. 
#Is there a better architecture or a different way of doing this that would work better.