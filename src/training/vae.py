import tensorflow as tf
from tensorflow.keras import layers

class VAE(tf.keras.Model):
    def __init__(self, latent_dim):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()

    def build_encoder(self):
        # Define the encoder layers
        encoder_input = tf.keras.Input(shape=(128, 128, 3))
        x = layers.Conv2D(32, 3, activation='relu', strides=2, padding='same')(encoder_input)
        x = layers.Conv2D(64, 3, activation='relu', strides=2, padding='same')(x)
        x = layers.Flatten()(x)
        # Output the mean and log variance of the latent space
        mean = layers.Dense(self.latent_dim)(x)
        log_var = layers.Dense(self.latent_dim)(x)
        encoder_output = mean, log_var
        return tf.keras.Model(encoder_input, encoder_output)

    def build_decoder(self):
        # Define the decoder layers
        decoder_input = tf.keras.Input(shape=(self.latent_dim,))
        x = layers.Dense(16 * 16 * 64, activation='relu')(decoder_input)
        x = layers.Reshape((16, 16, 64))(x)
        x = layers.Conv2DTranspose(64, 3, activation='relu', strides=2, padding='same')(x)
        x = layers.Conv2DTranspose(32, 3, activation='relu', strides=2, padding='same')(x)
        # Reconstruct the image using convolutional layers
        decoder_output = layers.Conv2DTranspose(3, 3, activation='sigmoid', padding='same')(x)
        return tf.keras.Model(decoder_input, decoder_output)

    def encode(self, x):
        # Encode the input image into a latent space representation
        mean, log_var = self.encoder(x)
        epsilon = tf.random.normal(shape=tf.shape(mean))
        z = mean + tf.exp(0.5 * log_var) * epsilon
        return z, mean, log_var

    def decode(self, z):
        # Decode the latent vector and reconstruct the image
        reconstructed = self.decoder(z)
        return reconstructed

    def call(self, x):
        # Forward pass through the VAE model
        z, mean, log_var = self.encode(x)
        reconstructed = self.decode(z)
        return reconstructed, mean, log_var

    
    def loss(self,x, reconstructed_x, mean, log_var):
        # Define the loss function
        # Reconstruction loss
        reconstruction_loss = tf.reduce_mean(tf.square(x - reconstructed_x))
        # KL divergence loss
        kl_loss = -0.5 * tf.reduce_mean(1 + log_var - tf.square(mean) - tf.exp(log_var))
        # Total loss
        total_loss = reconstruction_loss + kl_loss
        return total_loss

