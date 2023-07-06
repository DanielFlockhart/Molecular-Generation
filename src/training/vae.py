import tensorflow as tf
from tensorflow.keras import layers

class VAE(tf.keras.Model):
    def __init__(self, latent_dim):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()

    def build_encoder(self):
        encoder_input = tf.keras.Input(shape=(128, 128, 3))
        x = layers.Conv2D(32, 3, activation='relu', strides=2, padding='same')(encoder_input)
        x = layers.Conv2D(64, 3, activation='relu', strides=2, padding='same')(x)
        x = layers.Flatten()(x)
        mean = layers.Dense(self.latent_dim, activation='linear')(x)
        log_var = layers.Dense(self.latent_dim, activation='linear')(x)
        encoder_output = mean, log_var
        return tf.keras.Model(encoder_input, encoder_output)

    def build_decoder(self):
        decoder_input = tf.keras.Input(shape=(self.latent_dim,))
        x = layers.Dense(8 * 8 * 256, activation='relu')(decoder_input)
        x = layers.Reshape((8, 8, 256))(x)
        x = layers.Conv2DTranspose(128, 3, activation='relu', strides=4, padding='same')(x)
        x = layers.Conv2DTranspose(64, 3, activation='relu', strides=4, padding='same')(x)
        decoder_output = layers.Conv2DTranspose(3, 3, activation='sigmoid', padding='same')(x)
        return tf.keras.Model(decoder_input, decoder_output)

    def encode(self, x):
        mean, log_var = self.encoder(x)
        epsilon = tf.random.normal(shape=tf.shape(mean))
        z = mean + tf.exp(0.5 * log_var) * epsilon
        return z, mean, log_var

    def decode(self, z):
        reconstructed = self.decoder(z)
        return reconstructed

    def call(self, x):
        z, mean, log_var = self.encode(x)
        reconstructed = self.decode(z)

        reconstruction_loss = tf.reduce_mean(tf.square(x - reconstructed))
        kl_loss = -0.5 * tf.reduce_mean(1 + log_var - tf.square(mean) - tf.exp(log_var))
        total_loss = reconstruction_loss + kl_loss
        self.add_loss(total_loss)

        return reconstructed, mean, log_var
