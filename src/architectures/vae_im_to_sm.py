import tensorflow as tf
from tensorflow.keras import layers

class VariationalAutoencoder(tf.keras.Model):
    def __init__(self, input_dim, latent_dim, output_dim):
        super(VariationalAutoencoder, self).__init__()
        self.encoder = self.build_encoder(input_dim, latent_dim)
        self.decoder = self.build_decoder(latent_dim, output_dim)
        self.training = True

    def build_encoder(self, input_dim, latent_dim):
        encoder_inputs = tf.keras.Input(shape=input_dim)
        x = layers.Reshape(target_shape=(input_dim[0], 1, 1))(encoder_inputs)
        x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        x = layers.MaxPooling2D((2, 2), padding='same')(x)
        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = layers.MaxPooling2D((2, 2), padding='same')(x)
        x = layers.Flatten()(x)
        z_mean = layers.Dense(latent_dim)(x)
        z_log_var = layers.Dense(latent_dim)(x)
        return tf.keras.Model(encoder_inputs, [z_mean, z_log_var], name='encoder')

    def build_decoder(self, latent_dim, output_dim):
        latent_inputs = tf.keras.Input(shape=(latent_dim,))
        x = layers.Dense(128, activation='relu')(latent_inputs)
        x = layers.Dense(256, activation="relu")(x)
        x = layers.Dense(512, activation="relu")(x)
        output = layers.Dense(output_dim[0], activation="sigmoid")(x)
        
        return tf.keras.Model(latent_inputs, output, name='decoder')

    def sampling(self, args):
        z_mean, z_log_var = args
        epsilon = tf.random.normal(shape=tf.shape(z_mean))
        z = z_mean + tf.exp(0.5 * z_log_var) * epsilon
        return z

    def call(self, inputs):
        z_mean, z_log_var = self.encoder(inputs)
        if self.training:
            z = self.sampling([z_mean, z_log_var])
        else:
            z = z_mean
        reconstructed = self.decoder(z)
        return reconstructed

    def compute_loss(self, inputs, targets):
        z_mean, z_log_var = self.encoder(inputs)
        z = self.sampling([z_mean, z_log_var])

        # Reconstruction loss
        reconstruction_loss = tf.reduce_mean(tf.keras.losses.mean_squared_error(targets, self.decoder(z)))

        # KL divergence loss
        kl_loss = -0.5 * tf.reduce_mean(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))

        total_loss = reconstruction_loss + kl_loss
        return total_loss


def generate_molecule(vector,model):
    '''
    Generate a molecule from a vector and condition using the trained model.
    '''
    vector = np.expand_dims(vector, axis=0)
    condition = np.expand_dims(condition, axis=0)
    # Call the decoder to generate the image
    image = model.predict(vector)


    image = tf.reshape(image,(100,100,1))
    image = np.squeeze(image)

    # Convert the image to a NumPy array and cast to uint8
    image = (image * 255).astype(np.uint8)

    # Create a PIL Image from the NumPy array
    pil_image = Image.fromarray(image, mode='L')
    return pil_image    

if __name__ == "__main__":
    # Define input and output dimensions
    input_dim = (10000)  # Modify this to match your input vector dimension
    latent_dim = 64  # Dimension of the latent space
    output_dim = (768,)  # Modify this to match your input vector dimension

    # Create the VAE model
    vae = VariationalAutoencoder(input_dim, latent_dim, output_dim)

    # Compile the model
    vae.compile(optimizer='adam', loss=vae.compute_loss)

    # Print model summary
    vae.summary()
    import numpy as np
    from PIL import Image
    input_vector = np.random.rand(10000)
    out_vector = generate_molecule(input_vector,vae)
    