import tensorflow as tf
from tensorflow.keras import layers

class Generator(tf.keras.Model):
    def __init__(self):
        super(Generator, self).__init__()
        # Define the layers for the generator
        self.dense = layers.Dense(7 * 7 * 256, input_shape=(128,))
        self.reshape = layers.Reshape((7, 7, 256))
        self.conv1 = layers.Conv2DTranspose(128, 5, strides=1, padding='same')
        self.bn1 = layers.BatchNormalization()
        self.conv2 = layers.Conv2DTranspose(64, 5, strides=2, padding='same')
        self.bn2 = layers.BatchNormalization()
        self.conv3 = layers.Conv2DTranspose(3, 5, strides=2, padding='same')

    def call(self, inputs):
        # Forward pass through the generator layers
        x = self.dense(inputs)
        x = self.reshape(x)
        x = tf.nn.relu(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = tf.nn.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = tf.nn.relu(x)
        x = tf.nn.tanh(self.conv3(x))
        return x
    
    def loss(self, fake_output):
        # Define the loss function for the generator
        return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(fake_output), logits=fake_output))


class Discriminator(tf.keras.Model):
    def __init__(self):
        super(Discriminator, self).__init__()
        # Define the layers for the discriminator
        self.conv1 = layers.Conv2D(64, 5, strides=2, padding='same', input_shape=[28, 28, 1])
        self.dropout1 = layers.Dropout(0.3)
        self.conv2 = layers.Conv2D(128, 5, strides=2, padding='same')
        self.dropout2 = layers.Dropout(0.3)
        self.flatten = layers.Flatten()
        self.dense = layers.Dense(1)

    def call(self, inputs):
        # Forward pass through the discriminator layers
        x = tf.nn.leaky_relu(self.conv1(inputs))
        x = self.dropout1(x)
        x = tf.nn.leaky_relu(self.conv2(x))
        x = self.dropout2(x)
        x = self.flatten(x)
        x = self.dense(x)
        return x
    
    def loss(self, real_output, fake_output):
        # Define the loss function for the discriminator
        real_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(real_output), logits=real_output))
        fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(fake_output), logits=fake_output))
        total_loss = real_loss + fake_loss
        return total_loss
    
# Define the GAN model
class GAN(tf.keras.Model):
    def __init__(self, generator, discriminator):
        super(GAN, self).__init__()
        self.generator = generator
        self.discriminator = discriminator

    def call(self, inputs):
        # Forward pass through both networks
        generated_images = self.generator(inputs)
        discriminator_output = self.discriminator(generated_images)
        return discriminator_output