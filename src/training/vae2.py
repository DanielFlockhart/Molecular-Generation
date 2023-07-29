import torch
import torch.nn as nn
import torch.nn.functional as F

class VariationalAutoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim, output_dim, temperature=1):
        super(VariationalAutoencoder, self).__init__()
        self.output_dim = output_dim
        self.temperature = temperature
        self.encoder = self.build_encoder(input_dim, latent_dim)
        self.decoder = self.build_decoder(latent_dim, self.output_dim)

    def build_encoder(self, input_dim, latent_dim):
        '''
        Building the encoder part of the model
        '''
        encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim),
            nn.Sigmoid()
        )
        return encoder

    def build_decoder(self, latent_dim, output_dim):
        '''
        Build the decoder part of the model
        '''
        decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim[0] * output_dim[1] * output_dim[2]),
            nn.ReLU(),
            nn.Unflatten(1, (output_dim[0], output_dim[1], output_dim[2])),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=1, padding=0),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=1, padding=0),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=1, padding=0),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid(),
            nn.Flatten(),
            nn.Linear(output_dim[0] * output_dim[1] * output_dim[2], output_dim[0] * output_dim[1] * output_dim[2]),
            nn.Sigmoid()
        )
        return decoder

    def sampling(self, z_mean, z_log_var):
        '''
        Samples from the latent space using the reparameterization trick
        '''
        epsilon = torch.randn_like(z_mean)
        scaled_epsilon = epsilon * self.temperature
        return z_mean + torch.exp(0.5 * z_log_var) * scaled_epsilon

    def forward(self, inputs):
        '''
        Runs the model
        '''
        latent = self.encoder(inputs)
        reconstructed = self.decoder(latent)
        return reconstructed

    def compute_loss(self, inputs, targets, reconstructed, sample_weight=None, training=False):
        '''
        Calculates the loss of the model
        '''
        bce_loss = F.binary_cross_entropy(reconstructed, targets, reduction='mean')
        mse_loss = F.mse_loss(reconstructed, targets, reduction='mean')
        reconstruction_loss = bce_loss + mse_loss
        print(" - total_loss:", reconstruction_loss.item())
        return reconstruction_loss
