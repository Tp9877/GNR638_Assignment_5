import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, latent_dim1=32, latent_dim2=16):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1)  # (28x28) -> (14x14)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)  # (14x14) -> (7x7)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)

        # First latent layer
        self.fc_mu1 = nn.Linear(128, latent_dim1)
        self.fc_logvar1 = nn.Linear(128, latent_dim1)

        # Second latent layer (depends on first)
        self.fc_mu2 = nn.Linear(latent_dim1, latent_dim2)
        self.fc_logvar2 = nn.Linear(latent_dim1, latent_dim2)

    def forward(self, x):
        x = F.tanh(self.conv1(x))
        x = F.tanh(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = F.tanh(self.fc1(x))

        # First-level latent variables
        mu1 = self.fc_mu1(x)
        logvar1 = self.fc_logvar1(x).clamp(min=-10, max=10)  # Clamping to avoid extreme values
        logvar1 = logvar1.to(torch.float32)

        z1 = self.reparameterize(mu1, logvar1)

        # Second-level latent variables
        mu2 = self.fc_mu2(z1)
        logvar2 = self.fc_logvar2(z1).clamp(min=-10, max=10)  # Clamping to avoid extreme values
        logvar2 = logvar2.to(torch.float32)
        z2 = self.reparameterize(mu2, logvar2)

        return z1, mu1, logvar1, z2, mu2, logvar2


    def reparameterize(self, mu, logvar):
        logvar = logvar.clamp(-10, 10)  # Prevent extreme values
        std = torch.exp(0.5 * logvar) + 1e-6  # Avoid zero std
        eps = torch.randn_like(std)  # Sample noise
        return mu + eps * std  # Adding epsilon to prevent division by zero



class Decoder(nn.Module):
    def __init__(self, latent_dim1=32, latent_dim2=16):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim1 + latent_dim2, 128)
        self.fc2 = nn.Linear(128, 64 * 7 * 7)
        self.deconv1 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)  # (7x7) -> (14x14)
        self.deconv2 = nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1)  # (14x14) -> (28x28)

    def forward(self, z1, z2):
        x = F.tanh(self.fc1(torch.cat([z1, z2], dim=1)))
        x = F.tanh(self.fc2(x))
        x = x.view(x.size(0), 64, 7, 7)
        x = F.tanh(self.deconv1(x))
        x = self.deconv2(x)  # Remove sigmoid (since BCEWithLogits will handle it)
  # Output between 0 and 1
        return x


class VAE(nn.Module):
    def __init__(self, latent_dim1=32, latent_dim2=16):
        super(VAE, self).__init__()
        self.encoder = Encoder(latent_dim1, latent_dim2)
        self.decoder = Decoder(latent_dim1, latent_dim2)

    def forward(self, x):
        z1, mu1, logvar1, z2, mu2, logvar2 = self.encoder(x)
        recon_x = self.decoder(z1, z2)
        return recon_x, mu1, logvar1, mu2, logvar2
