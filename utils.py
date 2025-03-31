import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

def kl_divergence(mu, logvar):
    logvar = logvar.clamp(-10, 10)  # Clamp to prevent instability
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp().clamp(max=1e6))
def loss_function(recon_x, x, mu, logvar):
    recon_loss = torch.nn.functional.mse_loss(recon_x, x, reduction='sum')
    kl_loss = kl_divergence(mu, logvar)  # Use stable KL computation
    return recon_loss + kl_loss
