# GNR638_Assignment_5

                     ┌──────────────────────────┐
                     │    Input Image (28x28)   │
                     └──────────────────────────┘
                                  │
            ┌─────────────────────┴─────────────────────┐
            │                                           │
    ┌──────────────────┐                       ┌──────────────────┐
    │  Conv2D (32, 4x4)│                       │  Conv2D (64, 4x4)│
    │  Stride=2, ReLU  │                       │  Stride=2, ReLU  │
    └──────────────────┘                       └──────────────────┘
             └─────────────────────────────────────────┘         
                                  │
                     ┌──────────────────────────┐
                     │   Flatten (7x7x64 → 128) │
                     └──────────────────────────┘
                                  │
                    ┌──────────────────────────┐
                    │   Fully Connected (128)  │
                    └──────────────────────────┘
                                  │
                   ┌─────────────────────────────┐
                   │   First Latent Space (z1)   │
                   │      μ1, logvar1 → z1       │
                   └─────────────────────────────┘
                                  │
                   ┌─────────────────────────────┐
                   │   Second Latent Space (z2)  │
                   │      μ2, logvar2 → z2       │
                   └─────────────────────────────┘
                                  │
                     ┌──────────────────────────┐
                     │    Concatenate z1, z2    │
                     └──────────────────────────┘
                                  │
                    ┌──────────────────────────┐
                    │   Fully Connected (128)  │
                    └──────────────────────────┘
                                  │
                     ┌──────────────────────────┐
                     │ Unflatten (128 → 7x7x64) │
                     └──────────────────────────┘
                                  │
            ┌─────────────────────┴─────────────────────┐
            │                                           │
    ┌──────────────────┐                       ┌──────────────────┐
    │Deconv2D (64, 4x4)│                       │Deconv2D (32, 4x4)│
    │  Stride=2, ReLU  │                       │  Stride=2, ReLU  │
    └──────────────────┘                       └──────────────────┘
             └─────────────────────────────────────────┘
                                  │
                     ┌──────────────────────────┐
                     │   Output Image (28x28)   │
                     └──────────────────────────┘


  Above is representation of heirarichal vae that is implemented by us in the code.



  ## Overview
This project implements a **Hierarchical Variational Autoencoder (H-VAE)** on the **MNIST dataset**. Unlike a standard VAE, the H-VAE uses a **two-level latent structure (`z1`, `z2`)** to capture multi-scale representations of data.

## Architecture
The H-VAE consists of:

1. **Encoder**: Extracts features and maps them to a two-level latent space (`z1`, `z2`).
2. **Reparameterization Trick**: Ensures sampling is differentiable for backpropagation.
3. **Decoder**: Reconstructs the image from latent variables (`z1`, `z2`).
4. **Loss Function**: Combines **Reconstruction Loss** and **KL Divergence Loss** to regularize latent space.

---

## Model Components

### **Encoder**
- Uses **two convolutional layers** to extract spatial features.
- A **fully connected (FC) layer** maps these features to a bottleneck representation.
- Computes **two sets of means (`μ1, μ2`) and variances (`logσ1², logσ2²`)**:
  - `z1` is sampled from (`μ1`, `logσ1²`).
  - `z2` is sampled from (`μ2`, `logσ2²`), conditioned on `z1`.
- **Tanh activation** is used for non-linearity.

### **Reparameterization Trick**
- Ensures **differentiability** during sampling:
  - `z = μ + σ * ε`, where `ε ~ N(0,1)`.
- Log-variance is **clamped** to prevent numerical instability.

### **Decoder**
- **Fully connected layers** reshape the latent vector back into an image-like representation.
- **Transpose convolutions** (deconvolutions) upsample the feature map back to **28x28 resolution**.
- **Tanh activation** is used instead of ReLU to provide a smooth transition.

---

## Loss Function
The loss function consists of:
1. **Reconstruction Loss** (Binary Cross-Entropy): Measures how close the reconstructed image is to the input.
2. **KL Divergence Loss**: Regularizes the latent space to approximate a normal distribution.
   

Total loss:
\[
\mathcal{L} = \mathcal{L}_{\text{recon}} + \mathcal{KL}_{\text{div}}
\]

