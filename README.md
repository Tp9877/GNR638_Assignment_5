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


  Above is representation of  Hierarchical Variational Autoencoder that is implemented by us in the code.



  ## Overview
This project implements a **Hierarchical Variational Autoencoder (H-VAE)** on the **MNIST dataset**. Unlike a standard VAE, the H-VAE uses a **two-level latent structure (`z1`, `z2`)** to capture multi-scale representations of data.

## Architecture
The H-VAE consists of:

1. **Encoder**: Extracts features and maps them to a two-level latent space (`z1`, `z2`). Note (`z1`, `z2`) fixed to ('32','16').
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
   

Total loss: L_recon + KL_div


## Training
1. Load **MNIST dataset** (Note: the data is not normalized).
2. Train the VAE using **Adam optimizer** with `lr=0.001`. (We varied lr from 1e-4 to 1e-2 and we got the best result for lr=1e-3)
3. Optimize the **total loss** for better reconstruction and meaningful latent spaces.
4. Monitor **loss convergence** over 100 epochs.

---

## Generating New Images
- Sample **random latent vectors (`z1`, `z2`)** from a normal distribution.
- Pass them through the **decoder** to generate **new MNIST-style digits**.

---
## Results - Generated Images

### **1) VAE Trained on Entire MNIST Dataset**
After training on the entire MNIST dataset, the model is able to generate diverse handwritten digits, capturing the full distribution of the data. The results show the VAE's ability to model various digits and their characteristics.

![Generated Digits on Full MNIST](https://github.com/user-attachments/assets/aa2c71a0-06c8-46f4-9bba-c44deb3e9411)

### **2) VAE Trained on Class "7" of MNIST Dataset**
When trained only on images belonging to the **digit "7"**, the VAE focuses on generating digits that resemble the characteristics of the "7" class. The images generated by the VAE trained solely on the class "7" are more consistent and focused on that specific class.

![Generated Digits on Class 7](https://github.com/user-attachments/assets/b527d606-1f5d-4b3b-b994-f97449ae5fc7)


## Google Colab

[Hierarchical VAE](https://colab.research.google.com/drive/1sudGbTVUL-sPG5fg6WBzskHmvYykhGPq?usp=sharing)
