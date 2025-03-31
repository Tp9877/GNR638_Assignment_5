import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from model import VAE
from utils import loss_function

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # Normalize to mean=0, std=1 for stability
])

train_dataset = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2, pin_memory=True)

indices_7 = [i for i, (img, label) in enumerate(train_dataset) if label == 7]

# Create a subset dataset with only class 7
train_dataset_7 = Subset(train_dataset, indices_7)

# Create DataLoader
train_loader_7 = DataLoader(train_dataset_7, batch_size=64, shuffle=True, num_workers=2, pin_memory=True)


vae = VAE(latent_dim1=32, latent_dim2=16).to(device)
optimizer = torch.optim.Adam(vae.parameters(), lr=1e-3)  # Lower LR for stability


num_epochs = 100
scaler = torch.amp.GradScaler()

#Training


vae.train()
for epoch in range(num_epochs):
    for batch_idx, (data, _) in enumerate(train_loader):  # change it to train_loader_7 if we want to train only on Class "7" of MNIST Dataset. Currently trained on entire MNIST dataset 
        data = data.to(device)  # Move tensor to GPU if available


        optimizer.zero_grad()  # Reset gradients

        # Forward pass
        recon_x, _, _, mu, logvar = vae(data)  # Ignore extra values


        # Compute loss
        loss = loss_function(recon_x, data, mu, logvar)

        # Backward pass
        loss.backward()

        # Apply gradient clipping
        torch.nn.utils.clip_grad_norm_(vae.parameters(), max_norm=5)

        # Update parameters
        optimizer.step()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

#Generation of Samples

vae.eval()
latent_dim1=32
latent_dim2=16
with torch.no_grad():
    z1 = torch.randn(100, latent_dim1).to(device)  # Latent dim 1
    z2 = torch.randn(100, latent_dim2).to(device)  # Latent dim 2
    z = torch.cat([z1, z2], dim=1)
    print("Concatenated Latent Shape:", z.shape)
    print("Generated Latent Shapes:", z1.shape, z2.shape)  # Debugging step
    samples = vae.decoder(z1, z2).cpu()


fig, axes = plt.subplots(10, 10, figsize=(10, 10))
for i, ax in enumerate(axes.flat):
    ax.imshow(samples[i, 0], cmap="gray")
    ax.axis("off")

plt.show()

