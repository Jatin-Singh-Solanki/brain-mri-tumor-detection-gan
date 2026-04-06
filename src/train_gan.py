import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from dcgan_model import Generator, Discriminator


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_size = 16
image_size = 64
noise_dim = 100
epochs = 5
learning_rate_g = 0.0002
learning_rate_d = 0.0001

os.makedirs("outputs/generated_images", exist_ok=True)

transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

dataset = datasets.ImageFolder(root="data/raw", transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

generator = Generator(noise_dim=noise_dim).to(device)
discriminator = Discriminator().to(device)

criterion = nn.BCELoss()

optimizer_g = torch.optim.Adam(generator.parameters(), lr=learning_rate_g, betas=(0.5, 0.999))
optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=learning_rate_d, betas=(0.5, 0.999))


def save_generated_images(epoch, generator, noise_dim, device):
    generator.eval()

    with torch.no_grad():
        noise = torch.randn(8, noise_dim, device=device)
        fake_images = generator(noise).cpu()

    plt.figure(figsize=(12, 6))

    for i in range(8):
        plt.subplot(2, 4, i + 1)
        plt.imshow(fake_images[i].squeeze(), cmap="gray")
        plt.axis("off")

    plt.tight_layout()
    plt.savefig(f"outputs/generated_images/epoch_{epoch + 1}.png")
    plt.close()

    generator.train()


for epoch in range(epochs):
    for batch_idx, (real_images, _) in enumerate(dataloader):
        real_images = real_images.to(device)
        current_batch_size = real_images.size(0)

        real_labels = torch.full((current_batch_size, 1), 0.9, device=device)
        fake_labels = torch.full((current_batch_size, 1), 0.1, device=device)

        optimizer_d.zero_grad()

        real_outputs = discriminator(real_images)
        d_loss_real = criterion(real_outputs, real_labels)

        noise = torch.randn(current_batch_size, noise_dim, device=device)
        fake_images = generator(noise)
        fake_outputs = discriminator(fake_images.detach())
        d_loss_fake = criterion(fake_outputs, fake_labels)

        d_loss = (d_loss_real + d_loss_fake) / 2
        d_loss.backward()
        optimizer_d.step()

        optimizer_g.zero_grad()

        noise = torch.randn(current_batch_size, noise_dim, device=device)
        fake_images = generator(noise)
        outputs = discriminator(fake_images)

        generator_target_labels = torch.full((current_batch_size, 1), 0.9, device=device)
        g_loss = criterion(outputs, generator_target_labels)

        g_loss.backward()
        optimizer_g.step()

        if batch_idx % 100 == 0:
            print(
                f"Epoch [{epoch + 1}/{epochs}] "
                f"Batch [{batch_idx}/{len(dataloader)}] "
                f"D Loss: {d_loss.item():.4f} "
                f"G Loss: {g_loss.item():.4f}"
            )

    save_generated_images(epoch, generator, noise_dim, device)

print("GAN training completed")