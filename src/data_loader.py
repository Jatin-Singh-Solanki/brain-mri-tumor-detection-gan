import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

data_dir = "data/raw"

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor()
])

dataset = datasets.ImageFolder(root=data_dir, transform=transform)

dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

print("Total images:", len(dataset))
print("Classes:", dataset.classes)

for images, labels in dataloader:
    print("Batch shape:", images.shape)
    print("First 5 labels:", labels[:5].tolist())
    break
