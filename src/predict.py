import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

image_size = 64


class CNNClassifier(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Flatten(),

            nn.Linear(64 * 16 * 16, 128),
            nn.ReLU(),

            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)


transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
])

model = CNNClassifier().to(device)
model.load_state_dict(torch.load("models/brain_tumor_classifier.pth", map_location=device))
model.eval()

image_path = "data/raw/yes/0050cbda7dc3d3709039151cc80040_big_gallery.jpeg"

image = Image.open(image_path)
image = transform(image).unsqueeze(0).to(device)

with torch.no_grad():
    output = model(image)
    prediction = output.item()

if prediction >= 0.5:
    print(f"Prediction Score: {prediction:.4f}")
    print("Prediction: Tumor")
else:
    print(f"Prediction Score: {prediction:.4f}")
    print("Prediction: No Tumor")