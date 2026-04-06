import torch
import torchvision
import numpy as np
import cv2
import pandas as pd

print("PyTorch version:", torch.__version__)
print("Torchvision version:", torchvision.__version__)
print("NumPy version:", np.__version__)
print("OpenCV version:", cv2.__version__)
print("Pandas version:", pd.__version__)
print("CUDA available:", torch.cuda.is_available())
print("Setup is working fine")