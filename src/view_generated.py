import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

image_dir = "outputs/generated_images"
image_files = sorted(os.listdir(image_dir))

plt.figure(figsize=(15, 8))

for i, image_file in enumerate(image_files):
    image_path = os.path.join(image_dir, image_file)
    image = mpimg.imread(image_path)

    plt.subplot(2, 3, i + 1)
    plt.imshow(image)
    plt.title(image_file)
    plt.axis("off")

plt.tight_layout()
plt.show()