from pathlib import Path

data_dir = Path("data/raw")
yes_dir = data_dir / "yes"
no_dir = data_dir / "no"

yes_images = list(yes_dir.glob("*"))
no_images = list(no_dir.glob("*"))

print(f"Tumor images: {len(yes_images)}")
print(f"No tumor images: {len(no_images)}")

if len(yes_images) > 0:
    print("Sample tumor image:", yes_images[0].name)

if len(no_images) > 0:
    print("Sample no tumor image:", no_images[0].name)