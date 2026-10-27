
from huggingface_hub import hf_hub_download
from pathlib import Path
import zipfile

repo_id = "mohanty/PlantVillage"
data_dir = Path("data")
data_dir.mkdir(exist_ok=True)

print("Downloading the image archive...")

zip_path = hf_hub_download(
    repo_id=repo_id,
    repo_type="dataset",
    filename="data.zip"
)

print("Archive downloaded:", zip_path)
print("Extracting images...")

with zipfile.ZipFile(zip_path, "r") as archive:
    archive.extractall(data_dir)

print("Extraction completed!")

print("Checking for extracted images...")
images = list(data_dir.rglob("*.JPG"))
images += list(data_dir.rglob("*.jpg"))
images += list(data_dir.rglob("*.png"))

print("Images found:", len(images))

if images:
    print("Example image:", images[0])
else:
    print("No images found. Check the extracted folder structure.")
