import os
import json
from fastai.vision.all import *
from pathlib import Path
from PIL import Image
from tqdm import tqdm

def prepare_data(limit=100):
    data_dir = Path("Data")
    image_dir = data_dir / "images"
    metadata_file = data_dir / "metadata.jsonl"
    
    if metadata_file.exists() and image_dir.exists() and len(list(image_dir.glob("*.jpg"))) >= limit:
        print("Dataset already prepared.")
        return

    image_dir.mkdir(parents=True, exist_ok=True)
    
    print("Downloading PETS dataset via fastai...")
    path = untar_data(URLs.PETS)
    images_path = path/"images"
    files = get_image_files(images_path)
    
    print(f"Saving {limit} images to Data/...")
    
    count = 0
    with open(metadata_file, "w") as f:
        for img_path in tqdm(files[:limit]):
            try:
                img = Image.open(img_path).convert("RGB")
                # Caption is the name of the file (breed)
                caption = " ".join(img_path.stem.split("_")[:-1])
                
                img_name = f"{count:05d}.jpg"
                save_path = image_dir / img_name
                img.save(save_path)
                
                entry = {"file_name": img_name, "text": f"a photo of a {caption}"}
                f.write(json.dumps(entry) + "\n")
                
                count += 1
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                continue

    print(f"Successfully saved {count} images and metadata.")

if __name__ == "__main__":
    prepare_data(limit=100)