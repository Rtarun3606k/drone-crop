import os
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed

# Set your dataset path here
DATASET_DIR = "/home/bruh/Documents/BNN2/split"

# Define minimum resolution for deletion (approx. 4K)
MIN_WIDTH = 3800
MIN_HEIGHT = 2100

# Supported image formats
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp")

def is_large_image(image_path):
    try:
        with Image.open(image_path) as img:
            width, height = img.size
            return width >= MIN_WIDTH or height >= MIN_HEIGHT
    except Exception as e:
        print(f"Error reading {image_path}: {e}")
        return False

def process_image(file_path):
    if is_large_image(file_path):
        try:
            os.remove(file_path)
            print(f"Deleted: {file_path}")
            return 1
        except Exception as e:
            print(f"Error deleting {file_path}: {e}")
    return 0

def delete_large_images_multithreaded(dataset_dir, max_workers=8):
    all_files = []
    for root, _, files in os.walk(dataset_dir):
        for file in files:
            if file.lower().endswith(IMAGE_EXTENSIONS):
                all_files.append(os.path.join(root, file))

    deleted_count = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_image, path) for path in all_files]
        for future in as_completed(futures):
            deleted_count += future.result()

    print(f"\n✅ Done. Deleted {deleted_count} images with resolution around or above 4K.")

if __name__ == "__main__":
    delete_large_images_multithreaded(DATASET_DIR)
