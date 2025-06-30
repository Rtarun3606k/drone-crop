#!/usr/bin/env python3

import os

# Test the dataset paths
paths = [
    "/home/dragoon/Downloads/MH-SoyaHealthVision An Indian UAV and Leaf Image Dataset for Integrated Crop Health Assessment/Soyabean_UAV-Based_Image_Dataset/Soyabean_Mosaic/",
    "/home/dragoon/Downloads/MH-SoyaHealthVision An Indian UAV and Leaf Image Dataset for Integrated Crop Health Assessment/Soyabean_UAV-Based_Image_Dataset/rust/",
    "/home/dragoon/Downloads/MH-SoyaHealthVision An Indian UAV and Leaf Image Dataset for Integrated Crop Health Assessment/Soyabean_UAV-Based_Image_Dataset/Healthy_Soyabean/"
]

class_names = ['Soyabean_Mosaic', 'rust', 'healthy']

print("🔍 Dataset Path Verification")
print("=" * 50)

for i, (path, class_name) in enumerate(zip(paths, class_names)):
    print(f"\nClass {i}: {class_name}")
    print(f"Path: {path}")
    
    if os.path.exists(path):
        try:
            files = os.listdir(path)
            image_files = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff'))]
            print(f"✅ Directory exists")
            print(f"📁 Total files: {len(files)}")
            print(f"🖼️ Image files: {len(image_files)}")
            
            if image_files:
                print(f"📋 First 5 images: {image_files[:5]}")
        except Exception as e:
            print(f"❌ Error reading directory: {e}")
    else:
        print(f"❌ Directory does not exist")

print("\n" + "=" * 50)
print("Path verification complete!")
