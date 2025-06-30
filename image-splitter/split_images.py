#!/usr/bin/env python3
"""
Image Splitting Script
This script splits each image in the data folder into 16 smaller images (4x4 grid)
and saves them in data-split folder with the same directory structure.
"""

import os
import sys
from PIL import Image
from pathlib import Path

def split_image(image_path, output_dir, splits_per_row=4, splits_per_col=4):
    """
    Split an image into smaller images in a grid pattern.
    
    Args:
        image_path (str): Path to the input image
        output_dir (str): Directory to save the split images
        splits_per_row (int): Number of horizontal splits (default: 4)
        splits_per_col (int): Number of vertical splits (default: 4)
    """
    try:
        # Open the image
        with Image.open(image_path) as img:
            # Get image dimensions
            width, height = img.size
            
            # Calculate dimensions for each split
            split_width = width // splits_per_row
            split_height = height // splits_per_col
            
            # Get the base filename without extension
            base_name = Path(image_path).stem
            extension = Path(image_path).suffix
            
            # Create output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)
            
            # Split the image
            for row in range(splits_per_col):
                for col in range(splits_per_row):
                    # Calculate crop coordinates
                    left = col * split_width
                    top = row * split_height
                    right = left + split_width
                    bottom = top + split_height
                    
                    # Crop the image
                    cropped_img = img.crop((left, top, right, bottom))
                    
                    # Create filename for the split image
                    split_filename = f"{base_name}_split_{row:02d}_{col:02d}{extension}"
                    split_path = os.path.join(output_dir, split_filename)
                    
                    # Save the split image
                    cropped_img.save(split_path)
                    
            print(f"Successfully split {image_path} into {splits_per_row * splits_per_col} images")
            return True
            
    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")
        return False

def process_directory(input_dir, output_dir):
    """
    Process all images in a directory and its subdirectories.
    
    Args:
        input_dir (str): Input directory path
        output_dir (str): Output directory path
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Supported image extensions
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    
    total_images = 0
    processed_images = 0
    
    # Walk through all directories and files
    for root, dirs, files in os.walk(input_path):
        for file in files:
            # Check if file is an image
            if Path(file).suffix.lower() in image_extensions:
                total_images += 1
                
                # Get relative path from input directory
                rel_path = Path(root).relative_to(input_path)
                
                # Create corresponding output directory
                output_subdir = output_path / rel_path
                
                # Full paths
                input_file_path = os.path.join(root, file)
                
                # Split the image
                if split_image(input_file_path, str(output_subdir)):
                    processed_images += 1
    
    print(f"\nProcessing complete!")
    print(f"Total images found: {total_images}")
    print(f"Successfully processed: {processed_images}")
    print(f"Failed: {total_images - processed_images}")

def main():
    """Main function"""
    # Define paths
    script_dir = Path(__file__).parent
    input_dir = script_dir / "data"
    output_dir = script_dir / "data-split"
    
    # Check if input directory exists
    if not input_dir.exists():
        print(f"Error: Input directory '{input_dir}' does not exist!")
        sys.exit(1)
    
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print("Starting image splitting process...")
    print("Each image will be split into 16 smaller images (4x4 grid)")
    print("-" * 60)
    
    # Process all images
    process_directory(str(input_dir), str(output_dir))

if __name__ == "__main__":
    main()
