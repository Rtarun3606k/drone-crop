#!/usr/bin/env python3
"""
Multi-threaded Image Augmentation Script for Doubling Class Sizes

This script augments images in a directory structure to double the number of images
in each class using multi-threading for improved performance.

# Basic usage
python augment_double.py data-split/

# With custom number of worker threads
python augment_double.py data-split/ --workers 8

# With verbose logging
python augment_double.py data-split/ --verbose

# With custom random seed for reproducibility
python augment_double.py data-split/ --seed 123
"""

import os
import shutil
import random
import threading
from pathlib import Path
import argparse
import logging
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image, ImageOps, ImageFilter, ImageEnhance
import time
from typing import List, Tuple, Dict

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Thread-safe counter for progress tracking
class ThreadSafeCounter:
    def __init__(self):
        self._value = 0
        self._lock = threading.Lock()
    
    def increment(self):
        with self._lock:
            self._value += 1
            return self._value
    
    @property
    def value(self):
        with self._lock:
            return self._value

class MultiThreadedImageAugmenter:
    """Multi-threaded image augmenter for doubling class sizes."""
    
    def __init__(self, base_path: str, max_workers: int = 4, seed: int = 42):
        """Initialize the augmenter."""
        self.base_path = Path(base_path)
        self.max_workers = max_workers
        self.seed = seed
        random.seed(seed)
        
        # Thread-safe counters
        self.processed_counter = ThreadSafeCounter()
        self.total_images = 0
        
    def get_image_files(self, directory: Path) -> List[Path]:
        """Get all image files from a directory."""
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        image_files = []
        
        for file_path in directory.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in image_extensions:
                image_files.append(file_path)
                
        return image_files
    
    def count_images_per_class(self) -> Dict[str, int]:
        """Count images in each class directory."""
        class_counts = {}
        
        for class_dir in self.base_path.iterdir():
            if class_dir.is_dir():
                image_files = self.get_image_files(class_dir)
                class_counts[class_dir.name] = len(image_files)
                logger.info(f"Class '{class_dir.name}': {len(image_files)} images")
                
        return class_counts
    
    def augment_single_image(self, image_path: Path, output_path: Path) -> bool:
        """Apply augmentation to a single image."""
        try:
            # Open image
            img = Image.open(image_path)
            
            # Convert to RGB if necessary
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Apply random augmentations
            augmentations = []
            
            # Rotation (0, 90, 180, 270 degrees)
            if random.random() < 0.7:
                angle = random.choice([0, 90, 180, 270])
                if angle != 0:
                    img = img.rotate(angle, expand=True)
                    augmentations.append(f"rot{angle}")
            
            # Horizontal flip
            if random.random() < 0.5:
                img = ImageOps.mirror(img)
                augmentations.append("hflip")
            
            # Vertical flip
            if random.random() < 0.3:
                img = ImageOps.flip(img)
                augmentations.append("vflip")
            
            # Brightness adjustment
            if random.random() < 0.6:
                factor = random.uniform(0.7, 1.3)
                enhancer = ImageEnhance.Brightness(img)
                img = enhancer.enhance(factor)
                augmentations.append(f"bright{factor:.2f}")
            
            # Contrast adjustment
            if random.random() < 0.6:
                factor = random.uniform(0.7, 1.3)
                enhancer = ImageEnhance.Contrast(img)
                img = enhancer.enhance(factor)
                augmentations.append(f"cont{factor:.2f}")
            
            # Color saturation
            if random.random() < 0.4:
                factor = random.uniform(0.8, 1.2)
                enhancer = ImageEnhance.Color(img)
                img = enhancer.enhance(factor)
                augmentations.append(f"color{factor:.2f}")
            
            # Gaussian blur
            if random.random() < 0.3:
                radius = random.uniform(0.5, 2.0)
                img = img.filter(ImageFilter.GaussianBlur(radius=radius))
                augmentations.append(f"blur{radius:.1f}")
            
            # Sharpness
            if random.random() < 0.3:
                factor = random.uniform(0.8, 1.5)
                enhancer = ImageEnhance.Sharpness(img)
                img = enhancer.enhance(factor)
                augmentations.append(f"sharp{factor:.2f}")
            
            # Save the augmented image
            img.save(output_path, quality=95, optimize=True)
            
            # Update progress counter
            current_count = self.processed_counter.increment()
            if current_count % 10 == 0:
                progress = (current_count / self.total_images) * 100
                logger.info(f"Progress: {current_count}/{self.total_images} ({progress:.1f}%)")
            
            logger.debug(f"Augmented {image_path.name} -> {output_path.name} with: {', '.join(augmentations)}")
            return True
            
        except Exception as e:
            logger.error(f"Error augmenting {image_path}: {str(e)}")
            return False
    
    def generate_augmented_filename(self, original_path: Path, index: int) -> str:
        """Generate a unique filename for augmented image."""
        stem = original_path.stem
        suffix = original_path.suffix
        return f"{stem}_aug_double_{index:05d}{suffix}"
    
    def augment_class_images(self, class_name: str, image_files: List[Path]) -> int:
        """Augment images for a single class to double the count."""
        class_dir = self.base_path / class_name
        original_count = len(image_files)
        target_count = original_count  # We want to generate the same number to double
        
        logger.info(f"Augmenting class '{class_name}': generating {target_count} new images")
        
        # Prepare augmentation tasks
        augmentation_tasks = []
        
        for i in range(target_count):
            # Select a random source image to augment
            source_image = random.choice(image_files)
            
            # Generate output filename
            output_filename = self.generate_augmented_filename(source_image, i)
            output_path = class_dir / output_filename
            
            # Ensure we don't overwrite existing files
            counter = 0
            while output_path.exists():
                counter += 1
                stem = source_image.stem
                suffix = source_image.suffix
                output_filename = f"{stem}_aug_double_{i:05d}_{counter:03d}{suffix}"
                output_path = class_dir / output_filename
            
            augmentation_tasks.append((source_image, output_path))
        
        # Execute augmentation tasks with thread pool
        successful_augmentations = 0
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_task = {
                executor.submit(self.augment_single_image, source, target): (source, target)
                for source, target in augmentation_tasks
            }
            
            # Process completed tasks
            for future in as_completed(future_to_task):
                source, target = future_to_task[future]
                try:
                    if future.result():
                        successful_augmentations += 1
                except Exception as e:
                    logger.error(f"Task failed for {source} -> {target}: {str(e)}")
        
        logger.info(f"Class '{class_name}': Successfully augmented {successful_augmentations}/{target_count} images")
        return successful_augmentations
    
    def augment_all_classes(self) -> Dict[str, int]:
        """Augment all classes to double their image counts."""
        if not self.base_path.exists():
            raise ValueError(f"Base path does not exist: {self.base_path}")
        
        start_time = time.time()
        
        # Get initial class counts
        logger.info("Analyzing dataset structure...")
        class_counts = self.count_images_per_class()
        
        if not class_counts:
            logger.warning("No classes found in the dataset!")
            return {}
        
        # Calculate total images to process
        self.total_images = sum(class_counts.values())
        logger.info(f"Total images to generate: {self.total_images}")
        
        # Process each class
        results = {}
        
        for class_name, original_count in class_counts.items():
            logger.info(f"\n{'='*50}")
            logger.info(f"Processing class: {class_name}")
            logger.info(f"Original count: {original_count}")
            
            class_dir = self.base_path / class_name
            image_files = self.get_image_files(class_dir)
            
            if not image_files:
                logger.warning(f"No images found in class '{class_name}'")
                results[class_name] = 0
                continue
            
            # Augment this class
            augmented_count = self.augment_class_images(class_name, image_files)
            results[class_name] = augmented_count
            
            # Log updated counts
            new_total = len(self.get_image_files(class_dir))
            logger.info(f"Class '{class_name}' now has {new_total} images (was {original_count})")
        
        # Final summary
        end_time = time.time()
        duration = end_time - start_time
        
        logger.info(f"\n{'='*50}")
        logger.info("AUGMENTATION COMPLETE!")
        logger.info(f"Total processing time: {duration:.2f} seconds")
        logger.info(f"Images processed: {self.processed_counter.value}")
        
        # Show final counts
        final_counts = self.count_images_per_class()
        logger.info("\nFinal class distribution:")
        for class_name, count in final_counts.items():
            original = class_counts.get(class_name, 0)
            logger.info(f"  {class_name}: {count} images (was {original}, +{count - original})")
        
        return results

def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(description="Multi-threaded Image Augmentation for Doubling Class Sizes")
    parser.add_argument("input_dir", help="Input directory containing class subdirectories")
    parser.add_argument("--workers", "-w", type=int, default=4, 
                       help="Number of worker threads (default: 4)")
    parser.add_argument("--seed", "-s", type=int, default=42,
                       help="Random seed for reproducible results (default: 42)")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Validate input directory
    input_path = Path(args.input_dir)
    if not input_path.exists():
        logger.error(f"Input directory does not exist: {input_path}")
        return 1
    
    if not input_path.is_dir():
        logger.error(f"Input path is not a directory: {input_path}")
        return 1
    
    # Create augmenter and run
    try:
        logger.info(f"Starting image augmentation with {args.workers} worker threads")
        logger.info(f"Input directory: {input_path}")
        logger.info(f"Random seed: {args.seed}")
        
        augmenter = MultiThreadedImageAugmenter(
            base_path=str(input_path),
            max_workers=args.workers,
            seed=args.seed
        )
        
        results = augmenter.augment_all_classes()
        
        if results:
            logger.info("Augmentation completed successfully!")
            return 0
        else:
            logger.error("No augmentation was performed")
            return 1
            
    except Exception as e:
        logger.error(f"Augmentation failed: {str(e)}")
        return 1

if __name__ == "__main__":
    exit(main())