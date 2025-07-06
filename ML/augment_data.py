#!/usr/bin/env python3
"""
Data Augmentation Script to Balance Class Distribution (Multi-threaded)

This script analyzes class imbalance in datasets and applies data augmentation 
to minority classes to achieve balanced datasets where all classes have equal 
number of images. Uses multi-threading for faster processing.

Usage Examples:
    # Analyze dataset only
    python augment_data.py /path/to/data --analyze-only
    
    # Balance dataset (dry run first to see what will happen)
    python augment_data.py /path/to/data --dry-run
    
    # Actually balance the dataset with default CPU count threads
    python augment_data.py /path/to/data
    
    # Balance with backup, specific target count, and custom thread count
    python augment_data.py /path/to/data --create-backup --target-count 1000 --workers 8

Expected directory structure:
    /path/to/data/
    ├── Class1/
    │   ├── image1.jpg
    │   ├── image2.jpg
    │   └── ...
    ├── Class2/
    │   ├── image1.jpg
    │   └── ...
    └── Class3/
        └── ...
"""

import os
import shutil
from pathlib import Path
import random
from collections import defaultdict
import argparse
import logging
from typing import Dict, List
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Third-party imports
try:
    import cv2
    import numpy as np
    from PIL import Image, ImageEnhance, ImageFilter
    import albumentations as A
except ImportError as e:
    print(f"ERROR: Missing required dependency: {e}")
    print("Please install required packages:")
    print("pip install opencv-python numpy pillow albumentations")
    exit(1)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DataAugmenter:
    """Class to handle data augmentation for balancing datasets."""
    
    def __init__(self, base_path: str, seed: int = 42, max_workers: int = None):
        """
        Initialize the DataAugmenter.
        
        Args:
            base_path: Path to the data directory containing train/test folders
            seed: Random seed for reproducibility
            max_workers: Maximum number of worker threads (default: CPU count)
        """
        self.base_path = Path(base_path)
        self.seed = seed
        self.max_workers = max_workers or os.cpu_count()
        self.lock = threading.Lock()  # For thread-safe operations
        random.seed(seed)
        np.random.seed(seed)
        
        # Define augmentation pipeline
        self.augmentation_pipeline = A.Compose([
            A.OneOf([
                A.RandomRotate90(p=1.0),
                A.Rotate(limit=45, p=1.0),
                A.Transpose(p=1.0),
            ], p=0.8),
            
            A.OneOf([
                A.HorizontalFlip(p=1.0),
                A.VerticalFlip(p=1.0),
            ], p=0.7),
            
            A.OneOf([
                A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=1.0),
                A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0),
                A.RandomGamma(gamma_limit=(80, 120), p=1.0),
            ], p=0.6),
            
            A.OneOf([
                A.GaussNoise(p=1.0),
                A.GaussianBlur(blur_limit=(3, 7), p=1.0),
                A.MotionBlur(blur_limit=7, p=1.0),
            ], p=0.4),
            
            A.OneOf([
                A.ElasticTransform(alpha=1, sigma=50, p=1.0),
                A.GridDistortion(num_steps=5, distort_limit=0.3, p=1.0),
                A.OpticalDistortion(distort_limit=0.05, p=1.0),
            ], p=0.3),
            
            A.OneOf([
                A.CoarseDropout(num_holes_range=(1, 8), hole_height_range=(0.05, 0.15), hole_width_range=(0.05, 0.15), p=1.0),
            ], p=0.2),
        ])
    
    def count_images_per_class(self) -> Dict[str, int]:
        """
        Count images in each class.
        
        Returns:
            Dictionary mapping class names to image counts
        """
        class_counts = {}
        
        for class_dir in self.base_path.iterdir():
            if class_dir.is_dir():
                image_files = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png"))
                class_counts[class_dir.name] = len(image_files)
                
        return class_counts
    
    def analyze_imbalance(self) -> Dict[str, int]:
        """
        Analyze class imbalance in the dataset.
        
        Returns:
            Dictionary of class counts
        """
        class_counts = self.count_images_per_class()
        
        logger.info("Class distribution analysis:")
        for class_name, count in sorted(class_counts.items()):
            logger.info(f"  {class_name}: {count:,} images")
            
        return class_counts
    
    def augment_image_task(self, args_tuple) -> tuple:
        """
        Thread-safe wrapper for augment_image method.
        
        Args:
            args_tuple: Tuple containing (source_image_path, output_path, augmentation_id)
            
        Returns:
            Tuple of (success: bool, output_path: Path, error_message: str or None)
        """
        source_image, output_path, aug_id = args_tuple
        
        try:
            # Create a separate random generator for each thread
            local_random = random.Random(self.seed + threading.current_thread().ident + aug_id)
            
            # Read image
            image = cv2.imread(str(source_image))
            if image is None:
                return False, output_path, f"Could not read image: {source_image}"
            
            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Apply augmentation
            augmented = self.augmentation_pipeline(image=image)
            augmented_image = augmented['image']
            
            # Convert back to BGR and save
            augmented_image = cv2.cvtColor(augmented_image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(output_path), augmented_image)
            
            return True, output_path, None
            
        except Exception as e:
            return False, output_path, str(e)

    def augment_image(self, image_path: Path, output_path: Path, augmentation_id: int) -> bool:
        """
        Apply augmentation to a single image and save it.
        
        Args:
            image_path: Path to source image
            output_path: Path to save augmented image
            augmentation_id: Unique ID for this augmentation
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Read image
            image = cv2.imread(str(image_path))
            if image is None:
                logger.warning(f"Could not read image: {image_path}")
                return False
            
            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Apply augmentation
            augmented = self.augmentation_pipeline(image=image)
            augmented_image = augmented['image']
            
            # Convert back to BGR and save
            augmented_image = cv2.cvtColor(augmented_image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(output_path), augmented_image)
            
            return True
            
        except Exception as e:
            logger.error(f"Error augmenting {image_path}: {e}")
            return False
    
    def balance_dataset(self, target_count: int = None, dry_run: bool = False):
        """
        Balance the dataset by augmenting minority classes to have equal number of images.
        Uses multi-threading for faster processing.
        
        Args:
            target_count: Target number of images per class. If None, uses max class count
            dry_run: If True, only show what would be done without actually doing it
        """
        class_counts = self.count_images_per_class()
        
        if not class_counts:
            logger.error(f"No classes found in {self.base_path}")
            return
        
        # Determine target count - use the maximum class count to ensure all classes are balanced
        if target_count is None:
            target_count = max(class_counts.values())
        
        logger.info(f"Balancing dataset to {target_count:,} images per class using {self.max_workers} threads")
        logger.info("Current distribution:")
        for class_name, count in sorted(class_counts.items()):
            logger.info(f"  {class_name}: {count:,} images")
        
        for class_name, current_count in class_counts.items():
            if current_count >= target_count:
                logger.info(f"Class '{class_name}' already has {current_count:,} images (>= {target_count:,})")
                continue
            
            needed_augmentations = target_count - current_count
            logger.info(f"Class '{class_name}': need {needed_augmentations:,} more images")
            
            if dry_run:
                continue
            
            class_dir = self.base_path / class_name
            original_images = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png"))
            
            if not original_images:
                logger.warning(f"No images found in {class_dir}")
                continue
            
            logger.info(f"Generating {needed_augmentations:,} augmented images for '{class_name}' using multi-threading...")
            
            # Prepare all augmentation tasks
            augmentation_tasks = []
            for i in range(needed_augmentations):
                # Randomly select an original image
                source_image = random.choice(original_images)
                
                # Generate output filename
                base_name = source_image.stem
                output_name = f"{base_name}_aug_{i:05d}.jpg"
                output_path = class_dir / output_name
                
                # Skip if file already exists
                if output_path.exists():
                    continue
                
                augmentation_tasks.append((source_image, output_path, i))
            
            if not augmentation_tasks:
                logger.info(f"All augmented images already exist for '{class_name}'")
                continue
            
            # Process tasks in parallel
            successful_augmentations = 0
            failed_augmentations = 0
            
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Submit all tasks
                future_to_task = {
                    executor.submit(self.augment_image_task, task): task 
                    for task in augmentation_tasks
                }
                
                # Process completed tasks
                for future in as_completed(future_to_task):
                    success, output_path, error_msg = future.result()
                    
                    if success:
                        successful_augmentations += 1
                        # Progress reporting
                        if successful_augmentations % 100 == 0:
                            logger.info(f"  Generated {successful_augmentations}/{len(augmentation_tasks)} images...")
                    else:
                        failed_augmentations += 1
                        if error_msg:
                            logger.warning(f"Failed to augment: {error_msg}")
            
            logger.info(f"Successfully generated {successful_augmentations:,} augmented images for '{class_name}'")
            if failed_augmentations > 0:
                logger.warning(f"Failed to generate {failed_augmentations:,} images for '{class_name}'")
    
    def create_backup(self):
        """
        Create a backup of the original dataset.
        """
        backup_path = self.base_path.parent / f"{self.base_path.name}_backup"
        
        if backup_path.exists():
            logger.info(f"Backup already exists at {backup_path}")
            return
        
        logger.info(f"Creating backup of dataset...")
        shutil.copytree(self.base_path, backup_path)
        logger.info(f"Backup created at {backup_path}")
    
    def restore_from_backup(self):
        """
        Restore dataset from backup.
        """
        backup_path = self.base_path.parent / f"{self.base_path.name}_backup"
        
        if not backup_path.exists():
            logger.error(f"No backup found at {backup_path}")
            return
        
        logger.info(f"Restoring dataset from backup...")
        if self.base_path.exists():
            shutil.rmtree(self.base_path)
        shutil.copytree(backup_path, self.base_path)
        logger.info(f"Dataset restored from {backup_path}")


def main():
    """Main function to run the data augmentation script."""
    parser = argparse.ArgumentParser(description="Augment data to balance class distribution - all classes will have equal number of images")
    parser.add_argument("data_path", nargs='?', default="/home/bruh/Documents/data-augmented/data-split",
                       help="Path to the data directory containing class subdirectories")
    parser.add_argument("--target-count", type=int, default=None,
                       help="Target number of images per class (default: max class count)")
    parser.add_argument("--dry-run", action="store_true",
                       help="Show what would be done without actually doing it")
    parser.add_argument("--create-backup", action="store_true",
                       help="Create backup before augmentation")
    parser.add_argument("--restore-backup", action="store_true",
                       help="Restore from backup")
    parser.add_argument("--analyze-only", action="store_true",
                       help="Only analyze class distribution without augmenting")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")
    parser.add_argument("--workers", type=int, default=None,
                       help="Number of worker threads (default: CPU count)")
    
    args = parser.parse_args()
    
    # Validate data path
    data_path = Path(args.data_path)
    if not data_path.exists():
        logger.error(f"Data path does not exist: {data_path}")
        return
    
    if not data_path.is_dir():
        logger.error(f"Data path is not a directory: {data_path}")
        return
    
    # Check if there are any subdirectories (classes)
    class_dirs = [d for d in data_path.iterdir() if d.is_dir()]
    if not class_dirs:
        logger.error(f"No class subdirectories found in: {data_path}")
        return
    
    logger.info(f"Found {len(class_dirs)} classes in {data_path}")
    
    # Initialize augmenter with multi-threading support
    augmenter = DataAugmenter(args.data_path, args.seed, args.workers)
    
    # Handle restore from backup
    if args.restore_backup:
        augmenter.restore_from_backup()
        return
    
    # Analyze current distribution
    logger.info("Analyzing current class distribution...")
    class_counts = augmenter.analyze_imbalance()
    
    if args.analyze_only:
        return
    
    # Create backup if requested
    if args.create_backup:
        augmenter.create_backup()
    
    # Balance dataset to ensure all classes have equal number of images
    target = args.target_count if args.target_count else max(class_counts.values())
    logger.info(f"\nStarting augmentation to balance all classes to {target:,} images each...")
    augmenter.balance_dataset(target, args.dry_run)
    
    # Analyze final distribution
    if not args.dry_run:
        logger.info("\n" + "="*50)
        logger.info("FINAL RESULTS - Class distribution after balancing:")
        final_counts = augmenter.analyze_imbalance()
        
        # Check if balancing was successful
        counts = list(final_counts.values())
        if len(set(counts)) == 1:
            logger.info(f"✅ SUCCESS: All classes now have exactly {counts[0]:,} images each!")
        else:
            logger.warning("⚠️  Classes still have different counts. Some augmentations may have failed.")
        logger.info("="*50)


if __name__ == "__main__":
    main()
