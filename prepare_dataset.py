#!/usr/bin/env python3
"""
Dataset Preparation Utility for Cats vs Dogs
============================================

This script prepares the dataset for modern transfer learning by:
- Creating proper train/validation splits
- Organizing data into expected directory structure
- Cleaning up any corrupted images
- Providing dataset statistics

Usage: python prepare_dataset.py
"""

import os
import shutil
import random
from pathlib import Path
from PIL import Image
import argparse

def validate_image(image_path):
    """Check if image is valid and can be opened."""
    try:
        with Image.open(image_path) as img:
            img.verify()
        return True
    except (IOError, SyntaxError):
        print(f"Warning: Corrupted image found: {image_path}")
        return False

def prepare_dataset(source_dir="training_set/training_set", 
                   target_dir="data", 
                   val_split=0.2, 
                   seed=42):
    """
    Prepare dataset with proper train/validation split.
    
    Args:
        source_dir: Source directory containing cats/ and dogs/ folders
        target_dir: Target directory for organized dataset
        val_split: Fraction of data to use for validation (0.2 = 20%)
        seed: Random seed for reproducible splits
    """
    
    print("🐱🐶 Preparing Cats vs Dogs Dataset")
    print("=" * 50)
    
    # Set random seed for reproducible splits
    random.seed(seed)
    
    source_path = Path(source_dir)
    target_path = Path(target_dir)
    
    if not source_path.exists():
        raise FileNotFoundError(f"Source directory not found: {source_path}")
    
    # Create target directory structure
    directories = [
        target_path / "train" / "cats",
        target_path / "train" / "dogs", 
        target_path / "validation" / "cats",
        target_path / "validation" / "dogs"
    ]
    
    for dir_path in directories:
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"✓ Created directory: {dir_path}")
    
    # Process each class
    stats = {}
    
    for class_name in ["cats", "dogs"]:
        print(f"\n📁 Processing {class_name}...")
        
        source_class_dir = source_path / class_name
        if not source_class_dir.exists():
            print(f"Warning: {source_class_dir} not found, skipping...")
            continue
        
        # Get all image files
        image_files = list(source_class_dir.glob("*.jpg")) + list(source_class_dir.glob("*.jpeg"))
        
        # Filter out corrupted images
        valid_images = [img for img in image_files if validate_image(img)]
        corrupted_count = len(image_files) - len(valid_images)
        
        if corrupted_count > 0:
            print(f"⚠️  Found {corrupted_count} corrupted images, skipping them")
        
        # Shuffle for random split
        random.shuffle(valid_images)
        
        # Calculate split
        n_val = int(len(valid_images) * val_split)
        n_train = len(valid_images) - n_val
        
        train_images = valid_images[:n_train]
        val_images = valid_images[n_train:]
        
        # Copy training images
        print(f"📋 Copying {len(train_images)} training images...")
        for img_path in train_images:
            dest_path = target_path / "train" / class_name / img_path.name
            shutil.copy2(img_path, dest_path)
        
        # Copy validation images
        print(f"📋 Copying {len(val_images)} validation images...")
        for img_path in val_images:
            dest_path = target_path / "validation" / class_name / img_path.name
            shutil.copy2(img_path, dest_path)
        
        # Store statistics
        stats[class_name] = {
            "total": len(valid_images),
            "train": len(train_images),
            "validation": len(val_images),
            "corrupted": corrupted_count
        }
        
        print(f"✅ {class_name}: {len(train_images)} train, {len(val_images)} validation")
    
    # Print final statistics
    print("\n📊 DATASET STATISTICS")
    print("=" * 50)
    
    total_train = sum(stats[cls]["train"] for cls in stats)
    total_val = sum(stats[cls]["validation"] for cls in stats)
    total_images = total_train + total_val
    
    print(f"Training images:   {total_train}")
    print(f"Validation images: {total_val}")
    print(f"Total images:      {total_images}")
    print(f"Validation split:  {val_split:.1%}")
    
    print("\nPer class breakdown:")
    for class_name, class_stats in stats.items():
        print(f"  {class_name.capitalize()}: "
              f"{class_stats['train']} train, "
              f"{class_stats['validation']} val "
              f"({class_stats['corrupted']} corrupted)")
    
    print(f"\n✅ Dataset prepared successfully!")
    print(f"📁 Data organized in: {target_path.absolute()}")
    print("\nDirectory structure:")
    print(f"  {target_dir}/")
    print(f"    ├── train/")
    print(f"    │   ├── cats/     ({stats.get('cats', {}).get('train', 0)} images)")
    print(f"    │   └── dogs/     ({stats.get('dogs', {}).get('train', 0)} images)")
    print(f"    └── validation/")
    print(f"        ├── cats/     ({stats.get('cats', {}).get('validation', 0)} images)")
    print(f"        └── dogs/     ({stats.get('dogs', {}).get('validation', 0)} images)")

def main():
    parser = argparse.ArgumentParser(description="Prepare Cats vs Dogs dataset")
    parser.add_argument("--source", default="training_set/training_set", 
                       help="Source directory path (default: training_set/training_set)")
    parser.add_argument("--target", default="data",
                       help="Target directory path (default: data)")
    parser.add_argument("--val-split", type=float, default=0.2,
                       help="Validation split ratio (default: 0.2)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducible splits (default: 42)")
    
    args = parser.parse_args()
    
    prepare_dataset(
        source_dir=args.source,
        target_dir=args.target, 
        val_split=args.val_split,
        seed=args.seed
    )

if __name__ == "__main__":
    main()