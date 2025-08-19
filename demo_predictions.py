#!/usr/bin/env python3
"""
Demo: Test Predictions on Cats vs Dogs Model
============================================

This script demonstrates how to use the trained model for inference.
It loads the trained model and makes predictions on test images.
"""

import sys
from pathlib import Path

# Import our modern training module
from modern_cats_dogs_pytorch import predict_image

def main():
    """Demo predictions with the trained model."""
    
    print("🐱🐶 Cats vs Dogs Model - Prediction Demo")
    print("=" * 50)
    
    model_path = 'best_cats_dogs_model.pth'
    
    # Check if model exists
    if not Path(model_path).exists():
        print(f"❌ Model not found: {model_path}")
        print("🔧 Please train the model first:")
        print("   python modern_cats_dogs_pytorch.py")
        return
    
    print(f"✅ Found trained model: {model_path}")
    
    # Example: Test on validation images if available
    val_cats_dir = Path('data/validation/cats')
    val_dogs_dir = Path('data/validation/dogs')
    
    if val_cats_dir.exists() and val_dogs_dir.exists():
        print("\n🧪 Testing on validation images...")
        
        # Test on a few cat images
        cat_images = list(val_cats_dir.glob('*.jpg'))[:3]
        print(f"\n🐱 Testing {len(cat_images)} cat images:")
        
        for img_path in cat_images:
            print(f"\n📷 Testing: {img_path.name}")
            pred_class, confidence = predict_image(model_path, img_path, show_image=False)
            result = "✅ CORRECT" if pred_class == 0 else "❌ WRONG"
            print(f"   {result} (Expected: CAT)")
        
        # Test on a few dog images
        dog_images = list(val_dogs_dir.glob('*.jpg'))[:3]
        print(f"\n🐶 Testing {len(dog_images)} dog images:")
        
        for img_path in dog_images:
            print(f"\n📷 Testing: {img_path.name}")
            pred_class, confidence = predict_image(model_path, img_path, show_image=False)
            result = "✅ CORRECT" if pred_class == 1 else "❌ WRONG"
            print(f"   {result} (Expected: DOG)")
    
    else:
        print("\n⚠️  Validation images not found.")
        print("🔧 Please prepare the dataset first:")
        print("   python prepare_dataset.py")
    
    print("\n💡 To test on your own images:")
    print("   predict_image('best_cats_dogs_model.pth', 'path/to/your/image.jpg')")
    
    print("\n🚀 Model ready for production use!")

if __name__ == "__main__":
    main()