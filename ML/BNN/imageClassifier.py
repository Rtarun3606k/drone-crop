#!/usr/bin/env python3
"""
Image Classifier using trained Inception-TL model
Usage: python classify_images.py --image path/to/image.jpg
       python classify_images.py --folder path/to/folder --max_images 10
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import tensorflow as tf

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')

class ImageClassifier:
    def __init__(self, model_path='Inception-TL.h5', class_names=None):
        """
        Initialize the image classifier
        
        Args:
            model_path: Path to the trained model file
            class_names: List of class names (default: ['bikes', 'autorickshaw'])
        """
        self.model_path = model_path
        self.class_names = class_names or ['bikes', 'autorickshaw']
        self.model = None
        self.load_model()
    
    def load_model(self):
        """Load the trained model"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        print(f"Loading model from {self.model_path}...")
        self.model = load_model(self.model_path)
        print("✅ Model loaded successfully!")
        print(f"🏷️  Classes: {self.class_names}")
    
    def preprocess_image(self, image_path):
        """
        Preprocess image for prediction
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Preprocessed image array
        """
        img = load_img(image_path, target_size=(256, 256))
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0  # Normalize to [0,1]
        return img_array, img
    
    def classify_single_image(self, image_path, show_image=True, save_result=False):
        """
        Classify a single image
        
        Args:
            image_path: Path to the image file
            show_image: Whether to display the image
            save_result: Whether to save the result plot
            
        Returns:
            Dictionary with prediction results
        """
        try:
            # Preprocess image
            img_array, original_img = self.preprocess_image(image_path)
            
            # Make prediction
            prediction = self.model.predict(img_array, verbose=0)[0][0]
            
            # Convert to class and confidence
            if prediction > 0.5:
                predicted_class = self.class_names[1]
                confidence = prediction * 100
            else:
                predicted_class = self.class_names[0]
                confidence = (1 - prediction) * 100
            
            result = {
                'image_path': image_path,
                'predicted_class': predicted_class,
                'confidence': confidence,
                'raw_score': prediction
            }
            
            # Display results
            print(f"🖼️  Image: {os.path.basename(image_path)}")
            print(f"🎯 Prediction: {predicted_class}")
            print(f"📊 Confidence: {confidence:.1f}%")
            print(f"🔢 Raw score: {prediction:.4f}")
            
            # Confidence interpretation
            if confidence > 90:
                print("   🟢 Very confident prediction")
            elif confidence > 70:
                print("   🟡 Moderately confident prediction")  
            else:
                print("   🔴 Low confidence prediction")
            
            # Show/save image
            if show_image or save_result:
                plt.figure(figsize=(8, 6))
                plt.imshow(original_img)
                plt.title(f'Prediction: {predicted_class}\nConfidence: {confidence:.1f}%', 
                         fontsize=14, fontweight='bold')
                plt.axis('off')
                
                if save_result:
                    output_path = f"prediction_{os.path.basename(image_path)}.png"
                    plt.savefig(output_path, bbox_inches='tight', dpi=150)
                    print(f"💾 Result saved to: {output_path}")
                
                if show_image:
                    plt.show()
                else:
                    plt.close()
            
            return result
            
        except Exception as e:
            print(f"❌ Error processing {image_path}: {e}")
            return None
    
    def classify_folder(self, folder_path, max_images=10, show_images=True, save_summary=False):
        """
        Classify all images in a folder
        
        Args:
            folder_path: Path to folder containing images
            max_images: Maximum number of images to process
            show_images: Whether to display images
            save_summary: Whether to save classification summary
            
        Returns:
            List of prediction results
        """
        if not os.path.exists(folder_path):
            print(f"❌ Folder not found: {folder_path}")
            return []
        
        # Get image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
        image_files = []
        
        for file in os.listdir(folder_path):
            if any(file.lower().endswith(ext) for ext in image_extensions):
                image_files.append(file)
        
        if not image_files:
            print(f"❌ No image files found in {folder_path}")
            return []
        
        # Limit number of images
        image_files = sorted(image_files)[:max_images]
        
        print(f"🖼️  Processing {len(image_files)} images from {folder_path}")
        print("="*60)
        
        # Process images
        results = []
        
        if show_images:
            cols = 5
            rows = (len(image_files) + cols - 1) // cols
            plt.figure(figsize=(4*cols, 3*rows))
        
        for i, filename in enumerate(image_files):
            image_path = os.path.join(folder_path, filename)
            result = self.classify_single_image(image_path, show_image=False)
            
            if result:
                results.append(result)
                
                if show_images:
                    img = load_img(image_path, target_size=(256, 256))
                    plt.subplot(rows, cols, i + 1)
                    plt.imshow(img)
                    plt.title(f'{filename}\n{result["predicted_class"]}\n{result["confidence"]:.1f}%', 
                             fontsize=10)
                    plt.axis('off')
            
            print("-" * 60)
        
        if show_images:
            plt.tight_layout()
            plt.show()
        
        # Print summary
        print("\n" + "="*60)
        print("CLASSIFICATION SUMMARY")
        print("="*60)
        
        class_counts = {}
        for result in results:
            class_name = result['predicted_class']
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        print(f"📊 Total images processed: {len(results)}")
        for class_name, count in class_counts.items():
            percentage = (count / len(results)) * 100
            print(f"   {class_name}: {count} images ({percentage:.1f}%)")
        
        # Save summary if requested
        if save_summary:
            summary_path = f"classification_summary_{os.path.basename(folder_path)}.txt"
            with open(summary_path, 'w') as f:
                f.write("IMAGE CLASSIFICATION SUMMARY\n")
                f.write("="*50 + "\n\n")
                f.write(f"Folder: {folder_path}\n")
                f.write(f"Total images: {len(results)}\n\n")
                
                for result in results:
                    f.write(f"{result['image_path']:<50} → {result['predicted_class']:<15} ({result['confidence']:.1f}%)\n")
            
            print(f"💾 Summary saved to: {summary_path}")
        
        return results

def main():
    parser = argparse.ArgumentParser(description='Classify images using trained Inception-TL model')
    parser.add_argument('--model', default='Inception-TL.h5', help='Path to model file')
    parser.add_argument('--image', help='Path to single image to classify')
    parser.add_argument('--folder', help='Path to folder containing images')
    parser.add_argument('--max_images', type=int, default=10, help='Maximum images to process from folder')
    parser.add_argument('--classes', nargs='+', default=['bikes', 'autorickshaw'], help='Class names')
    parser.add_argument('--no_display', action='store_true', help='Don\'t display images')
    parser.add_argument('--save_results', action='store_true', help='Save prediction results')
    parser.add_argument('--save_summary', action='store_true', help='Save classification summary')
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.image and not args.folder:
        print("❌ Please provide either --image or --folder argument")
        parser.print_help()
        return
    
    try:
        # Initialize classifier
        classifier = ImageClassifier(args.model, args.classes)
        
        if args.image:
            # Classify single image
            print("="*60)
            print("SINGLE IMAGE CLASSIFICATION")
            print("="*60)
            classifier.classify_single_image(
                args.image, 
                show_image=not args.no_display,
                save_result=args.save_results
            )
        
        elif args.folder:
            # Classify folder
            print("="*60)
            print("FOLDER CLASSIFICATION")
            print("="*60)
            classifier.classify_folder(
                args.folder, 
                max_images=args.max_images,
                show_images=not args.no_display,
                save_summary=args.save_summary
            )
    
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())