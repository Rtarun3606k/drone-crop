import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
import os
from datetime import datetime
import glob
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

from paths import Soyabean_Mosaic_array, Healthyhealthy_images, Rust_array

# Import the model classes from predictor.py
from predictor import BinaryNeuralNetwork, predict_plant_disease

def get_random_images_from_dataset(base_paths, image_arrays, num_samples_per_class=1000):
    """
    Get random images from each class
    
    Args:
        base_paths (list): List of base directory paths for each class
        image_arrays (list): List of image filename arrays for each class
        num_samples_per_class (int): Number of random samples per class
    
    Returns:
        list: List of dictionaries with image_path, true_class, true_class_name
    """
    
    class_names = ['Soyabean_Mosaic', 'healthy', 'rust']
    random_samples = []
    
    # Ensure we have matching arrays
    data_mapping = [
        (base_paths[0], Soyabean_Mosaic_array, 'Soyabean_Mosaic', 0),
        (base_paths[1], Rust_array, 'rust', 1),  # Note: using index 1 for rust
        (base_paths[2], Healthyhealthy_images, 'healthy', 2)  # Note: using index 2 for healthy
    ]
    
    for base_path, image_array, class_name, class_idx in data_mapping:
        # Get random samples from this class
        available_images = image_array[:min(len(image_array), 100)]  # Limit to first 100 if too many
        random_indices = random.sample(range(len(available_images)), 
                                     min(num_samples_per_class, len(available_images)))
        
        for idx in random_indices:
            image_filename = available_images[idx]
            full_path = os.path.join(base_path, image_filename)
            
            # Check if file exists
            if os.path.exists(full_path):
                random_samples.append({
                    'image_path': full_path,
                    'image_filename': image_filename,
                    'true_class_idx': class_idx,
                    'true_class_name': class_name
                })
    
    return random_samples

def random_prediction_test(model_path, num_samples_per_class=1000, save_dir='random_test_results'):
    """
    Perform random prediction testing and generate comprehensive analysis
    
    Args:
        model_path (str): Path to the trained model
        num_samples_per_class (int): Number of random samples per class to test
        save_dir (str): Directory to save results
    
    Returns:
        dict: Test results and statistics
    """
    
    # Create results directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    # Define paths and class names (update these to match your actual dataset structure)
    constant_paths = [
        "/home/dragoon/Downloads/MH-SoyaHealthVision An Indian UAV and Leaf Image Dataset for Integrated Crop Health Assessment/Soyabean_UAV-Based_Image_Dataset/Soyabean_Mosaic/",
        "/home/dragoon/Downloads/MH-SoyaHealthVision An Indian UAV and Leaf Image Dataset for Integrated Crop Health Assessment/Soyabean_UAV-Based_Image_Dataset/rust/",
        "/home/dragoon/Downloads/MH-SoyaHealthVision An Indian UAV and Leaf Image Dataset for Integrated Crop Health Assessment/Soyabean_UAV-Based_Image_Dataset/Healthy_Soyabean/"
    ]
    
    class_names = ['Soyabean_Mosaic', 'rust', 'healthy']
    
    print(f"Starting random prediction test with {num_samples_per_class} samples per class...")
    
    # Debug: Check paths and arrays
    print("Checking dataset paths and arrays...")
    for i, path in enumerate(constant_paths):
        print(f"  Path {i}: {path}")
        print(f"  Exists: {os.path.exists(path)}")
        if os.path.exists(path):
            actual_files = [f for f in os.listdir(path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            print(f"  Actual files found: {len(actual_files)}")
        print()
    
    print(f"Array lengths: Soyabean_Mosaic={len(Soyabean_Mosaic_array)}, Rust={len(Rust_array)}, Healthy={len(Healthyhealthy_images)}")
    
    # Get random samples
    random_samples = get_random_images_from_dataset(constant_paths, 
                                                   [Soyabean_Mosaic_array, Rust_array, Healthyhealthy_images], 
                                                   num_samples_per_class)
    
    print(f"Selected {len(random_samples)} random images for testing")
    
    # Perform predictions
    results = []
    correct_predictions = 0
    total_predictions = 0
    
    print("Making predictions...")
    for i, sample in enumerate(random_samples):
        try:
            # Make prediction
            prediction_result = predict_plant_disease(sample['image_path'], model_path)
            
            if 'error' not in prediction_result:
                # Check if prediction is correct
                is_correct = prediction_result['predicted_class_name'] == sample['true_class_name']
                if is_correct:
                    correct_predictions += 1
                
                # Store result
                result_entry = {
                    'sample_id': i + 1,
                    'image_path': sample['image_path'],
                    'image_filename': sample['image_filename'],
                    'true_class_idx': sample['true_class_idx'],
                    'true_class_name': sample['true_class_name'],
                    'predicted_class_idx': prediction_result['predicted_class_idx'],
                    'predicted_class_name': prediction_result['predicted_class_name'],
                    'is_correct': is_correct,
                    'confidence': prediction_result['confidence'],
                    'soyabean_mosaic_prob': prediction_result['probabilities'][0],
                    'healthy_prob': prediction_result['probabilities'][1],
                    'rust_prob': prediction_result['probabilities'][2],
                    'raw_logit_soyabean': prediction_result['raw_logits'][0],
                    'raw_logit_healthy': prediction_result['raw_logits'][1],
                    'raw_logit_rust': prediction_result['raw_logits'][2]
                }
                
                results.append(result_entry)
                total_predictions += 1
                
                # Progress update
                if (i + 1) % 10 == 0:
                    current_accuracy = (correct_predictions / total_predictions) * 100
                    print(f"Processed {i + 1}/{len(random_samples)} images. Current accuracy: {current_accuracy:.1f}%")
            
            else:
                print(f"Error predicting {sample['image_path']}: {prediction_result['error']}")
                
        except Exception as e:
            print(f"Exception predicting {sample['image_path']}: {str(e)}")
    
    # Calculate final statistics
    final_accuracy = (correct_predictions / total_predictions) * 100 if total_predictions > 0 else 0
    
    print(f"\nPrediction completed!")
    print(f"Total predictions: {total_predictions}")
    print(f"Correct predictions: {correct_predictions}")
    print(f"Final accuracy: {final_accuracy:.2f}%")
    
    # Create DataFrame and save to CSV
    results_df = pd.DataFrame(results)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"{save_dir}/random_prediction_results_{timestamp}.csv"
    results_df.to_csv(csv_filename, index=False)
    print(f"Results saved to: {csv_filename}")
    
    # Generate comprehensive analysis and visualizations
    generate_prediction_analysis(results_df, save_dir, timestamp, class_names)
    
    # Return summary statistics
    summary_stats = {
        'total_predictions': total_predictions,
        'correct_predictions': correct_predictions,
        'accuracy': final_accuracy,
        'results_dataframe': results_df,
        'csv_file': csv_filename
    }
    
    return summary_stats

def generate_prediction_analysis(results_df, save_dir, timestamp, class_names):
    """
    Generate comprehensive analysis graphs and statistics
    """
    
    print("Generating analysis graphs...")
    
    # Check if we have any results
    if len(results_df) == 0:
        print("No results to analyze!")
        return
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    try:
        # Create comprehensive figure
        fig = plt.figure(figsize=(20, 16))
        
        # 1. Overall Accuracy
        plt.subplot(3, 4, 1)
        accuracy = (results_df['is_correct'].sum() / len(results_df)) * 100
        plt.bar(['Correct', 'Incorrect'], 
                [results_df['is_correct'].sum(), len(results_df) - results_df['is_correct'].sum()],
                color=['green', 'red'], alpha=0.7)
        plt.title(f'Overall Accuracy: {accuracy:.1f}%', fontsize=14, fontweight='bold')
        plt.ylabel('Number of Predictions')
        for i, v in enumerate([results_df['is_correct'].sum(), len(results_df) - results_df['is_correct'].sum()]):
            plt.text(i, v + 0.5, str(v), ha='center', va='bottom', fontweight='bold')
        
        # 2. Confusion Matrix
        plt.subplot(3, 4, 2)
        cm = confusion_matrix(results_df['true_class_name'], results_df['predicted_class_name'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        # 3. Per-Class Accuracy
        plt.subplot(3, 4, 3)
        class_accuracy = results_df.groupby('true_class_name')['is_correct'].mean() * 100
        bars = plt.bar(class_accuracy.index, class_accuracy.values, 
                       color=['skyblue', 'lightgreen', 'lightcoral'], alpha=0.8)
        plt.title('Per-Class Accuracy', fontsize=14, fontweight='bold')
        plt.ylabel('Accuracy (%)')
        plt.xticks(rotation=45)
        for bar, acc in zip(bars, class_accuracy.values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                    f'{acc:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # 4. Confidence Distribution
        plt.subplot(3, 4, 4)
        plt.hist(results_df['confidence'], bins=20, alpha=0.7, color='purple', edgecolor='black')
        plt.title('Confidence Distribution', fontsize=14, fontweight='bold')
        plt.xlabel('Confidence Score')
        plt.ylabel('Frequency')
        plt.axvline(results_df['confidence'].mean(), color='red', linestyle='--', 
                    label=f'Mean: {results_df["confidence"].mean():.3f}')
        plt.legend()
        
        # 5. Correct vs Incorrect Confidence
        plt.subplot(3, 4, 5)
        correct_conf = results_df[results_df['is_correct'] == True]['confidence']
        incorrect_conf = results_df[results_df['is_correct'] == False]['confidence']
        
        if len(correct_conf) > 0:
            plt.hist(correct_conf, bins=15, alpha=0.7, label='Correct', color='green')
        if len(incorrect_conf) > 0:
            plt.hist(incorrect_conf, bins=15, alpha=0.7, label='Incorrect', color='red')
        plt.title('Confidence: Correct vs Incorrect', fontsize=14, fontweight='bold')
        plt.xlabel('Confidence Score')
        plt.ylabel('Frequency')
        plt.legend()
        
        # 6. Class Distribution in Test Set
        plt.subplot(3, 4, 6)
        class_counts = results_df['true_class_name'].value_counts()
        plt.pie(class_counts.values, labels=class_counts.index, autopct='%1.1f%%', 
                colors=['lightblue', 'lightgreen', 'lightcoral'])
        plt.title('Test Set Class Distribution', fontsize=14, fontweight='bold')
        
        # 7. Prediction Confidence by Class
        plt.subplot(3, 4, 7)
        sns.boxplot(data=results_df, x='true_class_name', y='confidence')
        plt.title('Confidence by True Class', fontsize=14, fontweight='bold')
        plt.xlabel('True Class')
        plt.ylabel('Confidence')
        plt.xticks(rotation=45)
        
        # 8. Error Analysis by Confidence Threshold
        plt.subplot(3, 4, 8)
        thresholds = np.arange(0.5, 1.0, 0.05)
        accuracies = []
        samples_kept = []
        
        for threshold in thresholds:
            high_conf_results = results_df[results_df['confidence'] >= threshold]
            if len(high_conf_results) > 0:
                acc = (high_conf_results['is_correct'].sum() / len(high_conf_results)) * 100
                accuracies.append(acc)
                samples_kept.append(len(high_conf_results))
            else:
                accuracies.append(0)
                samples_kept.append(0)
        
        plt.plot(thresholds, accuracies, 'b-', marker='o', linewidth=2, label='Accuracy')
        plt.title('Accuracy vs Confidence Threshold', fontsize=14, fontweight='bold')
        plt.xlabel('Confidence Threshold')
        plt.ylabel('Accuracy (%)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # 9. Samples Retained vs Threshold
        plt.subplot(3, 4, 9)
        plt.plot(thresholds, samples_kept, 'r-', marker='s', linewidth=2)
        plt.title('Samples Retained vs Threshold', fontsize=14, fontweight='bold')
        plt.xlabel('Confidence Threshold')
        plt.ylabel('Number of Samples')
        plt.grid(True, alpha=0.3)
        
        # 10. Probability Distribution for Each Class
        plt.subplot(3, 4, 10)
        prob_cols = ['soyabean_mosaic_prob', 'healthy_prob', 'rust_prob']
        for i, col in enumerate(prob_cols):
            if col in results_df.columns:
                plt.hist(results_df[col], bins=20, alpha=0.5, label=class_names[i])
        plt.title('Probability Distributions', fontsize=14, fontweight='bold')
        plt.xlabel('Probability')
        plt.ylabel('Frequency')
        plt.legend()
        
        # 11. Misclassification Analysis
        plt.subplot(3, 4, 11)
        misclassified = results_df[results_df['is_correct'] == False]
        if len(misclassified) > 0:
            misclass_counts = misclassified.groupby(['true_class_name', 'predicted_class_name']).size().unstack(fill_value=0)
            sns.heatmap(misclass_counts, annot=True, fmt='d', cmap='Reds')
            plt.title('Misclassification Patterns', fontsize=14, fontweight='bold')
            plt.ylabel('True Class')
            plt.xlabel('Predicted Class')
        else:
            plt.text(0.5, 0.5, 'No Misclassifications!', ha='center', va='center', 
                    transform=plt.gca().transAxes, fontsize=16, fontweight='bold')
            plt.title('Misclassification Patterns', fontsize=14, fontweight='bold')
        
        # 12. Performance Summary
        plt.subplot(3, 4, 12)
        metrics = ['Accuracy', 'Avg Confidence', 'High Conf Correct']
        high_conf_correct = len(results_df[(results_df['confidence'] > 0.8) & (results_df['is_correct'] == True)])
        values = [
            accuracy,
            results_df['confidence'].mean() * 100,
            (high_conf_correct / len(results_df)) * 100
        ]
        
        colors = ['blue', 'green', 'orange']
        bars = plt.bar(range(len(metrics)), values, color=colors, alpha=0.7)
        plt.title('Performance Summary (%)', fontsize=14, fontweight='bold')
        plt.xticks(range(len(metrics)), metrics, rotation=45)
        plt.ylabel('Percentage')
        
        # Add value labels
        for bar, value in zip(bars, values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                    f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # Add text annotation for total samples
        plt.text(0.02, 0.98, f'Total Samples: {len(results_df)}', 
                 transform=plt.gca().transAxes, fontsize=10, fontweight='bold',
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        
        # Save the comprehensive analysis
        analysis_filename = f"{save_dir}/prediction_analysis_{timestamp}.png"
        plt.savefig(analysis_filename, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Analysis graphs saved to: {analysis_filename}")
        
    except Exception as e:
        print(f"Error generating plots: {str(e)}")
        print("Continuing with text-based analysis...")
        accuracy = (results_df['is_correct'].sum() / len(results_df)) * 100
        print(f"Overall Accuracy: {accuracy:.2f}%")
    
    # Generate detailed classification report
    print("\nDetailed Classification Report:")
    print("=" * 60)
    
    try:
        # Get unique classes present in the data
        unique_true_classes = sorted(results_df['true_class_name'].unique())
        unique_pred_classes = sorted(results_df['predicted_class_name'].unique())
        all_unique_classes = sorted(list(set(unique_true_classes + unique_pred_classes)))
        
        print(f"Classes found in data: {all_unique_classes}")
        print(f"Expected classes: {class_names}")
        
        # Use labels parameter to specify all possible classes
        report = classification_report(results_df['true_class_name'], 
                                     results_df['predicted_class_name'], 
                                     labels=class_names,
                                     target_names=class_names,
                                     zero_division=0)
        print(report)
        
        # Save classification report to file
        report_filename = f"{save_dir}/classification_report_{timestamp}.txt"
        with open(report_filename, 'w') as f:
            f.write("Random Prediction Test - Classification Report\n")
            f.write("=" * 50 + "\n")
            f.write(f"Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Samples: {len(results_df)}\n")
            f.write(f"Overall Accuracy: {accuracy:.2f}%\n")
            f.write(f"Classes found in data: {all_unique_classes}\n")
            f.write(f"Expected classes: {class_names}\n\n")
            f.write(report)
        
        print(f"Classification report saved to: {report_filename}")
        
    except Exception as e:
        print(f"Error generating classification report: {str(e)}")
        print("Generating basic accuracy statistics instead...")
        
        # Basic accuracy by class
        print("\nBasic Per-Class Statistics:")
        for class_name in class_names:
            class_data = results_df[results_df['true_class_name'] == class_name]
            if len(class_data) > 0:
                class_accuracy = (class_data['is_correct'].sum() / len(class_data)) * 100
                print(f"  {class_name}: {len(class_data)} samples, {class_accuracy:.1f}% accuracy")
            else:
                print(f"  {class_name}: 0 samples in test set")
        
        # Save basic report
        report_filename = f"{save_dir}/classification_report_{timestamp}.txt"
        with open(report_filename, 'w') as f:
            f.write("Random Prediction Test - Basic Report\n")
            f.write("=" * 50 + "\n")
            f.write(f"Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Samples: {len(results_df)}\n")
            f.write(f"Overall Accuracy: {accuracy:.2f}%\n\n")
            f.write("Per-Class Statistics:\n")
            for class_name in class_names:
                class_data = results_df[results_df['true_class_name'] == class_name]
                if len(class_data) > 0:
                    class_accuracy = (class_data['is_correct'].sum() / len(class_data)) * 100
                    f.write(f"  {class_name}: {len(class_data)} samples, {class_accuracy:.1f}% accuracy\n")
                else:
                    f.write(f"  {class_name}: 0 samples in test set\n")
        
        print(f"Basic report saved to: {report_filename}")
    
    print(f"Classification report saved to: {report_filename}")

def test_all_images_comprehensive(model_path, save_dir='comprehensive_test_results'):
    """
    Test ALL images in the dataset for comprehensive evaluation
    
    Args:
        model_path (str): Path to the trained model
        save_dir (str): Directory to save results
    
    Returns:
        dict: Comprehensive test results and statistics
    """
    
    # Create results directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Define paths and class mappings
    base_paths = [
        "/home/dragoon/Downloads/MH-SoyaHealthVision An Indian UAV and Leaf Image Dataset for Integrated Crop Health Assessment/Soyabean_UAV-Based_Image_Dataset/Soyabean_Mosaic/",
        "/home/dragoon/Downloads/MH-SoyaHealthVision An Indian UAV and Leaf Image Dataset for Integrated Crop Health Assessment/Soyabean_UAV-Based_Image_Dataset/rust/",
        "/home/dragoon/Downloads/MH-SoyaHealthVision An Indian UAV and Leaf Image Dataset for Integrated Crop Health Assessment/Soyabean_UAV-Based_Image_Dataset/Healthy_Soyabean/"
    ]
    
    # Image arrays from paths.py
    image_arrays = [Soyabean_Mosaic_array, Rust_array, Healthyhealthy_images]
    class_names = ['Soyabean_Mosaic', 'rust', 'healthy']
    
    print("🎯 Starting COMPREHENSIVE testing of ALL images in dataset...")
    print("=" * 70)
    
    # Get ALL images from dataset
    all_samples = []
    class_counts = {}
    
    for i, (base_path, image_array, class_name) in enumerate(zip(base_paths, image_arrays, class_names)):
        class_samples = []
        
        print(f"Processing {class_name} class...")
        
        for image_filename in image_array:
            full_path = os.path.join(base_path, image_filename)
            
            # Check if file exists
            if os.path.exists(full_path):
                class_samples.append({
                    'image_path': full_path,
                    'image_filename': image_filename,
                    'true_class_idx': i,
                    'true_class_name': class_name
                })
        
        all_samples.extend(class_samples)
        class_counts[class_name] = len(class_samples)
        print(f"  Found {len(class_samples)} {class_name} images")
    
    total_images = len(all_samples)
    print(f"\n📊 Total images to test: {total_images}")
    print(f"Class distribution: {class_counts}")
    
    # Perform predictions on ALL images
    results = []
    correct_predictions = 0
    total_predictions = 0
    
    print("\n🔍 Making predictions on ALL images...")
    print("=" * 50)
    
    # Progress tracking
    progress_interval = max(1, total_images // 20)  # Update every 5%
    
    for i, sample in enumerate(all_samples):
        try:
            # Make prediction
            prediction_result = predict_plant_disease(sample['image_path'], model_path)
            
            if 'error' not in prediction_result:
                # Check if prediction is correct
                is_correct = prediction_result['predicted_class_name'] == sample['true_class_name']
                if is_correct:
                    correct_predictions += 1
                
                # Store comprehensive result
                result_entry = {
                    'sample_id': i + 1,
                    'image_path': sample['image_path'],
                    'image_filename': sample['image_filename'],
                    'true_class_idx': sample['true_class_idx'],
                    'true_class_name': sample['true_class_name'],
                    'predicted_class_idx': prediction_result['predicted_class_idx'],
                    'predicted_class_name': prediction_result['predicted_class_name'],
                    'is_correct': is_correct,
                    'confidence': prediction_result['confidence'],
                    'soyabean_mosaic_prob': prediction_result['probabilities'][0],
                    'rust_prob': prediction_result['probabilities'][1],
                    'healthy_prob': prediction_result['probabilities'][2],
                    'raw_logit_soyabean': prediction_result['raw_logits'][0],
                    'raw_logit_rust': prediction_result['raw_logits'][1],
                    'raw_logit_healthy': prediction_result['raw_logits'][2],
                    'max_prob': max(prediction_result['probabilities']),
                    'prediction_entropy': -sum([p * np.log(p + 1e-8) for p in prediction_result['probabilities']])
                }
                
                results.append(result_entry)
                total_predictions += 1
                
                # Progress update
                if (i + 1) % progress_interval == 0 or i == total_images - 1:
                    current_accuracy = (correct_predictions / total_predictions) * 100
                    progress = ((i + 1) / total_images) * 100
                    print(f"Progress: {progress:.1f}% ({i+1}/{total_images}) | Current accuracy: {current_accuracy:.2f}%")
            
            else:
                print(f"❌ Error predicting {sample['image_path']}: {prediction_result['error']}")
                
        except Exception as e:
            print(f"❌ Exception predicting {sample['image_path']}: {str(e)}")
    
    # Calculate comprehensive statistics
    final_accuracy = (correct_predictions / total_predictions) * 100 if total_predictions > 0 else 0
    
    print(f"\n✅ COMPREHENSIVE TESTING COMPLETED!")
    print(f"Total images processed: {total_predictions}")
    print(f"Correct predictions: {correct_predictions}")
    print(f"Overall accuracy: {final_accuracy:.2f}%")
    
    # Create comprehensive DataFrame
    results_df = pd.DataFrame(results)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save comprehensive CSV
    csv_filename = f"{save_dir}/comprehensive_all_images_results_{timestamp}.csv"
    results_df.to_csv(csv_filename, index=False)
    print(f"📄 Comprehensive results saved to: {csv_filename}")
    
    # Generate comprehensive analysis
    generate_comprehensive_analysis(results_df, save_dir, timestamp, class_names, class_counts)
    
    # Return comprehensive summary
    summary_stats = {
        'total_images_tested': total_predictions,
        'correct_predictions': correct_predictions,
        'overall_accuracy': final_accuracy,
        'class_counts': class_counts,
        'results_dataframe': results_df,
        'csv_file': csv_filename
    }
    
    return summary_stats

def generate_comprehensive_analysis(results_df, save_dir, timestamp, class_names, class_counts):
    """
    Generate comprehensive analysis for ALL images testing
    """
    
    print("\n📊 Generating comprehensive analysis graphs...")
    
    # Set up plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create massive comprehensive figure
    fig = plt.figure(figsize=(24, 20))
    
    # 1. Overall Dataset Statistics
    plt.subplot(4, 5, 1)
    plt.bar(class_counts.keys(), class_counts.values(), 
            color=['lightblue', 'lightcoral', 'lightgreen'], alpha=0.8)
    plt.title('Dataset Size by Class', fontsize=14, fontweight='bold')
    plt.ylabel('Number of Images')
    plt.xticks(rotation=45)
    for i, (k, v) in enumerate(class_counts.items()):
        plt.text(i, v + 10, str(v), ha='center', va='bottom', fontweight='bold')
    
    # 2. Overall Accuracy
    plt.subplot(4, 5, 2)
    accuracy = (results_df['is_correct'].sum() / len(results_df)) * 100
    correct_count = results_df['is_correct'].sum()
    incorrect_count = len(results_df) - correct_count
    
    plt.bar(['Correct', 'Incorrect'], [correct_count, incorrect_count],
            color=['green', 'red'], alpha=0.7)
    plt.title(f'Overall Accuracy: {accuracy:.1f}%', fontsize=14, fontweight='bold')
    plt.ylabel('Number of Predictions')
    for i, v in enumerate([correct_count, incorrect_count]):
        plt.text(i, v + max(correct_count, incorrect_count) * 0.01, str(v), 
                ha='center', va='bottom', fontweight='bold')
    
    # 3. Confusion Matrix
    plt.subplot(4, 5, 3)
    cm = confusion_matrix(results_df['true_class_name'], results_df['predicted_class_name'], 
                         labels=class_names)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    # 4. Per-Class Accuracy
    plt.subplot(4, 5, 4)
    class_accuracy = results_df.groupby('true_class_name')['is_correct'].mean() * 100
    bars = plt.bar(class_accuracy.index, class_accuracy.values, 
                   color=['skyblue', 'lightcoral', 'lightgreen'], alpha=0.8)
    plt.title('Per-Class Accuracy', fontsize=14, fontweight='bold')
    plt.ylabel('Accuracy (%)')
    plt.xticks(rotation=45)
    for bar, acc in zip(bars, class_accuracy.values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                f'{acc:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # 5. Confidence Distribution
    plt.subplot(4, 5, 5)
    plt.hist(results_df['confidence'], bins=30, alpha=0.7, color='purple', edgecolor='black')
    plt.title('Confidence Distribution (All Images)', fontsize=14, fontweight='bold')
    plt.xlabel('Confidence Score')
    plt.ylabel('Frequency')
    plt.axvline(results_df['confidence'].mean(), color='red', linestyle='--', 
                label=f'Mean: {results_df["confidence"].mean():.3f}')
    plt.legend()
    
    # 6. Confidence by Class
    plt.subplot(4, 5, 6)
    sns.boxplot(data=results_df, x='true_class_name', y='confidence')
    plt.title('Confidence by True Class', fontsize=14, fontweight='bold')
    plt.xlabel('True Class')
    plt.ylabel('Confidence')
    plt.xticks(rotation=45)
    
    # 7. Correct vs Incorrect Confidence
    plt.subplot(4, 5, 7)
    correct_conf = results_df[results_df['is_correct'] == True]['confidence']
    incorrect_conf = results_df[results_df['is_correct'] == False]['confidence']
    
    plt.hist(correct_conf, bins=20, alpha=0.7, label='Correct', color='green')
    if len(incorrect_conf) > 0:
        plt.hist(incorrect_conf, bins=20, alpha=0.7, label='Incorrect', color='red')
    plt.title('Confidence: Correct vs Incorrect', fontsize=14, fontweight='bold')
    plt.xlabel('Confidence Score')
    plt.ylabel('Frequency')
    plt.legend()
    
    # 8. Prediction Entropy Distribution
    plt.subplot(4, 5, 8)
    plt.hist(results_df['prediction_entropy'], bins=25, alpha=0.7, color='orange', edgecolor='black')
    plt.title('Prediction Entropy Distribution', fontsize=14, fontweight='bold')
    plt.xlabel('Entropy')
    plt.ylabel('Frequency')
    plt.axvline(results_df['prediction_entropy'].mean(), color='red', linestyle='--',
                label=f'Mean: {results_df["prediction_entropy"].mean():.3f}')
    plt.legend()
    
    # 9. Accuracy vs Confidence Threshold
    plt.subplot(4, 5, 9)
    thresholds = np.arange(0.3, 1.0, 0.05)
    accuracies = []
    samples_kept = []
    
    for threshold in thresholds:
        high_conf_results = results_df[results_df['confidence'] >= threshold]
        if len(high_conf_results) > 0:
            acc = (high_conf_results['is_correct'].sum() / len(high_conf_results)) * 100
            accuracies.append(acc)
            samples_kept.append(len(high_conf_results))
        else:
            accuracies.append(0)
            samples_kept.append(0)
    
    plt.plot(thresholds, accuracies, 'b-', marker='o', linewidth=2, label='Accuracy')
    plt.title('Accuracy vs Confidence Threshold', fontsize=14, fontweight='bold')
    plt.xlabel('Confidence Threshold')
    plt.ylabel('Accuracy (%)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # 10. Samples Retained vs Threshold
    plt.subplot(4, 5, 10)
    plt.plot(thresholds, samples_kept, 'r-', marker='s', linewidth=2)
    plt.title('Samples Retained vs Threshold', fontsize=14, fontweight='bold')
    plt.xlabel('Confidence Threshold')
    plt.ylabel('Number of Samples')
    plt.grid(True, alpha=0.3)
    
    # 11. Per-Class Precision, Recall, F1-Score
    plt.subplot(4, 5, 11)
    from sklearn.metrics import precision_recall_fscore_support
    
    precision, recall, fscore, support = precision_recall_fscore_support(
        results_df['true_class_name'], results_df['predicted_class_name'], 
        labels=class_names, average=None, zero_division=0
    )
    
    x = np.arange(len(class_names))
    width = 0.25
    
    plt.bar(x - width, precision, width, label='Precision', alpha=0.8)
    plt.bar(x, recall, width, label='Recall', alpha=0.8)
    plt.bar(x + width, fscore, width, label='F1-Score', alpha=0.8)
    
    plt.title('Per-Class Metrics', fontsize=14, fontweight='bold')
    plt.xlabel('Class')
    plt.ylabel('Score')
    plt.xticks(x, class_names, rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 12. Misclassification Heatmap
    plt.subplot(4, 5, 12)
    misclassified = results_df[results_df['is_correct'] == False]
    if len(misclassified) > 0:
        misclass_counts = misclassified.groupby(['true_class_name', 'predicted_class_name']).size().unstack(fill_value=0)
        sns.heatmap(misclass_counts, annot=True, fmt='d', cmap='Reds', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Misclassification Patterns', fontsize=14, fontweight='bold')
        plt.ylabel('True Class')
        plt.xlabel('Predicted Class')
    else:
        plt.text(0.5, 0.5, 'No Misclassifications!', ha='center', va='center', 
                transform=plt.gca().transAxes, fontsize=16, fontweight='bold', color='green')
        plt.title('Misclassification Patterns', fontsize=14, fontweight='bold')
    
    # 13. Top Confidence Predictions by Class
    plt.subplot(4, 5, 13)
    top_conf_by_class = []
    for class_name in class_names:
        class_data = results_df[results_df['true_class_name'] == class_name]
        avg_conf = class_data['confidence'].mean()
        top_conf_by_class.append(avg_conf)
    
    bars = plt.bar(class_names, top_conf_by_class, 
                   color=['skyblue', 'lightcoral', 'lightgreen'], alpha=0.8)
    plt.title('Average Confidence by Class', fontsize=14, fontweight='bold')
    plt.ylabel('Average Confidence')
    plt.xticks(rotation=45)
    for bar, conf in zip(bars, top_conf_by_class):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{conf:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # 14. Error Rate by Class
    plt.subplot(4, 5, 14)
    error_rates = []
    for class_name in class_names:
        class_data = results_df[results_df['true_class_name'] == class_name]
        error_rate = (1 - class_data['is_correct'].mean()) * 100
        error_rates.append(error_rate)
    
    bars = plt.bar(class_names, error_rates, 
                   color=['salmon', 'orange', 'yellow'], alpha=0.8)
    plt.title('Error Rate by Class', fontsize=14, fontweight='bold')
    plt.ylabel('Error Rate (%)')
    plt.xticks(rotation=45)
    for bar, err in zip(bars, error_rates):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                f'{err:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # 15. Model Performance Summary
    plt.subplot(4, 5, 15)
    summary_metrics = [
        accuracy,
        np.mean(precision) * 100,
        np.mean(recall) * 100,
        np.mean(fscore) * 100
    ]
    metric_names = ['Accuracy', 'Avg Precision', 'Avg Recall', 'Avg F1-Score']
    
    bars = plt.bar(metric_names, summary_metrics, 
                   color=['blue', 'green', 'orange', 'red'], alpha=0.7)
    plt.title('Overall Performance Summary', fontsize=14, fontweight='bold')
    plt.ylabel('Score (%)')
    plt.xticks(rotation=45)
    plt.ylim(0, 100)
    
    for bar, value in zip(bars, summary_metrics):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # 16. Sample Distribution Analysis
    plt.subplot(4, 5, 16)
    plt.pie(class_counts.values(), labels=class_counts.keys(), autopct='%1.1f%%', 
            colors=['lightblue', 'lightcoral', 'lightgreen'])
    plt.title('Dataset Distribution', fontsize=14, fontweight='bold')
    
    # 17. High/Low Confidence Analysis
    plt.subplot(4, 5, 17)
    high_conf = results_df[results_df['confidence'] > 0.8]
    low_conf = results_df[results_df['confidence'] < 0.6]
    medium_conf = results_df[(results_df['confidence'] >= 0.6) & (results_df['confidence'] <= 0.8)]
    
    conf_categories = ['High (>0.8)', 'Medium (0.6-0.8)', 'Low (<0.6)']
    conf_counts = [len(high_conf), len(medium_conf), len(low_conf)]
    
    plt.bar(conf_categories, conf_counts, 
            color=['green', 'yellow', 'red'], alpha=0.7)
    plt.title('Confidence Categories', fontsize=14, fontweight='bold')
    plt.ylabel('Number of Predictions')
    plt.xticks(rotation=45)
    for i, v in enumerate(conf_counts):
        plt.text(i, v + max(conf_counts) * 0.01, str(v), 
                ha='center', va='bottom', fontweight='bold')
    
    # 18. Probability Distribution Analysis
    plt.subplot(4, 5, 18)
    prob_cols = ['soyabean_mosaic_prob', 'rust_prob', 'healthy_prob']
    for i, col in enumerate(prob_cols):
        plt.hist(results_df[col], bins=20, alpha=0.5, label=class_names[i])
    plt.title('Class Probability Distributions', fontsize=14, fontweight='bold')
    plt.xlabel('Probability')
    plt.ylabel('Frequency')
    plt.legend()
    
    # 19. Entropy vs Accuracy
    plt.subplot(4, 5, 19)
    plt.scatter(results_df['prediction_entropy'], results_df['confidence'], 
               c=results_df['is_correct'], cmap='RdYlGn', alpha=0.6)
    plt.colorbar(label='Correct Prediction')
    plt.title('Entropy vs Confidence', fontsize=14, fontweight='bold')
    plt.xlabel('Prediction Entropy')
    plt.ylabel('Confidence')
    
    # 20. Final Statistics Summary
    plt.subplot(4, 5, 20)
    plt.text(0.1, 0.9, 'COMPREHENSIVE TEST SUMMARY', fontsize=16, fontweight='bold',
             transform=plt.gca().transAxes)
    plt.text(0.1, 0.8, f'Total Images: {len(results_df):,}', fontsize=12,
             transform=plt.gca().transAxes)
    plt.text(0.1, 0.7, f'Overall Accuracy: {accuracy:.2f}%', fontsize=12,
             transform=plt.gca().transAxes)
    plt.text(0.1, 0.6, f'Avg Confidence: {results_df["confidence"].mean():.3f}', fontsize=12,
             transform=plt.gca().transAxes)
    plt.text(0.1, 0.5, f'Std Confidence: {results_df["confidence"].std():.3f}', fontsize=12,
             transform=plt.gca().transAxes)
    plt.text(0.1, 0.4, f'High Conf (>0.8): {len(high_conf)} ({len(high_conf)/len(results_df)*100:.1f}%)', fontsize=10,
             transform=plt.gca().transAxes)
    plt.text(0.1, 0.3, f'Perfect Score: {(accuracy == 100.0)}', fontsize=12,
             transform=plt.gca().transAxes)
    plt.axis('off')
    
    plt.tight_layout()
    
    # Save comprehensive analysis
    analysis_filename = f"{save_dir}/comprehensive_analysis_{timestamp}.png"
    plt.savefig(analysis_filename, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"📊 Comprehensive analysis saved to: {analysis_filename}")
    
    # Generate and save detailed report
    save_comprehensive_report(results_df, save_dir, timestamp, class_names, class_counts)

def save_comprehensive_report(results_df, save_dir, timestamp, class_names, class_counts):
    """
    Save comprehensive text report
    """
    
    report_filename = f"{save_dir}/comprehensive_report_{timestamp}.txt"
    
    with open(report_filename, 'w') as f:
        f.write("COMPREHENSIVE BINARY NEURAL NETWORK EVALUATION REPORT\n")
        f.write("=" * 60 + "\n")
        f.write(f"Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total Images Tested: {len(results_df):,}\n\n")
        
        f.write("DATASET COMPOSITION:\n")
        f.write("-" * 30 + "\n")
        total_dataset = sum(class_counts.values())
        for class_name, count in class_counts.items():
            percentage = (count / total_dataset) * 100
            f.write(f"{class_name}: {count:,} images ({percentage:.1f}%)\n")
        
        f.write(f"\nOVERALL PERFORMANCE:\n")
        f.write("-" * 30 + "\n")
        accuracy = (results_df['is_correct'].sum() / len(results_df)) * 100
        f.write(f"Overall Accuracy: {accuracy:.2f}%\n")
        f.write(f"Correct Predictions: {results_df['is_correct'].sum():,}\n")
        f.write(f"Incorrect Predictions: {len(results_df) - results_df['is_correct'].sum():,}\n")
        f.write(f"Average Confidence: {results_df['confidence'].mean():.4f}\n")
        f.write(f"Confidence Std Dev: {results_df['confidence'].std():.4f}\n")
        
        f.write(f"\nPER-CLASS PERFORMANCE:\n")
        f.write("-" * 30 + "\n")
        for class_name in class_names:
            class_data = results_df[results_df['true_class_name'] == class_name]
            if len(class_data) > 0:
                class_accuracy = (class_data['is_correct'].sum() / len(class_data)) * 100
                avg_conf = class_data['confidence'].mean()
                f.write(f"{class_name}:\n")
                f.write(f"  Samples: {len(class_data):,}\n")
                f.write(f"  Accuracy: {class_accuracy:.2f}%\n")
                f.write(f"  Avg Confidence: {avg_conf:.4f}\n")
                f.write(f"  Correct: {class_data['is_correct'].sum():,}\n")
                f.write(f"  Incorrect: {len(class_data) - class_data['is_correct'].sum():,}\n\n")
        
        # Add confusion matrix
        f.write("CONFUSION MATRIX:\n")
        f.write("-" * 30 + "\n")
        cm = confusion_matrix(results_df['true_class_name'], results_df['predicted_class_name'], 
                             labels=class_names)
        f.write("Predicted ->  " + "  ".join([f"{name:>12}" for name in class_names]) + "\n")
        for i, true_class in enumerate(class_names):
            f.write(f"{true_class:>12}  " + "  ".join([f"{cm[i][j]:>12}" for j in range(len(class_names))]) + "\n")
        
        f.write(f"\nCONFIDENCE ANALYSIS:\n")
        f.write("-" * 30 + "\n")
        high_conf = results_df[results_df['confidence'] > 0.8]
        medium_conf = results_df[(results_df['confidence'] >= 0.6) & (results_df['confidence'] <= 0.8)]
        low_conf = results_df[results_df['confidence'] < 0.6]
        
        f.write(f"High Confidence (>0.8): {len(high_conf):,} ({len(high_conf)/len(results_df)*100:.1f}%)\n")
        f.write(f"Medium Confidence (0.6-0.8): {len(medium_conf):,} ({len(medium_conf)/len(results_df)*100:.1f}%)\n")
        f.write(f"Low Confidence (<0.6): {len(low_conf):,} ({len(low_conf)/len(results_df)*100:.1f}%)\n")
        
        if len(high_conf) > 0:
            high_conf_acc = (high_conf['is_correct'].sum() / len(high_conf)) * 100
            f.write(f"High Confidence Accuracy: {high_conf_acc:.2f}%\n")
    
    print(f"📄 Comprehensive report saved to: {report_filename}")

# ========================================================================================
# MAIN PREDICTION TEST FUNCTIONS
# ========================================================================================

def run_random_prediction_test(sample_size=1000):
    """
    Main function to run the random prediction test
    
    Args:
        sample_size (int): Number of random samples per class to test (default: 1000)
    """
    
    # Check for available model files
    model_files = glob.glob("results/bnn_plant_disease_model_*.pth")
    if not model_files:
        print("❌ No trained model files found in results/ directory!")
        print("Please run the training notebook first to generate a model.")
        return None
    
    # Use the most recent model file
    model_path = max(model_files, key=os.path.getctime)
    print(f"Using model: {model_path}")
    
    print("🚀 Starting Random Prediction Test for Binary Neural Network")
    print(f"📊 Sample size: {sample_size} images per class")
    print("=" * 60)
    
    # Check if model file exists
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        return None
    
    # Run the test
    try:
        test_results = random_prediction_test(
            model_path=model_path,
            num_samples_per_class=sample_size,  # Adjustable sample size for random testing
            save_dir='random_test_results'
        )
        
        print("\n" + "=" * 60)
        print("📊 RANDOM PREDICTION TEST SUMMARY")
        print("=" * 60)
        print(f"Total Predictions: {test_results['total_predictions']}")
        print(f"Correct Predictions: {test_results['correct_predictions']}")
        print(f"Accuracy: {test_results['accuracy']:.2f}%")
        print(f"Results saved to: {test_results['csv_file']}")
        print("=" * 60)
        
        return test_results
        
    except Exception as e:
        print(f"❌ Error running prediction test: {str(e)}")
        return None

def run_comprehensive_test():
    """
    Run comprehensive test on ALL images in the dataset
    """
    
    # Check for available model files
    model_files = glob.glob("results/bnn_plant_disease_model_*.pth")
    if not model_files:
        print("❌ No trained model files found in results/ directory!")
        print("Please run the training notebook first to generate a model.")
        return None
    
    # Use the most recent model file
    model_path = max(model_files, key=os.path.getctime)
    print(f"Using model: {model_path}")
    
    print("🎯 Starting COMPREHENSIVE test on ALL dataset images...")
    print("=" * 70)
    
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        return None
    
    try:
        # Run comprehensive test on ALL images
        comprehensive_results = test_all_images_comprehensive(
            model_path=model_path,
            save_dir='comprehensive_test_results'
        )
        
        print("\n" + "=" * 70)
        print("🏆 COMPREHENSIVE TEST SUMMARY")
        print("=" * 70)
        print(f"Total Images Tested: {comprehensive_results['total_images_tested']:,}")
        print(f"Correct Predictions: {comprehensive_results['correct_predictions']:,}")
        print(f"Overall Accuracy: {comprehensive_results['overall_accuracy']:.2f}%")
        print("\nClass Distribution:")
        for class_name, count in comprehensive_results['class_counts'].items():
            print(f"  {class_name}: {count:,} images")
        print(f"\nResults saved to: {comprehensive_results['csv_file']}")
        print("=" * 70)
        
        return comprehensive_results
        
    except Exception as e:
        print(f"❌ Error during comprehensive testing: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

# ========================================================================================
# CONVENIENCE FUNCTIONS FOR EASY TESTING
# ========================================================================================

def quick_test(sample_size=100):
    """Quick test with small sample size for fast results"""
    print(f"🚀 Quick Test - {sample_size} samples per class")
    return run_random_prediction_test(sample_size=sample_size)

def standard_test(sample_size=1000):
    """Standard test with default sample size"""
    print(f"🚀 Standard Test - {sample_size} samples per class")
    return run_random_prediction_test(sample_size=sample_size)

def large_test(sample_size=2000):
    """Large test with high sample size for thorough evaluation"""
    print(f"🚀 Large Test - {sample_size} samples per class")
    return run_random_prediction_test(sample_size=sample_size)

def complete_test():
    """Test ALL images in the dataset"""
    print("🚀 Complete Test - ALL images in dataset")
    return run_comprehensive_test()

def custom_test(sample_size):
    """Custom test with user-specified sample size"""
    print(f"🚀 Custom Test - {sample_size} samples per class")
    return run_random_prediction_test(sample_size=sample_size)

# ========================================================================================
# MAIN EXECUTION SECTION
# ========================================================================================

def main():
    """
    Main function to run prediction tests
    
    Options:
    1. Random Test: Test a specified number of random samples per class
    2. Comprehensive Test: Test ALL images in the dataset
    """
    
    print("🤖 Binary Neural Network - Plant Disease Prediction Testing")
    print("=" * 70)
    print("Select test type:")
    print("1. Random Test (adjustable sample size)")
    print("2. Comprehensive Test (ALL images)")
    print("3. Quick Random Test (100 samples per class)")
    print("4. Large Random Test (2000 samples per class)")
    print("=" * 70)
    
    choice = input("Enter your choice (1-4) or press Enter for default random test (1000 samples): ").strip()
    
    if choice == "1" or choice == "":
        # Random test with custom sample size
        if choice == "1":
            try:
                sample_size = int(input("Enter sample size per class (default 1000): ") or "1000")
            except ValueError:
                sample_size = 1000
                print("Invalid input, using default sample size: 1000")
        else:
            sample_size = 1000
        
        print(f"\n🎯 Starting Random Test with {sample_size} samples per class...")
        result = run_random_prediction_test(sample_size=sample_size)
        
    elif choice == "2":
        print("\n🎯 Starting Comprehensive Test on ALL images...")
        result = run_comprehensive_test()
        
    elif choice == "3":
        print("\n🎯 Starting Quick Random Test with 100 samples per class...")
        result = run_random_prediction_test(sample_size=100)
        
    elif choice == "4":
        print("\n🎯 Starting Large Random Test with 2000 samples per class...")
        result = run_random_prediction_test(sample_size=2000)
        
    else:
        print("❌ Invalid choice. Running default random test...")
        result = run_random_prediction_test(sample_size=1000)
    
    if result:
        print("\n✅ Test completed successfully!")
    else:
        print("\n❌ Test failed!")

if __name__ == "__main__":
    main()