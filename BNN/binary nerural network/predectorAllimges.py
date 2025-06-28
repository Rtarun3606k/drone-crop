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
        (base_paths[2], Healthyhealthy_images, 'healthy', 1),  # Note: using index 2 for healthy
        (base_paths[1], Rust_array, 'rust', 2)  # Note: using index 1 for rust
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
        "/home/dragoon/coding/drone-crop/BNN/DataSet/Plant/Soyabean_Mosaic/",
        "/home/dragoon/coding/drone-crop/BNN/DataSet/Plant/rust/",
        "/home/dragoon/coding/drone-crop/BNN/DataSet/Plant/healthy/"
    ]
    
    class_names = ['Soyabean_Mosaic', 'healthy', 'rust']
    
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

def run_random_prediction_test():
    """
    Main function to run the random prediction test
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
    print("=" * 60)
    
    # Check if model file exists
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        return None
    
    # Run the test
    try:
        test_results = random_prediction_test(
            model_path=model_path,
            num_samples_per_class=1000,  # Test 500 random samples per class (increased for comprehensive testing)
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

if __name__ == "__main__":
    # Run the random prediction test
    results = run_random_prediction_test()