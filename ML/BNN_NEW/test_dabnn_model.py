#!/usr/bin/env python3
"""
Test script for the trained DABNN model
Loads the best saved model and evaluates it on test data
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import time
import os
from pathlib import Path
import argparse
import warnings
warnings.filterwarnings('ignore')

# Import model classes (same as training script)
class BinaryActivation(torch.autograd.Function):
    """Binary activation function for BNN"""
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return torch.sign(input)
    
    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input.le(-1)] = 0
        grad_input[input.ge(1)] = 0
        return grad_input

class BinaryLinear(nn.Module):
    """Binary linear layer"""
    def __init__(self, in_features, out_features, bias=True):
        super(BinaryLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.1)
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
    
    def forward(self, input):
        binary_weight = torch.sign(self.weight)
        output = F.linear(input, binary_weight, self.bias)
        return output

class BinaryConv2d(nn.Module):
    """Binary convolution layer"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super(BinaryConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size) * 0.1)
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('bias', None)
    
    def forward(self, input):
        binary_weight = torch.sign(self.weight)
        output = F.conv2d(input, binary_weight, self.bias, self.stride, self.padding)
        return output

class ChannelAttention(nn.Module):
    """Channel attention mechanism from the paper"""
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.PReLU(),
            nn.Linear(in_channels // reduction, in_channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        b, c, _, _ = x.size()
        
        # Channel attention
        avg_out = self.fc(self.avg_pool(x).view(b, c))
        max_out = self.fc(self.max_pool(x).view(b, c))
        out = avg_out + max_out
        
        return self.sigmoid(out).view(b, c, 1, 1) * x

class SpatialAttention(nn.Module):
    """Spatial attention mechanism from the paper"""
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        
        out = self.conv(x_cat)
        return self.sigmoid(out) * x

class DualAttention(nn.Module):
    """Dual attention combining channel and spatial attention"""
    def __init__(self, in_channels):
        super(DualAttention, self).__init__()
        self.channel_attention = ChannelAttention(in_channels)
        self.spatial_attention = SpatialAttention()
    
    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x

class DABConv(nn.Module):
    """Dual Attention Binary Convolution module"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(DABConv, self).__init__()
        self.binary_conv = BinaryConv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.dual_attention = DualAttention(out_channels)
        self.prelu = nn.PReLU()
    
    def forward(self, x):
        x = self.binary_conv(x)
        x = self.bn(x)
        x = self.dual_attention(x)
        x = self.prelu(x)
        return x

class BNNBasicBlock(nn.Module):
    """BNN Basic Block with residual connection"""
    def __init__(self, in_channels, out_channels, stride=1):
        super(BNNBasicBlock, self).__init__()
        
        # First path
        self.dabconv1 = DABConv(in_channels, out_channels, 3, stride, 1)
        self.dabconv2 = DABConv(out_channels, out_channels, 1, 1, 0)
        
        # Second path
        self.dabconv3 = DABConv(out_channels, out_channels, 3, 1, 1)
        self.dabconv4 = DABConv(out_channels, out_channels, 1, 1, 0)
        
        # Residual connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        identity = self.shortcut(x)
        
        out = self.dabconv1(x)
        out = self.dabconv2(out)
        out = self.dabconv3(out)
        out = self.dabconv4(out)
        
        out += identity
        return out

class DABNN(nn.Module):
    """Dual Attention Binary Neural Network"""
    def __init__(self, num_classes=4):
        super(DABNN, self).__init__()
        
        # Stem block
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3, bias=False),  # Initial quantized conv
            nn.BatchNorm2d(64),
            nn.PReLU(),
            DABConv(64, 64, 3, 1, 1),  # 3x3 DABconv
            DABConv(64, 128, 1, 2, 0)  # 1x1 DABconv with downsampling
        )
        
        # Feature extractor - 6 BNN basic blocks
        self.features = nn.ModuleList([
            BNNBasicBlock(128, 128),
            BNNBasicBlock(128, 256, 2),
            BNNBasicBlock(256, 256),
            BNNBasicBlock(256, 512, 2),
            BNNBasicBlock(512, 512),
            BNNBasicBlock(512, 512)
        ])
        
        # Classification layer
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.PReLU(),
            BinaryLinear(512, num_classes)
        )
    
    def forward(self, x):
        x = self.stem(x)
        
        for block in self.features:
            x = block(x)
        
        x = self.classifier(x)
        return x

def load_model(model_path, device):
    """Load the trained model"""
    print(f"Loading model from {model_path}...")
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Extract model configuration
    model_config = checkpoint.get('model_config', {'num_classes': 4, 'img_size': 128})
    num_classes = model_config['num_classes']
    img_size = model_config.get('img_size', 128)
    
    # Initialize model
    model = DABNN(num_classes=num_classes).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Extract other information
    class_names = checkpoint.get('class_names', [f'Class_{i}' for i in range(num_classes)])
    best_val_acc = checkpoint.get('best_val_acc', None)
    test_acc = checkpoint.get('test_acc', None)
    
    print(f"Model loaded successfully!")
    print(f"Number of classes: {num_classes}")
    print(f"Image size: {img_size}")
    print(f"Class names: {class_names}")
    if best_val_acc:
        print(f"Best validation accuracy during training: {best_val_acc:.2f}%")
    if test_acc:
        print(f"Test accuracy during training: {test_acc:.2f}%")
    
    return model, class_names, img_size

def create_test_loader(data_path, img_size, batch_size=32, test_ratio=1.0):
    """Create test data loader"""
    
    # Define transforms
    test_transforms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load dataset
    dataset = ImageFolder(data_path, transform=test_transforms)
    
    # If test_ratio < 1.0, use only a portion of the dataset
    if test_ratio < 1.0:
        total_size = len(dataset)
        test_size = int(test_ratio * total_size)
        remaining_size = total_size - test_size
        test_dataset, _ = random_split(
            dataset, [test_size, remaining_size],
            generator=torch.Generator().manual_seed(42)
        )
    else:
        test_dataset = dataset
    
    # Create data loader
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    print(f"Test dataset size: {len(test_dataset)}")
    print(f"Test batches: {len(test_loader)}")
    print(f"Dataset classes: {dataset.classes}")
    
    return test_loader, dataset.classes

def evaluate_model(model, test_loader, device, class_names):
    """Evaluate the model on test data"""
    print("\nEvaluating model...")
    
    model.eval()
    all_preds = []
    all_targets = []
    all_probs = []
    correct = 0
    total = 0
    
    start_time = time.time()
    
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            
            # Forward pass
            outputs = model(inputs)
            probs = F.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)
            
            # Collect predictions and targets
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            
            # Calculate accuracy
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # Print progress
            if (batch_idx + 1) % 10 == 0:
                print(f"Processed {batch_idx + 1}/{len(test_loader)} batches", end='\r')
    
    eval_time = time.time() - start_time
    print(f"\nEvaluation completed in {eval_time:.2f} seconds")
    
    # Calculate metrics
    test_accuracy = 100.0 * correct / total
    
    # Generate detailed classification report
    report = classification_report(
        all_targets, all_preds, 
        target_names=class_names,
        digits=4,
        output_dict=True
    )
    
    # Generate confusion matrix
    cm = confusion_matrix(all_targets, all_preds)
    
    return test_accuracy, report, cm, all_preds, all_targets, all_probs

def print_detailed_results(test_accuracy, report, class_names):
    """Print detailed accuracy results"""
    print("\n" + "="*80)
    print("DETAILED TEST RESULTS")
    print("="*80)
    
    print(f"\nOverall Test Accuracy: {test_accuracy:.4f}%")
    print(f"Overall Test Error Rate: {100-test_accuracy:.4f}%")
    
    print(f"\nPer-Class Results:")
    print("-" * 60)
    print(f"{'Class':<20} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
    print("-" * 60)
    
    for class_name in class_names:
        if class_name in report:
            metrics = report[class_name]
            print(f"{class_name:<20} {metrics['precision']:<12.4f} {metrics['recall']:<12.4f} "
                  f"{metrics['f1-score']:<12.4f} {metrics['support']:<10}")
    
    print("-" * 60)
    macro_avg = report['macro avg']
    print(f"{'Macro Average':<20} {macro_avg['precision']:<12.4f} {macro_avg['recall']:<12.4f} "
          f"{macro_avg['f1-score']:<12.4f} {report['macro avg']['support']:<10}")
    
    weighted_avg = report['weighted avg']
    print(f"{'Weighted Average':<20} {weighted_avg['precision']:<12.4f} {weighted_avg['recall']:<12.4f} "
          f"{weighted_avg['f1-score']:<12.4f} {report['weighted avg']['support']:<10}")

def plot_confusion_matrix(cm, class_names, save_path=None):
    """Plot and save confusion matrix"""
    plt.figure(figsize=(10, 8))
    
    # Create heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Number of Samples'})
    
    plt.title('Confusion Matrix - DABNN Model Test Results', fontsize=16, fontweight='bold')
    plt.xlabel('Predicted Class', fontsize=12)
    plt.ylabel('True Class', fontsize=12)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    
    plt.show()

def plot_class_accuracies(report, class_names, save_path=None):
    """Plot per-class accuracies"""
    # Extract per-class recall (which is the same as per-class accuracy)
    recalls = [report[class_name]['recall'] for class_name in class_names if class_name in report]
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(class_names, recalls, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
    
    # Add value labels on bars
    for i, (bar, recall) in enumerate(zip(bars, recalls)):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{recall:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.title('Per-Class Accuracy - DABNN Model', fontsize=16, fontweight='bold')
    plt.xlabel('Disease Classes', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.ylim(0, 1.1)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Class accuracies plot saved to {save_path}")
    
    plt.show()

def main():
    """Main testing function"""
    parser = argparse.ArgumentParser(description='Test DABNN model')
    parser.add_argument('--model_path', type=str, default='best_dabnn_model.pth',
                        help='Path to the saved model')
    parser.add_argument('--data_path', type=str, default='/home/bruh/Documents/BNN2/split',
                        help='Path to test dataset')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for testing')
    parser.add_argument('--test_ratio', type=float, default=1.0,
                        help='Ratio of dataset to use for testing (0.1 for 10%)')
    parser.add_argument('--save_plots', action='store_true',
                        help='Save plots to files')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    print("\n" + "="*80)
    print("DABNN MODEL TESTING")
    print("="*80)
    
    try:
        # Load model
        model, expected_class_names, img_size = load_model(args.model_path, device)
        
        # Create test data loader
        test_loader, dataset_class_names = create_test_loader(
            args.data_path, img_size, args.batch_size, args.test_ratio
        )
        
        # Use the class names from dataset (they should match the model)
        class_names = dataset_class_names
        
        # Verify class compatibility
        if len(class_names) != len(expected_class_names):
            print(f"Warning: Dataset has {len(class_names)} classes, but model expects {len(expected_class_names)}")
        
        # Evaluate model
        test_accuracy, report, cm, preds, targets, probs = evaluate_model(
            model, test_loader, device, class_names
        )
        
        # Print detailed results
        print_detailed_results(test_accuracy, report, class_names)
        
        # Generate plots
        print("\nGenerating visualizations...")
        
        if args.save_plots:
            plot_confusion_matrix(cm, class_names, 'confusion_matrix_test.png')
            plot_class_accuracies(report, class_names, 'class_accuracies_test.png')
        else:
            plot_confusion_matrix(cm, class_names)
            plot_class_accuracies(report, class_names)
        
        # Summary
        print("\n" + "="*80)
        print("SUMMARY")
        print("="*80)
        print(f"Test Dataset: {args.data_path}")
        print(f"Model: {args.model_path}")
        print(f"Test Samples: {len(targets)}")
        print(f"Overall Accuracy: {test_accuracy:.4f}%")
        print(f"Macro F1-Score: {report['macro avg']['f1-score']:.4f}")
        print(f"Weighted F1-Score: {report['weighted avg']['f1-score']:.4f}")
        
        return test_accuracy, report, cm
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please check the model path and data path.")
        return None, None, None
    except Exception as e:
        print(f"Error during testing: {e}")
        return None, None, None

if __name__ == "__main__":
    main()
