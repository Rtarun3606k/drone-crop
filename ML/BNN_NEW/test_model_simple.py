#!/usr/bin/env python3
"""
Simple test script for DABNN model - focused on accuracy metrics
Usage: python test_model_simple.py [--data_path /path/to/test/data] [--model_path /path/to/model.pth]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import argparse
import time
from pathlib import Path

# Import the model architecture (you'll need to have the model classes available)
# For simplicity, I'm including the minimal required classes here
class BinaryActivation(torch.autograd.Function):
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
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super(BinaryConv2d, self).__init__()
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size) * 0.1)
        self.stride = stride
        self.padding = padding
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('bias', None)
    
    def forward(self, input):
        binary_weight = torch.sign(self.weight)
        return F.conv2d(input, binary_weight, self.bias, self.stride, self.padding)

class ChannelAttention(nn.Module):
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
        avg_out = self.fc(self.avg_pool(x).view(b, c))
        max_out = self.fc(self.max_pool(x).view(b, c))
        out = avg_out + max_out
        return self.sigmoid(out).view(b, c, 1, 1) * x

class SpatialAttention(nn.Module):
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
    def __init__(self, in_channels):
        super(DualAttention, self).__init__()
        self.channel_attention = ChannelAttention(in_channels)
        self.spatial_attention = SpatialAttention()
    
    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x

class DABConv(nn.Module):
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
    def __init__(self, in_channels, out_channels, stride=1):
        super(BNNBasicBlock, self).__init__()
        self.dabconv1 = DABConv(in_channels, out_channels, 3, stride, 1)
        self.dabconv2 = DABConv(out_channels, out_channels, 1, 1, 0)
        self.dabconv3 = DABConv(out_channels, out_channels, 3, 1, 1)
        self.dabconv4 = DABConv(out_channels, out_channels, 1, 1, 0)
        
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
    def __init__(self, num_classes=4):
        super(DABNN, self).__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3, bias=False),
            nn.BatchNorm2d(64),
            nn.PReLU(),
            DABConv(64, 64, 3, 1, 1),
            DABConv(64, 128, 1, 2, 0)
        )
        
        self.features = nn.ModuleList([
            BNNBasicBlock(128, 128),
            BNNBasicBlock(128, 256, 2),
            BNNBasicBlock(256, 256),
            BNNBasicBlock(256, 512, 2),
            BNNBasicBlock(512, 512),
            BNNBasicBlock(512, 512)
        ])
        
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

def test_model(model_path, data_path, batch_size=32):
    """Test the DABNN model and return accuracy metrics"""
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model from: {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    
    # Get model configuration
    model_config = checkpoint.get('model_config', {'num_classes': 4, 'img_size': 128})
    num_classes = model_config['num_classes']
    img_size = model_config.get('img_size', 128)
    class_names = checkpoint.get('class_names', [f'Class_{i}' for i in range(num_classes)])
    
    # Initialize and load model
    model = DABNN(num_classes=num_classes).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Model loaded - Classes: {num_classes}, Image size: {img_size}")
    print(f"Expected classes: {class_names}")
    
    # Data transforms
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load test dataset
    print(f"Loading test data from: {data_path}")
    test_dataset = ImageFolder(data_path, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    print(f"Test dataset - Total samples: {len(test_dataset)}")
    print(f"Dataset classes: {test_dataset.classes}")
    print(f"Samples per class: {[len([x for x in test_dataset.targets if x == i]) for i in range(len(test_dataset.classes))]}")
    
    # Test the model
    print("\\nTesting model...")
    correct = 0
    total = 0
    class_correct = [0] * num_classes
    class_total = [0] * num_classes
    
    start_time = time.time()
    
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # Per-class accuracy
            for i in range(targets.size(0)):
                label = targets[i]
                class_correct[label] += (predicted[i] == targets[i]).item()
                class_total[label] += 1
            
            if (batch_idx + 1) % 10 == 0:
                print(f"Progress: {batch_idx + 1}/{len(test_loader)} batches", end='\\r')
    
    test_time = time.time() - start_time
    overall_accuracy = 100.0 * correct / total
    
    # Results
    print(f"\\n{'='*60}")
    print("TEST RESULTS")
    print(f"{'='*60}")
    print(f"Overall Accuracy: {overall_accuracy:.2f}% ({correct}/{total})")
    print(f"Test time: {test_time:.2f} seconds")
    print(f"Samples per second: {total/test_time:.1f}")
    
    print(f"\\nPer-Class Accuracy:")
    print(f"{'Class':<15} {'Correct':<8} {'Total':<8} {'Accuracy':<10}")
    print("-" * 45)
    
    for i, class_name in enumerate(test_dataset.classes):
        if class_total[i] > 0:
            class_acc = 100.0 * class_correct[i] / class_total[i]
            print(f"{class_name:<15} {class_correct[i]:<8} {class_total[i]:<8} {class_acc:<10.2f}%")
        else:
            print(f"{class_name:<15} {'0':<8} {'0':<8} {'N/A':<10}")
    
    # Additional metrics
    print(f"\\nAdditional Metrics:")
    print(f"Error Rate: {100-overall_accuracy:.2f}%")
    print(f"Model Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    return {
        'overall_accuracy': overall_accuracy,
        'class_accuracies': [100.0 * class_correct[i] / class_total[i] if class_total[i] > 0 else 0 
                           for i in range(num_classes)],
        'class_names': test_dataset.classes,
        'total_samples': total,
        'correct_samples': correct,
        'test_time': test_time
    }

def main():
    parser = argparse.ArgumentParser(description='Test DABNN model')
    parser.add_argument('--model_path', type=str, default='best_dabnn_model.pth',
                        help='Path to the saved model file')
    parser.add_argument('--data_path', type=str, default='/home/bruh/Documents/BNN2/split',
                        help='Path to test dataset directory')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for testing')
    
    args = parser.parse_args()
    
    # Check if files exist
    if not Path(args.model_path).exists():
        print(f"Error: Model file not found at {args.model_path}")
        return
    
    if not Path(args.data_path).exists():
        print(f"Error: Data directory not found at {args.data_path}")
        return
    
    # Run test
    try:
        results = test_model(args.model_path, args.data_path, args.batch_size)
        print(f"\\nTest completed successfully!")
        print(f"Final Accuracy: {results['overall_accuracy']:.2f}%")
    except Exception as e:
        print(f"Error during testing: {e}")
        print("Make sure your dataset directory contains subdirectories for each class")

if __name__ == "__main__":
    main()
