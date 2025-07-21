import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
import json
import threading
from queue import Queue
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
from torchvision import transforms

def load_config(config_path=None):
    """
    Load model configuration from JSON file
    
    Parameters:
    config_path (str): Path to config file. If None, uses default in same directory
    
    Returns:
    dict: Configuration dictionary
    """
    if config_path is None:
        config_path = os.path.join(os.path.dirname(__file__), 'model_config.json')
    
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Config file not found: {config_path}")
        return {
            "models": {
                "dabnn": {
                    "path": "../../BNN_NEW/best_dabnn_model.pth",
                    "type": "DABNN",
                    "description": "Dual Attention Binary Neural Network"
                }
            },
            "default_model": "dabnn"
        }

def get_model_path(model_name=None, config=None):
    """
    Get model path from configuration
    
    Parameters:
    model_name (str): Name of the model in config. If None, uses default
    config (dict): Configuration dictionary. If None, loads from file
    
    Returns:
    str: Path to model file
    """
    if config is None:
        config = load_config()
    
    if model_name is None:
        model_name = config.get('default_model', 'dabnn')
    
    model_info = config.get('models', {}).get(model_name, {})
    return model_info.get('path', '')

# Binary activation function for DABNN
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

# Define the basic BinaryLinear (original version without scale)
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
        return F.linear(input, binary_weight, self.bias)

# Binary convolution layer for DABNN
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

# Attention mechanisms for DABNN
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

# New DABNN model
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


# Legacy BNN model for backward compatibility
class BinaryNeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, num_hidden_layers=1, embedding_size=512, dropout_rate=0.35):
        super(BinaryNeuralNetwork, self).__init__()
        
        # Progressive embedding
        self.embedding = nn.Sequential(
            nn.Linear(input_size, input_size // 4),
            nn.ReLU(),
            nn.BatchNorm1d(input_size // 4),
            nn.Dropout(dropout_rate),
            nn.Linear(input_size // 4, input_size // 8),
            nn.ReLU(),
            nn.BatchNorm1d(input_size // 8),
            nn.Dropout(dropout_rate),
            nn.Linear(input_size // 8, input_size // 16),
            nn.ReLU(),
            nn.BatchNorm1d(input_size // 16),
            nn.Dropout(dropout_rate),
            nn.Linear(input_size // 16, embedding_size),
            nn.ReLU(),
            nn.BatchNorm1d(embedding_size),
        )
        
        # Binary layers without scale parameter
        self.input_binary = BinaryLinear(embedding_size, hidden_size)
        self.hidden_layers = nn.ModuleList([
            BinaryLinear(hidden_size, hidden_size) for _ in range(num_hidden_layers)
        ])
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(hidden_size) for _ in range(num_hidden_layers + 1)
        ])
        
        self.output_layer = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, x):
        # Flatten input
        if len(x.shape) > 2:
            x = x.view(x.size(0), -1)
            
        x = self.embedding(x)
        
        x = self.input_binary(x)
        x = self.batch_norms[0](x)
        x = torch.sign(F.hardtanh(x))
        x = self.dropout(x)
        
        for i, binary_layer in enumerate(self.hidden_layers):
            x = binary_layer(x)
            x = self.batch_norms[i+1](x)
            x = torch.sign(F.hardtanh(x))
            x = self.dropout(x)
        
        x = self.output_layer(x)
        return x


def load_model(model_path, device='cpu'):
    """
    Load model and automatically detect whether it's DABNN or legacy BNN
    
    Parameters:
    model_path (str): Path to the model file
    device: Device to load the model on
    
    Returns:
    tuple: (model, model_config) where model_config contains metadata
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    print(f"Loading model from {model_path}...")
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Try to detect model type from checkpoint structure
    if 'model_config' in checkpoint:
        # New DABNN model
        model_config = checkpoint['model_config']
        num_classes = model_config.get('num_classes', 4)
        img_size = model_config.get('img_size', 128)
        class_names = checkpoint.get('class_names', ['Healthy_Soyabean', 'Soyabean Semilooper_Pest_Attack', 'Soyabean_Mosaic', 'rust'])
        
        print("Detected DABNN model")
        model = DABNN(num_classes=num_classes).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        model_config.update({
            'model_type': 'DABNN',
            'class_names': class_names,
            'img_size': img_size,
            'input_channels': 3
        })
        
    else:
        # Legacy BNN model
        print("Detected legacy BNN model")
        
        # Default config for legacy model
        model_config = {
            'model_type': 'BNN',
            'num_classes': 4,
            'img_size': 64,
            'input_channels': 3,
            'class_names': ['Healthy_Soyabean', 'Soyabean Semilooper_Pest_Attack', 'Soyabean_Mosaic', 'rust']
        }
        
        model = BinaryNeuralNetwork(
            input_size=12288,  # 3x64x64
            hidden_size=256,
            num_classes=4,
            num_hidden_layers=1,
            embedding_size=512
        )
        
        # Load weights
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.to(device)
    
    model.eval()
    
    print(f"Model loaded successfully!")
    print(f"Model type: {model_config['model_type']}")
    print(f"Number of classes: {model_config['num_classes']}")
    print(f"Image size: {model_config['img_size']}")
    print(f"Class names: {model_config['class_names']}")
    
    return model, model_config


def get_transform(img_size, model_type='DABNN'):
    """
    Get appropriate image transform based on model type
    
    Parameters:
    img_size (int): Target image size
    model_type (str): Model type ('DABNN' or 'BNN')
    
    Returns:
    torchvision.transforms.Compose: Transform pipeline
    """
    if model_type == 'DABNN':
        # DABNN uses 128x128 images with standard ImageNet normalization
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        # Legacy BNN uses 64x64 images
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


# Modified function to handle a single image prediction with auto model detection
def predict_single_image(image_path, model, model_config, device='cpu'):
    """
    Process a single image and return prediction results
    
    Parameters:
    image_path (str): Path to the image file
    model: The loaded model (BNN or DABNN)
    model_config (dict): Model configuration containing metadata
    device: Device to run inference on
    
    Returns:
    dict: Prediction results for this image
    """
    try:
        # Load and process image
        image = Image.open(image_path).convert('RGB')
        
        # Get appropriate transform based on model type
        transform = get_transform(model_config['img_size'], model_config['model_type'])
        image_tensor = transform(image).unsqueeze(0).to(device)
        
        # Run inference
        with torch.no_grad():
            output = model(image_tensor)
            probs = F.softmax(output, dim=1)
            confidence, pred_idx = torch.max(probs, 1)
        
        # Get class names from model config
        class_names = model_config['class_names']
        predicted_class = class_names[pred_idx.item()]
        confidence_score = confidence.item() * 100
        
        # Create probabilities dictionary
        all_probs = {class_names[i]: float(probs[0][i].item() * 100) for i in range(len(class_names))}
        
        return {
            'image_name': os.path.basename(image_path),
            'predicted_class': predicted_class,
            'confidence': float(confidence_score),
            'probabilities': all_probs,
            'model_type': model_config['model_type'],
            'status': 'success'
        }
    except Exception as e:
        return {
            'image_name': os.path.basename(image_path),
            'status': 'error',
            'error': str(e)
        }

# Batch prediction function using threading with auto model detection
def batch_predict(image_paths, model_path, num_threads=4, output_json_path=None):
    """
    Process multiple images in parallel and return combined results
    
    Parameters:
    image_paths (list): List of paths to image files
    model_path (str): Path to the saved model file
    num_threads (int): Number of threads to use
    output_json_path (str): Path to save JSON output (optional)
    
    Returns:
    list: List of prediction results for all images
    """
    # Load the model once for all threads to use
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model with auto-detection
    model, model_config = load_model(model_path, device)
    
    results = []
    
    # Process images in parallel
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        # Submit all tasks and collect futures
        future_to_path = {executor.submit(predict_single_image, path, model, model_config, device): path 
                         for path in image_paths}
        
        # Process results as they complete
        total_images = len(image_paths)
        completed = 0
        
        print(f"Processing {total_images} images with {num_threads} threads...")
        
        for future in future_to_path:
            result = future.result()
            results.append(result)
            
            # Progress reporting
            completed += 1
            if completed % 10 == 0 or completed == total_images:
                print(f"Processed {completed}/{total_images} images")
    
    # Save results to JSON if output path provided
    if output_json_path:
        with open(output_json_path, 'w') as json_file:
            json.dump(results, json_file, indent=2)
        print(f"Results saved to {output_json_path}")
    
    return results

if __name__ == "__main__":
    # Updated to use the new DABNN model from BNN_NEW folder
    image_path = "../../BNN_NEW/test_image.jpg"  # Update with actual test image path
    model_path = "../../BNN_NEW/best_dabnn_model.pth"  # New DABNN model
    
    # For batch prediction, update paths as needed:
    # Directory containing images to process
    image_dir = "../../BNN_NEW/test_data"  # Update with actual test data path
    output_json = "dabnn_results.json"
    
    # Check if the new model exists, otherwise fall back to legacy model
    if not os.path.exists(model_path):
        print(f"DABNN model not found at {model_path}")
        print("Falling back to legacy BNN model...")
        # Fallback to legacy model path (update this to your actual legacy model path)
        # model_path = "../../BNN/optimized_bnn_plant_disease_64x64.pt"
    
    # Check if test directory exists, create sample test if needed
    if os.path.exists(image_dir):
        # Get all jpg files in directory
        image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        if image_paths:
            print(f"Found {len(image_paths)} images for batch processing")
            
            # Run batch prediction
            try:
                # Check if analysis module exists for enhanced processing
                try:
                    from analysis import process_batch_with_analysis
                    results = process_batch_with_analysis(image_paths, model_path, num_threads=8, output_json_path=output_json)
                except ImportError:
                    print("Analysis module not found, using standard batch processing")
                    results = batch_predict(image_paths, model_path, num_threads=8, output_json_path=output_json)
                
                print(f"Batch processing complete. Processed {len(results)} images.")
                
                # Print summary
                successful_predictions = [r for r in results if r['status'] == 'success']
                if successful_predictions:
                    print(f"Successful predictions: {len(successful_predictions)}")
                    print(f"Model type used: {successful_predictions[0].get('model_type', 'Unknown')}")
                    
                    # Show class distribution
                    class_counts = {}
                    for result in successful_predictions:
                        pred_class = result['predicted_class']
                        class_counts[pred_class] = class_counts.get(pred_class, 0) + 1
                    
                    print("Prediction distribution:")
                    for class_name, count in class_counts.items():
                        print(f"  {class_name}: {count} images")
                
            except Exception as e:
                print(f"Error during batch processing: {e}")
        else:
            print(f"No images found in {image_dir}")
    else:
        print(f"Test directory not found: {image_dir}")
        print("Please update the image_dir path to point to your test images")
    
    # For single image prediction (if test image exists)
    if os.path.exists(image_path):
        try:
            # Create a single-element list and use the batch function
            single_result = batch_predict([image_path], model_path, num_threads=1)[0]
            
            print(f"\nSingle image prediction results:")
            print(f"Image: {single_result['image_name']}")
            print(f"Predicted class: {single_result['predicted_class']}")
            print(f"Confidence: {single_result['confidence']:.2f}%")
            print(f"Model type: {single_result.get('model_type', 'Unknown')}")
            print("\nAll probabilities:")
            for class_name, prob in single_result['probabilities'].items():
                print(f"  {class_name}: {prob:.2f}%")
        except Exception as e:
            print(f"Error during single image prediction: {e}")
    else:
        print(f"Test image not found: {image_path}")
        print("Please update the image_path to point to a valid test image")

