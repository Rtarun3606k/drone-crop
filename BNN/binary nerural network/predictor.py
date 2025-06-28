import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import random

from paths import Soyabean_Mosaic_array,Healthyhealthy_images,Rust_array

# Binary Activation Function
class BinaryActivation(torch.autograd.Function):
    """
    Binary activation function using the sign function.
    Forward: sign(x) = {-1 if x < 0, +1 if x >= 0}
    Backward: Straight-through estimator (STE) - passes gradients through unchanged
    """
    
    @staticmethod
    def forward(ctx, input):
        return torch.sign(input)
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

def binary_activation(x):
    """Wrapper function for binary activation"""
    return BinaryActivation.apply(x)

# Binary Linear Layer
class BinaryLinear(nn.Module):
    """
    Binary Linear layer with binary weights.
    Weights are binarized using the sign function during forward pass.
    """
    
    def __init__(self, in_features, out_features, bias=True):
        super(BinaryLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Initialize weights using normal distribution
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.1)
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
    
    def forward(self, input):
        # Binarize weights using sign function
        binary_weight = torch.sign(self.weight)
        
        # Perform linear transformation with binary weights
        output = F.linear(input, binary_weight, self.bias)
        
        return output

# Binary Neural Network Model
class BinaryNeuralNetwork(nn.Module):
    """
    Binary Neural Network for multiclass plant disease classification.
    """
    
    def __init__(self, input_size=3*64*64, hidden_size=512, num_classes=3):
        super(BinaryNeuralNetwork, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        
        # First layer: Regular linear layer (input preprocessing)
        self.input_layer = nn.Linear(input_size, hidden_size)
        
        # Hidden layer: Binary linear layer
        self.hidden_layer = BinaryLinear(hidden_size, hidden_size)
        
        # Output layer: Regular linear layer for final classification
        self.output_layer = nn.Linear(hidden_size, num_classes)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        # Flatten input if it's not already flattened
        if len(x.shape) > 2:
            x = x.view(x.size(0), -1)  # Flatten to (batch_size, input_size)
        
        # Input layer with ReLU activation
        x = F.relu(self.input_layer(x))
        x = self.dropout(x)
        
        # Hidden layer with binary weights and binary activation
        x = self.hidden_layer(x)
        x = binary_activation(x)  # Binary activation function
        x = self.dropout(x)
        
        # Output layer (no activation - raw logits)
        logits = self.output_layer(x)
        
        return logits

def predict_plant_disease(image_path, model_path=None, model=None, class_names=None):
    """
    Predict plant disease from image path using trained Binary Neural Network
    
    Args:
        image_path (str): Path to the image file
        model_path (str, optional): Path to saved model weights
        model (torch.nn.Module, optional): Pre-loaded model
        class_names (list, optional): List of class names
    
    Returns:
        dict: Raw prediction results containing logits, probabilities, and predicted class
    """
    
    # Default class names (update based on your dataset)
    if class_names is None:
        # class_names = ['healthy', 'rust', 'Soyabean_Mosaic']
        class_names = ['Soyabean_Mosaic','healthy','rust']
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model if not provided
    if model is None:
        if model_path is None:
            raise ValueError("Either model or model_path must be provided")
        
        # Initialize model architecture (must match training)
        model = BinaryNeuralNetwork(
            input_size=3*64*64,
            hidden_size=512,
            num_classes=len(class_names)
        )
        
        # Load trained weights
        model.load_state_dict(torch.load(model_path, map_location=device))
    
    model.to(device)
    model.eval()
    
    # Image preprocessing pipeline (must match training preprocessing)
    transform = transforms.Compose([
        transforms.Resize((64, 64)),  # Resize to 64x64
        transforms.ToTensor(),        # Convert to tensor [0,1]
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])  # ImageNet normalization
    ])
    
    try:
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        input_tensor = transform(image).unsqueeze(0).to(device)  # Add batch dimension
        
        # Make prediction
        with torch.no_grad():
            # Get raw logits
            logits = model(input_tensor)
            
            # Get probabilities using softmax
            probabilities = F.softmax(logits, dim=1)
            
            # Get predicted class
            predicted_class_idx = torch.argmax(logits, dim=1).item()
            predicted_class_name = class_names[predicted_class_idx]
            
            # Get confidence score
            confidence = probabilities[0][predicted_class_idx].item()
            
        # Return raw prediction results
        raw_results = {
            'image_path': image_path,
            'predicted_class_idx': predicted_class_idx,
            'predicted_class_name': predicted_class_name,
            'confidence': confidence,
            'raw_logits': logits.cpu().numpy().flatten().tolist(),
            'probabilities': probabilities.cpu().numpy().flatten().tolist(),
            'class_names': class_names,
            'all_class_scores': {
                class_names[i]: probabilities[0][i].item() 
                for i in range(len(class_names))
            }
        }
        
        return raw_results
        
    except Exception as e:
        return {
            'error': str(e),
            'image_path': image_path,
            'success': False
        }

def predict_batch_plant_disease(image_paths, model_path=None, model=None, class_names=None):
    """
    Predict plant disease for multiple images
    """
    results = []
    
    # Load model once for batch processing
    if model is None and model_path is not None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if class_names is None:
            class_names = ['healthy', 'rust', 'Soyabean_Mosaic']
            
        model = BinaryNeuralNetwork(
            input_size=3*64*64,
            hidden_size=512,
            num_classes=len(class_names)
        )
        model.load_state_dict(torch.load(model_path, map_location=device))
    
    # Predict each image
    for image_path in image_paths:
        result = predict_plant_disease(image_path, model=model, class_names=class_names)
        results.append(result)
    
    return results

def quick_predict(image_path, model_path):
    """
    Quick prediction with minimal output
    """
    result = predict_plant_disease(image_path, model_path)
    
    if 'error' in result:
        return f"Error: {result['error']}"
    
    return {
        'prediction': result['predicted_class_name'],
        'confidence': f"{result['confidence']:.2%}",
        'raw_scores': result['all_class_scores']
    }

# Usage examples:
if __name__ == "__main__":
  # Example", 1: Single image prediction
   # image_path", = "/home/dragoon/coding/drone-crop/BNN/DataSet/Plant/Soyabean_Mosaic/DJI_20231216171115_0123_D_000089.jpg"
    image_path = "/home/dragoon/coding/drone-crop/BNN/DataSet/Plant/Soyabean_Mosaic/DJI_20231216171115_0123_D_000025.jpg"

    constant_path = ["/home/dragoon/Downloads/MH-SoyaHealthVision An Indian UAV and Leaf Image Dataset for Integrated Crop Health Assessment/Soyabean_UAV-Based_Image_Dataset/Soyabean_Mosaic/","/home/dragoon/Downloads/MH-SoyaHealthVision An Indian UAV and Leaf Image Dataset for Integrated Crop Health Assessment/Soyabean_UAV-Based_Image_Dataset/rust/","/home/dragoon/Downloads/MH-SoyaHealthVision An Indian UAV and Leaf Image Dataset for Integrated Crop Health Assessment/Soyabean_UAV-Based_Image_Dataset/healthy/","/home/dragoon/Downloads/MH-SoyaHealthVision An Indian UAV and Leaf Image Dataset for Integrated Crop Health Assessment/Soyabean_UAV-Based_Image_Dataset/Healthy_Soyabean/"]

    #", image_path = "/home/dragoon/coding/drone-crop/BNN/DataSet/Plant/rust/DJI_20231215155433_0107_D_000115.jpg"
    model_path = "results/bnn_plant_disease_model_20250628_125321.pth"
    
    # Get raw prediction

    



    for i in Soyabean_Mosaic_array:

        raw_result = predict_plant_disease(constant_path+i, model_path)
        print("Raw Prediction Result:")
        print(raw_result)
    
    # Example 2: Quick prediction
        quick_result = quick_predict(image_path, model_path)
        print("\nQuick Prediction:")
        print(quick_result)
        val = 0
        val = val + 1
        print(f"\nPrediction {val} completed for {image_path}")
        if val == 5:
            break
    
    # Example 3: Batch prediction
    # image_paths = [
    #     "/path/to/image1.jpg",
    #     "/path/to/image2.jpg",
    #     "/path/to/image3.jpg"
    # ]
    
    # batch_results = predict_batch_plant_disease(image_paths, model_path)
    # print(f"\nBatch Predictions for {len(batch_results)} images completed")