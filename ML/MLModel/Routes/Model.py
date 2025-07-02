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
# Define the basic BinaryLinear (original version without scale)
class BinaryLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(BinaryLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
            
    def forward(self, input):
        binary_weight = torch.sign(self.weight)
        return F.linear(input, binary_weight, self.bias)

# Original BNN model matching the trained weights
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


# Modified function to handle a single image prediction
def predict_single_image(image_path, model, device='cpu'):
    """
    Process a single image and return prediction results
    
    Parameters:
    image_path (str): Path to the image file
    model: The loaded BNN model
    device: Device to run inference on
    
    Returns:
    dict: Prediction results for this image
    """
    try:
        # Load and process image
        image = Image.open(image_path).convert('RGB')
        
        # Images of any size will be resized to 64x64
        transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        image_tensor = transform(image).unsqueeze(0).to(device)
        
        # Run inference
        with torch.no_grad():
            output = model(image_tensor)
            probs = F.softmax(output, dim=1)
            confidence, pred_idx = torch.max(probs, 1)
        
        # Get class names
        class_names = ['Healthy_Soyabean', 'Soyabean Semilooper_Pest_Attack', 'Soyabean_Mosaic', 'rust']
        predicted_class = class_names[pred_idx.item()]
        confidence_score = confidence.item() * 100
        
        # Create probabilities dictionary
        all_probs = {class_names[i]: float(probs[0][i].item() * 100) for i in range(len(class_names))}
        
        return {
            'image_name': os.path.basename(image_path),
            'predicted_class': predicted_class,
            'confidence': float(confidence_score),
            'probabilities': all_probs,
            'status': 'success'
        }
    except Exception as e:
        return {
            'image_name': os.path.basename(image_path),
            'status': 'error',
            'error': str(e)
        }

# Batch prediction function using threading
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
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Load the model once for all threads to use
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    checkpoint = torch.load(model_path, map_location=device)
    
    # Create model
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
    
    results = []
    
    # Process images in parallel
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        # Submit all tasks and collect futures
        future_to_path = {executor.submit(predict_single_image, path, model, device): path 
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
    # For single image prediction
    image_path = "/home/dragoon/Downloads/MH-SoyaHealthVision An Indian UAV and Leaf Image Dataset for Integrated Crop Health Assessment/Soyabean_UAV-Based_Image_Dataset/Soyabean_Mosaic/DJI_20231215160828_0110_D_000011.jpg"
    model_path = "/home/dragoon/coding/drone-crop/ML/BNN/optimized_bnn_plant_disease_64x64.pt"
    
    # For batch prediction, uncomment and modify:
    # Directory containing images to process
    # image_dir = "/home/dragoon/Downloads/MH-SoyaHealthVision An Indian UAV and Leaf Image Dataset for Integrated Crop Health Assessment/Soyabean_UAV-Based_Image_Dataset/Soyabean_Mosaic/"
    # image_dir = "/home/dragoon/Downloads/MH-SoyaHealthVision An Indian UAV and Leaf Image Dataset for Integrated Crop Health Assessment/Soyabean_UAV-Based_Image_Dataset/Soyabean Semilooper_Pest_Attack/"
    # image_dir = "/home/dragoon/Downloads/MH-SoyaHealthVision An Indian UAV and Leaf Image Dataset for Integrated Crop Health Assessment/Soyabean_UAV-Based_Image_Dataset/rust/"
    image_dir = "/home/dragoon/Downloads/MH-SoyaHealthVision An Indian UAV and Leaf Image Dataset for Integrated Crop Health Assessment/Soyabean_UAV-Based_Image_Dataset/Healthy_Soyabean"
    output_json = "Healthy_Soyabean.json"
    
    # Get all jpg files in directory
    image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir) 
                  if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    # Run batch prediction
    results = batch_predict(image_paths, model_path, num_threads=8, output_json_path=output_json)
    print(f"Batch processing complete. Processed {len(results)} images.")
    
    # For single image prediction
    try:
        # Create a single-element list and use the batch function
        single_result = batch_predict([image_path], model_path, num_threads=1)[0]
        
        print(f"Predicted class: {single_result['predicted_class']}")
        print(f"Confidence: {single_result['confidence']:.2f}%")
        print("\nAll probabilities:")
        for class_name, prob in single_result['probabilities'].items():
            print(f"  {class_name}: {prob:.2f}%")
    except Exception as e:
        print(f"Error during prediction: {e}")

