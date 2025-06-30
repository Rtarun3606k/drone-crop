import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import os
import csv
import pandas as pd
from tqdm import tqdm


if torch.cuda.is_available():
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU device count: {torch.cuda.device_count()}")
    print(f"GPU device name: {torch.cuda.get_device_name(0)}")
    device = torch.device('cuda')
else:
    print("CUDA not available, using CPU")
    device = torch.device('cpu')
# Binary Activation Function
class BinaryActivation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return torch.sign(input)
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

def binary_activation(x):
    return BinaryActivation.apply(x)

# Binary Linear Layer
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

# Binary Neural Network Model
class BinaryNeuralNetwork(nn.Module):
    def __init__(self, input_size=3*224*224, hidden_size=512, num_classes=4):  # 4 classes to match your model
        super(BinaryNeuralNetwork, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.input_layer = nn.Linear(input_size, hidden_size)
        self.hidden_layer = BinaryLinear(hidden_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        if len(x.shape) > 2:
            x = x.view(x.size(0), -1)
        x = F.relu(self.input_layer(x))
        x = self.dropout(x)
        x = self.hidden_layer(x)
        x = binary_activation(x)
        x = self.dropout(x)
        logits = self.output_layer(x)
        return logits

def predict_image(image_path, model_path, class_names):
    # Load the model


    model = BinaryNeuralNetwork(input_size=3*224*224, hidden_size=512, num_classes=4)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Image preprocessing
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load and preprocess image
    try:
        image = Image.open(image_path).convert('RGB')
        image_tensor = transform(image).unsqueeze(0)
        
        # Make prediction
        with torch.no_grad():
            output = model(image_tensor)
            probabilities = F.softmax(output, dim=1)[0]
            predicted_class = torch.argmax(output, dim=1).item()
        
        result = {
            'image_path': image_path,
            'predicted_class_idx': predicted_class,
            'predicted_class': class_names[predicted_class],
            'confidence': probabilities[predicted_class].item(),
        }
        
        # Add probabilities for each class
        for i, class_name in enumerate(class_names):
            result[f'prob_{class_name}'] = probabilities[i].item()
            
        return result
    
    except Exception as e:
        return {
            'image_path': image_path,
            'error': str(e)
        }

def predict_batch_and_save_csv(image_folder, image_files, model_path, output_csv, original_class_name):
    """
    Predict a batch of images and save results to CSV
    
    Args:
        image_folder: Base folder containing images
        image_files: List of image filenames
        model_path: Path to the trained model
        output_csv: Path to save the CSV results
        original_class_name: Name of the original class (entered manually)
    """
    
    # Define class names matching your model's output
    class_names = ['Healthy_Soyabean', 'Soyabean Semilooper_Pest_Attack', 'Soyabean_Mosaic', 'rust']
    
    # Prepare results list
    results = []
    
    # Process each image
    print(f"Processing {len(image_files)} images from class '{original_class_name}'...")
    for i, img_file in enumerate(tqdm(image_files)):
        full_path = os.path.join(image_folder, img_file)
        
        # Make prediction
        try:
            result = predict_image(full_path, model_path, class_names)
            
            # Add original class
            result['original_class'] = original_class_name
            
            # Add to results list
            results.append(result)
            
            # Print progress
            if (i + 1) % 10 == 0 or i == 0:
                print(f"Processed {i+1}/{len(image_files)} images")
                
        except Exception as e:
            print(f"Error processing {img_file}: {e}")
            results.append({
                'image_path': full_path,
                'original_class': original_class_name,
                'error': str(e)
            })
    
    # Create DataFrame and save to CSV
    if results:
        # Check if file exists to decide if we need headers
        file_exists = os.path.isfile(output_csv)
        
        # Write to CSV (append mode if file exists)
        df = pd.DataFrame(results)
        df.to_csv(output_csv, mode='a' if file_exists else 'w', 
                 header=not file_exists, index=False)
        
        print(f"Results for {len(results)} images saved to {output_csv}")
        
        # Calculate quick statistics
        if 'predicted_class' in df.columns and 'original_class' in df.columns:
            correct = df[df['predicted_class'] == df['original_class']].shape[0]
            accuracy = (correct / len(df)) * 100
            print(f"Class accuracy: {accuracy:.2f}% ({correct}/{len(df)} correct)")
    else:
        print("No results to save!")

# Example usage
if __name__ == "__main__":
    class_names = ['Healthy_Soyabean', 'Soyabean Semilooper_Pest_Attack', 'Soyabean_Mosaic', 'rust']
    # Paths
    model_path = "results/bnn_plant_disease_model_20250629_141950.pth"  # Update with your model path
    image_base_path = "/home/dragoon/Downloads/MH-SoyaHealthVision An Indian UAV and Leaf Image Dataset for Integrated Crop Health Assessment/Soyabean_UAV-Based_Image_Dataset/Healthy_Soyabean/"  # Update with your image folder
    output_csv = "bnn_predictions_results.csv"
    
    # Import your image lists
    from paths import Healthyhealthy_images, Soyabean_Mosaic_array, Rust_array
    
    # Process healthy images
    # predict_batch_and_save_csv(
    #     image_folder=image_base_path,
    #     image_files=Healthyhealthy_images,
    #     model_path=model_path,
    #     output_csv=output_csv,
    #     original_class_name="Healthy_Soyabean"  # This is what you'd enter manually
    # )
    
    # To process other classes, uncomment and modify these:
    """
    # Process Mosaic images
    predict_batch_and_save_csv(
        image_folder="/path/to/mosaic/folder",
        image_files=Soyabean_Mosaic_array,
        model_path=model_path,
        output_csv=output_csv,
        original_class_name="Soyabean_Mosaic"
    )
    
    # Process Rust images
    predict_batch_and_save_csv(
        image_folder="/path/to/rust/folder",
        image_files=Rust_array,
        model_path=model_path,
        output_csv=output_csv,
        original_class_name="rust"
    )
    """
    
    # Print final message
    # print(f"\nAll predictions completed and saved to {output_csv}")
    # print("You can now open this CSV file to analyze the results!")

    # image_path_downlaoded = "/home/dragoon/Downloads/masoy-oil.jpg"
    # image_path_downlaoded = "/home/dragoon/Downloads/e7c3557bf30e3a2e9c52b66feeccca20.png"
    # image_path_downlaoded = "/home/dragoon/Downloads/80105131.jpg"
    image_path_downlaoded = "/home/dragoon/Downloads/Soy_Rust_2.jpeg"

    # result = predict_image(class_names=class_names, image_path=image_base_path+"image_012.jpg", model_path=model_path)
    result = predict_image(class_names=class_names, image_path=image_path_downlaoded, model_path=model_path)
    print(f"Prediction for image: {result['image_path']}\nresult: {result['predicted_class']}\n confidence {result['confidence']:.2f}")