from PIL import Image
import os
import re
from datetime import datetime
import csv

# constantPath = "/home/dragoon/Downloads/MH-SoyaHealthVision An Indian UAV and Leaf Image Dataset for Integrated Crop Health Assessment/Soyabean_UAV-Based_Image_Dataset/rust/"
# constantPath = "/home/dragoon/Downloads/MH-SoyaHealthVision An Indian UAV and Leaf Image Dataset for Integrated Crop Health Assessment/Soyabean_UAV-Based_Image_Dataset/Soyabean_Mosaic/"
constantPath = "/home/dragoon/Downloads/MH-SoyaHealthVision An Indian UAV and Leaf Image Dataset for Integrated Crop Health Assessment/Soyabean_UAV-Based_Image_Dataset/Healthy_Soyabean/"
# Define an absolute path for the CSV file
csv_output_path = "/home/dragoon/coding/drone-crop/xgboost/UAVHealthy_Soyabean.csv"

def parse_dji_filename(filename):
    """Parse DJI filename to extract date and time"""
    # DJI filename pattern: DJI_YYYYMMDDHHMMSS_XXXX_X_XXXXXX.jpg
    pattern = r'DJI_(\d{4})(\d{2})(\d{2})(\d{2})(\d{2})(\d{2})_'
    match = re.search(pattern, filename)
    
    if match:
        year, month, day, hour, minute, second = match.groups()
        dt = datetime(int(year), int(month), int(day), int(hour), int(minute), int(second))
        return dt
    return None

def get_image_path(files):
    history = []
    for i in files:
        image_path = constantPath + i

        try:
            # Open image to get basic info
            image = Image.open(image_path)

            print("=" * 60)
            print("IMAGE METADATA EXTRACTION")
            print("=" * 60)
            print(f"Processing file: {i}")

            # Basic image info
            print("BASIC IMAGE INFORMATION:")
            print("-" * 30)
            print(f"Image Format: {image.format}")
            print(f"Image Size: {image.size}")
            print()

            # File system timestamps
            print("FILE SYSTEM TIMESTAMPS:")
            print("-" * 30)
            try:
                stat = os.stat(image_path)
                created_time = datetime.fromtimestamp(stat.st_ctime)
                modified_time = datetime.fromtimestamp(stat.st_mtime)
                accessed_time = datetime.fromtimestamp(stat.st_atime)
                
                print(f"Created                  : {created_time}")
                print(f"Modified                 : {modified_time}")
                print(f"Accessed                 : {accessed_time}")
            except Exception as e:
                print(f"Error getting file timestamps: {e}")
                created_time = modified_time = accessed_time = None

            print()

            # Information from filename
            print("INFORMATION FROM FILENAME:")
            print("-" * 30)
            filename = os.path.basename(image_path)
            print(f"Filename                 : {filename}")

            # Parse date/time from filename
            parsed_datetime = parse_dji_filename(filename)
            if parsed_datetime:
                print(f"Parsed Date/Time         : {parsed_datetime}")
                formatted_date = parsed_datetime.strftime('%A, %B %d, %Y at %I:%M:%S %p')
                print(f"Formatted Date           : {formatted_date}")
            else:
                print("Could not parse date/time from filename")
                parsed_datetime = None
                formatted_date = None
                
            # Create dictionary for CSV writer (for all files, regardless of pattern match)
            dataDict = {
                'Filename': filename,
                'Format': image.format if image.format else "Unknown",
                'Size': str(image.size),
                'Created': created_time,
                'Modified': modified_time,
                'Accessed': accessed_time,
                'Parsed Date/Time': parsed_datetime,
                'Formatted Date': formatted_date
            }
            history.append(dataDict)
                
        except Exception as e:
            print(f"Error processing {i}: {e}")
            continue

    return history

def csvWriter(files):
    try:
        with open(csv_output_path, mode='w', newline='') as csvfile:
            fieldnames = ['Filename', 'Format', 'Size', 'Created', 'Modified', 'Accessed', 'Parsed Date/Time', 'Formatted Date']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            
            for file_data in files:
                writer.writerow(file_data)
        
        print(f"CSV file successfully written to: {csv_output_path}")
        # Check if the file exists
        if os.path.exists(csv_output_path):
            print(f"Verified: File exists with size {os.path.getsize(csv_output_path)} bytes")
        else:
            print(f"Warning: File was not created at {csv_output_path}")
            
    except Exception as e:
        print(f"Error writing CSV file: {e}")

# Main code execution
try:
    # Make sure the directory exists
    if not os.path.exists(constantPath):
        print(f"Error: Directory not found: {constantPath}")
        exit(1)
        
    # Change to the directory where the images are stored
    os.chdir(constantPath)
    files = os.listdir(constantPath)

    # Filter only image files
    image_files = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png', '.tiff', '.bmp'))]
    print(f"Found {len(image_files)} image files")

    if len(image_files) == 0:
        print("No image files found. Please check the directory path.")
        exit(1)

    history = get_image_path(image_files)
    print(f"Processed {len(history)} images successfully")

    if len(history) == 0:
        print("Warning: No image metadata was extracted.")
    else:
        csvWriter(history)
    
except Exception as e:
    print(f"An unexpected error occurred: {e}")