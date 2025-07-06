import os
import shutil
import sys

# ensure MLModel root is on PYTHONPATH before importing
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from MongoDB.DataBaseConnection import getAllIncompleteBatches
from Routes.Model import  batch_predict



def ModelRunner():
    """
    Fetch all incomplete batches and copy their zip files into data/Zips.
    """
    # ensure destination exists
    # project base
    project_root = '/home/dragoon/coding/drone-crop'
    # target data/Zips folder under MLModel
    zips_dir = os.path.join(project_root, 'ML', 'MLModel', 'data', 'Zips')
    unzip_dir = os.path.join(project_root, 'ML', 'MLModel', 'data', 'unzip')
    os.makedirs(unzip_dir, exist_ok=True)
    os.makedirs(zips_dir, exist_ok=True)
    os.makedirs(project_root+'/predectionResults', exist_ok=True)

    batches = getAllIncompleteBatches()
    if not batches:
        print("No incomplete batches found.")
        return

    for batch in batches:
        # full source path from imagesZipURL
        src = os.path.join(project_root, batch['imagesZipURL'])
        if not os.path.isfile(src):
            print(f"Source file not found: {src}")
            continue

        dst = os.path.join(zips_dir, os.path.basename(src))
        try:
            shutil.copy(src, dst)
            print(f"Copied {os.path.basename(src)} → {dst}")
            os.system(f"unzip -o '{dst}' -d '{unzip_dir}'")
            
            model_path = "/home/dragoon/coding/drone-crop/ML/BNN/optimized_bnn_plant_disease_64x64.pt"            
            output_json = f"/home/dragoon/coding/drone-crop/predectionResults/{batch['sessionId']}.json"

            # Get all jpg files in directory
            image_paths = [os.path.join(unzip_dir, f) for f in os.listdir(unzip_dir) 
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

            # Run batch prediction
            results = batch_predict(image_paths, model_path, num_threads=8, output_json_path=output_json)
            print(f"Batch processing complete. Processed {len(results)} images.")

            os.system(f"rm -rf '{unzip_dir}'/*")  # Clean up unzip directory after processing
            os.system(f"rm -rf '{zips_dir}'/*")  # Clean up unzip directory after processing

        except Exception as e:
            print(f"Failed to copy {src}: {e}")

if __name__ == "__main__":
    ModelRunner()
    print("ModelRunner completed.")