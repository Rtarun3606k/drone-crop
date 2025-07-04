
import os
import shutil
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from MongoDB.DataBaseConnection import getAllIncompleteBatches


# Add the parent directory (MLModel) to Python path
root_dir = '/home/dragoon/coding/drone-crop/'


def ModelRunner():    
    """
    This function is a placeholder for the model runner logic.
    It should be implemented to run the machine learning model.
    """
    incompletebatches = getAllIncompleteBatches()
    print("Incomplete batches fetched:", incompletebatches)

    if incompletebatches:
        print("Incomplete batches found:", incompletebatches)
        for batch in incompletebatches:
            
            shutil.copy(root_dir+batch['imagesZipURL'],'Zips')


if __name__ == "__main__":
    # Ensure the Zips directory exists
    # if not os.path.exists('Zips'):
    os.makedirs(root_dir+'ML/MLModel/data', exist_ok=True)
    
    # Run the model runner function
    ModelRunner()