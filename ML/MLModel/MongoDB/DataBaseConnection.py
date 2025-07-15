from pymongo import MongoClient
from bson.objectid import ObjectId 
from dotenv import load_dotenv
import os
import logging
import datetime

# Load environment variables from .env file
load_dotenv()

MongoDBConnection = MongoClient(os.getenv("MONGODB_URI", "mongodb://localhost:27017/"))

db = MongoDBConnection["droneCrop"]

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

print("MongoDB connection established successfully.")

def getAllIncompleteBatches():
    """
    Fetch all incomplete batches from the 'Batch' collection.
    """

    try:
        data = db['Batch'].find({"isModelCompleted": False , "hasExecutionFailed": False})
        coursor = list(data)
        # data = db['Batch'].find()
        print("Fetched incomplete batches successfully.", len(coursor),coursor, "batches found.")
        return coursor
    except Exception as e:
        print(f"An error occurred while fetching incomplete batches: {e}")
        return []
    
def updatebatchStatus(batch_id, status):
    """
    Update the status of a batch in the 'Batch' collection.
    
    :param batch_id: The ID of the batch to update (ObjectId or string).
    :param status: The new status ('completed' or 'failed').
    """
    try:
        # Convert string to ObjectId if necessary
        if isinstance(batch_id, str):
            batch_id = ObjectId(batch_id)
        
        if status == 'failed':
            result = db['Batch'].update_one(
                {"_id": batch_id},
                {"$set": {
                    "isModelCompleted": False, 
                    "hasExecutionFailed": True,
                    # "updatedAt": datetime.utcnow()
                }}
            )
        elif status == 'completed':
            result = db['Batch'].update_one(
                {"_id": batch_id},
                {"$set": {
                    "isModelCompleted": True, 
                    "hasExecutionFailed": False,  # Fixed typo: removed extra space
                    # "updatedAt": datetime.utcnow()
                }}
            )
        else:
            result = db['Batch'].update_one(
                {"_id": batch_id},
                {"$set": {
                    "isModelCompleted": True, 
                    "hasExecutionFailed": False,  # Fixed typo: removed extra space
                   
                }}
            )
            logger.error(f"Invalid status: {status}. Use 'completed' or 'failed'.")
            return False
        
        logger.info(f"Updated batch {batch_id} with status: {status}")
        return result.modified_count > 0
        
    except Exception as e:
        logger.error(f"An error occurred while updating batch status: {e}")
        return False
    
if __name__ == "__main__":
    # query = db['Batch'].delete_many({"isModelCompleted": False})
    # print(f"Deleted {query.deleted_count} incomplete batches from the database.")
    # query = db['Batch'].find({"isModelCompleted": False})
    # print(f"Found {query.count()} incomplete batches in the database.")
    # print("Incomplete batches:", list(query))  # Example usage to ensure the connection works
    print(getAllIncompleteBatches())  # Example usage to ensure the connection works

# data  = getAllIncompleteBatches()
# # print("Function to get all incomplete batches defined successfully.",getAllIncompleteBatches()[0],listLen)
# print("First incomplete batch:", data[0] if data else "No incomplete batches found.")

# print("Fetched incomplete batches successfully.", data, "batches found.")

# updateStatus = updatebatchStatus(data[0]['_id'], False)

# print(f"Batch status updated successfully. {updateStatus} document(s) modified.")
# print("Updated batch:", list(db['Batch'].find()))


# query = db['User'].find()
# # Example usage
# print(list(query))  # Example usage to ensure the connection works

# query = db['Batch'].update_one({"_id": ObjectId("6866119f01db44ded886ea11")}, {"$set": {"name": "pymongo"}})

# print("Update operation completed successfully.")
# print(query.modified_count, "document(s) updated.",query)  # Print the number of documents updated
# query = db['Batch'].find()
# print(list(query))  # Example usage to ensure the connection works


# print(len(list(query)))  # Example usage to ensure the connection works
# print(list(query)[0])  # Example usage to ensure the connection works
# print(get_collection("User"),list(query))  # Example usage to ensure the connection works