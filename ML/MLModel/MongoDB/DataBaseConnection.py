from pymongo import MongoClient
from bson.objectid import ObjectId 
from dotenv import load_dotenv
import os
import logging

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
        data = db['Batch'].find({"isCompletedModel": False})
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
    
    :param batch_id: The ID of the batch to update.
    :param status: The new status to set for the batch.
    """
    try:
        if status =='failed':
            result = db['Batch'].update_one(
                {"_id": ObjectId(batch_id)},
                {"$set": {"isCompletedModel": False, "execFailed": True}}
            )
        result = db['Batch'].update_one(
            {"_id": ObjectId(batch_id)},
            {"$set": {"isCompletedModel": True, "execFailed": False}}
        )
        return result.modified_count
    except Exception as e:
        print(f"An error occurred while updating batch status: {e}")
        
        return False
    
if __name__ == "__main__":
    query = db['Batch'].delete_many({"isCompletedModel": False})
    print(f"Deleted {query.deleted_count} incomplete batches from the database.")

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