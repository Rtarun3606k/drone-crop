from pymongo import MongoClient
from bson.objectid import ObjectId 
from dotenv import load_dotenv
from datetime import datetime
import os
import logging

# Load environment variables from .env file
load_dotenv()

MongoDBConnection = MongoClient(os.getenv("MONGODB_URI", "mongodb://localhost:27017/"))

db = MongoDBConnection["droneCrop"]

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

predection_path = '/home/dragoon/coding/drone-crop/predectionResults'

print("MongoDB connection established successfully.")

def getAllIncompleteBatches():
    """
    Fetch all incomplete batches from the 'Batch' collection.
    """

    try:
        data = db['Batch'].find({"isDescCompleted": False, "isModelCompleted": True, "hasExecutionFailed": False})
        coursor = list(data)
        # data = db['Batch'].find()
        print("Fetched incomplete batches successfully.", len(coursor), "batches found.")
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
                {"$set": {"isDescCompleted": False, "hasExecutionFailed": True}}
            )
        result = db['Batch'].update_one(
            {"_id": ObjectId(batch_id)},
            {"$set": {"isDescCompleted": True, "hasExecutionFailed": False}}
        )
        return result.modified_count
    except Exception as e:
        print(f"An error occurred while updating batch status: {e}")
        
        return False
    
def uploadResponseToDB(id, response):
    """
    Upload the response to the database for a specific batch.
    
    :param id: The ID of the batch to update.
    :param response: The response data to upload.
    """
    try:
        result = db['Batch'].update_one(
            {"_id": ObjectId(id)},
            {"$set": {"description": response}}
        )
        return result.modified_count
    except Exception as e:
        print(f"An error occurred while uploading response to DB: {e}")
        return False
    

def uploadPrefferedLanguageResponseToDB(id, response):
    """
    Upload the preferred language response to the database for a specific batch.
    
    :param id: The ID of the batch to update.
    :param response: The preferred language response data to upload.
    """
    try:
        result = db['Batch'].update_one(
            {"_id": ObjectId(id)},
            {"$set": {"langDescription": response}}
        )
        return result.modified_count
    except Exception as e:
        print(f"An error occurred while uploading preferred language response to DB: {e}")
        return False
    


def create_description(batch_id, language, long_description, 
                        short_description, word_count = None, 
                        confidence = None):
    """
    Create a new description document
    
    Args:
        batch_id: The batch ID
        language: Language code
        long_description: Long form description
        short_description: Short form description
        word_count: Optional word count
        confidence: Optional AI confidence score
        
    Returns:
        The created description ID or None if failed
    """
    try:
        description_doc = {
            "batchId": ObjectId(batch_id),
            "language": language,
            "longDescription": long_description,
            "shortDescription": short_description,
            "createdAt": datetime.utcnow(),
            "updatedAt": datetime.utcnow()
        }
        
        if word_count is not None:
            description_doc["wordCount"] = word_count
        if confidence is not None:
            description_doc["confidence"] = confidence
        
        result = db["Description"].insert_one(description_doc)
        return str(result.inserted_id)
    except Exception as e:
        print(f"Error creating description: {e}")
        return None

if __name__ == "__main__":
    # query = db['Batch'].delete_many({"isCompletedModel": False})
    # query = db['Batch'].delete_many({"isCompletedModel": False})
    query = getAllIncompleteBatches()
    # print(f"Deleted {query.deleted_count} incomplete batches from the database.")

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