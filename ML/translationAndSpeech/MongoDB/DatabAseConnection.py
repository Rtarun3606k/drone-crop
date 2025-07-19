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

#predection_path = '/home/dragoon/coding/drone-crop/predectionResults'
predection_path = '/home/bruh/Documents/drone-crop/predectionResults'

print("MongoDB connection established successfully.")

def getAllIncompleteBatches():
    """
    Fetch all incomplete batches with their descriptions.
    Returns list of dictionaries with _id, preferredLanguage, and descriptions array.
    """
    try:
        pipeline = [
            {
                "$match": {
                    "isAudioCompleted": False,
                    "isDescCompleted": True,
                    "hasExecutionFailed": False
                }
            },
            {
                "$lookup": {
                    "from": "Description",
                    "let": {"batchId": "$_id"},
                    "pipeline": [
                        {
                            "$match": {
                                "$expr": {
                                    "$eq": [
                                        {"$toObjectId": "$batchId"},
                                        "$$batchId"
                                    ]
                                }
                            }
                        }
                    ],
                    "as": "descriptions"
                }
            },
            {
                "$match": {
                    "descriptions": {"$ne": []}
                }
            },
            {
                "$project": {
                    "_id": 1,
                    "preferredLanguage": 1,
                    "sessionId": 1,
                    "descriptions": {
                        "$map": {
                            "input": "$descriptions",
                            "as": "desc",
                            "in": {
                                "language": "$$desc.language",
                                "longDescription": "$$desc.longDescription",
                                "shortDescription": "$$desc.shortDescription"
                            }
                        }
                    }
                }
            }
        ]
        
        cursor = list(db['Batch'].aggregate(pipeline))
        return cursor
    except Exception as e:
        logger.error(f"Error fetching incomplete batches: {e}")
        return []
    
    
def updatebatchStatus(batch_id, status,audioURL=None):
    """
    Update the status of a batch in the 'Batch' collection.
    
    :param batch_id: The ID of the batch to update.
    :param status: The new status to set for the batch.
    """
    try:
        if status =='failed':
            result = db['Batch'].update_one(
                {"_id": ObjectId(batch_id)},
                {"$set": {"isAudioCompleted": False, "hasExecutionFailed": True}}
            )
        result = db['Batch'].update_one(
            {"_id": ObjectId(batch_id)},
            {"$set": {"isAudioCompleted": True, "hasExecutionFailed": False,'audioURL':audioURL if audioURL else None}}
        )
        return result.modified_count
    except Exception as e:
        print(f"An error occurred while updating batch status: {e}")
        
        return False
    
def findDescByID(batch_id):
    """
    Find descriptions by batch ID.
    
    :param batch_id: The ID of the batch to find descriptions for.
    :return: List of descriptions for the given batch ID.
    """
    try:
        descriptions = list(db['Description'].find({"batchId": ObjectId(batch_id)}))
        return descriptions
    except Exception as e:
        print(f"An error occurred while fetching descriptions: {e}")
        return []

    

    
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