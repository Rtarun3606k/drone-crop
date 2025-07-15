from pymongo import MongoClient
from bson.objectid import ObjectId 
from dotenv import load_dotenv
import os
import logging
from datetime import datetime

# Load environment variables from .env file
load_dotenv()

MongoDBConnection = MongoClient(os.getenv("MONGODB_URI", "mongodb://localhost:27017/"))

db = MongoDBConnection["droneCrop"]

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

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

def updatebatchStatus(batch_id, status, audioURL=None):
    """
    Update batch status.
    """
    try:
        if status == 'failed':
            result = db['Batch'].update_one(
                {"_id": ObjectId(batch_id)},
                {"$set": {
                    "isAudioCompleted": False, 
                    "hasExecutionFailed": True,
                    "updatedAt": datetime.utcnow()
                }}
            )
        else:
            result = db['Batch'].update_one(
                {"_id": ObjectId(batch_id)},
                {"$set": {
                    "isAudioCompleted": True, 
                    "hasExecutionFailed": False,
                    "audioURL": audioURL,
                    "updatedAt": datetime.utcnow()
                }}
            )
        
        return result.modified_count > 0
    except Exception as e:
        logger.error(f"Error updating batch status: {e}")
        return False

if __name__ == "__main__":
    batches = getAllIncompleteBatches()
    print(f"Found {len(batches)} batches")
    if batches:
        print("Sample:", {
            "_id": str(batches[0]["_id"]),
            "preferredLanguage": batches[0]["preferredLanguage"], 
            "descriptions_count": batches[0]["descriptions"]
        })