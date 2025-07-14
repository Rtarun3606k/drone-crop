from pymongo import MongoClient
from bson import ObjectId
from typing import Dict, List, Optional, Union
from datetime import datetime

class DescriptionQueryService:
    """
    Service class for querying Description documents using PyMongo
    """
    
    def __init__(self, connection_string: str = "mongodb://localhost:27017/", database_name: str = "droneCrop"):
        """
        Initialize the MongoDB connection
        
        Args:
            connection_string: MongoDB connection string
            database_name: Name of the database
        """
        self.client = MongoClient(connection_string)
        self.db = self.client[database_name]
        self.descriptions_collection = self.db["Description"]
        self.batches_collection = self.db["Batch"]
    
    def get_descriptions_by_batch_id(self, batch_id: str) -> List[Dict]:
        """
        Get all descriptions for a specific batch
        
        Args:
            batch_id: The batch ID to query descriptions for
            
        Returns:
            List of description documents
        """
        try:
            descriptions = list(self.descriptions_collection.find({
                "batchId": batch_id
            }))
            return descriptions
        except Exception as e:
            print(f"Error querying descriptions by batch ID: {e}")
            return []
    
    def get_batch_with_descriptions(self, batch_id: str) -> Dict:
        """
        Get batch information along with its descriptions
        
        Args:
            batch_id: The batch ID to query
            
        Returns:
            Dictionary containing batch info and descriptions
        """
        try:
            # Get batch information
            batch = self.batches_collection.find_one({"_id": ObjectId(batch_id)})
            if not batch:
                return {"error": "Batch not found"}
            
            # Get descriptions for this batch
            descriptions = list(self.descriptions_collection.find({
                "batchId": batch_id
            }))
            
            # Organize descriptions by language
            descriptions_by_lang = {}
            for desc in descriptions:
                lang = desc.get("language", "En")
                descriptions_by_lang[lang] = desc
            
            return {
                "batch": batch,
                "descriptions": descriptions_by_lang,
                "total_descriptions": len(descriptions)
            }
            
        except Exception as e:
            print(f"Error querying batch with descriptions: {e}")
            return {"error": str(e)}
    
    def get_preferred_and_english_descriptions(self, batch_id: str) -> Dict:
        """
        Get both English and preferred language descriptions for a batch
        
        Args:
            batch_id: The batch ID to query
            
        Returns:
            Dictionary with English and preferred language descriptions
        """
        try:
            # First get the batch to know the preferred language
            batch = self.batches_collection.find_one({"_id": ObjectId(batch_id)})
            if not batch:
                return {"error": "Batch not found"}
            
            preferred_language = batch.get("preferredLanguage", "En")
            
            # Query descriptions for both English and preferred language
            descriptions = list(self.descriptions_collection.find({
                "batchId": batch_id,
                "language": {"$in": ["En", preferred_language]}
            }))
            
            # Organize by language
            result = {
                "batch_id": batch_id,
                "preferred_language": preferred_language,
                "english_description": None,
                "preferred_description": None
            }
            
            for desc in descriptions:
                if desc["language"] == "En":
                    result["english_description"] = {
                        "id": str(desc["_id"]),
                        "long_description": desc.get("longDescription"),
                        "short_description": desc.get("shortDescription"),
                        "word_count": desc.get("wordCount"),
                        "confidence": desc.get("confidence"),
                        "created_at": desc.get("createdAt"),
                        "updated_at": desc.get("updatedAt")
                    }
                elif desc["language"] == preferred_language and preferred_language != "En":
                    result["preferred_description"] = {
                        "id": str(desc["_id"]),
                        "long_description": desc.get("longDescription"),
                        "short_description": desc.get("shortDescription"),
                        "word_count": desc.get("wordCount"),
                        "confidence": desc.get("confidence"),
                        "created_at": desc.get("createdAt"),
                        "updated_at": desc.get("updatedAt")
                    }
            
            return result
            
        except Exception as e:
            print(f"Error querying preferred and English descriptions: {e}")
            return {"error": str(e)}
    
    def get_description_by_language(self, batch_id: str, language: str) -> Optional[Dict]:
        """
        Get description for a specific batch and language
        
        Args:
            batch_id: The batch ID
            language: Language code (En, Ta, Hi, Te, Ml, Kn)
            
        Returns:
            Description document or None if not found
        """
        try:
            description = self.descriptions_collection.find_one({
                "batchId": batch_id,
                "language": language
            })
            return description
        except Exception as e:
            print(f"Error querying description by language: {e}")
            return None
    
    def get_incomplete_batches_for_descriptions(self) -> List[Dict]:
        """
        Get all batches that need description generation (isDescCompleted = False)
        
        Returns:
            List of incomplete batches
        """
        try:
            incomplete_batches = list(self.batches_collection.find({
                "isDescCompleted": False,
                "hasExecutionFailed": False
            }))
            return incomplete_batches
        except Exception as e:
            print(f"Error querying incomplete batches: {e}")
            return []
    
    def update_description(self, description_id: str, updates: Dict) -> bool:
        """
        Update a description document
        
        Args:
            description_id: The description ID to update
            updates: Dictionary of fields to update
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Add updatedAt timestamp
            updates["updatedAt"] = datetime.utcnow()
            
            result = self.descriptions_collection.update_one(
                {"_id": ObjectId(description_id)},
                {"$set": updates}
            )
            return result.modified_count > 0
        except Exception as e:
            print(f"Error updating description: {e}")
            return False
    
    def create_description(self, batch_id: str, language: str, long_description: str, 
                          short_description: str, word_count: Optional[int] = None, 
                          confidence: Optional[float] = None) -> Optional[str]:
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
                "batchId": batch_id,
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
            
            result = self.descriptions_collection.insert_one(description_doc)
            return str(result.inserted_id)
        except Exception as e:
            print(f"Error creating description: {e}")
            return None
    
    def get_descriptions_by_criteria(self, criteria: Dict) -> List[Dict]:
        """
        Get descriptions based on custom criteria
        
        Args:
            criteria: MongoDB query criteria
            
        Returns:
            List of matching description documents
        """
        try:
            descriptions = list(self.descriptions_collection.find(criteria))
            return descriptions
        except Exception as e:
            print(f"Error querying descriptions by criteria: {e}")
            return []

# Usage example
if __name__ == "__main__":
    # Initialize the service
    desc_service = DescriptionQueryService()
    
    # Example usage
    batch_id = "your_batch_id_here"
    
    # Get both English and preferred language descriptions
    result = desc_service.get_preferred_and_english_descriptions(batch_id)
    print("Preferred and English descriptions:", result)
    
    # Get all descriptions for a batch
    all_descriptions = desc_service.get_descriptions_by_batch_id(batch_id)
    print("All descriptions:", all_descriptions)
    
    # Get batch with descriptions
    batch_with_desc = desc_service.get_batch_with_descriptions(batch_id)
    print("Batch with descriptions:", batch_with_desc)
    
    # Get incomplete batches
    incomplete = desc_service.get_incomplete_batches_for_descriptions()
    print("Incomplete batches:", len(incomplete))