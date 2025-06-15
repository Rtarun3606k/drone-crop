from flask import Blueprint, request, jsonify
import requests
import uuid
import os
import json
from dotenv import load_dotenv
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

translate = Blueprint('translate', __name__)

@translate.route('/translate', methods=['POST'])
def translate_text():
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({"error": "No text provided"}), 400

    text = data['text']
    source_language = data.get('source_language', None)  # Optional: source language
    target_language = data.get('target_language', 'en')  # Default target language is English
    
    try:
        # Get Azure Translator credentials from environment variables
        key = os.getenv('AZURE_TRANSLATOR_API_KEY')
        endpoint = os.getenv('AZURE_TRANSLATOR_API_ENDPOINT')
        location = os.getenv('AZURE_TRANSLATOR_API_LOCATION')
        
        # Log credentials (exclude actual key value for security)
        logger.info(f"Endpoint: {endpoint}")
        logger.info(f"Location: {location}")
        logger.info(f"Key exists: {bool(key)}")
        
        if not key or not endpoint or not location:
            return jsonify({"error": "Translation service configuration is missing"}), 500
        
        # Ensure location doesn't have quotes
        location = location.replace('"', '')
        
        # Set up the API request
        path = '/translate'
        constructed_url = endpoint + path
        
        params = {
            'api-version': '3.0',
            'to': target_language
        }
        
        # Add source language if provided
        if source_language:
            params['from'] = source_language
        
        headers = {
            'Ocp-Apim-Subscription-Key': key,
            'Ocp-Apim-Subscription-Region': location,
            'Content-type': 'application/json',
            'X-ClientTraceId': str(uuid.uuid4())
        }
        
        # Log request details (excluding sensitive information)
        logger.info(f"Request URL: {constructed_url}")
        logger.info(f"Request params: {params}")
        logger.info("Headers sent (excluding key)")
        
        # Request body
        body = [{
            'text': text
        }]
        
        # Make the API request with retry logic
        max_retries = 3
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                response = requests.post(constructed_url, params=params, headers=headers, json=body, timeout=10)
                
                # Log response details
                logger.info(f"Response status: {response.status_code}")
                
                # Check for specific error codes
                if response.status_code == 401:
                    logger.error("Authentication failed: Check your API key and region")
                    return jsonify({
                        "error": "Authentication failed with Azure Translator API. Please check your API key and region.",
                        "status_code": 401,
                        "details": "Verify that your AZURE_TRANSLATOR_API_KEY is correct and AZURE_TRANSLATOR_API_LOCATION matches your resource region."
                    }), 401
                
                if response.status_code == 403:
                    logger.error("Authorization failed: Check your subscription permissions")
                    return jsonify({
                        "error": "Authorization failed with Azure Translator API. Please check your subscription permissions.",
                        "status_code": 403
                    }), 403
                
                # Raise for other status codes
                response.raise_for_status()
                
                translation_result = response.json()
                
                # Process the response
                result = {
                    "original_text": text,
                    "original_language": source_language or "auto-detected",
                    "target_language": target_language,
                    "translated_text": translation_result[0]["translations"][0]["text"]
                }
                
                # If language was auto-detected, include it in the response
                if not source_language and "detectedLanguage" in translation_result[0]:
                    result["detected_language"] = translation_result[0]["detectedLanguage"]["language"]
                    result["detection_confidence"] = translation_result[0]["detectedLanguage"]["score"]
                
                return jsonify(result), 200
                
            except requests.exceptions.RequestException as e:
                retry_count += 1
                logger.error(f"Request failed (attempt {retry_count}/{max_retries}): {str(e)}")
                
                if retry_count >= max_retries:
                    # Try to get response details if available
                    error_details = {}
                    if 'response' in locals():
                        try:
                            error_details = response.json()
                        except:
                            error_details = {"text": response.text[:200] if response.text else "No response body"}
                    
                    logger.error(f"All retries failed. Error: {str(e)}, Details: {error_details}")
                    
                    return jsonify({
                        "error": f"Translation service error: {str(e)}",
                        "status_code": response.status_code if 'response' in locals() else None,
                        "details": error_details
                    }), 500
    
    except Exception as e:
        logger.exception("Unexpected error in translate_text:")
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500


@translate.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "ok"}), 200


@translate.route('/languages', methods=['GET'])
def get_available_languages():
    try:
        # Get Azure Translator credentials from environment variables
        key = os.getenv('AZURE_TRANSLATOR_API_KEY')
        endpoint = os.getenv('AZURE_TRANSLATOR_API_ENDPOINT')
        location = os.getenv('AZURE_TRANSLATOR_API_LOCATION')
        
        if not key or not endpoint or not location:
            return jsonify({"error": "Translation service configuration is missing"}), 500
            
        # Set up the API request for available languages
        path = '/languages'
        constructed_url = endpoint + path
        
        params = {
            'api-version': '3.0',
            'scope': 'translation'  # We're interested in translation languages
        }
        
        headers = {
            'Ocp-Apim-Subscription-Key': key,
            'Ocp-Apim-Subscription-Region': location
        }
        
        response = requests.get(constructed_url, params=params, headers=headers, timeout=10)
        response.raise_for_status()
        
        return jsonify(response.json()), 200
        
    except Exception as e:
        return jsonify({"error": f"Failed to fetch available languages: {str(e)}"}), 500


@translate.route('/debug', methods=['GET'])
def debug_translator_config():
    """
    Debug endpoint to check Azure Translator configuration without making a translation call.
    Only use this in development/troubleshooting - it shows partial API key for verification.
    """
    try:
        # Get Azure Translator credentials from environment variables
        key = os.getenv('AZURE_TRANSLATOR_API_KEY')
        endpoint = os.getenv('AZURE_TRANSLATOR_API_ENDPOINT')
        location = os.getenv('AZURE_TRANSLATOR_API_LOCATION')
        
        # Check if credentials exist
        config = {
            "key_exists": bool(key),
            "key_length": len(key) if key else 0,
            "key_prefix": key[:5] + '...' if key else None,
            "endpoint": endpoint,
            "location": location,
            "dotenv_loaded": os.path.exists('.env')
        }
        
        # Test connection to Azure Translator API without making a translation
        if key and endpoint and location:
            try:
                # Just check if we can connect to the endpoint (languages API doesn't require translation)
                check_url = f"{endpoint}/languages?api-version=3.0"
                
                headers = {
                    'Ocp-Apim-Subscription-Key': key,
                    'Ocp-Apim-Subscription-Region': location.replace('"', '')
                }
                
                response = requests.get(check_url, headers=headers, timeout=5)
                config["connection_status"] = response.status_code
                config["connection_message"] = "Success" if response.status_code == 200 else f"Failed with status {response.status_code}"
                
            except Exception as conn_err:
                config["connection_status"] = "error"
                config["connection_message"] = str(conn_err)
        
        return jsonify(config), 200
        
    except Exception as e:
        return jsonify({"error": f"Debug configuration check failed: {str(e)}"}), 500

