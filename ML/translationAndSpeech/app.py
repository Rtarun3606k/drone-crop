from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# add Routes
from Routes.transalate import translate
from Routes.speechConversion import sheech_conversion 
from Routes.Summerizer import summerisizer 


# Initialize Flask app and CORS
app = Flask(__name__)
CORS(app)


@app.route('/', methods=['GET'])
def index():
    # Log environment variables (partial key for security)
    translate_key = os.getenv('AZURE_TRANSLATOR_API_KEY')
    speech_key = os.getenv('AZURE_SPEECH_API_KEY')
    logger.info(f"Translation API Key exists: {bool(translate_key)}")
    logger.info(f"Translation API Location: {os.getenv('AZURE_TRANSLATOR_API_LOCATION')}")
    logger.info(f"Translation API Endpoint: {os.getenv('AZURE_TRANSLATOR_API_ENDPOINT')}")
    logger.info(f"Speech API Key exists: {bool(speech_key)}")
    logger.info(f"Speech API Location: {os.getenv('AZURE_SPEECH_API_LOCATION')}")
    logger.info(f"Speech API Endpoint: {os.getenv('AZURE_SPEECH_API_ENDPOINT')}")
    
    return jsonify({
        "message": "Welcome to the Flask API!",
        "services": {
            "translation": "/translate",
            "speech": "/speech"
        }
    })


# Register the blueprint
app.register_blueprint(translate, url_prefix='/translate')
app.register_blueprint(sheech_conversion, url_prefix='/speech')
app.register_blueprint(summerisizer, url_prefix='/summery')
