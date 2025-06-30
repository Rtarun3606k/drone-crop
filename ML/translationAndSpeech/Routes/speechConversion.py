from flask import Blueprint, request, jsonify, send_file
import os
import logging
import uuid
import time
import io
import threading
import requests
import base64
from dotenv import load_dotenv
import json
import tempfile

# Try to import the Speech SDK, but provide a fallback if it fails
try:
    import azure.cognitiveservices.speech as speechsdk
    SPEECH_SDK_AVAILABLE = True
except (ImportError, RuntimeError):
    SPEECH_SDK_AVAILABLE = False
    logging.warning("Azure Speech SDK not available or failed to initialize. Using REST API fallback.")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

sheech_conversion = Blueprint('speech_conversion', __name__)

# Cache to store audio results and processing status
audio_cache = {}

def generate_speech(text, target_language, voice_name=None, session_id=None):
    """
    Asynchronously generate speech from text using Azure Speech Services
    
    Parameters:
    - text: Text to convert to speech
    - target_language: Target language code (ka, hi, ta, te, ml, en-US, etc.)
    - voice_name: Name of the voice to use (default: determined by target language)
    - session_id: Unique identifier for this conversion request
    
    Returns:
    - session_id: The ID used to retrieve the result later
    """
    if not session_id:
        session_id = str(uuid.uuid4())
    
    # Language to voice mapping, with Emma as default for English
    language_voice_map = {
        "en": "en-US-EmmaMultilingualNeural",
        "en-US": "en-US-EmmaMultilingualNeural",
        "hi": "hi-IN-SwaraNeural",      # Hindi
        "ka": "kn-IN-SapnaNeural",      # Kannada
        "ta": "ta-IN-PallaviNeural",    # Tamil
        "te": "te-IN-ShrutiNeural",     # Telugu
        "ml": "ml-IN-SobhanaNeural"     # Malayalam
    }
    
    # If no voice specified, determine based on language
    if not voice_name:
        # First try exact match
        voice_name = language_voice_map.get(target_language)
        
        if not voice_name:
            # Try language prefix match (e.g., "hi-IN-Female" -> use "hi" language)
            prefix = target_language.split('-')[0] if '-' in target_language else target_language
            voice_name = language_voice_map.get(prefix)
            
            # Default to Emma if no match
            if not voice_name:
                voice_name = language_voice_map["en-US"]
                logger.warning(f"No voice found for language {target_language}, using Emma")
    
    # Initialize status in cache
    audio_cache[session_id] = {
        "status": "processing",
        "progress": 0,
        "result": None,
        "error": None,
        "language": target_language,
        "voice": voice_name
    }
    
    def speech_synthesis_task():
        try:
            # Get configuration from environment variables
            speech_key = os.getenv('AZURE_SPEECH_API_KEY')
            speech_region = os.getenv('AZURE_SPEECH_API_LOCATION')
            
            if not speech_key or not speech_region:
                raise Exception("Speech service configuration is missing")
            
            # Ensure audio directory exists
            audio_dir = os.path.join(os.getcwd(), 'audio')
            os.makedirs(audio_dir, exist_ok=True)
            
            # Use a file in the audio directory with the session ID
            file_name = os.path.join(audio_dir, f"speech_{session_id}.mp3")
            logger.info(f"Will save audio to: {file_name}")

            # REST API method - Using the working implementation from pure_rest_api.py
            logger.info(f"Using REST API for speech synthesis for session {session_id}")
            audio_cache[session_id]["progress"] = 10
            
            # Construct the TTS endpoint URL using the PROVEN working format from pure_rest_api.py
            tts_url = f"https://{speech_region}.tts.speech.microsoft.com/cognitiveservices/v1"
            logger.info(f"Using TTS URL: {tts_url}")
            
            headers = {
                "Ocp-Apim-Subscription-Key": speech_key,
                "Content-Type": "application/ssml+xml",
                "X-Microsoft-OutputFormat": "audio-16khz-128kbitrate-mono-mp3",
                "User-Agent": "DroneAppAPI"
            }
            
            # Create SSML content - properly escape text
            import html
            safe_text = html.escape(text)
            
            # Get language code for SSML
            ssml_lang = target_language
            
            # If language is just a 2-char code, add appropriate region
            lang_region_map = {
                "hi": "hi-IN",
                "ka": "kn-IN",  # Note: Kannada is "kn" in SSML
                "kn": "kn-IN", 
                "ta": "ta-IN",
                "te": "te-IN",
                "ml": "ml-IN"
            }
            
            if len(ssml_lang) == 2:
                ssml_lang = lang_region_map.get(ssml_lang, f"{ssml_lang}-US")
            
            # Using the working SSML format from pure_rest_api.py
            ssml = f"""<speak version='1.0' xml:lang='{ssml_lang}'>
    <voice name='{voice_name}'>
        {safe_text}
    </voice>
</speak>"""
            
            # Log the SSML and URL for debugging
            logger.info(f"Using TTS URL: {tts_url}")
            logger.info(f"Using voice: {voice_name} for language: {target_language}")
            logger.info(f"SSML format: {ssml[:100]}...")
            
            audio_cache[session_id]["progress"] = 30
            
            # Make REST API call
            start_time = time.time()
            response = requests.post(tts_url, headers=headers, data=ssml.encode('utf-8'))
            elapsed_time = time.time() - start_time
            logger.info(f"REST API request completed in {elapsed_time:.2f} seconds")
            
            # Log the response status
            logger.info(f"REST API response status: {response.status_code}")
            if response.status_code != 200:
                logger.error(f"REST API error response: {response.text[:200]}")
            
            audio_cache[session_id]["progress"] = 70
            
            # Check if the request was successful
            if response.status_code == 200:
                # Store the audio data
                audio_data = response.content
                
                audio_cache[session_id] = {
                    "status": "completed",
                    "progress": 100,
                    "result": audio_data,
                    "language": target_language,
                    "voice": voice_name,
                    "file_path": file_name,
                    "error": None
                }
                
                logger.info(f"REST API speech synthesis completed for {session_id}, audio size: {len(audio_data)} bytes")
                
                # Save to file in the audio directory
                try:
                    with open(file_name, "wb") as f:
                        f.write(audio_data)
                    logger.info(f"Audio saved to file: {file_name}")
                except Exception as e:
                    logger.error(f"Error saving audio to file: {str(e)}")
            else:
                error_message = f"REST API request failed with status {response.status_code}: {response.text}"
                logger.error(error_message)
                audio_cache[session_id]["status"] = "failed"
                audio_cache[session_id]["error"] = error_message
                
        except Exception as e:
            logger.exception(f"Error in speech synthesis task for {session_id}")
            audio_cache[session_id]["status"] = "failed"
            audio_cache[session_id]["error"] = str(e)
    
    # Start processing in a separate thread
    thread = threading.Thread(target=speech_synthesis_task)
    thread.daemon = True
    thread.start()
    
    return session_id

@sheech_conversion.route('/text-to-speech', methods=['POST'])
def text_to_speech():
    """
    Endpoint to initiate text-to-speech conversion
    
    Accepts JSON with:
    - text: Text to convert to speech
    - target_language: Target language code (optional, default: en-US)
      Supported languages include: en-US (English), hi (Hindi), ka (Kannada), 
      ta (Tamil), te (Telugu), ml (Malayalam)
    - voice: Voice name (optional, default: determined by language)
    
    Returns:
    - session_id: Use this to poll for results
    """
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({"error": "No text provided"}), 400
        
        text = data['text']
        target_language = data.get('target_language', 'en-US')
        voice = data.get('voice')  # Now optional, determined based on language if not provided
        
        # Check if client expects the session ID with '-uuid' suffix
        append_uuid_suffix = request.args.get('append_uuid_suffix', 'false').lower() == 'true'
        
        # Start the speech synthesis process
        session_id = generate_speech(text, target_language, voice)
        
        # Append '-uuid' suffix if requested for backward compatibility
        response_session_id = f"{session_id}-uuid" if append_uuid_suffix else session_id
        
        return jsonify({
            "session_id": response_session_id,
            "status": "processing",
            "language": target_language,
            "voice": audio_cache[session_id].get("voice", "auto-selected"),
            "message": "Speech synthesis started. Use the session_id to check status and retrieve the result."
        }), 202
        
    except Exception as e:
        logger.exception("Error in text-to-speech endpoint")
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

@sheech_conversion.route('/status/<session_id>', methods=['GET'])
def check_status(session_id):
    """
    Check the status of a text-to-speech conversion
    
    Parameters:
    - session_id: The ID returned from the text-to-speech endpoint
    
    Returns:
    - Status information for the conversion
    """
    try:
        # Remove '-uuid' suffix if present
        clean_session_id = session_id.replace('-uuid', '')
        
        if clean_session_id not in audio_cache:
            return jsonify({"error": "Session not found", "session_id": clean_session_id}), 404
        
        status_info = audio_cache[clean_session_id].copy()
        
        # Remove binary data from the response
        if "result" in status_info:
            status_info["result"] = bool(status_info["result"])
        
        return jsonify({
            "session_id": clean_session_id,
            "status": status_info["status"],
            "progress": status_info["progress"],
            "language": status_info.get("language", "unknown"),
            "voice": status_info.get("voice", "unknown"),
            "error": status_info["error"],
            "ready": status_info["status"] == "completed"
        }), 200
        
    except Exception as e:
        logger.exception("Error checking status")
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

@sheech_conversion.route('/audio/<session_id>', methods=['GET'])
def get_audio(session_id):
    """
    Retrieve the generated audio file
    
    Parameters:
    - session_id: The ID returned from the text-to-speech endpoint
    
    Returns:
    - Audio file as MP3
    """
    try:
        # Remove '-uuid' suffix if present
        clean_session_id = session_id.replace('-uuid', '')
        
        if clean_session_id not in audio_cache:
            return jsonify({"error": "Session not found", "session_id": clean_session_id}), 404
        
        cache_entry = audio_cache[clean_session_id]
        
        if cache_entry["status"] == "processing":
            return jsonify({"error": "Audio generation is still in progress", "progress": cache_entry["progress"]}), 202
        
        if cache_entry["status"] == "failed":
            return jsonify({"error": "Audio generation failed", "details": cache_entry["error"]}), 500
        
        if not cache_entry["result"]:
            return jsonify({"error": "Audio data not available"}), 500
        
        # First check if we have a file path and the file exists
        if "file_path" in cache_entry and os.path.exists(cache_entry["file_path"]):
            return send_file(
                cache_entry["file_path"],
                mimetype="audio/mpeg",
                as_attachment=True,
                download_name=f"speech_{clean_session_id}.mp3"
            )
        
        # Fallback to in-memory audio data if file doesn't exist
        audio_data = io.BytesIO(cache_entry["result"])
        audio_data.seek(0)
        
        return send_file(
            audio_data,
            mimetype="audio/mpeg",
            as_attachment=True,
            download_name=f"speech_{clean_session_id}.mp3"
        )
        
    except Exception as e:
        logger.exception("Error retrieving audio")
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

@sheech_conversion.route('/voices', methods=['GET'])
def list_voices():
    """
    Get available voices for text-to-speech using the REST API
    
    Returns:
    - List of available voices with their properties
    """
    try:
        speech_key = os.getenv('AZURE_SPEECH_API_KEY')
        speech_region = os.getenv('AZURE_SPEECH_API_LOCATION')
        
        if not speech_key or not speech_region:
            return jsonify({"error": "Speech service configuration is missing"}), 500
        
        # Use the working REST API approach to get actual voices
        voices_url = f"https://{speech_region}.tts.speech.microsoft.com/cognitiveservices/v1/voices/list"
        
        headers = {
            "Ocp-Apim-Subscription-Key": speech_key
        }
        
        logger.info(f"Fetching voices from {voices_url}")
        
        try:
            # Use timeout to prevent hanging requests
            response = requests.get(voices_url, headers=headers, timeout=10)
            
            if response.status_code == 200:
                all_voices = response.json()
                
                # Transform the data for the API response
                formatted_voices = []
                
                # Keep track of required voices found
                emma_found = False
                required_languages = {
                    "hi-IN": False,  # Hindi
                    "kn-IN": False,  # Kannada
                    "ta-IN": False,  # Tamil
                    "te-IN": False,  # Telugu
                    "ml-IN": False   # Malayalam
                }
                
                for voice in all_voices:
                    # Check if the voice is neural and has good quality
                    if "Neural" in voice.get("VoiceType", ""):
                        name = voice.get("ShortName", "")
                        locale = voice.get("Locale", "")
                        
                        # Check if this is an Emma voice or one of our required languages
                        is_emma = "Emma" in name
                        is_required_language = any(locale.startswith(lang) for lang in required_languages.keys())
                        
                        if is_emma:
                            emma_found = True
                            
                        if is_required_language:
                            for lang in required_languages.keys():
                                if locale.startswith(lang):
                                    required_languages[lang] = True
                        
                        # Check if this is one of our specifically needed voices
                        is_recommended = (
                            is_emma or
                            name == "hi-IN-SwaraNeural" or  # Hindi
                            name == "kn-IN-SapnaNeural" or  # Kannada
                            name == "ta-IN-PallaviNeural" or  # Tamil
                            name == "te-IN-ShrutiNeural" or  # Telugu
                            name == "ml-IN-SobhanaNeural"   # Malayalam
                        )
                            
                        formatted_voice = {
                            "name": name,
                            "display_name": voice.get("DisplayName", name),
                            "gender": voice.get("Gender", "Unknown"),
                            "locale": locale,
                            "language_name": voice.get("LocaleName", "Unknown"),
                            "voice_type": voice.get("VoiceType", "Unknown"),
                            "recommended": is_recommended,
                            "language_code": locale.split("-")[0] if "-" in locale else locale
                        }
                        
                        formatted_voices.append(formatted_voice)
                
                # Filter by language or voice name if provided in query params
                language_filter = request.args.get('language')
                name_filter = request.args.get('name')
                
                if language_filter:
                    formatted_voices = [v for v in formatted_voices if language_filter.lower() in v["locale"].lower()]
                
                if name_filter:
                    formatted_voices = [v for v in formatted_voices if name_filter.lower() in v["name"].lower()]
                
                # Make sure Emma and our required voices are always included
                fallback_voices = []
                
                if not emma_found:
                    fallback_voices.append({
                        "name": "en-US-EmmaMultilingualNeural",
                        "display_name": "Emma Multilingual",
                        "gender": "Female",
                        "locale": "en-US",
                        "language_name": "English (United States)",
                        "language_code": "en",
                        "voice_type": "Neural",
                        "description": "A friendly, sincere voice with a light-hearted and pleasant tone that's ideal for education",
                        "recommended": True
                    })
                
                required_voice_map = {
                    "hi-IN": {
                        "name": "hi-IN-SwaraNeural",
                        "display_name": "Swara (Hindi)",
                        "gender": "Female",
                        "locale": "hi-IN",
                        "language_name": "Hindi (India)",
                        "language_code": "hi",
                        "voice_type": "Neural",
                        "recommended": True
                    },
                    "kn-IN": {
                        "name": "kn-IN-SapnaNeural",
                        "display_name": "Sapna (Kannada)",
                        "gender": "Female",
                        "locale": "kn-IN",
                        "language_name": "Kannada (India)",
                        "language_code": "ka",
                        "voice_type": "Neural",
                        "recommended": True
                    },
                    "ta-IN": {
                        "name": "ta-IN-PallaviNeural",
                        "display_name": "Pallavi (Tamil)",
                        "gender": "Female",
                        "locale": "ta-IN",
                        "language_name": "Tamil (India)",
                        "language_code": "ta",
                        "voice_type": "Neural",
                        "recommended": True
                    },
                    "te-IN": {
                        "name": "te-IN-ShrutiNeural",
                        "display_name": "Shruti (Telugu)",
                        "gender": "Female",
                        "locale": "te-IN",
                        "language_name": "Telugu (India)",
                        "language_code": "te",
                        "voice_type": "Neural",
                        "recommended": True
                    },
                    "ml-IN": {
                        "name": "ml-IN-SobhanaNeural",
                        "display_name": "Sobhana (Malayalam)",
                        "gender": "Female",
                        "locale": "ml-IN",
                        "language_name": "Malayalam (India)",
                        "language_code": "ml",
                        "voice_type": "Neural",
                        "recommended": True
                    }
                }
                
                # Add any missing required voices
                for lang, found in required_languages.items():
                    if not found:
                        fallback_voices.append(required_voice_map[lang])
                
                # Add required voices at the beginning for emphasis
                formatted_voices = fallback_voices + formatted_voices
                
                # Create a summary of supported languages
                supported_languages = [
                    {"code": "en", "name": "English", "voices": [v for v in formatted_voices if v["locale"].startswith("en")]},
                    {"code": "hi", "name": "Hindi", "voices": [v for v in formatted_voices if v["locale"].startswith("hi")]},
                    {"code": "ka", "name": "Kannada", "voices": [v for v in formatted_voices if v["locale"].startswith("kn")]},
                    {"code": "ta", "name": "Tamil", "voices": [v for v in formatted_voices if v["locale"].startswith("ta")]},
                    {"code": "te", "name": "Telugu", "voices": [v for v in formatted_voices if v["locale"].startswith("te")]},
                    {"code": "ml", "name": "Malayalam", "voices": [v for v in formatted_voices if v["locale"].startswith("ml")]}
                ]
                
                return jsonify({
                    "voices": formatted_voices,
                    "total": len(formatted_voices),
                    "supported_languages": supported_languages,
                    "source": "Azure Speech REST API",
                    "timestamp": time.time()
                }), 200
                
            else:
                logger.error(f"Failed to retrieve voices: {response.status_code}, {response.text}")
                
                # Fall back to hardcoded required voices
                fallback_voices = [
                    {
                        "name": "en-US-EmmaMultilingualNeural",
                        "display_name": "Emma Multilingual",
                        "gender": "Female",
                        "locale": "en-US",
                        "language_name": "English (United States)",
                        "language_code": "en",
                        "description": "A friendly, sincere voice with a light-hearted and pleasant tone that's ideal for education",
                        "voice_type": "Neural",
                        "recommended": True,
                        "fallback": True
                    },
                    {
                        "name": "hi-IN-SwaraNeural",
                        "display_name": "Swara (Hindi)",
                        "gender": "Female",
                        "locale": "hi-IN",
                        "language_name": "Hindi (India)",
                        "language_code": "hi",
                        "voice_type": "Neural",
                        "recommended": True,
                        "fallback": True
                    },
                    {
                        "name": "kn-IN-SapnaNeural",
                        "display_name": "Sapna (Kannada)",
                        "gender": "Female",
                        "locale": "kn-IN",
                        "language_name": "Kannada (India)",
                        "language_code": "ka",
                        "voice_type": "Neural",
                        "recommended": True,
                        "fallback": True
                    },
                    {
                        "name": "ta-IN-PallaviNeural",
                        "display_name": "Pallavi (Tamil)",
                        "gender": "Female",
                        "locale": "ta-IN",
                        "language_name": "Tamil (India)",
                        "language_code": "ta",
                        "voice_type": "Neural",
                        "recommended": True,
                        "fallback": True
                    },
                    {
                        "name": "te-IN-ShrutiNeural",
                        "display_name": "Shruti (Telugu)",
                        "gender": "Female",
                        "locale": "te-IN",
                        "language_name": "Telugu (India)",
                        "language_code": "te",
                        "voice_type": "Neural",
                        "recommended": True,
                        "fallback": True
                    },
                    {
                        "name": "ml-IN-SobhanaNeural",
                        "display_name": "Sobhana (Malayalam)",
                        "gender": "Female",
                        "locale": "ml-IN",
                        "language_name": "Malayalam (India)",
                        "language_code": "ml",
                        "voice_type": "Neural",
                        "recommended": True,
                        "fallback": True
                    }
                ]
                
                # Add supported languages summary using fallback voices
                supported_languages = [
                    {"code": "en", "name": "English", "voices": [v for v in fallback_voices if v["locale"].startswith("en")]},
                    {"code": "hi", "name": "Hindi", "voices": [v for v in fallback_voices if v["locale"].startswith("hi")]},
                    {"code": "ka", "name": "Kannada", "voices": [v for v in fallback_voices if v["locale"].startswith("kn")]},
                    {"code": "ta", "name": "Tamil", "voices": [v for v in fallback_voices if v["locale"].startswith("ta")]},
                    {"code": "te", "name": "Telugu", "voices": [v for v in fallback_voices if v["locale"].startswith("te")]},
                    {"code": "ml", "name": "Malayalam", "voices": [v for v in fallback_voices if v["locale"].startswith("ml")]}
                ]
                
                return jsonify({
                    "voices": fallback_voices,
                    "total": len(fallback_voices),
                    "supported_languages": supported_languages,
                    "source": "Fallback (API Error)",
                    "error_code": response.status_code,
                    "timestamp": time.time(),
                    "note": "Using fallback voices due to API error"
                }), 200
                
        except Exception as api_error:
            logger.exception("Error fetching voices from API")
            
            # Fall back to hardcoded required voices with the same structure
            fallback_voices = [
                {
                    "name": "en-US-EmmaMultilingualNeural",
                    "display_name": "Emma Multilingual",
                    "gender": "Female",
                    "locale": "en-US",
                    "language_name": "English (United States)",
                    "language_code": "en",
                    "voice_type": "Neural",
                    "recommended": True,
                    "fallback": True
                },
                {
                    "name": "hi-IN-SwaraNeural",
                    "display_name": "Swara (Hindi)",
                    "gender": "Female",
                    "locale": "hi-IN",
                    "language_name": "Hindi (India)",
                    "language_code": "hi",
                    "voice_type": "Neural",
                    "recommended": True,
                    "fallback": True
                },
                {
                    "name": "kn-IN-SapnaNeural",
                    "display_name": "Sapna (Kannada)",
                    "gender": "Female",
                    "locale": "kn-IN",
                    "language_name": "Kannada (India)",
                    "language_code": "ka",
                    "voice_type": "Neural",
                    "recommended": True,
                    "fallback": True
                },
                {
                    "name": "ta-IN-PallaviNeural",
                    "display_name": "Pallavi (Tamil)",
                    "gender": "Female",
                    "locale": "ta-IN",
                    "language_name": "Tamil (India)",
                    "language_code": "ta",
                    "voice_type": "Neural",
                    "recommended": True,
                    "fallback": True
                },
                {
                    "name": "te-IN-ShrutiNeural",
                    "display_name": "Shruti (Telugu)",
                    "gender": "Female",
                    "locale": "te-IN",
                    "language_name": "Telugu (India)",
                    "language_code": "te",
                    "voice_type": "Neural",
                    "recommended": True,
                    "fallback": True
                },
                {
                    "name": "ml-IN-SobhanaNeural",
                    "display_name": "Sobhana (Malayalam)",
                    "gender": "Female",
                    "locale": "ml-IN",
                    "language_name": "Malayalam (India)",
                    "language_code": "ml",
                    "voice_type": "Neural",
                    "recommended": True,
                    "fallback": True
                }
            ]
            
            # Add supported languages summary using fallback voices
            supported_languages = [
                {"code": "en", "name": "English", "voices": [v for v in fallback_voices if v["locale"].startswith("en")]},
                {"code": "hi", "name": "Hindi", "voices": [v for v in fallback_voices if v["locale"].startswith("hi")]},
                {"code": "ka", "name": "Kannada", "voices": [v for v in fallback_voices if v["locale"].startswith("kn")]},
                {"code": "ta", "name": "Tamil", "voices": [v for v in fallback_voices if v["locale"].startswith("ta")]},
                {"code": "te", "name": "Telugu", "voices": [v for v in fallback_voices if v["locale"].startswith("te")]},
                {"code": "ml", "name": "Malayalam", "voices": [v for v in fallback_voices if v["locale"].startswith("ml")]}
            ]
            
            return jsonify({
                "voices": fallback_voices,
                "total": len(fallback_voices),
                "supported_languages": supported_languages,
                "source": "Fallback (Exception)",
                "error": str(api_error),
                "timestamp": time.time(),
                "note": "Using fallback voices due to API error"
            }), 200
        
    except Exception as e:
        logger.exception("Error retrieving voices")
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

@sheech_conversion.route('/debug', methods=['GET'])
def debug():
    """
    Debug endpoint to check Azure Speech Services configuration and connectivity
    """
    try:
        speech_key = os.getenv('AZURE_SPEECH_API_KEY')
        speech_region = os.getenv('AZURE_SPEECH_API_LOCATION')
        
        # Use the proven working endpoint format
        tts_endpoint = f"https://{speech_region}.tts.speech.microsoft.com/"
        
        config = {
            "key_exists": bool(speech_key),
            "key_length": len(speech_key) if speech_key else 0,
            "key_prefix": speech_key[:5] + '...' + speech_key[-5:] if speech_key else None,
            "region": speech_region,
            "tts_endpoint": tts_endpoint,
            "dotenv_loaded": os.path.exists('.env'),
            "speech_sdk_available": SPEECH_SDK_AVAILABLE,
            "implementation": "REST API only",  # Always use REST API since it's working
            "system_info": {
                "os": os.name,
                "platform": os.sys.platform,
                "python_version": os.sys.version
            },
            "timestamp": time.time(),
            "utc_time": time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())
        }
        
        # Test TTS REST API connection with voices list endpoint
        if speech_key and speech_region:
            try:
                # Test 1: Voices API
                voices_url = f"{tts_endpoint}cognitiveservices/v1/voices/list"
                
                headers = {
                    'Ocp-Apim-Subscription-Key': speech_key
                }
                
                logger.info(f"Testing voices API connection: {voices_url}")
                start_time = time.time()
                voices_response = requests.get(voices_url, headers=headers, timeout=10)
                elapsed_time = time.time() - start_time
                
                voices_count = 0
                emma_found = False
                if voices_response.status_code == 200:
                    try:
                        voices_data = voices_response.json()
                        voices_count = len(voices_data)
                        emma_found = any("Emma" in voice.get("ShortName", "") for voice in voices_data)
                    except Exception as e:
                        logger.error(f"Error parsing voices response: {str(e)}")
                
                config["voices_api_test"] = {
                    "status_code": voices_response.status_code,
                    "success": voices_response.status_code == 200,
                    "response_time_seconds": round(elapsed_time, 2),
                    "endpoint": voices_url,
                    "voice_count": voices_count,
                    "emma_found": emma_found
                }
                
                # Test 2: TTS Synthesis API
                tts_url = f"{tts_endpoint}cognitiveservices/v1"
                
                headers = {
                    "Ocp-Apim-Subscription-Key": speech_key,
                    "Content-Type": "application/ssml+xml",
                    "X-Microsoft-OutputFormat": "audio-16khz-128kbitrate-mono-mp3",
                    "User-Agent": "DroneAppDebug"
                }
                
                # Small SSML for testing
                test_ssml = """<speak version='1.0' xml:lang='en-US'>
    <voice name='en-US-EmmaMultilingualNeural'>
        This is a test.
    </voice>
</speak>"""
                
                logger.info(f"Testing TTS API connection: {tts_url}")
                start_time = time.time()
                tts_response = requests.post(tts_url, headers=headers, data=test_ssml.encode('utf-8'), timeout=10)
                elapsed_time = time.time() - start_time
                
                audio_size = 0
                if tts_response.status_code == 200:
                    audio_size = len(tts_response.content)
                
                config["tts_api_test"] = {
                    "status_code": tts_response.status_code,
                    "success": tts_response.status_code == 200,
                    "response_time_seconds": round(elapsed_time, 2),
                    "endpoint": tts_url,
                    "audio_size_bytes": audio_size,
                    "audio_generated": audio_size > 0,
                    "ssml_sample": test_ssml
                }
                
                # Overall status
                config["overall_status"] = {
                    "all_tests_passed": voices_response.status_code == 200 and tts_response.status_code == 200,
                    "voices_api_working": voices_response.status_code == 200,
                    "tts_api_working": tts_response.status_code == 200,
                    "recommendation": "REST API is working correctly" if (voices_response.status_code == 200 and tts_response.status_code == 200) else "REST API is experiencing issues"
                }
                
            except requests.exceptions.Timeout:
                logger.error("Timeout error when testing API connections")
                config["api_connection_error"] = {
                    "status": "timeout",
                    "error": "Connection timed out when testing API"
                }
            
            except requests.exceptions.ConnectionError as conn_err:
                logger.exception("Connection error when testing API")
                config["api_connection_error"] = {
                    "status": "connection_error",
                    "error": f"Connection failed: {str(conn_err)}"
                }
                
            except Exception as other_err:
                logger.exception("Error testing API connections")
                config["api_connection_error"] = {
                    "status": "error",
                    "error": str(other_err)
                }
        
        return jsonify(config), 200
        
    except Exception as e:
        logger.exception("Error in debug endpoint")
        return jsonify({
            "error": f"An error occurred: {str(e)}",
            "timestamp": time.time(),
            "utc_time": time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())
        }), 500

@sheech_conversion.route('/health', methods=['GET'])
def health_check():
    """
    Health check endpoint
    """
    try:
        speech_key = os.getenv('AZURE_SPEECH_API_KEY')
        speech_region = os.getenv('AZURE_SPEECH_API_LOCATION')
        
        health_status = {
            "status": "ok", 
            "service": "Speech Conversion API",
            "implementation": "REST API",
            "message": "Speech conversion service is running",
            "config_loaded": bool(speech_key and speech_region),
            "timestamp": time.time(),
            "utc_time": time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())
        }
        
        # Quick API availability check if detailed is requested
        if request.args.get('detailed') == 'true' and speech_key and speech_region:
            try:
                # Just do a simple request to the voices endpoint with a short timeout
                voices_url = f"https://{speech_region}.tts.speech.microsoft.com/cognitiveservices/v1/voices/list"
                headers = {'Ocp-Apim-Subscription-Key': speech_key}
                
                response = requests.get(voices_url, headers=headers, timeout=2)
                
                health_status["api_check"] = {
                    "available": response.status_code == 200,
                    "status_code": response.status_code
                }
            except Exception:
                health_status["api_check"] = {
                    "available": False,
                    "error": "Could not connect to Speech Services API"
                }
        
        return jsonify(health_status), 200
        
    except Exception as e:
        logger.exception("Error in health check")
        return jsonify({
            "status": "error",
            "message": f"Error during health check: {str(e)}",
            "timestamp": time.time()
        }), 500