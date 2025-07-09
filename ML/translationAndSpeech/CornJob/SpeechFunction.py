"""
Speech Function Module for Text Translation and Speech Synthesis

This module provides functionality to translate text and convert it to speech,
saving the audio to a specified file path.
"""

import os
import uuid
import time
import logging
import requests
import html
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


def translate_and_synthesize_speech(text, target_language, output_file_path, source_language=None, voice_name=None):
    """
    Translate text to target language and convert it to speech, saving to specified file path.
    
    Parameters:
    - text (str): Text to translate and convert to speech
    - target_language (str): Target language code (e.g., 'hi', 'ka', 'ta', 'te', 'ml', 'en')
    - output_file_path (str): Full path where the audio file should be saved
    - source_language (str, optional): Source language code for translation (auto-detect if None)
    - voice_name (str, optional): Specific voice to use (auto-selected based on language if None)
    
    Returns:
    - dict: Result containing success status, file path, and any error messages
    """
    
    try:
        # Step 1: Translate the text if target language is not English
        translated_text = text
        translation_result = None
        
        if target_language not in ['en', 'en-US']:
            logger.info(f"Translating text to {target_language}")
            translation_result = translate_text(text, target_language, source_language)
            
            if not translation_result['success']:
                return {
                    'success': False,
                    'error': f"Translation failed: {translation_result['error']}",
                    'file_path': None
                }
            
            translated_text = translation_result['translated_text']
            logger.info(f"Translation successful: {translated_text[:50]}...")
        
        # Step 2: Convert translated text to speech
        logger.info(f"Converting text to speech in {target_language}")
        speech_result = text_to_speech(translated_text, target_language, output_file_path, voice_name)
        
        if not speech_result['success']:
            return {
                'success': False,
                'error': f"Speech synthesis failed: {speech_result['error']}",
                'file_path': None
            }
        
        # Return success result
        return {
            'success': True,
            'file_path': speech_result['file_path'],
            'original_text': text,
            'translated_text': translated_text,
            'target_language': target_language,
            'voice_used': speech_result['voice_used'],
            'translation_detected_language': translation_result['detected_language'] if translation_result else None,
            'audio_size_bytes': speech_result['audio_size_bytes'],
            'message': f"Successfully translated and synthesized speech to {output_file_path}"
        }
        
    except Exception as e:
        logger.exception("Error in translate_and_synthesize_speech")
        return {
            'success': False,
            'error': f"Unexpected error: {str(e)}",
            'file_path': None
        }


def translate_text(text, target_language, source_language=None):
    """
    Translate text using Azure Translator API
    
    Parameters:
    - text (str): Text to translate
    - target_language (str): Target language code
    - source_language (str, optional): Source language code (auto-detect if None)
    
    Returns:
    - dict: Translation result with success status and translated text
    """
    try:
        # Get Azure Translator credentials
        key = os.getenv('AZURE_TRANSLATOR_API_KEY')
        endpoint = os.getenv('AZURE_TRANSLATOR_API_ENDPOINT')
        location = os.getenv('AZURE_TRANSLATOR_API_LOCATION')
        
        if not key or not endpoint or not location:
            return {
                'success': False,
                'error': "Translation service configuration is missing",
                'translated_text': None,
                'detected_language': None
            }
        
        # Clean location string
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
        
        # Request body
        body = [{'text': text}]
        
        # Make the API request
        response = requests.post(constructed_url, params=params, headers=headers, json=body, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            if result and len(result) > 0 and 'translations' in result[0]:
                translated_text = result[0]['translations'][0]['text']
                detected_language = result[0].get('detectedLanguage', {}).get('language', 'unknown')
                
                logger.info(f"Translation successful: {text[:30]}... -> {translated_text[:30]}...")
                
                return {
                    'success': True,
                    'translated_text': translated_text,
                    'detected_language': detected_language,
                    'error': None
                }
            else:
                return {
                    'success': False,
                    'error': "Invalid response format from translation API",
                    'translated_text': None,
                    'detected_language': None
                }
        else:
            error_msg = f"Translation API error: {response.status_code} - {response.text}"
            logger.error(error_msg)
            return {
                'success': False,
                'error': error_msg,
                'translated_text': None,
                'detected_language': None
            }
            
    except Exception as e:
        logger.exception("Error in translate_text")
        return {
            'success': False,
            'error': f"Translation error: {str(e)}",
            'translated_text': None,
            'detected_language': None
        }


def text_to_speech(text, target_language, output_file_path, voice_name=None):
    """
    Convert text to speech using Azure Speech Services and save to file
    
    Parameters:
    - text (str): Text to convert to speech
    - target_language (str): Target language code
    - output_file_path (str): Full path where the audio file should be saved
    - voice_name (str, optional): Specific voice to use
    
    Returns:
    - dict: Speech synthesis result with success status and file path
    """
    try:
        # Get Azure Speech credentials
        speech_key = os.getenv('AZURE_SPEECH_API_KEY')
        speech_region = os.getenv('AZURE_SPEECH_API_LOCATION')
        
        if not speech_key or not speech_region:
            return {
                'success': False,
                'error': "Speech service configuration is missing",
                'file_path': None,
                'voice_used': None,
                'audio_size_bytes': 0
            }
        
        # Language to voice mapping
        language_voice_map = {
            "en": "en-US-EmmaMultilingualNeural",
            "en-US": "en-US-EmmaMultilingualNeural",
            "hi": "hi-IN-SwaraNeural",      # Hindi
            "ka": "kn-IN-SapnaNeural",      # Kannada
            "ta": "ta-IN-PallaviNeural",    # Tamil
            "te": "te-IN-ShrutiNeural",     # Telugu
            "ml": "ml-IN-SobhanaNeural"     # Malayalam
        }
        
        # Determine voice if not specified
        if not voice_name:
            # First try exact match
            voice_name = language_voice_map.get(target_language)
            
            if not voice_name:
                # Try language prefix match
                prefix = target_language.split('-')[0] if '-' in target_language else target_language
                voice_name = language_voice_map.get(prefix)
                
                # Default to Emma if no match
                if not voice_name:
                    voice_name = language_voice_map["en-US"]
                    logger.warning(f"No voice found for language {target_language}, using Emma")
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
        
        # Construct the TTS endpoint URL
        tts_url = f"https://{speech_region}.tts.speech.microsoft.com/cognitiveservices/v1"
        
        headers = {
            "Ocp-Apim-Subscription-Key": speech_key,
            "Content-Type": "application/ssml+xml",
            "X-Microsoft-OutputFormat": "audio-16khz-128kbitrate-mono-mp3",
            "User-Agent": "DroneAppAPI"
        }
        
        # Create SSML content - properly escape text
        safe_text = html.escape(text)
        
        # Get language code for SSML
        ssml_lang = target_language
        
        # Map language codes for SSML
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
        
        # Create SSML
        ssml = f"""<speak version='1.0' xml:lang='{ssml_lang}'>
    <voice name='{voice_name}'>
        {safe_text}
    </voice>
</speak>"""
        
        logger.info(f"Using TTS URL: {tts_url}")
        logger.info(f"Using voice: {voice_name} for language: {target_language}")
        
        # Make REST API call
        start_time = time.time()
        response = requests.post(tts_url, headers=headers, data=ssml.encode('utf-8'), timeout=30)
        elapsed_time = time.time() - start_time
        
        logger.info(f"REST API request completed in {elapsed_time:.2f} seconds")
        logger.info(f"REST API response status: {response.status_code}")
        
        # Check if the request was successful
        if response.status_code == 200:
            # Save the audio data to file
            audio_data = response.content
            
            with open(output_file_path, "wb") as f:
                f.write(audio_data)
            
            logger.info(f"Audio saved to file: {output_file_path}, size: {len(audio_data)} bytes")
            
            return {
                'success': True,
                'file_path': output_file_path,
                'voice_used': voice_name,
                'audio_size_bytes': len(audio_data),
                'error': None
            }
        else:
            error_message = f"REST API request failed with status {response.status_code}: {response.text}"
            logger.error(error_message)
            return {
                'success': False,
                'error': error_message,
                'file_path': None,
                'voice_used': voice_name,
                'audio_size_bytes': 0
            }
            
    except Exception as e:
        logger.exception("Error in text_to_speech")
        return {
            'success': False,
            'error': f"Speech synthesis error: {str(e)}",
            'file_path': None,
            'voice_used': voice_name if 'voice_name' in locals() else None,
            'audio_size_bytes': 0
        }


def get_supported_languages():
    """
    Get list of supported languages for translation and speech synthesis
    
    Returns:
    - dict: Dictionary of supported languages with their codes and names
    """
    return {
        'translation_languages': {
            'en': 'English',
            'hi': 'Hindi',
            'ka': 'Kannada',
            'ta': 'Tamil',
            'te': 'Telugu',
            'ml': 'Malayalam',
            'fr': 'French',
            'es': 'Spanish',
            'de': 'German',
            'ja': 'Japanese',
            'ko': 'Korean',
            'zh': 'Chinese (Simplified)',
            'ar': 'Arabic',
            'ru': 'Russian'
        },
        'speech_languages': {
            'en': 'English (Emma Multilingual)',
            'hi': 'Hindi (Swara)',
            'ka': 'Kannada (Sapna)',
            'ta': 'Tamil (Pallavi)',
            'te': 'Telugu (Shruti)',
            'ml': 'Malayalam (Sobhana)'
        },
        'voice_mapping': {
            'en': 'en-US-EmmaMultilingualNeural',
            'hi': 'hi-IN-SwaraNeural',
            'ka': 'kn-IN-SapnaNeural',
            'ta': 'ta-IN-PallaviNeural',
            'te': 'te-IN-ShrutiNeural',
            'ml': 'ml-IN-SobhanaNeural'
        }
    }


# Example usage and testing function
def test_speech_function():
    """
    Test function to demonstrate usage of translate_and_synthesize_speech
    """
    # Test parameters
    test_text = "Hello, this is a test message for speech synthesis."
    test_language = "hi"  # Hindi
    test_output_path = "/home/dragoon/coding/drone-crop/ML/translationAndSpeech/audio/test_output.mp3"
    
    print("Testing translate_and_synthesize_speech function...")
    result = translate_and_synthesize_speech(
        text=test_text,
        target_language=test_language,
        output_file_path=test_output_path
    )
    
    print(f"Result: {result}")
    return result


if __name__ == "__main__":
    # Run test if script is executed directly
    test_result = test_speech_function()
    if test_result['success']:
        print(f"✅ Test successful! Audio saved to: {test_result['file_path']}")
    else:
        print(f"❌ Test failed: {test_result['error']}")
