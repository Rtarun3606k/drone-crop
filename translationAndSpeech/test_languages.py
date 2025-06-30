#!/usr/bin/env python3
"""
Test script for the multilingual text-to-speech API
This script tests text-to-speech conversion for multiple Indian languages
"""

import os
import requests
import time
import json

# API Base URL - Change this to match your Flask server
BASE_URL = "http://localhost:5000/speech"

# Sample texts in different languages
texts = {
    "en": "Hello, this is a test of the English language text-to-speech service.",
    "hi": "नमस्ते, यह हिंदी भाषा टेक्स्ट-टू-स्पीच सेवा का एक परीक्षण है।",
    "ka": "ನಮಸ್ಕಾರ, ಇದು ಕನ್ನಡ ಭಾಷೆಯ ಪಠ್ಯ-ಯಿಂದ-ಧ್ವನಿ ಸೇವೆಯ ಪರೀಕ್ಷೆಯಾಗಿದೆ.",
    "ta": "வணக்கம், இது தமிழ் மொழி உரை-இருந்து-பேச்சு சேவையின் சோதனை.",
    "te": "నమస్కారం, ఇది తెలుగు భాషా వచన-నుండి-స్పీచ్ సేవ యొక్క పరీక్ష.",
    "ml": "നമസ്കാരം, ഇത് മലയാളം ഭാഷാ ടെക്സ്റ്റ്-ടു-സ്പീച്ച് സേവനത്തിന്റെ ഒരു പരീക്ഷണമാണ്."
}

# Ensure output directory exists
os.makedirs('audio', exist_ok=True)

def test_language(lang_code, text):
    """Test text-to-speech for a specific language"""
    print(f"\n[{lang_code}] Testing {lang_code} text-to-speech...")
    print(f"Text: {text}")
    
    # Step 1: Get list of available voices
    print(f"Fetching voices for {lang_code}...")
    try:
        voices_response = requests.get(f"{BASE_URL}/voices?language={lang_code}")
        if voices_response.status_code != 200:
            print(f"Failed to get voices: {voices_response.status_code}")
            return False
            
        voices_data = voices_response.json()
        lang_voices = []
        
        # Find matching voices for this language
        for voice in voices_data["voices"]:
            locale = voice.get("locale", "").lower()
            code = voice.get("language_code", "").lower()
            if locale.startswith(lang_code.lower()) or code == lang_code.lower():
                lang_voices.append(voice)
        
        if not lang_voices:
            print(f"No voices found for language {lang_code}")
            return False
            
        # Use the first recommended voice or just the first one
        selected_voice = next((v for v in lang_voices if v.get("recommended")), lang_voices[0])
        voice_name = selected_voice["name"]
        print(f"Selected voice: {voice_name}")
        
        # Step 2: Convert text to speech
        print("Converting text to speech...")
        tts_payload = {
            "text": text,
            "target_language": lang_code,
            "voice": voice_name
        }
        
        tts_response = requests.post(f"{BASE_URL}/text-to-speech", json=tts_payload)
        if tts_response.status_code != 202:
            print(f"Failed to start text-to-speech conversion: {tts_response.status_code}")
            return False
            
        tts_data = tts_response.json()
        session_id = tts_data["session_id"]
        print(f"Session ID: {session_id}")
        
        # Step 3: Poll for status
        status_url = f"{BASE_URL}/status/{session_id}"
        max_polls = 30
        polls = 0
        
        while polls < max_polls:
            status_response = requests.get(status_url)
            if status_response.status_code != 200:
                print(f"Failed to check status: {status_response.status_code}")
                time.sleep(1)
                polls += 1
                continue
                
            status_data = status_response.json()
            if status_data["status"] == "completed":
                print(f"Conversion completed! Progress: {status_data['progress']}%")
                break
                
            if status_data["status"] == "failed":
                print(f"Conversion failed: {status_data.get('error', 'Unknown error')}")
                return False
                
            print(f"In progress: {status_data['progress']}%")
            time.sleep(1)
            polls += 1
            
        if polls >= max_polls:
            print("Timeout waiting for conversion to complete")
            return False
            
        # Step 4: Download the audio
        print("Downloading audio file...")
        audio_url = f"{BASE_URL}/audio/{session_id}"
        audio_response = requests.get(audio_url)
        
        if audio_response.status_code != 200:
            print(f"Failed to download audio: {audio_response.status_code}")
            return False
            
        output_file = f"audio/{lang_code}_test.mp3"
        with open(output_file, "wb") as f:
            f.write(audio_response.content)
            
        print(f"Audio saved to {output_file}")
        return True
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return False

def main():
    """Run tests for all languages"""
    print("===== Testing Multilingual Text-to-Speech API =====")
    
    results = {}
    for lang, text in texts.items():
        success = test_language(lang, text)
        results[lang] = "Success" if success else "Failed"
    
    print("\n===== Test Results =====")
    for lang, result in results.items():
        print(f"{lang}: {result}")

if __name__ == "__main__":
    main()
