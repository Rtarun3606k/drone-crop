#!/usr/bin/env python3
"""
Azure Text-to-Speech using only REST API (no SDK)
Based on official Microsoft REST API documentation
"""

import os
import requests
import time
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

print("Azure Text-to-Speech REST API Example")
print("===================================\n")

# Get configuration from environment variables
speech_key = os.getenv('AZURE_SPEECH_API_KEY')
speech_region = os.getenv('AZURE_SPEECH_API_LOCATION')

if not speech_key or not speech_region:
    print("Error: Missing Azure Speech service credentials.")
    print("Please set AZURE_SPEECH_API_KEY and AZURE_SPEECH_API_LOCATION in your .env file.")
    exit(1)

print(f"Using:")
print(f"- API Key: {speech_key[:5]}...{speech_key[-5:]}")
print(f"- Region: {speech_region}")

# ---- Step 1: Get available voices ----
print("\nStep 1: Getting available voices...")

# Construct the voices list URL
voices_url = f"https://{speech_region}.tts.speech.microsoft.com/cognitiveservices/v1/voices/list"

headers = {
    "Ocp-Apim-Subscription-Key": speech_key
}

try:
    response = requests.get(voices_url, headers=headers)
    
    if response.status_code == 200:
        voices = response.json()
        emma_voices = [v for v in voices if "Emma" in v.get("ShortName", "")]
        
        print(f"Successfully retrieved {len(voices)} voices")
        if emma_voices:
            print(f"Found {len(emma_voices)} Emma voices:")
            for voice in emma_voices:
                print(f"  - {voice.get('ShortName', 'Unknown')}: {voice.get('LocalName', 'Unknown')}")
            
            # Use the first Emma voice found
            voice_name = emma_voices[0].get('ShortName', 'en-US-EmmaMultilingualNeural')
        else:
            print("No Emma voices found, using default")
            voice_name = "en-US-EmmaMultilingualNeural"
    else:
        print(f"Failed to get voices: {response.status_code}")
        print(f"Response: {response.text}")
        voice_name = "en-US-EmmaMultilingualNeural"  # Use default
except Exception as e:
    print(f"Error getting voices: {str(e)}")
    voice_name = "en-US-EmmaMultilingualNeural"  # Use default

# ---- Step 2: Convert text to speech ----
print(f"\nStep 2: Converting text to speech using voice: {voice_name}...")

# Text to synthesize
text = "Hello! This is a test of the Azure Speech REST API using Emma's voice."

# Construct the TTS endpoint URL
tts_url = f"https://{speech_region}.tts.speech.microsoft.com/cognitiveservices/v1"

headers = {
    "Ocp-Apim-Subscription-Key": speech_key,
    "Content-Type": "application/ssml+xml",
    "X-Microsoft-OutputFormat": "audio-16khz-128kbitrate-mono-mp3",
    "User-Agent": "DroneAppPythonRESTDemo"
}

# Create SSML content
ssml = f"""<speak version='1.0' xml:lang='en-US'>
    <voice name='{voice_name}'>
        {text}
    </voice>
</speak>"""

print(f"SSML Content:\n{ssml}\n")
print(f"Sending request to {tts_url}...")

try:
    start_time = time.time()
    response = requests.post(tts_url, headers=headers, data=ssml.encode('utf-8'))
    elapsed_time = time.time() - start_time
    
    print(f"Request completed in {elapsed_time:.2f} seconds")
    print(f"Status code: {response.status_code}")
    
    if response.status_code == 200:
        # Save the audio data
        output_file = "rest_output.mp3"
        with open(output_file, "wb") as f:
            f.write(response.content)
        
        file_size = os.path.getsize(output_file)
        print(f"Success! Audio file created: {output_file} ({file_size} bytes)")
        print(f"Full path: {os.path.abspath(output_file)}")
    else:
        print(f"Error: {response.status_code}")
        print(f"Response: {response.text}")
except Exception as e:
    print(f"Error during speech synthesis: {str(e)}")

print("\nDone!")
