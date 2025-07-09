# from CornJob.SpeechFunction import translate_and_synthesize_speech
from SpeechFunction import translate_and_synthesize_speech,text_to_speech

if __name__ == "__main__":
    # Example usage of the translate_and_synthesize_speech function
    result = translate_and_synthesize_speech(
        text="Hello, how are you today?",
        target_language="hi",
        output_file_path="/home/dragoon/coding/drone-crop/ML/translationAndSpeech/audio/abcd.mp3"
    )
    
    if result['success']:
        print(f"✅ Success! Audio saved to: {result['file_path']}")
        print(f"   Original: {result['original_text']}")
        print(f"   Translated: {result['translated_text']}")
        print(f"   Voice used: {result['voice_used']}")
        print(f"   File size: {result['audio_size_bytes']} bytes")
    else:
        print(f"❌ Failed: {result['error']}")

    result = text_to_speech(
        text="ನಮಸ್ಕಾರ, ನೀವು ಇಂದು ಹೇಗಿದ್ದೀರಿ?",
        target_language="kn",
        output_file_path="/home/dragoon/coding/drone-crop/ML/translationAndSpeech/audio/abcdefg.mp3"
    )