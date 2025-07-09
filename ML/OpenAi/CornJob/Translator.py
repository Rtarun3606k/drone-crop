import requests


def translationJob(text,lang):
    """
    Translates the given text to the specified language using a translation service.
    
    Args:
        text (str): The text to be translated.
        lang (str): The target language code (e.g., 'en', 'fr', 'es').
    
    Returns:
        str: The translated text.
    """
    transalor_url = " http://127.0.0.1:5003/translate/translate"
    payload ={
    "text": text,
    "source_language": "en",  
    "target_language": lang   
    }

    headers = {
        "Content-Type": "application/json"
    }
    response = requests.post(transalor_url, json=payload, headers=headers)
    if response.status_code == 200:
        translated_text = response.json().get('translated_text', '')
        return translated_text
    else:
        print(f"Error: {response.status_code} - {response.text}")
        return None
    
if __name__ == "__main__":
    text = "Take 3 dropps of muffin every 6 hours for 10 days"
    lang = ['hi', 'kn', 'ta', 'te', 'ml']  # List of languages to translate to

    for i in range(5):

        translated_text = translationJob(text, lang[i])
        if translated_text:
            print(f"Translated Text: {translated_text}")
        else:
            print("Translation failed.")
