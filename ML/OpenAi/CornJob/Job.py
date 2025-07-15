import json
# from CornJob.krishi_advisor import get_krishi_sathi_response
import os
from dotenv import load_dotenv
from MongoDB.DataBaseConnection import getAllIncompleteBatches,updatebatchStatus,uploadResponseToDB,uploadPrefferedLanguageResponseToDB,create_description
from CornJob.Translator import translationJob
# from CornJob.krishi import get_krishi_sathi_response
from CornJob.LOngAndShort import get_krishi_sathi_response

# Load environment variables
load_dotenv()

#predection_path = '/home/dragoon/coding/drone-crop/predectionResults'
predection_path = '/home/bruh/Documents/drone-crop/predectionResults'

def loadJsonFromFile(file_path, session_id):
    """
    Load JSON data from a file and return it as a dictionary.
    
    Args:
        file_path (str): The path to the JSON file.
        session_id (str): The session ID for logging or tracking purposes.
    
    Returns:
        dict: The loaded JSON data.
    """
    try:
        with open(file_path + f'/{session_id}_analysis.json', 'r') as file:
            data = json.load(file)
        print(f"Session {session_id}: Successfully loaded JSON data from {file_path}")
        return data
    except Exception as e:
        print(f"Session {session_id}: Error loading JSON data from {file_path}: {e}")
        return {} 

def analyze_crop_data_with_ai(json_data, user_question=None):
    """
    Analyze crop data using Azure OpenAI with proper parameters
    
    Args:
        json_data (dict): The crop analysis data
        user_question (str): Optional specific question about the crop
    
    Returns:
        str: AI analysis response
    """
    if not json_data:
        return "No data to analyze"
    
    # Format JSON data for the AI
    formatted_json = json.dumps(json_data, indent=2)
    
    # Create the user message
    if user_question:
        user_message = f"""Please analyze this crop health report and answer my specific question: {user_question}

Crop Analysis Report:
{formatted_json}

Please provide detailed analysis and recommendations based on this data."""
    else:
        user_message = f"""Please analyze this comprehensive crop health report and provide detailed insights and recommendations:

Crop Analysis Report:
{formatted_json}

Please provide:
1. Overall crop health assessment
2. Disease analysis and severity
3. Confidence analysis interpretation  
4. Specific treatment recommendations
5. Preventive measures for future"""
    
    try:
        # Call the function with all required parameters from environment variables
        # response = get_krishi_sathi_response(
        #     user_message=user_message,
        #     endpoint=os.getenv("ENDPOINT_URL"),
        #     api_key=os.getenv("AZURE_OPENAI_API_KEY"),
          
        #     search_key=os.getenv("SEARCH_KEY"),
            
        # )
        response = get_krishi_sathi_response(
            user_message
        )
        print(f"AI analysis response: {response}")
        return response
    except Exception as e:
        return f"Error analyzing data: {str(e)}"
    


def Job_generate_response():
    """
    Main function to generate responses for all incomplete batches.
    """
    print("Starting job to generate responses for all incomplete batches...")
    
    # Fetch all incomplete batches
    batches = getAllIncompleteBatches()
    
    if not batches:
        print("No incomplete batches found.")
        return
    
    for batch in batches:
        session_id = str(batch['sessionId'])
        batch_id = str(batch['_id'])
        preffered_language = batch.get('preferredLanguage', 'En')
        print(f"Processing batch with ID: {session_id}")
        
        # Load JSON data from file
        json_data = loadJsonFromFile(predection_path, session_id)
        
        if not json_data:
            print(f"No data found for batch {session_id}. Skipping...")
            continue
        
        # Analyze the crop data
        analysis = analyze_crop_data_with_ai(json_data=json_data)
        long_description=analysis['long_description']
        short_description=analysis['short_description']
        long_description = long_description.replace('\n', ' ')
        short_description = short_description.replace('\n', ' ')
        long_description= long_description.split(' ')
        short_description = short_description.split(' ')
        word_count = len(long_description) + len(short_description)
        # language = translationJob(analysis, preffered_language)  # Translate to Hindi
        LongDesc = translationJob(analysis['long_description'], preffered_language)  # Translate to English
        shortDesc = translationJob(analysis['short_description'], preffered_language)  # Translate to English

        if LongDesc is None or shortDesc is None:
            updatebatchStatus(batch_id, 'failed')
            print(f"Translation failed for batch {session_id}. Skipping...")
            continue
        LongDescWordCount = len(LongDesc.split(' '))
        shortDescWordCount = len(shortDesc.split(' '))
        print(f"Long Description Word Count: {LongDescWordCount}")
        print(f"Short Description Word Count: {shortDescWordCount}")
        languageWordCount = LongDescWordCount + shortDescWordCount
        language = {
            'longDesc' : LongDesc,
            'shortDesc' : shortDesc
        }
        
        if analysis and language:
            # Upload response to database
            # uploadResponseToDB(batch_id, analysis)
            create_description(batch_id=batch_id,long_description=analysis['long_description'],
                               short_description=analysis['short_description'],word_count=word_count,language='En',
                               confidence=analysis.get('confidence', None))

            print(f"Response uploaded for batch {session_id}.")
            # Upload translated response to database
            create_description(batch_id=batch_id,long_description=LongDesc,
                    short_description=shortDesc,word_count=languageWordCount,language=preffered_language,
                    confidence=analysis.get('confidence', None))
            print(f"Translated response uploaded for batch {session_id}.")
            
            # Update batch status
            updatebatchStatus(batch_id, 'completed')
            print(f"Batch {session_id} status updated to completed.")
        else:
            updatebatchStatus(batch_id, 'failed')
            print(f"Failed to analyze data for batch {session_id}.")
    
    print("Job completed successfully.")

if __name__ == "__main__":
    print("Loading JSON data from file...")
    
    session_id = "a8d5ccd9-4ded-431e-b5d8-bae8fb7c8774"
    json_data = loadJsonFromFile(predection_path, session_id)
    
    if json_data:
        print("JSON data loaded successfully.")
        # Fix: Use json.dumps() for printing, not json.dump()
        print(json.dumps(json_data, indent=4))
        
        print("\n" + "="*50)
        print("Analyzing crop data with AI...")
        
        # Analyze the crop data
        analysis = analyze_crop_data_with_ai(
            json_data=json_data,
            user_question="What immediate actions should I take based on this crop analysis?"
        )
        
        print("AI Analysis Result:")
        print(analysis)
    else:
        print("Failed to load JSON data.")