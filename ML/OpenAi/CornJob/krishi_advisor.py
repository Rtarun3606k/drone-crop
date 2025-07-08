import os
import base64
from openai import AzureOpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def get_krishi_sathi_response(user_message, endpoint=None, api_key=None, search_key=None):
    """
    Get response from Krishi Sathi agricultural advisor using Azure OpenAI with RAG.
    
    Args:
        user_message (str): The farmer's question or message
        endpoint (str, optional): Azure OpenAI endpoint URL
        api_key (str, optional): Azure OpenAI API key
        search_key (str, optional): Azure Search resource key
    
    Returns:
        str: Response from Krishi Sathi
    """
    # Use provided credentials or fallback to environment variables
    endpoint = endpoint or os.getenv("ENDPOINT_URL")
    deployment = os.getenv("DEPLOYMENT_NAME", "gpt-4o")
    search_endpoint = os.getenv("SEARCH_ENDPOINT")
    search_key = search_key or os.getenv("SEARCH_KEY")
    search_index = os.getenv("SEARCH_INDEX_NAME", "azureblob-index")
    subscription_key = api_key or os.getenv("AZURE_OPENAI_API_KEY")

    # Initialize Azure OpenAI client with key-based authentication
    client = AzureOpenAI(
        azure_endpoint=endpoint,
        api_key=subscription_key,
        api_version="2025-01-01-preview",
    )

    # Prepare the chat prompt
    system_prompt = """You are Krishi Sathi, an experienced and compassionate agricultural advisor who specializes in soybean cultivation and plant health. Your primary role is to assist farmers, especially those from rural and non-technical backgrounds, by providing clear, practical, and trustworthy advice on managing soybean diseases, pests, and overall farm health. Your answers must strictly follow the content from the "Soybean Plant Health Knowledge Base: A Comprehensive Guide (1).pdf".

Always communicate like a warm, respectful local expert—think of yourself as a Krishi Vigyan Kendra officer or a seasoned village farmer. Greet the user with "Namaste, Kisan bhai!" and explain things in simple, everyday language. Avoid technical jargon, and if a technical term is needed, explain it in simple words. Use bullet points, short paragraphs, and structured headings like:

What's the Problem?
How to Spot It (Symptoms)
What You Can Do (Remedies)
Good Practices to Prevent This

Never mention AI terms, confidence percentages, or file/image names. Don't guess answers—if the knowledge base doesn't have the info, say: "I don't have that information right now.
Also no need of formatting i just need plain text response.no need to making it like a chat response see example like heading ,bold, italic , emojies ,and that charchracter which are use in markdown formatting not needed is also not necessary like **xyz** and -- or - not necessary and not encouraged follow strictly.
 Please consult your local agricultural officer or Krishi Vigyan Kendra." Encourage farmers, offer hope, and always promote good farming habits like soil testing, crop rotation, pest monitoring, and certified seeds. End responses with kind, motivating words like: "Stay strong, Kisan bhai! With good care, your field will flourish." All advice must come strictly from the document "Soybean Plant Health Knowledge Base: A Comprehensive Guide (1).pdf".

Do not invent any information.

When asked about something outside the knowledge base, politely state:

"I don't have that information right now. Please consult your local Krishi Vigyan Kendra or agricultural expert."
"""

    chat_prompt = [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": user_message
        }
    ]

    # Generate the completion
    completion = client.chat.completions.create(
        model=deployment,
        messages=chat_prompt,
        max_tokens=800,
        temperature=0.7,
        top_p=0.95,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None,
        stream=False,
        extra_body={
            "data_sources": [{
                "type": "azure_search",
                "parameters": {
                    "endpoint": search_endpoint,
                    "index_name": "azureblob-index",
                    "semantic_configuration": "default",
                    "query_type": "simple",
                    "fields_mapping": {},
                    "in_scope": True,
                    "filter": None,
                    "strictness": 3,
                    "top_n_documents": 5,
                    "authentication": {
                        "type": "api_key",
                        "key": search_key
                    }
                }
            }]
        }
    )

    return completion.choices[0].message.content


# Example usage and main execution
if __name__ == "__main__":
    # Example usage
    user_question = "My soybean plants have some yellow spots on the leaves, and some leaves are falling off early. What should I do, Krishi Sathi?"
    
    # Call the function - it will automatically use credentials from .env file
    response = get_krishi_sathi_response(user_message=user_question)
    
    print("Krishi Sathi Response:")
    print(response)
    
    # You can also call it with different questions
    print("\n" + "="*50 + "\n")
    
    # Another example
    user_question2 = "What are the best practices for soybean cultivation?"
    response2 = get_krishi_sathi_response(user_message=user_question2)
    
    print("Krishi Sathi Response 2:")
    print(response2)
