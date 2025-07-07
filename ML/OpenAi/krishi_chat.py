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

    # Initialize Azure OpenAI client
    client = AzureOpenAI(
        azure_endpoint=endpoint,
        api_key=subscription_key,
        api_version="2025-01-01-preview",
    )

    # System prompt for Krishi Sathi
    system_prompt = """You are Krishi Sathi, an experienced and compassionate agricultural advisor who specializes in soybean cultivation and plant health. Your primary role is to assist farmers, especially those from rural and non-technical backgrounds, by providing clear, practical, and trustworthy advice on managing soybean diseases, pests, and overall farm health.

Always communicate like a warm, respectful local expert—think of yourself as a Krishi Vigyan Kendra officer or a seasoned village farmer. Greet the user with "Namaste, Kisan bhai!" and explain things in simple, everyday language. Avoid technical jargon, and if a technical term is needed, explain it in simple words.

Use bullet points, short paragraphs, and structured headings like:
- What's the Problem?
- How to Spot It (Symptoms)
- What You Can Do (Remedies)
- Good Practices to Prevent This

Never mention AI terms, confidence percentages, or file/image names. Don't guess answers—if you don't have specific information, say: "I don't have that information right now. Please consult your local agricultural officer or Krishi Vigyan Kendra." 

Encourage farmers, offer hope, and always promote good farming habits like soil testing, crop rotation, pest monitoring, and certified seeds. End responses with kind, motivating words like: "Stay strong, Kisan bhai! With good care, your field will flourish." 🌱"""

    # Prepare the chat messages
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message}
    ]

    # Generate the completion with RAG
    completion = client.chat.completions.create(
        model=deployment,
        messages=messages,
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
                    "index_name": search_index,
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


def simple_krishi_chat(message):
    """
    Simple function to chat with Krishi Sathi using default credentials.
    
    Args:
        message (str): Your question about soybean farming
    
    Returns:
        str: Krishi Sathi's response
    """
    return get_krishi_sathi_response(message)


# Example usage
if __name__ == "__main__":
    # Simple usage
    print("=== Simple Krishi Sathi Chat ===")
    
    # Example 1
    question1 = "My soybean plants have yellow leaves. What should I do?"
    print(f"Question: {question1}")
    print(f"Answer: {simple_krishi_chat(question1)}")
    
    print("\n" + "="*60 + "\n")
    
    # Example 2
    question2 = "How can I prevent pest attacks in my soybean field?"
    print(f"Question: {question2}")
    print(f"Answer: {simple_krishi_chat(question2)}")
    
    print("\n" + "="*60 + "\n")
    
    # Example 3 - Interactive mode
    print("=== Interactive Mode ===")
    print("Type 'exit' to quit")
    
    while True:
        user_input = input("\nAsk Krishi Sathi: ")
        if user_input.lower() in ['exit', 'quit', 'bye']:
            print("Dhanyawad, Kisan bhai! Take care of your fields! 🌱")
            break
        
        if user_input.strip():
            response = simple_krishi_chat(user_input)
            print(f"\nKrishi Sathi: {response}")
