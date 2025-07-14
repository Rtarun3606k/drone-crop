import os
from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential
from azure.ai.agents.models import ListSortOrder

def get_krishi_sathi_response(user_message, endpoint=None):
    """
    Get both long and short responses from the Krishi Sathi agricultural agent using Azure AI Foundry.
    
    Args:
        user_message (str): Farmer's question, or crop analysis result
        endpoint (str, optional): AI Foundry endpoint URL
    
    Returns:
        dict: Dictionary containing both long and short responses
              {
                  "long_description": str,
                  "short_description": str,
                  "status": str
              }
    """
    print("Starting chat with Krishi Sathi agent... krishi.py")
    endpoint = endpoint or os.getenv("AZURE_AGENT_ENDPOINT")
    agent_id = os.getenv("AGENT_ID")

    try:
        # Authenticate and initialize client
        project = AIProjectClient(
            credential=DefaultAzureCredential(),
            endpoint=endpoint
        )

        # Create a thread
        thread = project.agents.threads.create()

        # Enhanced user message requesting both formats
        enhanced_message = f"""
        {user_message}

        Please provide your response in two formats:

        1. LONG DESCRIPTION: A detailed, comprehensive analysis with specific recommendations, treatment options, and explanations (2-3 paragraphs)

        2. SHORT DESCRIPTION: A brief, actionable summary for quick reference (2-3 sentences maximum)

        Format your response exactly like this:
        LONG DESCRIPTION:
        [Your detailed response here]

        SHORT DESCRIPTION:
        [Your brief summary here]
        """

        # Send user message
        project.agents.messages.create(
            thread_id=thread.id,
            role="user",
            content=enhanced_message
        )

        # Start and process the run
        run = project.agents.runs.create_and_process(
            thread_id=thread.id,
            agent_id=agent_id
        )

        if run.status == "failed":
            return {
                "long_description": f"Run failed: {run.last_error}",
                "short_description": "Error occurred",
                "status": "failed"
            }
        
        # Retrieve the assistant's reply
        messages = project.agents.messages.list(
            thread_id=thread.id, 
            order=ListSortOrder.ASCENDING
        )

        # print(f"Messages retrieved: {len(messages)}")
        print(f"Messages retrieved: {messages}")

        for message in messages:
            if message.role == "assistant" and message.text_messages:
                response_text = message.text_messages[-1].text.value
                
                # Parse the response to extract long and short descriptions
                parsed_response = parse_dual_response(response_text)
                # print(f"response_text: {response_text}")
                return parsed_response

        return {
            "long_description": "No response received from the agent.",
            "short_description": "No response received.",
            "status": "no_response"
        }
        
    except Exception as e:
        print(f"Error in get_krishi_sathi_response: {str(e)}")
        return {
            "long_description": f"Error: {str(e)}",
            "short_description": "Error occurred",
            "status": "error"
        }

def parse_dual_response(response_text):
    """
    Parse the agent's response to extract long and short descriptions.
    
    Args:
        response_text (str): The full response from the agent
        
    Returns:
        dict: Parsed long and short descriptions
    """
    try:
        # Look for the structured format
        long_desc = ""
        short_desc = ""
        
        # Split the response by sections
        if "LONG DESCRIPTION:" in response_text and "SHORT DESCRIPTION:" in response_text:
            # Extract long description
            long_start = response_text.find("LONG DESCRIPTION:") + len("LONG DESCRIPTION:")
            short_start = response_text.find("SHORT DESCRIPTION:")
            long_desc = response_text[long_start:short_start].strip()
            
            # Extract short description
            short_desc = response_text[short_start + len("SHORT DESCRIPTION:"):].strip()
            
        else:
            # Fallback: if format not followed, split the response
            sentences = response_text.split('. ')
            if len(sentences) > 3:
                # Use first 3+ sentences as long, last 1-2 as short
                long_desc = '. '.join(sentences[:-2]) + '.'
                short_desc = '. '.join(sentences[-2:])
            else:
                # If response is short, duplicate it
                long_desc = response_text
                short_desc = response_text
        
        return {
            "long_description": long_desc.strip(),
            "short_description": short_desc.strip(),
            "status": "success"
        }
        
    except Exception as e:
        print(f"Error parsing dual response: {str(e)}")
        return {
            "long_description": response_text,
            "short_description": response_text[:200] + "..." if len(response_text) > 200 else response_text,
            "status": "parse_error"
        }

def get_krishi_sathi_response_legacy(user_message, endpoint=None):
    """
    Legacy function that returns a single response (for backward compatibility).
    
    Args:
        user_message (str): Farmer's question, or crop analysis result
        endpoint (str, optional): AI Foundry endpoint URL
    
    Returns:
        str: Agent's response (long description)
    """
    result = get_krishi_sathi_response(user_message, endpoint)
    return result.get("long_description", "No response received from the agent.")

# Example usage
if __name__ == "__main__":
    # Test the dual response function
    test_message = "My soybean plants have yellow leaves and some brown spots. What could be causing this?"
    
    result = get_krishi_sathi_response(test_message)
    
    print("=== KRISHI SATHI RESPONSE ===")
    print(f"Status: {result['status']}")
    print("\n=== LONG DESCRIPTION ===")
    print(result['long_description'])
    print("\n=== SHORT DESCRIPTION ===")
    print(result['short_description'])