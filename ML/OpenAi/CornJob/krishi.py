import os
from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential
from azure.ai.agents.models import ListSortOrder
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def get_krishi_sathi_response(user_message, endpoint=None):
    """
    Get a response from the Krishi Sathi agricultural agent using Azure AI Foundry.
    
    Args:
        user_message (str): Farmer's question, or crop analysis result
        endpoint (str, optional): AI Foundry endpoint URL
    
    Returns:
        str: Agent's response
    """
    print("Starting chat with Krishi Sathi agent... krishi.py")
    endpoint = endpoint or os.getenv("AZURE_AGENT_ENDPOINT")
    agent_id = os.getenv("AGENT_ID")

    # Authenticate and initialize client
    project = AIProjectClient(
        credential=DefaultAzureCredential(),
        endpoint=endpoint
    )

    # Create a thread
    thread = project.agents.threads.create()

    # Send user message
    project.agents.messages.create(
        thread_id=thread.id,
        role="user",
        content=user_message
    )

    # Start and process the run
    run = project.agents.runs.create_and_process(
        thread_id=thread.id,
        agent_id=agent_id
    )

    if run.status == "failed":
        return f"Run failed: {run.last_error}"
    
    # Retrieve and return the assistant's reply
    messages = project.agents.messages.list(
        thread_id=thread.id, 
        order=ListSortOrder.ASCENDING
    )

    for message in messages:
        if message.role == "assistant" and message.text_messages:
            return message.text_messages[-1].text.value

    return "No response received from the agent."


if __name__ == "__main__":
    response = get_krishi_sathi_response("How to control soybean semilooper ?")
    print("Krishi Sathi:", response)
