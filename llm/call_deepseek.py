import os
from azure.ai.inference.models import AssistantMessage, SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential
from dotenv import load_dotenv
import time
import traceback
from azure.core.exceptions import HttpResponseError

load_dotenv()

def load_deepseek_client(model_name: str):
    from azure.ai.inference import ChatCompletionsClient

    endpoint = os.getenv("DEEPSEEK_AZURE_ENDPOINT")
    api_version = os.getenv("DEEPSEEK_API_VERSION")

    deployment_name = model_name
    
    api_key = os.getenv("DEEPSEEK_V3_API_KEY")
    
    client = ChatCompletionsClient(
        endpoint=endpoint,
        credential=AzureKeyCredential(api_key),
        api_version=api_version
    )

    return client, deployment_name


def call_deepseek(deployment_name: str, 
                  prompt: str, 
                  max_tokens=1000, 
                  temperature=0, 
                  system_message=None, 
                  context=None):
    
    client, _= load_deepseek_client(model_name=deployment_name)
    
    try:
        #start_time = time.time()
        
        formatted_system_msg = system_message.format(context) if context else system_message

        response = client.complete(
            model=deployment_name,
            messages=[
                SystemMessage(content=formatted_system_msg),
                UserMessage(content=prompt)
            ],
            max_tokens=max_tokens,
            temperature=temperature
        )

        #end_time = time.time()

        #print("Deepseek: time per call:", end_time - start_time)
        # Note: azure.ai.inference does not currently return token usage metadata
        response_msg = response.choices[0].message.content

        input_tokens = response.usage.prompt_tokens
        output_tokens = response.usage.completion_tokens
                
        return response_msg, input_tokens, output_tokens

    except HttpResponseError as e:
        print("[DeepSeekClient] HttpResponseError:")
        traceback.print_exc()
        return f"HTTP_RESPONSE_ERROR: {str(e)}", None, -1

    except Exception as e:
        print("[DeepSeekClient] Unhandled exception during DeepSeek call")
        print("Prompt:", prompt)
        traceback.print_exc()
        raise e
