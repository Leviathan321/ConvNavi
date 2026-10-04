import os
from azure.ai.inference.models import AssistantMessage, SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential
from dotenv import load_dotenv
import time
import traceback
from azure.core.exceptions import HttpResponseError

load_dotenv()

OPENAI_DEPLOYMENTS = [
    "DeepSeek-V3.2",
    "DeepSeek-V4-Pro",
    "Kimi-K2-Thinking",
    "gpt-oss-120b",
]

def load_deepseek_client(model_name: str):
    deployment_name = model_name

    if deployment_name in OPENAI_DEPLOYMENTS:
        from openai import OpenAI

        endpoint = os.getenv("OPENAI_ENDPOINT")
        if endpoint and not endpoint.rstrip("/").endswith("/openai/v1"):
            endpoint = endpoint.rstrip("/") + "/openai/v1"
        client = OpenAI(
            api_key=os.getenv("OPENAI_KEY"),
            base_url=endpoint,
        )
        return client, deployment_name, "openai"

    from azure.ai.inference import ChatCompletionsClient

    endpoint = os.getenv("OPENAI_ENDPOINT")
    api_version = os.getenv("OPENAI_API_VERSION")

    api_key = os.getenv("OPENAI_KEY")
    
    client = ChatCompletionsClient(
        endpoint=endpoint,
        credential=AzureKeyCredential(api_key),
        api_version=api_version
    )

    return client, deployment_name, "azure"


def call_deepseek(deployment_name: str, 
                  prompt: str, 
                  max_tokens=1000, 
                  temperature=0, 
                  system_message=None, 
                  context=None):
    
    client, _, client_type = load_deepseek_client(model_name=deployment_name)
    
    try:
        #start_time = time.time()
        
        formatted_system_msg = system_message.format(context) if context else system_message

        if client_type == "openai":
            if deployment_name == "Kimi-K2-Thinking":
                max_tokens = max(max_tokens, 4096)

            response = client.chat.completions.create(
                model=deployment_name,
                messages=[
                    {"role": "system", "content": formatted_system_msg},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=max_tokens,
                temperature=temperature,
            )
        else:
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
        choice = response.choices[0]
        response_msg = choice.message.content
        if not response_msg or not response_msg.strip():
            raise RuntimeError(
                f"{deployment_name} returned no final content "
                f"(finish_reason={choice.finish_reason})."
            )

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
