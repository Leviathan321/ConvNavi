import os
from azure.ai.inference.models import AssistantMessage, SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential
from dotenv import load_dotenv
import time
import traceback
from azure.core.exceptions import HttpResponseError
import os
import traceback
from dotenv import load_dotenv

from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import HttpResponseError

OPENAI_DEPLOYMENTS = ["DeepSeek-V3.2", "DeepSeek-V4-Pro"]

load_dotenv()

def load_deepseek_client(model_name: str):

    deployment_name = model_name

    print(f"[INFO] Loading DeepSeek client for deployment: {deployment_name}")
    print(f"[INFO] OPENAI_DEPLOYMENTS: {OPENAI_DEPLOYMENTS}")

    if deployment_name in OPENAI_DEPLOYMENTS:
        from openai import OpenAI

        client = OpenAI(
            api_key=os.getenv("DEEPSEEK_V3_API_KEY"),
            base_url=os.getenv("DEEPSEEK_V3_ENDPOINT").rstrip("/") + "/",
        )

        return client, deployment_name, "openai"

    else:
        from azure.ai.inference import ChatCompletionsClient

        client = ChatCompletionsClient(
            endpoint=os.getenv("DEEPSEEK_V3_ENDPOINT"),
            credential=AzureKeyCredential(os.getenv("DEEPSEEK_V3_API_KEY")),
            api_version=os.getenv("DEEPSEEK_API_VERSION"),
        )

        return client, deployment_name, "azure"

def call_deepseek(
    deployment_name: str,
    prompt: str,
    max_tokens=1000,
    temperature=0,
    system_message=None,
    context=None,
):

    client, deployment_name, client_type = load_deepseek_client(deployment_name)

    try:

        formatted_system_msg = (
            system_message.format(context) if context else system_message
        )

        if client_type == "openai":

            response = client.chat.completions.create(
                model=deployment_name,
                messages=[
                    {
                        "role": "system",
                        "content": formatted_system_msg,
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
                max_tokens=max_tokens,
                temperature=temperature,
            )

            response_msg = response.choices[0].message.content

        else:

            response = client.complete(
                model=deployment_name,
                messages=[
                    SystemMessage(content=formatted_system_msg),
                    UserMessage(content=prompt),
                ],
                max_tokens=max_tokens,
                temperature=temperature,
            )

            response_msg = response.choices[0].message.content

        input_tokens = response.usage.prompt_tokens
        output_tokens = response.usage.completion_tokens

        return response_msg, input_tokens, output_tokens

    except HttpResponseError as e:
        print("[DeepSeekClient] HttpResponseError:")
        traceback.print_exc()
        return f"HTTP_RESPONSE_ERROR: {str(e)}", None, -1

    except Exception:
        print("[DeepSeekClient] Unhandled exception during DeepSeek call")
        print("Prompt:", prompt)
        traceback.print_exc()
        raise