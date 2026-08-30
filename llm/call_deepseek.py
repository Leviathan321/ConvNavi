import os
import traceback
from dotenv import load_dotenv

from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import HttpResponseError
from openai import OpenAI

load_dotenv()


def load_deepseek_client(model_name: str):
    if model_name == "DeepSeek-V4-Pro":
        endpoint = os.getenv("DEEPSEEK_AZURE_ENDPOINT")
        api_key = os.getenv("DEEPSEEK_V3_API_KEY")
        client = OpenAI(
            base_url=endpoint,
            api_key=api_key
        )

        return client, model_name

    else:
        from azure.ai.inference import ChatCompletionsClient

        endpoint = os.getenv("DEEPSEEK_AZURE_ENDPOINT")
        api_version = os.getenv("DEEPSEEK_API_VERSION")
        api_key = os.getenv("DEEPSEEK_V3_API_KEY")

        print("Using Azure AI Inference client")
        print("endpoint:", endpoint)

        client = ChatCompletionsClient(
            endpoint=endpoint,
            credential=AzureKeyCredential(api_key),
            api_version=api_version
        )

        return client, model_name


def call_deepseek(
    deployment_name: str,
    prompt: str,
    max_tokens=1000,
    temperature=0,
    system_message=None,
    context=None
):

    client, model_name = load_deepseek_client(deployment_name)

    try:
        formatted_system_msg = (
            system_message.format(context)
            if context
            else system_message
        )

        print("deploymentname:", deployment_name)
        print("endpoint:", os.getenv("DEEPSEEK_AZURE_ENDPOINT"))

        if model_name == "DeepSeek-V4-Pro":

            messages = []

            if formatted_system_msg:
                messages.append({
                    "role": "system",
                    "content": formatted_system_msg
                })

            messages.append({
                "role": "user",
                "content": prompt
            })

            response = client.chat.completions.create(
                model=deployment_name,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature
            )

            print("response:", response)

            response_msg = response.choices[0].message.content

            input_tokens = response.usage.prompt_tokens
            output_tokens = response.usage.completion_tokens

            return response_msg, input_tokens, output_tokens

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

            print("response:", response)

            response_msg = response.choices[0].message.content

            input_tokens = response.usage.prompt_tokens
            output_tokens = response.usage.completion_tokens

            return response_msg, input_tokens, output_tokens

    except HttpResponseError:
        print("[DeepSeekClient] HttpResponseError:")
        traceback.print_exc()
        return "HTTP_RESPONSE_ERROR", None, -1

    except Exception as e:
        print("[DeepSeekClient] Unhandled exception during DeepSeek call")
        print("Prompt:", prompt)
        traceback.print_exc()
        raise e