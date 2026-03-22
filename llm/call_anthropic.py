import os
import traceback
import requests
from dotenv import load_dotenv

load_dotenv()

ANTHROPIC_API_URL = "https://llm.api.caip.bmw.cloud/v1/chat/completions"


def call_anthropic(prompt,
                   max_tokens=200,
                   temperature=0,
                   system_prompt=None,
                   model="claude-4-sonnet"):

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY environment variable is not set")

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    try:
        response = requests.post(ANTHROPIC_API_URL, headers=headers, json=payload)
        response.raise_for_status()

        data = response.json()

        response_msg = data["choices"][0]["message"]["content"]

        usage = data.get("usage", {})
        input_tokens = usage.get("prompt_tokens", 0)
        output_tokens = usage.get("completion_tokens", 0)

        return response_msg, input_tokens, output_tokens

    except requests.exceptions.HTTPError as e:
        print(f"[AnthropicClient] HTTP error: {e}")
        traceback.print_exc()
        return f"HTTP_ERROR: {str(e)}", 0, 0

    except Exception as e:
        print("[AnthropicClient] Unhandled exception during Anthropic call")
        print("Prompt:", prompt)
        traceback.print_exc()
        raise e


if __name__ == "__main__":
    prompt = """
                rephrase the following user utterance, leaving the variables. give 10 examples:
                'bitte zur <BMW_App> <NP_appSynonyms> gehen'
            """
    print(call_anthropic(prompt=prompt))
