import os
import json
import traceback
from google.oauth2 import service_account
import google.genai as genai
from google.genai.types import GenerateContentConfig

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SERVICE_ACCOUNT_PATH = os.path.join(BASE_DIR, "key.json")
_CLIENT = None


def get_client():
    global _CLIENT
    if _CLIENT is None:
        try:
            with open(SERVICE_ACCOUNT_PATH, 'r') as file:
                service_account_data = json.load(file)

            creds = service_account.Credentials.from_service_account_info(
                service_account_data,
                scopes=["https://www.googleapis.com/auth/cloud-platform"]
            )

            _CLIENT = genai.Client(
                vertexai=True,
                credentials=creds,
                project=service_account_data["project_id"],
                location="global"
            )
        except Exception as e:
            print(f"Failed to initialize Gemini client: {e}")
            raise e
    return _CLIENT


def call_gemini(prompt,
                max_tokens=200,
                temperature=0,
                system_prompt=None,
                model="gemini-2.5-flash"):
    client = get_client()

    final_prompt = prompt
    if system_prompt:
        final_prompt = f"System: {system_prompt}\nUser: {prompt}"

    config = GenerateContentConfig(
        max_output_tokens=max_tokens,
        temperature=temperature
    )

    try:
        response = client.models.generate_content(
            model=model,
            contents=final_prompt,
            config=config
        )

        response_text = response.text

        input_tokens = 0
        output_tokens = 0

        meta = getattr(response, "usage_metadata", None)
        if meta is not None:
            input_tokens = getattr(meta, "prompt_token_count", None) or 0
            output_tokens = getattr(meta, "candidates_token_count", None) or 0

        return response_text, input_tokens, output_tokens

    except Exception as e:
        print(f"[GeminiClient] Error calling Gemini: {e}")
        traceback.print_exc()
        return f"GEMINI_ERROR: {str(e)}", 0, 0


if __name__ == "__main__":
    prompt = """
                rephrase the following user utterance, leaving the variables. give 10 examples:
                'bitte zur <BMW_App> <NP_appSynonyms> gehen'
            """
    print(call_gemini(prompt=prompt))
