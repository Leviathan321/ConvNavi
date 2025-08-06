from openai import AzureOpenAI
import os
from dotenv import load_dotenv

# Load variables from .env
load_dotenv()

azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
api_version = os.getenv("AZURE_OPENAI_API_VERSION")
api_key = os.getenv("AZURE_OPENAI_API_KEY")
deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT")

# Initialize Azure OpenAI client
client = AzureOpenAI(
    api_key=api_key,
    api_version=api_version,
    azure_endpoint=azure_endpoint
)


def call_llm(prompt, max_tokens = 200, temperature = 0, system_prompt = None):
    response = client.chat.completions.create(
        model=deployment_name,
        temperature=temperature,
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": "You are an in car conversational assistant." if system_prompt is None else system_prompt},
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message.content

if __name__ == "__main__":
    prompt = """
                rephrase the following user utterance, leaving the variables. give 10 examples:
                'bitte zur <BMW_App> <NP_appSynonyms> gehen'
            """
    print(call_llm(prompt=prompt))