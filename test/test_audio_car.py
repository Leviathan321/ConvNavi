import os
import requests
from openai import AzureOpenAI
from dotenv import load_dotenv

load_dotenv()  # by default it looks for a .env in the working directory

# Requires env vars:
#   AZURE_OPENAI_ENDPOINT
#   AZURE_OPENAI_API_KEY
#   AZURE_OPENAI_API_VERSION (optional, default below)
#   TTS_MODEL (your Azure OpenAI TTS deployment name)
#
# And your API running locally with /query_audio endpoint:
#   http://localhost:8000/query_audio

API_URL = os.getenv("NAV_API_URL", "http://localhost:8000/query_audio")

texts = ["What the hell is going on?? I want to have my fog lights on.",
         "Open all the windows in my car.",
         "Increase the temperature 22 degrees!",
         "What features does the car have?"]

client = AzureOpenAI(
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
    api_version="2025-03-01-preview"
)

for i, text in enumerate(texts):
    print(f"\n=== Testing query: '{text}' ===")
    # Generate audio with Azure OpenAI TTS
    tts = client.audio.speech.create(
        model="tts",
        voice=os.getenv("TTS_VOICE", "alloy"),
        input=text,
    )

    # save the audio localy for inspection in wav format

    # create out folder if it doesn't exist
    os.makedirs("out/car", exist_ok=True)

    with open(f"out/car/test_voice_{i}.wav", "wb") as f:
        f.write(tts.read())

    audio_bytes = tts.read()  # MP3 bytes

    # Send to your FastAPI endpoint (/query_audio)
    files = {"audio": ("query.mp3", audio_bytes, "audio/mpeg")}
    data = {
        "user_location": "39.955431,-75.154903",
        "user_id": "1",
        # "llm_type": "gpt-4.1-mini",  # optional
    }

    resp = requests.post(API_URL, files=files, data=data, timeout=120)
    print("status:", resp.status_code)
    print(resp.text)