from llm.call_openai import call_openai
from llm.call_ollama import call_ollama
import os
from dotenv import load_dotenv

load_dotenv()

def pass_llm(prompt, 
             model = os.getenv("LLM_MODEL"),
             max_tokens = 200, 
             temperature = 0, 
             system_prompt = "You are an in car conversational assistant."):
    # add here another model if used by ollama 
    if model == "llama3.2" or model == "mistral":
        return call_ollama(
                prompt=prompt,
                max_tokens=max_tokens,
                system_prompt=system_prompt,
                temperature=temperature,
                model=model)
    else:
        return call_openai(
                prompt=prompt,
                max_tokens=max_tokens,
                system_prompt=system_prompt,
                temperature=temperature,
                model=model)