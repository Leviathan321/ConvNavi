import subprocess
from llm.call_openai import call_openai, call_openai_gpt5_models
from llm.call_ollama import call_ollama
import os
from dotenv import load_dotenv
import traceback

load_dotenv()

TOTAL_TOKENS = 0

def ensure_model(model_name):
    try:
        # Check if the model is already pulled
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
        if model_name not in result.stdout:
            print(f"Model '{model_name}' not found. Pulling the model...")
            subprocess.run(["ollama", "pull", model_name], check=True)
        else:
            print(f"Model '{model_name}' is already available.")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred: {e}")
        
def pass_llm(prompt, 
             model=os.getenv("LLM_MODEL"),
             max_tokens=200, 
             temperature=0, 
             system_prompt="You are an in car conversational assistant."):
    """
    Call the selected LLM backend with error handling that logs inputs
    when a failure occurs.
    """
    global TOTAL_TOKENS

    try:
        if model in ("llama3","llama3.2", "mistral", "deepseek-v2",
                     "deepseek-r1", "qwen3:latest","qwen3:14b"):
            response, tokens = call_ollama(
                prompt=prompt,
                max_tokens=max_tokens,
                system_prompt=system_prompt,
                temperature=temperature,
                model=model
            )
        elif model in ("gpt-5", "gpt-5-chat","gpt-5-mini"):
            response, tokens = call_openai_gpt5_models(
                prompt=prompt,
                max_completion_tokens=max_tokens,
                system_prompt=system_prompt,
                temperature=temperature,
                model=model
            )
        elif model in ("gpt-4o", "gpt-35-turbo", "gpt-4", "gpt-4o-mini"):
            response, tokens = call_openai(
                prompt=prompt,
                max_tokens=max_tokens,
                system_prompt=system_prompt,
                temperature=temperature,
                model=model
            )
        else:
            raise ValueError("Model is not known.")
        
        TOTAL_TOKENS = TOTAL_TOKENS + tokens
        return response, tokens
    except Exception as e:
        print("=== LLM CALL FAILED ===")
        print(f"Model: {model}")
        print(f"Max tokens: {max_tokens}")
        print(f"Temperature: {temperature}")
        print(f"System prompt: {system_prompt}")
        print(f"User prompt: {prompt}")
        print("Error:", str(e))
        print("Traceback:")
        traceback.print_exc()
        raise  # re-raise so the caller can handle it

def get_total_tokens():
    return TOTAL_TOKENS