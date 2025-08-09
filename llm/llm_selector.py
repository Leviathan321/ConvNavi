from llm.call_openai import call_openai
from llm.call_ollama import call_ollama
import os
from dotenv import load_dotenv
import traceback

load_dotenv()

def pass_llm(prompt, 
             model=os.getenv("LLM_MODEL"),
             max_tokens=200, 
             temperature=0, 
             system_prompt="You are an in car conversational assistant."):
    """
    Call the selected LLM backend with error handling that logs inputs
    when a failure occurs.
    """
    try:
        if model in ("llama3.2", "mistral"):
            return call_ollama(
                prompt=prompt,
                max_tokens=max_tokens,
                system_prompt=system_prompt,
                temperature=temperature,
                model=model
            )
        else:
            return call_openai(
                prompt=prompt,
                max_tokens=max_tokens,
                system_prompt=system_prompt,
                temperature=temperature,
                model=model
            )

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
