import ollama
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B")  # For LLaMA 3
# Adjust to your model version version
def call_ollama(prompt, 
                system_prompt = "You are an in car conversational assistant.",
                max_tokens = 200, 
                temperature = 0, 
                model = "llama3.2"):
    # Define the conversation
    messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
    ]

    # Chat with the model
    response = ollama.chat(
        model=model,
        messages=messages,
        options={
            "num_predict": max_tokens,  # Limit the response to 100 tokens
            "temperature": temperature,    # Controls randomness (0 = deterministic, 1 = max randomness)
            "top_p": 0.9,          # Nucleus sampling (alternative to temperature)
            "repeat_penalty": 1.1  # Discourages repetitive text
        }
    )

    # Extract the output message content
    output_content = response["message"]["content"]

    input_tokens = tokenizer(prompt, truncation=True, padding=False)
    output_tokens = tokenizer(output_content, truncation=True, padding=False)

    # Calculate total token size (input + output)
    total_tokens = len(input_tokens["input_ids"]) + len(output_tokens["input_ids"])

    return output_content, total_tokens
