import ollama
import time

def call_ollama(prompt, 
                max_tokens, 
                temperature, 
                model = "llama3.2", 
                system_prompt = None):
    # Define the conversation
    messages = [
        {"role": "system", "content" : system_prompt},
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
    return output_content,  response["prompt_eval_count"] + response["eval_count"]

if __name__ == "__main__":
    input = "I am in the mood for nice music."
    prompt = """
                Rephrase the following user utterance by using synonyms. Use synonyms that dont change the meaning.
                Provide 2 variations. 

                **Example**

                Input: I am very hungry.
                Ouput: [I am so much hungry., I am very open minded for nice food.]

                **Your turn**

                Input: {}
                Output: 

                """
    print(call_ollama(prompt=prompt.format(input),
                      max_tokens=1000,
                      temperature=0.1,
                      system_prompt="You are a system for replacing words with synonyms." + \
                                    "Only output the result. Dont explain. The results should be human like and natural." +  \
                                    "Replace only single words."))