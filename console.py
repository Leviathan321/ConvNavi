from typing import Optional, Tuple
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import json
from main import get_embeddings_and_df, run_rag_navigation
import os

load_dotenv()  # by default it looks for a .env in the working directory

# Read USE_NLU and convert to boolean
USE_NLU = os.getenv("USE_NLU", "False").lower() in ["1", "true", "yes"]

app = FastAPI()

# Preload data (for example, on startup)
filter_city = "Philadelphia"
path_dataset = "data/raw/yelp_academic_dataset_business.json"
user_location = (39.955431, -75.154903)  # Philadelphia, PA

# we load the filter dataset and embeddings to save time if they exist
df_path="data/filtered_pois.csv"
emb_path="data/embeddings.npy"

embeddings, df= get_embeddings_and_df(path_dataset,
                                      filter_city=filter_city,
                                      nrows = 300000) # number entries to use

if __name__ == "__main__":
    import sys
    from models import SessionManager
    from main import generate_episodic_summary, append_navigation_memory_episode_async, EPISODIC_MEMORY_BATCH_SIZE
    
    session_manager = SessionManager.get_instance()
    
    while True:
        try:
            user_query = input("Enter query (or 'exit'): ").strip()
            if user_query.lower() == 'exit':
                # Store episodic summary before exiting
                session = session_manager.get_session("1")
                if session and session.len() >= EPISODIC_MEMORY_BATCH_SIZE:
                    history = session.get_history()
                    episodic_summary = generate_episodic_summary(history, llm_model="gpt-4o-mini")
                    if episodic_summary:
                        append_navigation_memory_episode_async(episodic_summary, conversation_id=session.id, llm_model="gpt-4o-mini")
                break
            output = run_rag_navigation(
                query=user_query,
                user_location=user_location,
                embeddings=embeddings,
                df=df,
                use_nlu = USE_NLU               
            )
            print("--------------------------------")
            print("output:", json.dumps(output, indent=2))
            print("--------------------------------")
            print(">", user_query)
            print("<", output["response"])
        except Exception as e:
            print(f"Error: {e}")
