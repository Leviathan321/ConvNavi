from typing import Optional, Tuple
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import json
from main import get_embeddings_and_df, run_rag_navigation

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
    while True:
        try:
            user_query = input("Enter query (or 'exit'): ").strip()
            if user_query.lower() == 'exit':
                break
            output = run_rag_navigation(
                query=user_query,
                user_location=user_location,
                embeddings=embeddings,
                df=df
            )
            print(json.dumps(output, indent=2))
        except Exception as e:
            print(f"Error: {e}")
