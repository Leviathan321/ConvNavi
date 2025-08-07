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

# Request schema
class QueryRequest(BaseModel):
    query: str
    user_location: Optional[Tuple[float, float]] = Field(default=user_location)

# Route
@app.post("/query")
def query_handler(request: QueryRequest):
    try:
        output = run_rag_navigation(
            query=request.query,
            user_location=user_location,
            embeddings=embeddings,
            df=df
        )
        return output
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
