import traceback
from typing import Any, Dict, List, Optional, Tuple
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import json
from main import get_embeddings_and_df, run_rag_navigation
from utils.check import check_if_poi_exists
from utils.format import sanitize_for_json
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

# Request schema
class QueryRequest(BaseModel):
    query: str
    user_location: Optional[Tuple[float, float]] = Field(default=user_location)

class POIQueryRequest(BaseModel):
    category: Optional[str] = None
    cuisine: Optional[str] = None
    price_level: Optional[str]= None
    radius_km: Optional[float] = None
    open_now: Optional[bool] = None
    rating: Optional[float] = None
    name: Optional[str] = None
    user_location: Optional[Tuple[float, float]] = Field(default=user_location)
    
class POIExistsResponse(BaseModel):
    exists: bool
    matching_pois: List[Dict[str, Any]]
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
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/poi_exists", response_model=POIExistsResponse)
def poi_exists(constraints: POIQueryRequest):
    try:
        constraints_dict = constraints.model_dump()
        print("['poi exists'] constraints:", constraints_dict)

        user_location = constraints_dict.pop("user_location")

        exists, matching_pois = check_if_poi_exists(df, constraints_dict, user_location)

        return POIExistsResponse(
            exists=exists,
            matching_pois=sanitize_for_json(matching_pois)
        )
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

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
                df=df,
                use_nlu=USE_NLU
            )
            print(json.dumps(output, indent=2))
        except Exception as e:
            print(f"Error: {e}")
