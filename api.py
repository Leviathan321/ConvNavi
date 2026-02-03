from enum import Enum
import traceback
from typing import Any, Dict, List, Optional, Tuple
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import json
from car.state import ENUM_MAP, CarState
from main import get_embeddings_and_df, run_rag_navigation
from models import Session, SessionManager
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
    llm_type: Optional[str] = "gpt-4o-mini"
    user_id: Optional[int] = 1
    
class POIQueryRequest(BaseModel):
    category: Optional[str] = None
    cuisine: Optional[str] = None
    price_level: Optional[str]= None
    radius_km: Optional[float] = None
    open_now: Optional[bool] = None
    rating: Optional[float] = None
    name: Optional[str] = None
    user_location: Optional[Tuple[float, float]] = Field(default=user_location)
    parking: Optional[str] = None
    
class POIExistsResponse(BaseModel):
    exists: bool
    matching_pois: List[Dict[str, Any]]

class InitialCarStateRequest(BaseModel):
    user_id: Optional[int] = 1
    car_state: Dict[str, Dict[str, Any]]  # mirrors CarState.state
    
# Route
@app.post("/query")
def query_handler(request: QueryRequest):
    try:
        print("Received query:", request)
        # set the llm to be used for answering
        if request.llm_type is not None:
            os.environ['LLM_MODEL'] = request.llm_type
            print("LLM model set: ", request.llm_type)
        llm_model = os.environ['LLM_MODEL']
        output = run_rag_navigation(
            query=request.query,
            user_location=user_location,
            embeddings=embeddings,
            df=df,
            use_nlu=USE_NLU,
            llm_model=llm_model,
            user_id=request.user_id
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

def set_initial_car_state(session: Session, raw_state: Dict[str, Dict[str, Any]]):
    """
    Replace the session's CarState with the given validated initial state.
    """
    car_state: CarState = session.car_state

    for subsystem, targets in raw_state.items():
        if subsystem not in car_state.state:
            raise ValueError(f"Unknown subsystem: {subsystem}")

        for target, value in targets.items():
            if target not in car_state.state[subsystem]:
                raise ValueError(f"Unknown target: {subsystem}.{target}")

            expected_type = ENUM_MAP[subsystem][target]

            # Enum case
            if issubclass(expected_type, Enum):
                car_state.state[subsystem][target] = expected_type(value)

            # Numeric case
            elif expected_type in (int, float):
                car_state.state[subsystem][target] = expected_type(value)

            else:
                raise ValueError(f"Unsupported type for {subsystem}.{target}")
            
@app.post("/carstate/init")
def init_car_state(request: InitialCarStateRequest):
    try:
        session_manager = SessionManager.get_instance()
        session = session_manager.get_session(request.user_id)

        if session is None:
            session = session_manager.create_session(request.user_id)

        set_initial_car_state(session, request.car_state)

        return {
            "success": True,
            "car_state": session.car_state.get_state(),
        }

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=str(e))
    
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
                use_nlu=USE_NLU,
                user_id = 1
            )
            print(json.dumps(output, indent=2))
        except Exception as e:
            print(f"Error: {e}")
