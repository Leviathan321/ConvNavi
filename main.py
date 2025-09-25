import math
from typing import Dict, List
from dotenv import load_dotenv
from json_repair import repair_json
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from geopy.distance import geodesic
from datetime import datetime
import json
import re
from llm.llm_selector import pass_llm, get_total_tokens # Your LLM call function
import os
from models import SessionManager, Turn
from sentence_transformers import SentenceTransformer, util
import torch
import traceback

from prompts import PROMPT_GENERATE_RECOMMENDATION, PROMPT_NLU, PROMPT_PARSE_CONSTRAINTS
from utils.file import load_jsonl_to_df
from utils.format import clean_json, extract_json

load_dotenv()

top_k = int(os.environ.get("TOP_K", 3))

# Load embedding model once
model = SentenceTransformer('all-MiniLM-L6-v2')

def embed_texts(texts):
    """Embed a list of texts and return tensors."""
    return model.encode(texts, convert_to_tensor=True)

def preprocess_poi_json(row):
    categories = row.get('category', '')
    rating = row.get('rating', None)
    price_level = row.get('price_level', None)
    return f"{row.get('name', '')}, a {categories} place rated {rating}/5 at {row.get('address', '')}. Price: {price_level if price_level else 'N/A'}."

def parse_query_to_constraints(query: str, history: str = ""):
    prompt = PROMPT_PARSE_CONSTRAINTS.format(history, query)
    #print("prompt:", prompt)
    response = pass_llm(prompt)[0]
    #print("response before repair:", response)
    response = extract_json(repair_json(response))
    #print("reponse after:", response)
    return response

def apply_structured_filters(df, intent, user_location, 
                             use_embeddings_category = False,
                             similarity_threshold_category=0.8):
    print("**** apply structured filters ****")
    print("intent: ", intent)
    df_filtered = df.copy()

    def filter_contains_in_fields(df_filtered, key_word, field_names):
        search_lower = key_word.lower()
        mask = pd.Series(False, index=df_filtered.index)
        
        for field in field_names:
            mask |= df_filtered[field].fillna("").str.lower().str.contains(search_lower, na=False)
        
        return df_filtered[mask]

    if len(df_filtered) > 0 and intent.get("category"):
        if use_embeddings_category:
            # more precise, but quite slow in laptop
            # TODO precompute embeddings of categories
            categories = df_filtered['category'].fillna("").tolist()
            category_embeddings = embed_texts(categories)
            intent_embedding = embed_texts([intent["category"]])[0]
            
            # Compute cosine similarity
            cos_scores = util.cos_sim(intent_embedding, category_embeddings)[0]
            indices = torch.where(cos_scores >= similarity_threshold_category)[0].tolist()
            df_filtered = df_filtered.iloc[indices]
        else:
            df_filtered = filter_contains_in_fields(df_filtered, intent["category"], 
                                                    field_names=["category", "name"])
        print(df_filtered.head())

    if len(df_filtered) > 0 and intent.get("name"):
        pattern = re.escape(intent["name"])
        df_filtered = df_filtered[df_filtered['name'].str.contains(pattern, case=False, na=False)]
        
        print(df_filtered.head())

    if len(df_filtered) > 0 and intent.get("cuisine"):
        # need to check in category
        pattern = re.escape(intent["cuisine"])
        df_filtered = filter_contains_in_fields(df_filtered, pattern,
                                            field_names=["category", "name"])
        # print(f"[Filter] Cuisine '{intent['cuisine']}'")
        print(df_filtered.head())

    if len(df_filtered) > 0 and intent.get("price_level"):
        if 'price_level' in df_filtered.columns:

            df_filtered = df_filtered[df_filtered['price_level'] == intent["price_level"]]
            # print(f"[Filter] Price level '{intent['price_level']}'")
        print(df_filtered.head())

    if len(df_filtered) > 0 and intent.get("radius_km") is not None:
        def within_radius(row):
            poi_loc = (row['latitude'], row['longitude'])
            return geodesic(user_location, poi_loc).km <= intent["radius_km"]
        df_filtered = df_filtered[df_filtered.apply(within_radius, axis=1)]
        # print(f"[Filter] Radius <= {intent['radius_km']} km")
        print(df_filtered.head())

    if len(df_filtered) > 0 and  intent.get("open_now") is True:
        now = datetime.now().strftime("%H:%M")
        def is_open(row):
            try:
                hours = row['opening_hours']
                if isinstance(hours, dict):
                    day_name = datetime.now().strftime('%A')
                    if day_name in hours:
                        time_range = hours[day_name]
                    else:
                        return False
                else:
                    time_range = hours
                start, end = time_range.split("-")
                return start <= now <= end
            except Exception:
                return False
        df_filtered = df_filtered[df_filtered.apply(is_open, axis=1)]
        # print(f"[Filter] Open now at {now}")
        print(df_filtered.head())

    if len(df_filtered) > 0 and intent.get("rating") is not None:
        df_filtered = df_filtered[df_filtered['rating'] >= intent["rating"]]
        # print(f"[Filter] Rating >= {intent['rating']}")

    if len(df_filtered) > 0 and intent.get("parking") is not None:
        df_filtered = df_filtered[df_filtered['parking']]


    print("*** Structured filter applied.")
    return df_filtered


def retrieve_top_k_semantically(query, df_filtered, embeddings, k=top_k):
    if df_filtered.empty:
        return df_filtered

    idx_map = df_filtered.index.tolist()
    sub_embeddings = np.array([embeddings[i] for i in idx_map])

    sub_index = faiss.IndexFlatL2(embeddings.shape[1])
    sub_index.add(sub_embeddings)

    query_vec = model.encode([query])
    D, I = sub_index.search(query_vec, k)

    top_indices = [idx_map[i] for i in I[0] if i < len(idx_map)]
    return df_filtered.loc[top_indices]


def generate_recommendation(query, pois_df):
    if pois_df.empty:
        return "Sorry, I cannot find any relevant places. Do you have other preferences in mind?"

    pois_text = "\n".join([
        f"{i + 1}. {row['text']}" for i, row in pois_df.iterrows()
    ])

    prompt = PROMPT_GENERATE_RECOMMENDATION.format(query, pois_text)
    response = pass_llm(prompt=prompt)[0]
    # print("response:", response)
    return response

def nlu(query: str, history: str = ""):
    prompt=PROMPT_NLU.format(query, history)
    response = pass_llm(prompt)[0]
    print(response)
    return extract_json(response)

def run_rag_navigation(query, user_location, embeddings, df, use_nlu = True):
    print("NLU active:", use_nlu)
    session_manager= SessionManager.get_instance()
    session = session_manager.get_active_session()
    if session is None or session.len() >= session.max_turns:
        print("Creating new session...")
        session = session_manager.create_session()
    
    history = session_manager.get_active_session().get_history()

    turn = Turn(question=query, answer=None, retrieved_pois=[])
    session.add_turn(turn)

    # NLU
    print("history:", history)
    nlu_parsed = {}
    if use_nlu:
        nlu_parsed = nlu(query, history)
        print("nlu: ", nlu_parsed)
    else:
        # mimic nlu
        nlu_parsed["intent"] = "POI"

    if nlu_parsed["intent"] != "POI":
        response = nlu_parsed["response"]
        pois_output = []
    else:
        poi_constraints = parse_query_to_constraints(query, history = history)
        print("[INFO] Parsed poi intent:", poi_constraints)
        df_filtered = apply_structured_filters(df, poi_constraints, user_location)
        
        retrieved_pois = retrieve_top_k_semantically(query, df_filtered, embeddings=embeddings, k=top_k)
        response = generate_recommendation(query, retrieved_pois)

        pois_output = retrieved_pois[[
            'name', 'category', 'rating', 'price_level', 'address', 'latitude', 'longitude', "parking"
        ]].to_dict(orient="records")
        pois_output = [clean_json(poi) for poi in pois_output]

    session.complete(response, retrieved_pois = pois_output)

    return {
        "response": response,
        "retrieved_pois": pois_output,
        "session_id": session.id,
        "tokens_total": get_total_tokens(),
        "price_total": round(3*get_total_tokens() / (10**6), 3), # some value, TODO use table
    }

import ast  # for safely evaluating the string dict

def load_dataset(path_dataset, nrows, filter_city):
    df = load_jsonl_to_df(path_dataset)  # load entire or sufficient dataset
    
    # Filter by city (case-insensitive, ignoring NaN)
    df_filtered = df[df['city'].str.contains(filter_city, case=False, na=False)].copy()
    
    # Reset index
    df_filtered = df_filtered.reset_index(drop=True)
    
    # Select top nrows from filtered data
    if nrows is not None:
        df_filtered = df_filtered.head(nrows)
    
    df_filtered.rename(columns={'stars': 'rating', 'categories': 'category', 'hours': 'opening_hours'}, inplace=True)

    def map_price_level(attributes: dict) -> str:
        try:
            if isinstance(attributes, dict):
                val = attributes.get("RestaurantsPriceRange2", None)
                if val is not None:
                    mapping = {"1": "$", "2": "$$", "3": "$$$", "4": "$$$$"}
                    return mapping.get(str(val))
        except Exception:
            pass
        return None

    df_filtered['price_level'] = df_filtered['attributes'].apply(map_price_level)
    df_filtered['text'] = df_filtered.apply(preprocess_poi_json, axis=1)

    # New: extract whether parking exists
    def has_parking(attributes: dict) -> bool:
        try:
            if isinstance(attributes, dict):
                parking_str = attributes.get("BusinessParking", None)
                if parking_str and isinstance(parking_str, str):
                    parking_dict = ast.literal_eval(parking_str)  # convert string → dict
                    return any(parking_dict.values())
        except Exception:
            pass
        return False
    
    df_filtered['parking'] = df_filtered['attributes'].apply(has_parking)
    return df_filtered


def create_embeddings(df, do_save = True):
    # Need to save and load later the embedding vector to save time.
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(df['text'].tolist(), show_progress_bar=True)

    faiss_index = faiss.IndexFlatL2(embeddings.shape[1])
    faiss_index.add(np.array(embeddings))

    return embeddings

def save_data(df, embeddings, df_path="data/filtered_pois.csv", emb_path="data/embeddings.npy"):
    df.to_csv(df_path, index=False)
    np.save(emb_path, embeddings)

def load_data(df_path="filtered_pois.csv", emb_path="embeddings.npy"):
    if os.path.exists(df_path) and os.path.exists(emb_path):
        df = pd.read_csv(df_path, encoding='latin1')
        embeddings = np.load(emb_path)
        return df, embeddings
    else:
        return None, None
    
def get_embeddings_and_df(path_dataset, 
                          filter_city,
                          df_path = "data/filtered_pois.csv", 
                          emb_path ="data/embeddings.npy",
                          nrows= None):
    df, embeddings = load_data(df_path, emb_path)
    print("Loading dataset.")
    if df is None or embeddings is None:
        df = load_dataset(path_dataset, nrows=nrows, filter_city=filter_city)
        embeddings = create_embeddings(df)
        save_data(df, embeddings, df_path, emb_path)
        print("Dataset loaded.")
    return embeddings, df

if __name__ == "__main__":
    user_city = "Philadelphia"
    path_dataset = "data/raw/yelp_academic_dataset_business.json"

    user_queries = [
        "I will have a date today and want try some burger restaurant.",
        "I am in the mood for some asian food close by rating 4.",
        "My parents will visit my city, any american restaurant to check out?",
        "I like to have some english breakfast, not expensive."
    ]
    user_location = (39.955431, -75.154903)  # Philadelphia, PA

    # we load the filter dataset and embeddings to save time if they exist
    df_path="data/filtered_pois.csv"
    emb_path="data/embeddings.npy"
    ############
    for query in user_queries:
        embeddings, df = get_embeddings_and_df(path_dataset,
                                               df_path, 
                                               emb_path)
        print("\n--- RAG Recommendation System ---\n")
        output = run_rag_navigation(query, user_location, embeddings, df=df)

        print(output["response"])
        print(json.dumps(output["retrieved_pois"], indent=2))
