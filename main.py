import math
from typing import Dict
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from geopy.distance import geodesic
from datetime import datetime
import json
import re
from llm.llm_selector import pass_llm  # Your LLM call function
import os

model = SentenceTransformer('all-MiniLM-L6-v2')

def load_jsonl_to_df(filepath, nrows=None):
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if nrows and i >= nrows:
                break
            obj = json.loads(line)
            data.append(obj)
    return pd.DataFrame(data)


def preprocess_poi_json(row):
    categories = row.get('category', '')
    rating = row.get('rating', None)
    price_level = row.get('price_level', None)
    return f"{row.get('name', '')}, a {categories} place rated {rating}/5 at {row.get('address', '')}. Price: {price_level if price_level else 'N/A'}."


def parse_query_to_constraints(query):
    prompt = f"""
        You are an assistant that extracts structured filters from natural language queries for POI search.
        You have to understand implicit requests, request which are produced by humans of different cultural background,
        language level, age, profession, mood.
        Try to consider all preferences or constraints the user provides in his request.
        Do not output any explanation or other irrevelant information.

        Query: '{query}'

        Return a JSON object with the following fields:
        - category: string or null (e.g., "Restaurants", "Mexican", "Italian", "Fast Food")
        - cuisine: string or null (e.g., "Mexican", "Burgers", "Thai")
        - price_level: one of "$", "$$", "$$$", or null (based on 'RestaurantsPriceRange2' where 1="$", 2="$$", etc.)
        - radius_km: float (e.g., 5.0) or null
        - open_now: true/false/null
        - rating: float between 1.0 and 5.0 or null

        Examples:

        Query: "Show me Italian restaurants open now with price range two dollars and rating at least 4."
        {{"category": "Restaurants", "cuisine": "Italian", "price_level": "$$", "radius_km": null, "open_now": true, "rating": 4.0}}

        Query: "I want Mexican places with rating above 3.5 within 3 kilometers."
        {{"category": null, "cuisine": "Mexican", "price_level": null, "radius_km": 3.0, "open_now": null, "rating": 3.5}}

        Query: "Find fast food open now with low prices and rating above 4."
        {{"category": "Fast Food", "cuisine": null, "price_level": "$", "radius_km": null, "open_now": true, "rating": 4.0}}

        Query: "Show high class restaurants and rating at least 3."
        {{"category": "Restaurants", "cuisine": null, "price_level": "$$$", "radius_km": null, "open_now": null, "rating": 3.0}}
    """
    # print(f"[Query] Prompt constructed: {prompt}")
    response = pass_llm(prompt)[0]
    
    print(f"[Query] Response: {response}")
    return extract_json(response)


def extract_json(text):
    try:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            json_str = match.group(0)
            return json.loads(json_str)
    except json.JSONDecodeError:
        pass
    return None


def apply_structured_filters(df, intent, user_location):
    df_filtered = df.copy()

    if intent.get("category"):
        df_filtered = df_filtered[df_filtered['category'].str.contains(intent["category"], case=False, na=False)]
        # print(f"[Filter] Category '{intent['category']}'")

    if intent.get("cuisine"):
        pattern = r'\b' + re.escape(intent["cuisine"]) + r'\b'
        df_filtered = df_filtered[df_filtered['category'].str.contains(pattern, case=False, na=False, regex=True)]
        # print(f"[Filter] Cuisine '{intent['cuisine']}'")

    if intent.get("price_level"):
        if 'price_level' in df_filtered.columns:
            df_filtered = df_filtered[df_filtered['price_level'] == intent["price_level"]]
            # print(f"[Filter] Price level '{intent['price_level']}'")

    if intent.get("radius_km") is not None:
        def within_radius(row):
            poi_loc = (row['latitude'], row['longitude'])
            return geodesic(user_location, poi_loc).km <= intent["radius_km"]
        df_filtered = df_filtered[df_filtered.apply(within_radius, axis=1)]
        # print(f"[Filter] Radius <= {intent['radius_km']} km")

    if intent.get("open_now") is True:
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

    if intent.get("rating") is not None:
        df_filtered = df_filtered[df_filtered['rating'] >= intent["rating"]]
        # print(f"[Filter] Rating >= {intent['rating']}")

    return df_filtered


def retrieve_top_k_semantically(query, df_filtered, embeddings, k=3):
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
        return "Sorry, I cannot find any relevant places."

    pois_text = "\n".join([
        f"{i + 1}. {row['text']}" for i, row in pois_df.iterrows()
    ])

    prompt = f"""User query: "{query}"
Here are some relevant places:
{pois_text}

Based on the query and the above options, recommend the most suitable place and summarize briefly in 20 words. Ask if you should navigate to that place."""
    response = pass_llm(prompt=prompt)[0]
    print("response:", response)
    return response

def clean_json(obj):
    """
    Recursively clean NaN, inf, -inf values in a dict/list structure
    """
    if isinstance(obj, dict):
        return {k: clean_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_json(v) for v in obj]
    elif isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None  # or a default value like 0 or ""
    return obj

def run_rag_navigation(query, user_location, embeddings, df):
    intent = parse_query_to_constraints(query)
    # print("[INFO] Parsed intent:", intent)

    df_filtered = apply_structured_filters(df, intent, user_location)
    
    retrieved_pois = retrieve_top_k_semantically(query, df_filtered, embeddings=embeddings, k=5)
   
    response = generate_recommendation(query, retrieved_pois)
    # print("response:", response)

    pois_output = retrieved_pois[[
        'name', 'category', 'rating', 'price_level', 'address', 'latitude', 'longitude'
    ]].to_dict(orient="records")
    pois_output = [clean_json(poi) for poi in pois_output]

    print("response:", response)
    print("retreived_pois:", retrieved_pois)
    return {"response": response, "retrieved_pois": pois_output}

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

    def map_price_level(attributes: Dict) -> str:
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
        df = pd.read_csv(df_path)
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
    if df is None or embeddings is None:
        df = load_dataset(path_dataset, nrows=nrows, filter_city=filter_city)
        embeddings = create_embeddings(df)
        save_data(df, embeddings, df_path, emb_path)
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
