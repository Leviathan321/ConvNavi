import time
from main import apply_structured_filters, get_embeddings_and_df, load_dataset

def check_if_poi_exists(df, constraints, user_location,
                        output_columns = ['name','category','rating','price_level','address'],
                        max_pois = 5):
    if df is None or df.empty:
        return False, []

    start_time = time.time()  # start timing
    df_filtered = apply_structured_filters(df, constraints, user_location)
    end_time = time.time()    # end timing
    print(f"[INFO] check_if_poi_exists execution time: {end_time - start_time:.4f} seconds")

    if output_columns is None:
        output_columns = df_filtered.columns.tolist()

    # ensure requested columns exist
    missing_cols = [col for col in output_columns if col not in df_filtered.columns]
    if missing_cols:
        raise KeyError(f"Missing columns in dataframe: {missing_cols}")

    pois_list = df_filtered[output_columns].to_dict(orient="records")
    exists = bool(pois_list)

    return exists, pois_list[:max_pois]

if __name__ == "__main__":
    # Load POI dataset
    path_dataset = "data/raw/yelp_academic_dataset_business.json"
    user_city = "Philadelphia"

    df_path="data/filtered_pois.csv"
    emb_path="data/embeddings.npy"

    embeddings, df = get_embeddings_and_df(
        path_dataset=path_dataset,
        df_path=df_path, 
        emb_path=emb_path,
        filter_city=user_city
    )

    # Example user location
    user_location = (39.955431, -75.154903)  # Philadelphia
    
    # Artificially generated constraints
    constraints = {
        "category": "supermarket",
        "cuisine": "french",
        "price_level": "low",
        "radius_km": None,
        "open_now": None,
        "rating": None,
        "name": None
    }

    # Check if at least one POI exists
    exists, matching_pois = check_if_poi_exists(df, constraints, user_location)

    if exists:
        print("Found at least one matching POI")
        print(matching_pois)
    else:
        print("No POI matches the constraints")
