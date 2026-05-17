import math
from typing import Any, Dict, List
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
import threading
from pathlib import Path
from car.state import ENUM_MAP, POSSIBLE_CAR_VALUES
from llm.llm_selector import pass_llm, get_total_tokens, get_total_costs, get_query_costs
import os
from models import SessionManager, Turn, Context, POIPreferences
from sentence_transformers import SentenceTransformer, util
import torch
import traceback
import ast

from prompts import (
    PROMPT_CAR_RESPONSE,
    PROMPT_GENERATE_RECOMMENDATION,
    PROMPT_NLU,
    PROMPT_PARSE_CONSTRAINTS,
    PROMPT_NLU_WITH_CAR,
    PROMPT_CAR_UPDATE,
    PROMPT_CLASSIFY_ACTION,
    PROMPT_POI_CONFIRM_SELECT,
    PROMPT_FILL_MISSING_CONSTRAINTS,
    PROMPT_CONSOLIDATE_NAV_UTTERANCE,
    PROMPT_GENERATE_EPISODIC_SUMMARY,
)
from utils.file import load_jsonl_to_df
from utils.format import clean_json, extract_json, extract_json_list, sanitize_for_json

load_dotenv()

top_k = int(os.environ.get("TOP_K", 3))
NAV_MEMORY_MAX_ENTRIES = int(os.environ.get("NAV_MEMORY_MAX_ENTRIES", 1000))
NAV_MEMORY_TOP_K = int(os.environ.get("NAV_MEMORY_TOP_K", 7))
USE_MEMORY = os.environ.get("USE_MEMORY", "true").strip().lower() not in ("0", "false", "no", "off")
EPISODIC_MEMORY_BATCH_SIZE = int(os.environ.get("EPISODIC_MEMORY_BATCH_SIZE", int(os.environ.get("MAX_TURNS", 10))))
EMBEDDING_DEVICE = os.environ.get("EMBEDDING_DEVICE", "auto").strip().lower()
if EMBEDDING_DEVICE == "auto":
    EMBEDDING_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
elif EMBEDDING_DEVICE == "cuda" and not torch.cuda.is_available():
    print("[WARN] EMBEDDING_DEVICE=cuda requested but CUDA is unavailable. Falling back to cpu.")
    EMBEDDING_DEVICE = "cpu"
NAV_MEMORY_PATH = Path(__file__).resolve().parent / "data" / "navigation_memory.csv"
NAV_MEMORY_EMBEDDINGS_PATH = Path(__file__).resolve().parent / "data" / "navigation_memory_embeddings.npy"
NAV_MEMORY_FAISS_PATH = Path(__file__).resolve().parent / "data" / "navigation_memory.faiss"
NAV_MEMORY_LOCK = threading.Lock()

print(f"episodic memory config - USE_MEMORY: {USE_MEMORY}, BATCH_SIZE: {EPISODIC_MEMORY_BATCH_SIZE}, EMBEDDING_DEVICE: {EMBEDDING_DEVICE}")

# Load embedding model once
model = SentenceTransformer('all-MiniLM-L6-v2', device=EMBEDDING_DEVICE)

INTENT_NO_NLU = os.getenv("INTENT_NO_NLU", "poi").upper()
POI_CONSTRAINT_FIELDS = [
    "category",
    "cuisine",
    "price_level",
    "radius_km",
    "open_now",
    "rating",
    "parking",
    "has_outdoor_seating",
    "noise_level",
    "good_for_kids",
    "name",
]
POI_OUTPUT_COLUMNS = [
    "name",
    "category",
    "rating",
    "price_level",
    "address",
    "has_outdoor_seating",
    "noise_level",
    "good_for_kids",
    "parking",
]

PROMPT_POI_INFO = """You are a helpful car navigation assistant. The user is asking a question about a place.

Conversation history:
{history}

Last recommended places:
{pois}

User question: "{query}"

Context is: "{context}"

Answer the user's question based on the available information about the places. 
If the information is not available, try to provide some information that makes sense.
Or direct to a dedicated service (e.g., a weather app, a traffic app).
Be concise unter 15 words.
"""


def embed_texts(texts):
    """Embed a list of texts and return tensors."""
    return model.encode(texts, convert_to_tensor=True)


def preprocess_poi_json(row):
    categories = row.get('category', '')
    rating = row.get('rating', None)
    price_level = row.get('price_level', None)
    return f"{row.get('name', '')}, a {categories} place rated {rating}/5 at {row.get('address', '')}. Price: {price_level if price_level else 'N/A'}."


def _parse_attributes(attributes):
    if isinstance(attributes, dict):
        return attributes
    if isinstance(attributes, str) and attributes.strip():
        try:
            parsed = ast.literal_eval(attributes)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            return {}
    return {}


def _normalize_bool_like(value):
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, np.integer)) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().strip('"').strip("'").lower()
        if normalized in ("true", "t", "1", "yes"):
            return True
        if normalized in ("false", "f", "0", "no"):
            return False
    return None


def _normalize_text_like(value):
    if value is None:
        return None
    if isinstance(value, float) and pd.isna(value):
        return None
    if isinstance(value, str):
        normalized = value.strip()
        normalized = re.sub(r"^u(['\"])(.*)\1$", r"\2", normalized)
        normalized = normalized.strip("'\"").strip()
        return normalized.lower() if normalized else None
    normalized = str(value).strip()
    return normalized.lower() if normalized else None


def _extract_attribute(attributes, attribute_name, value_type):
    attribute_map = _parse_attributes(attributes)
    raw_value = attribute_map.get(attribute_name)
    if value_type == "bool":
        return _normalize_bool_like(raw_value)
    if value_type == "text":
        return _normalize_text_like(raw_value)
    return raw_value


def _with_derived_poi_columns(df):
    if df is None or df.empty or "attributes" not in df.columns:
        return df

    df = df.copy()

    if "has_outdoor_seating" not in df.columns:
        df["has_outdoor_seating"] = df["attributes"].apply(
            lambda attributes: _extract_attribute(attributes, "OutdoorSeating", "bool")
        )

    if "noise_level" not in df.columns:
        df["noise_level"] = df["attributes"].apply(
            lambda attributes: _extract_attribute(attributes, "NoiseLevel", "text")
        )

    if "good_for_kids" not in df.columns:
        df["good_for_kids"] = df["attributes"].apply(
            lambda attributes: _extract_attribute(attributes, "GoodForKids", "bool")
        )

    if "parking" not in df.columns:
        def _has_parking(attributes):
            try:
                parking_value = _parse_attributes(attributes).get("BusinessParking")
                if parking_value:
                    parking_dict = ast.literal_eval(parking_value)
                    if isinstance(parking_dict, dict):
                        return any(bool(value) for value in parking_dict.values())
            except Exception:
                pass
            return False

        df["parking"] = df["attributes"].apply(_has_parking)

    return df


def _empty_memory_frame():
    return pd.DataFrame(columns=["time", "conversation_id", "summary"])


def load_navigation_memory(memory_path: Path = NAV_MEMORY_PATH):
    if not USE_MEMORY:
        return _empty_memory_frame()

    if not memory_path.exists():
        return _empty_memory_frame()

    try:
        memory_df = pd.read_csv(memory_path)
    except Exception:
        return _empty_memory_frame()

    if memory_df.empty:
        return _empty_memory_frame()

    for column in ["time", "conversation_id", "summary"]:
        if column not in memory_df.columns:
            memory_df[column] = ""

    return memory_df[["time", "conversation_id", "summary"]].fillna("")


def generate_episodic_summary(history: str, llm_model: str = "") -> str:
    """Generate an episodic summary from multiple conversation turns."""
    if not history or history == "[]":
        return ""
    
    prompt = PROMPT_GENERATE_EPISODIC_SUMMARY.format(history=history)
    response, _, _ = pass_llm(prompt=prompt, model=llm_model)
    summary = response.strip().strip('"').strip("'")
    if not summary:
        return ""
    return summary


def append_navigation_memory_episode(summary: str, conversation_id: str, memory_path: Path = NAV_MEMORY_PATH):
    """Store an episodic summary to navigation memory."""
    if not USE_MEMORY or not summary:
        return

    with NAV_MEMORY_LOCK:
        memory_df = load_navigation_memory(memory_path)
        embeddings = load_navigation_memory_embeddings(memory_df, memory_path)

        new_row = pd.DataFrame([
            {
                "time": datetime.utcnow().isoformat(timespec="seconds") + "Z",
                "conversation_id": str(conversation_id),
                "summary": summary,
            }
        ])

        new_embedding = np.asarray(embed_texts([summary]).cpu().numpy(), dtype=np.float32)

        memory_df = pd.concat([memory_df, new_row], ignore_index=True)
        embeddings = np.concatenate([embeddings, new_embedding], axis=0) if len(embeddings) else new_embedding

        if len(memory_df) > NAV_MEMORY_MAX_ENTRIES:
            memory_df = memory_df.iloc[-NAV_MEMORY_MAX_ENTRIES:].reset_index(drop=True)
            embeddings = embeddings[-NAV_MEMORY_MAX_ENTRIES:]

        normalized_embeddings = embeddings.astype(np.float32, copy=True)
        if len(normalized_embeddings) > 0:
            faiss.normalize_L2(normalized_embeddings)
        index = faiss.IndexFlatIP(normalized_embeddings.shape[1])
        index.add(normalized_embeddings)

        memory_path.parent.mkdir(parents=True, exist_ok=True)
        memory_df.to_csv(memory_path, index=False)
        np.save(NAV_MEMORY_EMBEDDINGS_PATH, embeddings)
        faiss.write_index(index, str(NAV_MEMORY_FAISS_PATH))
        
        # print(f"[INFO] Episodic memory stored for conversation {conversation_id}: {summary[:50]}...")


def append_navigation_memory_episode_async(summary: str, conversation_id: str, llm_model: str = ""):
    """Store an episodic summary asynchronously."""
    if not USE_MEMORY or not summary:
        return

    worker = threading.Thread(
        target=append_navigation_memory_episode,
        kwargs={
            "summary": summary,
            "conversation_id": conversation_id,
        },
        daemon=True,
    )
    worker.start()


def append_navigation_memory(utterance: str, conversation_id: str, llm_model: str = "", memory_path: Path = NAV_MEMORY_PATH):
    """Deprecated: Use append_navigation_memory_episode instead. Kept for backward compatibility."""
    if not USE_MEMORY:
        return

    # This function is kept for backward compatibility but does nothing now
    # Episodic summaries are stored when conversation stops or max turns reached


def load_navigation_memory_embeddings(memory_df: pd.DataFrame, memory_path: Path = NAV_MEMORY_PATH):
    if not USE_MEMORY:
        return np.empty((0, model.get_sentence_embedding_dimension()), dtype=np.float32)

    if memory_df.empty:
        return np.empty((0, model.get_sentence_embedding_dimension()), dtype=np.float32)

    if NAV_MEMORY_EMBEDDINGS_PATH.exists():
        try:
            embeddings = np.load(NAV_MEMORY_EMBEDDINGS_PATH)
            if len(embeddings) == len(memory_df):
                return embeddings
        except Exception:
            pass

    embeddings = np.asarray(embed_texts(memory_df["summary"].astype(str).tolist()).cpu().numpy(), dtype=np.float32)
    memory_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(NAV_MEMORY_EMBEDDINGS_PATH, embeddings)
    return embeddings


def load_navigation_memory_index(memory_df: pd.DataFrame, memory_path: Path = NAV_MEMORY_PATH):
    if not USE_MEMORY:
        return None

    if memory_df.empty:
        return None

    if NAV_MEMORY_FAISS_PATH.exists():
        try:
            index = faiss.read_index(str(NAV_MEMORY_FAISS_PATH))
            if index.ntotal == len(memory_df):
                return index
        except Exception:
            pass

    embeddings = load_navigation_memory_embeddings(memory_df, memory_path).astype(np.float32, copy=True)
    if len(embeddings) == 0:
        return None

    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    memory_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(NAV_MEMORY_FAISS_PATH))
    return index


def retrieve_navigation_memory(query: str, constraints: Dict[str, Any], missing_fields: List[str] = None,
                               top_k: int = NAV_MEMORY_TOP_K, memory_path: Path = NAV_MEMORY_PATH):
    if not USE_MEMORY:
        return []

    memory_df = load_navigation_memory(memory_path)
    if memory_df.empty:
        return []

    missing_fields = missing_fields or []
    missing_fields_text = ", ".join(missing_fields) if missing_fields else "none"
    search_text = (
        f"query: {query}\n"
        f"missing fields: {missing_fields_text}"
    )
    print("search_text:", search_text)
    search_embedding = np.asarray(embed_texts([search_text]).cpu().numpy(), dtype=np.float32)
    faiss.normalize_L2(search_embedding)

    index = load_navigation_memory_index(memory_df, memory_path)
    if index is None:
        return []

    top_n = min(top_k, len(memory_df))
    if top_n <= 0:
        return []

    scores, top_indices = index.search(search_embedding, top_n)
    retrieved = []
    for score, idx in zip(scores[0].tolist(), top_indices[0].tolist()):
        if idx < 0 or idx >= len(memory_df):
            continue
        row = memory_df.iloc[int(idx)]
        retrieved.append({
            "conversation_id": row.get("conversation_id", ""),
            "time": row.get("time", ""),
            "summary": row.get("summary", ""),
            "score": float(score),
        })

    return retrieved


def fill_missing_constraints_with_memory(query: str, history: str, constraints: Dict[str, Any], context: Context,
                                        llm_model: str = "", skip_fields: List[str] = None):
    canonical_fields = POI_CONSTRAINT_FIELDS

    constraint_snapshot = {field: constraints.get(field) if field in constraints else None for field in canonical_fields}

    skip_fields = set(skip_fields or [])

    # missing if key is absent or present with None value, excluding fields explicitly cleared this turn
    missing_fields = [
        field for field, value in constraint_snapshot.items()
        if value is None and field not in skip_fields
    ]
    print("[DEBUG] Current constraints:", constraints)
    print("[INFO] Missing constraint fields:", missing_fields)

    if not missing_fields:
        return constraints, 0, 0

    # STEP 1: Try to fill from context preferences first
    updated_constraints = dict(constraints)
    remaining_missing_fields = []
    
    for field in missing_fields:
        value_from_context = None
        
        # Check context.preferences.poi for all POI-related fields
        if hasattr(context.preferences.poi, field):
            context_value = getattr(context.preferences.poi, field)
            if context_value is not None:
                value_from_context = context_value
        
        if value_from_context is not None:
            updated_constraints[field] = value_from_context
            print(f"[DEBUG] Filled '{field}' from context: {value_from_context}")
        else:
            remaining_missing_fields.append(field)
    
    # STEP 2: Retrieve from memory for remaining missing fields
    if not remaining_missing_fields and USE_MEMORY:
        return updated_constraints, 0, 0

    retrieved_memory = []
    if USE_MEMORY:
        retrieved_memory = retrieve_navigation_memory(query, constraint_snapshot, missing_fields=remaining_missing_fields)
        if not retrieved_memory:
            return updated_constraints, 0, 0

    print("retrieved_memory:", retrieved_memory)

    prompt = PROMPT_FILL_MISSING_CONSTRAINTS.format(
        history=history,
        query=query,
        constraints=json.dumps(sanitize_for_json(constraint_snapshot), ensure_ascii=False, indent=2),
        missing_fields=json.dumps(remaining_missing_fields, ensure_ascii=False, indent=2),
        memory=json.dumps(sanitize_for_json(retrieved_memory), ensure_ascii=False, indent=2),
    )
    response, tokens_input, tokens_output = pass_llm(prompt=prompt, model=llm_model)
    print("[DEBUG] LLM response for filling missing constraints:", response)
    try:
        parsed = extract_json(repair_json(response))
    except Exception:
        parsed = None
    

    if not isinstance(parsed, dict):
        return updated_constraints, tokens_input, tokens_output

    for key, value in parsed.items():
        if key in skip_fields:
            continue
        if value is None:
            continue
        # set if absent or currently None
        if key not in updated_constraints or updated_constraints.get(key) is None:
            updated_constraints[key] = value

    print("[DEBUG] updated constraints after memory fill:", updated_constraints)
    return updated_constraints, tokens_input, tokens_output


def parse_query_to_constraints(query: str, history: str = "", llm_model: str = ""):
    prompt = PROMPT_PARSE_CONSTRAINTS.format(history, query)
    response, input_tokens, output_tokens = pass_llm(prompt, model=llm_model)
    response = extract_json(repair_json(response))
    return response, input_tokens, output_tokens


def  classify_action(query, history, llm_model=""):
    """Single lightweight classifier to determine user action within POI flow."""
    prompt = PROMPT_CLASSIFY_ACTION.format(history=history, query=query)
    response, tokens_input, tokens_output = pass_llm(prompt, model=llm_model)
    result = extract_json(repair_json(response))
    return result.get("action", "refine"), tokens_input, tokens_output


def apply_structured_filters(df, intent, user_location,
                             use_embeddings_category=False,
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
            categories = df_filtered['category'].fillna("").tolist()
            category_embeddings = embed_texts(categories)
            intent_embedding = embed_texts([intent["category"]])[0]
            cos_scores = util.cos_sim(intent_embedding, category_embeddings)[0]
            indices = torch.where(cos_scores >= similarity_threshold_category)[0].tolist()
            df_filtered = df_filtered.iloc[indices]
        else:
            df_filtered = filter_contains_in_fields(df_filtered, intent["category"],
                                                    field_names=["category", "name"])
        # print(df_filtered.head())

    if len(df_filtered) > 0 and intent.get("name"):
        pattern = re.escape(intent["name"])
        df_filtered = df_filtered[df_filtered['name'].str.contains(pattern, case=False, na=False)]
        print(df_filtered.head())

    if len(df_filtered) > 0 and intent.get("cuisine"):
        pattern = re.escape(intent["cuisine"])
        df_filtered = filter_contains_in_fields(df_filtered, pattern,
                                                field_names=["category", "name"])
        print(df_filtered.head())

    if len(df_filtered) > 0 and intent.get("price_level"):
        if 'price_level' in df_filtered.columns:
            df_filtered = df_filtered[df_filtered['price_level'] == intent["price_level"]]
        print(df_filtered.head())

    if len(df_filtered) > 0 and intent.get("has_outdoor_seating") is not None:
        expected_value = _normalize_bool_like(intent["has_outdoor_seating"])
        if expected_value is not None and 'has_outdoor_seating' in df_filtered.columns:
            df_filtered = df_filtered[
                df_filtered['has_outdoor_seating'].apply(_normalize_bool_like) == expected_value
            ]
        print(df_filtered.head())

    if len(df_filtered) > 0 and intent.get("noise_level"):
        expected_value = _normalize_text_like(intent["noise_level"])
        if expected_value and 'noise_level' in df_filtered.columns:
            df_filtered = df_filtered[
                df_filtered['noise_level'].fillna("").astype(str).str.lower().str.strip() == expected_value
            ]
        print(df_filtered.head())

    if len(df_filtered) > 0 and intent.get("good_for_kids") is not None:
        expected_value = _normalize_bool_like(intent["good_for_kids"])
        if expected_value is not None and 'good_for_kids' in df_filtered.columns:
            df_filtered = df_filtered[
                df_filtered['good_for_kids'].apply(_normalize_bool_like) == expected_value
            ]
        print(df_filtered.head())

    if len(df_filtered) > 0 and intent.get("radius_km") is not None:
        print("user_location:", user_location)
        def within_radius(row):
            poi_loc = (row['latitude'], row['longitude'])
            return geodesic(user_location, poi_loc).km <= intent["radius_km"]
        df_filtered = df_filtered[df_filtered.apply(within_radius, axis=1)]
        print(df_filtered.head())

    if len(df_filtered) > 0 and intent.get("open_now") is True:
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
        print(df_filtered.head())

    if len(df_filtered) > 0 and intent.get("rating") is not None:
        df_filtered = df_filtered[df_filtered['rating'] >= intent["rating"]]

    if len(df_filtered) > 0 and intent.get("parking") is not None:
        expected_value = _normalize_bool_like(intent["parking"])
        if expected_value is not None and 'parking' in df_filtered.columns:
            df_filtered = df_filtered[df_filtered['parking'].apply(_normalize_bool_like) == expected_value]

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


def generate_recommendation(query, pois_df, llm_model, history, context: Context = None):
    if pois_df.empty:
        return "Sorry, I cannot find any relevant places. Do you have other preferences in mind?", 0, 0

    pois_text = "\n".join([
        f"{i + 1}. {row['text']}" for i, row in pois_df.iterrows()
    ])

    # Format context if provided
    context_str = json.dumps(context.to_dict(), indent=2, ensure_ascii=False) if context else "{}"

    prompt = PROMPT_GENERATE_RECOMMENDATION.format(query, pois_text, history, context_str)
    response, tokens_input, tokens_output = pass_llm(prompt=prompt, model=llm_model)
    return response, tokens_input, tokens_output


def nlu(query: str, history: str = ""):
    prompt = PROMPT_NLU_WITH_CAR.format(query, history)
    response, tokens_input, tokens_output = pass_llm(prompt)
    return extract_json(response), tokens_input, tokens_output


def _build_return_dict(response, pois_output, session, user_id,
                       tokens_query_input, tokens_query_output, **extra):
    """Helper to build the standard return dictionary."""
    result = {
        "response": response,
        "retrieved_pois": pois_output,
        "session_id": session.id,
        "user_id": user_id,
        "tokens_total": get_total_tokens(),
        "tokens_query_input": tokens_query_input,
        "tokens_query_output": tokens_query_output,
        "price_query": get_query_costs(),
        "price_total": get_total_costs(),
    }
    result.update(extra)
    return result


def _finalize_turn(session, query, response, pois_output):
    """Helper to add and complete a turn."""
    session.add_turn(Turn(question=query, answer=None, retrieved_pois=[]))
    session.complete(response, retrieved_pois=pois_output)


def _handle_car_intent(query, session, history, llm_model,
                       tokens_query_input, tokens_query_output, user_id):
    """Handle the CAR intent flow."""
    print("[DEBUG] CAR intent")

    car_state = session.car_state

    prompt = PROMPT_CAR_UPDATE.format(
        current_state=car_state.get_state(),
        history=history,
        query=query,
        possible_values=POSSIBLE_CAR_VALUES,
    )

    output_str, tokens_input, tokens_output = pass_llm(
        prompt=prompt, model=llm_model
    )
    tokens_query_input += tokens_input
    tokens_query_output += tokens_output

    output_str = repair_json(output_str)
    result = json.loads(output_str)

    response = result.get("summary", "")
    changes = result.get("changes", [])

    for change in changes:
        subsystem = change.get("subsystem")
        target = change.get("target")
        value = change.get("value")

        if subsystem not in car_state.state:
            continue

        if target not in car_state.state[subsystem]:
            continue

        current_val = car_state.state[subsystem][target]

        if isinstance(current_val, (int, float)):
            if value == "increase":
                car_state.state[subsystem][target] += 1
            elif value == "decrease":
                car_state.state[subsystem][target] -= 1
            else:
                car_state.state[subsystem][target] = value
        else:
            enum_class = ENUM_MAP[subsystem][target]
            car_state.state[subsystem][target] = enum_class(value)

    pois_output = car_state.get_state()

    _finalize_turn(session, query, response, pois_output)

    return _build_return_dict(
        response, pois_output, session, user_id,
        tokens_query_input, tokens_query_output
    )


def _handle_poi_stop(query, session, user_id,
                     tokens_query_input, tokens_query_output,
                     llm_model=""):
    """Handle the STOP action within POI flow."""
    response = "Okay, ending the conversation."
    pois_output = []

    session.ended_by_user = True

    # Store episodic summary when conversation stops
    history = session.get_history()
    episodic_summary = generate_episodic_summary(history, llm_model=llm_model)
    if episodic_summary:
        append_navigation_memory_episode_async(episodic_summary, conversation_id=session.id, llm_model=llm_model)

    _finalize_turn(session, query, response, pois_output)

    return _build_return_dict(
        response, pois_output, session, user_id,
        tokens_query_input, tokens_query_output,
        conversation_finished=True
    )


def _handle_poi_confirm(query, session, user_id,
                        tokens_query_input, tokens_query_output,
                        llm_model=""):
    """Handle the CONFIRM action — start navigation to the POI the user confirmed (LLM-selected).

    Note: If the LLM cannot confidently select a POI, we do NOT ask the user to clarify here.
    We fall back to the top-ranked last POI (previous behavior), and still start navigation.
    """
    last_pois = session.get_last_retrieved_pois() or []

    if not last_pois:
        response = "I don't have a destination yet. Where would you like to go?"
        pois_output = []
        _finalize_turn(session, query, response, pois_output)
        return _build_return_dict(
            response, pois_output, session, user_id,
            tokens_query_input, tokens_query_output,
            navigation_started=False
        )

    # Build candidates with stable IDs if present; otherwise index fallback.
    candidates = []
    for i, p in enumerate(last_pois):
        poi_id = p.get("id")
        if poi_id is None:
            poi_id = str(i)  # fallback id
        candidates.append({
            "id": str(poi_id),
            "rank": i + 1,
            "name": p.get("name", ""),
            "address": p.get("address", ""),
            "category": p.get("category", ""),
            "rating": p.get("rating", None),
            "price_level": p.get("price_level", None),
        })

    history = session.get_history()
    pois_text = json.dumps(candidates, indent=2, ensure_ascii=False)

    prompt = PROMPT_POI_CONFIRM_SELECT.format(
        history=history,
        pois=pois_text,
        query=query,
    )

    llm_out, tokens_input, tokens_output = pass_llm(prompt=prompt, model=llm_model)
    tokens_query_input += tokens_input
    tokens_query_output += tokens_output

    # Parse model output robustly (same pattern as the rest of your code)
    try:
        llm_out = repair_json(llm_out)
        parsed = extract_json(llm_out)
    except Exception:
        parsed = {}

    selected_id = parsed.get("selected_poi_id", None)
    confidence = parsed.get("confidence", "low")

    selected_poi = None
    if selected_id is not None:
        # 1) Try matching on actual POI id (if present)
        for p in last_pois:
            if p.get("id") is not None and str(p.get("id")) == str(selected_id):
                selected_poi = p
                break

        # 2) If not found, allow index fallback id
        if selected_poi is None and str(selected_id).isdigit():
            idx = int(str(selected_id))
            if 0 <= idx < len(last_pois):
                selected_poi = last_pois[idx]

    # No user clarification: fallback to previous behavior
    if selected_poi is None or confidence == "low":
        selected_poi = last_pois[0]

    response = f"Starting navigation to {selected_poi.get('name')} at {selected_poi.get('address')}."
    pois_output = [selected_poi]

    _finalize_turn(session, query, response, pois_output)

    # Store episodic summary when user confirms (navigation starts)
    history = session.get_history()
    episodic_summary = generate_episodic_summary(history, llm_model=llm_model)
    if episodic_summary:
        append_navigation_memory_episode_async(episodic_summary, conversation_id=session.id, llm_model=llm_model)

    return _build_return_dict(
        response, pois_output, session, user_id,
        tokens_query_input, tokens_query_output,
        navigation_started=True
    )
def _handle_poi_info(query, session, user_id, history, llm_model,
                     tokens_query_input, tokens_query_output):
    """Handle the INFO action — answer questions about previously suggested POIs."""
    last_pois = session.get_last_retrieved_pois()

    if last_pois:
        pois_text = json.dumps(last_pois, indent=2)
    else:
        pois_text = "No places have been suggested yet."

    # Format context from Pydantic model
    context_str = json.dumps(session.user_context.to_dict(), indent=2, ensure_ascii=False)

    prompt = PROMPT_POI_INFO.format(
        history=history,
        pois=pois_text,
        query=query,
        context=context_str
    )

    response, tokens_input, tokens_output = pass_llm(
        prompt=prompt, model=llm_model
    )
    tokens_query_input += tokens_input
    tokens_query_output += tokens_output

    pois_output = last_pois if last_pois else []

    _finalize_turn(session, query, response, pois_output)

    return _build_return_dict(
        response, pois_output, session, user_id,
        tokens_query_input, tokens_query_output
    )

def _handle_poi_refine(query, session, user_location, embeddings, df,
                       llm_model, user_id, history,
                       tokens_query_input, tokens_query_output,
                       reset_constraints: bool = False):
    """Handle the REFINE action — parse new constraints, retrieve POIs.
       If reset_constraints=True: start from empty constraints (change of mind).
    """
    new_constraints, input_tokens, output_tokens = parse_query_to_constraints(
        query, history=history, llm_model=llm_model
    )

    print("[DEBUG] New constraints from query:", new_constraints)
    tokens_query_input += input_tokens
    tokens_query_output += output_tokens

    canonical_fields = POI_CONSTRAINT_FIELDS

    explicit_field_updates = {
        k: new_constraints.get(k)
        for k in canonical_fields
        if k in new_constraints
    }
    explicit_null_fields = [k for k, v in explicit_field_updates.items() if v is None]

    if reset_constraints:
        session.poi_constraints = {}

    if explicit_field_updates:
        # Apply explicit updates, including None values to clear previous constraints.
        session.poi_constraints.update(explicit_field_updates)

    # Keep context aligned when a field is explicitly cleared to avoid auto-refilling from old context.
    for field in explicit_null_fields:
        if hasattr(session.user_context.preferences.poi, field):
            setattr(session.user_context.preferences.poi, field, None)

    session.poi_constraints, input_tokens, output_tokens = fill_missing_constraints_with_memory(
        query=query,
        history=history,
        constraints=session.poi_constraints,
        context=session.user_context,
        llm_model=llm_model,
    )
    tokens_query_input += input_tokens
    tokens_query_output += output_tokens

    # Update context with filled constraints
    for field in POI_CONSTRAINT_FIELDS:
        if field in session.poi_constraints and session.poi_constraints[field] is not None:
            if hasattr(session.user_context.preferences.poi, field):
                setattr(session.user_context.preferences.poi, field, session.poi_constraints[field])
    
    print("[INFO] Accumulated POI constraints:", session.poi_constraints)
    print("[DEBUG] Updated context preferences:", session.user_context.preferences.poi)

    df_filtered = apply_structured_filters(
        df, session.poi_constraints, user_location
    )

    retrieved_pois = retrieve_top_k_semantically(
        query, df_filtered, embeddings=embeddings, k=top_k
    )

    print("session.user_context:", session.user_context)

    response, input_tokens, output_tokens = generate_recommendation(
        query, retrieved_pois, llm_model=llm_model, history=session.get_history(), 
        context=session.user_context
    )
    tokens_query_input += input_tokens
    tokens_query_output += output_tokens

    output_columns = [column for column in POI_OUTPUT_COLUMNS if column in retrieved_pois.columns]
    pois_output = retrieved_pois[output_columns].to_dict(orient="records")
    pois_output = [clean_json(poi) for poi in pois_output]

    _finalize_turn(session, query, response, pois_output)

    return _build_return_dict(
        response, pois_output, session, user_id,
        tokens_query_input, tokens_query_output
    )


def run_rag_navigation(
    query,
    user_location,
    embeddings,
    df,
    use_nlu=True,
    llm_model=os.environ["LLM_MODEL"],
    user_id=1,
    user_location_name: str = "Philadelphia, PA",
):
    print("[DEBUG] NLU active:", use_nlu)

    session_manager = SessionManager.get_instance()
    session = session_manager.get_session(user_id)

    if session is None or session.ended_by_user or session.len() >= session.max_turns:
        # Store episodic summary if we're ending a session
        if session is not None and session.len() >= session.max_turns:
            history = session.get_history()
            episodic_summary = generate_episodic_summary(history, llm_model=llm_model)
            if episodic_summary:
                append_navigation_memory_episode_async(episodic_summary, conversation_id=session.id, llm_model=llm_model)
                print(f"[INFO] Max turns reached. Episodic memory stored for session {session.id}")
        elif session is not None and session.ended_by_user:
            print(f"[INFO] Session {session.id} ended by user. Creating a new session.")
    
        print("[DEBUG] Creating new session...")
        session = session_manager.create_session(user_id)
        # Initialize context with location name
        session.user_context.location = user_location_name

    print("[DEBUG] User ID:", user_id, "Session ID:", session.id)
    print("[DEBUG] User Location:", session.user_context.location)

    history = session.get_history()

    tokens_query_input = 0
    tokens_query_output = 0

    # --------------------
    # NLU (if enabled, handles CAR vs POI split)
    # --------------------
    if use_nlu:
        nlu_parsed, tokens_input, tokens_output = nlu(query, history)
        tokens_query_input += tokens_input
        tokens_query_output += tokens_output
        intent = nlu_parsed.get("intent")
    else:
        intent = INTENT_NO_NLU

    # Non-POI, non-CAR intents (e.g. greetings, chitchat)
    if intent not in ("POI", "CAR"):
        response = nlu_parsed.get("response", "")
        pois_output = []
        _finalize_turn(session, query, response, pois_output)
        return _build_return_dict(
            response, pois_output, session, user_id,
            tokens_query_input, tokens_query_output
        )

    # CAR intent
    if intent == "CAR":
        return _handle_car_intent(
            query, session, history, llm_model,
            tokens_query_input, tokens_query_output, user_id
        )

    # --------------------
    # POI FLOW: single classifier → route
    # --------------------
    print("[DEBUG] Intent:", intent)
    if intent == "POI":
        # Note: Individual turns are no longer stored. Episodic summaries are stored when:
        action, tokens_input, tokens_output = classify_action(
            query, history, llm_model=llm_model
        )
        tokens_query_input += tokens_input
        tokens_query_output += tokens_output

        print(f"[DEBUG] Classified action: {action}")

        if action == "stop":
            return _handle_poi_stop(
                query, session, user_id,
                tokens_query_input, tokens_query_output,
                llm_model=llm_model
            )

        elif action == "confirm":
            return _handle_poi_confirm(
                query, session, user_id,
                tokens_query_input, tokens_query_output,
                llm_model=llm_model
            )

        elif action == "info":
            return _handle_poi_info(
                query, session, user_id, history, llm_model,
                tokens_query_input, tokens_query_output
            )

        elif action == "change_of_mind":
            # Same as refine, but start from empty constraints/state
            return _handle_poi_refine(
                query, session, user_location, embeddings, df,
                llm_model, user_id, history,
                tokens_query_input, tokens_query_output,
                reset_constraints=True
            )

        elif action == "refine":
            return _handle_poi_refine(
                query, session, user_location, embeddings, df,
                llm_model, user_id, history,
                tokens_query_input, tokens_query_output
            )
    raise ValueError(f"Unhandled intent: {intent}")


# --------------------
# DATA LOADING
# --------------------

def load_dataset(path_dataset, nrows, filter_city):
    df = load_jsonl_to_df(path_dataset)

    df_filtered = df[df['city'].str.contains(filter_city, case=False, na=False)].copy()
    df_filtered = df_filtered.reset_index(drop=True)

    if nrows is not None:
        df_filtered = df_filtered.head(nrows)

    df_filtered.rename(columns={
        'stars': 'rating',
        'categories': 'category',
        'hours': 'opening_hours'
    }, inplace=True)

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
    df_filtered['has_outdoor_seating'] = df_filtered['attributes'].apply(
        lambda attributes: _extract_attribute(attributes, "OutdoorSeating", "bool")
    )
    df_filtered['noise_level'] = df_filtered['attributes'].apply(
        lambda attributes: _extract_attribute(attributes, "NoiseLevel", "text")
    )
    df_filtered['good_for_kids'] = df_filtered['attributes'].apply(
        lambda attributes: _extract_attribute(attributes, "GoodForKids", "bool")
    )
    df_filtered['text'] = df_filtered.apply(preprocess_poi_json, axis=1)

    def has_parking(attributes: dict) -> bool:
        try:
            parking_value = _parse_attributes(attributes).get("BusinessParking")
            if parking_value:
                parking_dict = ast.literal_eval(parking_value)
                if isinstance(parking_dict, dict):
                    return any(bool(value) for value in parking_dict.values())
        except Exception:
            pass
        return False

    df_filtered['parking'] = df_filtered['attributes'].apply(has_parking)
    return df_filtered


def create_embeddings(df, do_save=True):
    embeddings = model.encode(df['text'].tolist(), show_progress_bar=True)

    faiss_index = faiss.IndexFlatL2(embeddings.shape[1])
    faiss_index.add(np.array(embeddings))

    return embeddings


def save_data(df, embeddings, df_path="data/filtered_pois.csv", emb_path="data/embeddings.npy"):
    df.to_csv(df_path, index=False)
    np.save(emb_path, embeddings)


def load_data(df_path="filtered_pois.csv", emb_path="embeddings.npy"):
    if os.path.exists(df_path) and os.path.exists(emb_path):
        df = pd.read_csv(df_path, encoding='utf-8')
        df = _with_derived_poi_columns(df)
        embeddings = np.load(emb_path)
        return df, embeddings
    else:
        return None, None


def get_embeddings_and_df(path_dataset,
                          filter_city,
                          df_path="data/filtered_pois.csv",
                          emb_path="data/embeddings.npy",
                          nrows=None):
    df, embeddings = load_data(df_path, emb_path)
    print("[INFO] Loading dataset.")
    if df is None or embeddings is None:
        df = load_dataset(path_dataset, nrows=nrows, filter_city=filter_city)
        embeddings = create_embeddings(df)
        save_data(df, embeddings, df_path, emb_path)
        print("[INFO] Dataset loaded.")
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

    df_path = "data/filtered_pois.csv"
    emb_path = "data/embeddings.npy"

    for query in user_queries:
        embeddings, df = get_embeddings_and_df(path_dataset,
                                               user_city,
                                               df_path,
                                               emb_path)
        print("\n--- RAG Recommendation System ---\n")
        output = run_rag_navigation(
            query, 
            user_location, 
            embeddings, 
            df=df,
            user_location_name=f"{user_city}, PA"
        )
        print(output["response"])
        print(json.dumps(output["retrieved_pois"], indent=2))