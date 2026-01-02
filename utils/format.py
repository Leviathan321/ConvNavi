import math
import json
import re

def sanitize_for_json(obj):
    """Recursively sanitize object to be JSON-compliant."""
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    elif isinstance(obj, tuple):
        return [sanitize_for_json(x) for x in obj]
    elif isinstance(obj, list):
        return [sanitize_for_json(x) for x in obj]
    elif isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    return obj

def extract_json(text):
    try:
        match = re.search(r"\{.*\}", text, re.DOTALL) 
        if match: 
            json_str = match.group(0) 
            return json.loads(json_str)
    except json.JSONDecodeError: 
        pass 
    return None

def extract_json_list(text):
    match = re.search(r"(\[.*\]|\{.*\})", text, re.DOTALL)
    if not match:
        raise json.JSONDecodeError("No JSON found", text, 0)

    json_str = match.group(1)
    parsed = json.loads(json_str)

    if isinstance(parsed, list):
        return parsed
    if isinstance(parsed, dict):
        return [parsed]

    raise json.JSONDecodeError("JSON is neither object nor array", json_str, 0)

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
