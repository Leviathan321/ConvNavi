import json
import pandas as pd

def load_jsonl_to_df(filepath, nrows=None):
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if nrows and i >= nrows:
                break
            obj = json.loads(line)
            data.append(obj)
    return pd.DataFrame(data)