import json
import pandas as pd
import csv
from datetime import datetime, timezone

def load_jsonl_to_df(filepath, nrows=None):
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if nrows and i >= nrows:
                break
            obj = json.loads(line)
            data.append(obj)
    return pd.DataFrame(data)


def append_navigation_memory(csv_path, utterance, conversation_id=1, time_iso=None):
    """Append an utterance to a navigation memory CSV with a timestamp and conversation id.

    The CSV is expected to have headers: time,conversation id,summary
    """
    if time_iso is None:
        # create an artificial UTC timestamp in ISO 8601 format
        time_iso = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')

    # Ensure file exists and has header
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            has_header = True
    except FileNotFoundError:
        has_header = False

    mode = 'a'
    with open(csv_path, mode, newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        if not has_header:
            writer.writerow(['time', 'conversation id', 'summary'])
        writer.writerow([time_iso, conversation_id, utterance])


def recreate_navigation_memory(csv_path):
    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['time', 'conversation id', 'summary'])