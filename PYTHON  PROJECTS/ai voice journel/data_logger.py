"""
data_logger.py
Save and load journal entries to a JSONL file. Each entry contains:
- id, timestamp, text, sentiment, summary
"""

import json
from pathlib import Path
from datetime import datetime
import uuid

DATA_FILE = Path("entries.jsonl")

def save_entry(text, sentiment, summary):
    entry = {
        "id": str(uuid.uuid4()),
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "text": text,
        "sentiment": sentiment,
        "summary": summary
    }
    with DATA_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return entry

def load_entries(limit=None):
    if not DATA_FILE.exists():
        return []
    out = []
    with DATA_FILE.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit and i>=limit:
                break
            out.append(json.loads(line))
    return out
