from typing import List, Dict, Any
import json
import logging
import os

logger = logging.getLogger(__name__)

def load_json(json_path: str) -> Any:
    """Load a JSON file."""
    if not os.path.exists(json_path):
        logger.error(f"JSON file not found: {json_path}")
        return None
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading JSON {json_path}: {e}")
        return None

def load_jsonl(jsonl_path: str) -> List[Dict[str, Any]]:
    """Load a JSONL file."""
    if not os.path.exists(jsonl_path):
        logger.error(f"JSONL file not found: {jsonl_path}")
        return []
        
    data = []
    try:
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        return data
    except Exception as e:
        logger.error(f"Error loading JSONL {jsonl_path}: {e}")
        return []
