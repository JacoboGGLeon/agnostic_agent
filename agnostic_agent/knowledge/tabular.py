from typing import List, Dict, Any
import pandas as pd
import logging
import os

logger = logging.getLogger(__name__)

def load_csv(csv_path: str) -> pd.DataFrame:
    """Load a CSV file into a DataFrame."""
    if not os.path.exists(csv_path):
        logger.error(f"CSV file not found: {csv_path}")
        return pd.DataFrame()
    return pd.read_csv(csv_path)

def query_csv_by_filter(csv_path: str, filters: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Simple exact match filter on CSV columns."""
    df = load_csv(csv_path)
    if df.empty:
        return []
    
    for col, val in filters.items():
        if col in df.columns:
            df = df[df[col] == val]
    
    return df.to_dict(orient="records")
