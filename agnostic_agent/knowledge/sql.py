from typing import List, Dict, Any, Optional
import sqlite3
import pandas as pd
import logging

logger = logging.getLogger(__name__)

def connect_sqlite(db_path: str) -> sqlite3.Connection:
    """Connect to a SQLite database."""
    try:
        conn = sqlite3.connect(db_path)
        return conn
    except Exception as e:
        logger.error(f"Error connecting to SQLite DB {db_path}: {e}")
        raise

def query_sqlite(db_path: str, query: str) -> List[Dict[str, Any]]:
    """Execute a read-only query on a SQLite DB."""
    conn = connect_sqlite(db_path)
    try:
        df = pd.read_sql_query(query, conn)
        return df.to_dict(orient="records")
    except Exception as e:
        logger.error(f"Query failed: {e}")
        return []
    finally:
        conn.close()
