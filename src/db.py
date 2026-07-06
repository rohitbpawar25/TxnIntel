import os
import pandas as pd
from sqlalchemy import create_engine, text
from src.config import DATABASE_URL

def get_engine():
    if not DATABASE_URL:
        raise ValueError("DATABASE_URL environment variable is not set.")
    return create_engine(DATABASE_URL)

def load_transactions_from_db() -> pd.DataFrame:
    engine = get_engine()
    query = "SELECT * FROM creditcard_transactions;"
    df = pd.read_sql(query, engine)
    return df
