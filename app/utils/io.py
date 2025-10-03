# app/utils/io.py
import pandas as pd
import io

def read_csv_bytes(b: bytes) -> pd.DataFrame:
    """
    Safely read CSV from bytes, stripping BOM and handling common encodings.
    """
    try:
        return pd.read_csv(io.BytesIO(b))
    except UnicodeDecodeError:
        return pd.read_csv(io.BytesIO(b), encoding="latin-1")
