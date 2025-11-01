# app/config.py
from __future__ import annotations
from pathlib import Path

# === Basic paths ===
BASE_DIR = Path(__file__).resolve().parent
DB_PATH = str(BASE_DIR / "data.db")

# === Dropbox direct-download URL (MUST end with ?dl=1) ===
# Example: https://www.dropbox.com/scl/fi/abc123/meta_data.csv?rlkey=...&dl=1
DROPBOX_CSV_URL = "https://www.dropbox.com/scl/fi/i8v1j2ms3hnci7leql2ll/meta_data.csv?rlkey=15sz8a4l847sawc6cyrmfzjk7&st=w1d4l3cr&dl=1"

# === Optional local fallback (use absolute path on your machine if you want)
# e.g., "/Users/you/Desktop/meta_data.csv"
BASE_CSV_PATH = "app/meta_data.csv"

# === App behavior ===
MAX_SQL_ROWS = 5000
DEFAULT_SQL_LIMIT = 500
