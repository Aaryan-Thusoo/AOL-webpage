# app/config.py
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# SQLite database path (keep your existing DB config)
DB_PATH = str(BASE_DIR / "data.db")

import os

# LocationIQ API key
LOCATIONIQ_API_KEY = os.getenv("LOCATIONIQ_API_KEY", "YOUR_API_KEY_HERE")

# ------- Base dataset config (Drive-first, local fallback) -------
BASE_DRIVE_FILE_ID = os.getenv("BASE_DRIVE_FILE_ID", "").strip()
BASE_CSV_PATH = os.getenv("BASE_CSV_PATH", str(BASE_DIR / "meta_data.csv"))
TABLE_NAME = "data"

# ============= AGGREGATION RULES (optional overrides) =============
AGGREGATION_RULES = { }
DEFAULT_NUMERIC_AGG = "sum"

# ============= Google Drive OAuth Config =============
GDRIVE_CREDENTIALS_FILE = os.environ.get(
    "GDRIVE_CREDENTIALS_FILE",
    os.path.abspath(os.path.join(BASE_DIR, "..", "credentials.json"))
)
GDRIVE_TOKEN_FILE = os.environ.get(
    "GDRIVE_TOKEN_FILE",
    os.path.abspath(os.path.join(BASE_DIR, "..", "token.json"))
)
GDRIVE_SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]

# -------- Performance caps (UI safety) --------
MAX_SQL_ROWS       = 50_000     # Hard cap returned to UI even if LIMIT is huge
DEFAULT_SQL_LIMIT  = 1_000      # If user forgets LIMIT in UI queries, we add this

# -------- Mapping caps --------
MAX_MAP_POINTS     = 1_200      # <= this -> popups; > this -> fast cluster / heatmap
HEATMAP_THRESHOLD  = 5_000      # switch to heatmap for very large sets

# -------- Export caps --------
EXPORT_MAX_ROWS    = 500_000    # Cap for CSV export to avoid OOM
