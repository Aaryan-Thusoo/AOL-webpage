# app/boot.py
from __future__ import annotations
import logging
from typing import Tuple
import requests

from .config import DROPBOX_CSV_URL, BASE_CSV_PATH
from .services import db
from .state import set_dataset_info

log = logging.getLogger(__name__)


def _load_from_remote(url: str, timeout: int = 180) -> Tuple[str, list[str], int]:
    """
    Download CSV bytes from a plain HTTP/HTTPS URL and load into SQLite 'data' table.
    Works with Dropbox (?dl=1), S3, GCS, etc.
    Returns: (message, columns, row_count)
    """
    if not url:
        raise RuntimeError("No remote CSV URL provided")
    log.info("Attempting remote CSV load: %s", url)
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    blob = r.content
    msg, cols, nrows = db.replace_data_table_from_csv_bytes(blob)
    log.info("Remote CSV load succeeded: %d rows, %d columns.", nrows, len(cols))
    return msg, cols, nrows


def _load_from_local(path: str) -> Tuple[str, list[str], int]:
    """
    Load CSV from a local file path (fallback).
    """
    log.info("Attempting local CSV load: %s", path)
    with open(path, "rb") as f:
        blob = f.read()
    msg, cols, nrows = db.replace_data_table_from_csv_bytes(blob)
    log.info("Local CSV load succeeded: %d rows, %d columns.", nrows, len(cols))
    return msg, cols, nrows


def autoload_initial_dataset() -> dict:
    """
    Try sources in priority order:
      1) Dropbox (remote URL)
      2) Local CSV fallback
    """
    # 1) Remote URL
    try:
        msg, cols, nrows = _load_from_remote(DROPBOX_CSV_URL)
        set_dataset_info(source="remote_csv", label=DROPBOX_CSV_URL, row_count=nrows, columns=cols)
        return {
            "source": "remote_csv",
            "url": DROPBOX_CSV_URL,
            "message": msg,
            "columns": cols,
            "row_count": nrows,
            "ok": True,
        }
    except Exception as e:
        log.error("Autoload from remote URL failed: %s", e)

    # 2) Local fallback
    try:
        msg, cols, nrows = _load_from_local(BASE_CSV_PATH)
        set_dataset_info(source="local_csv", label=BASE_CSV_PATH, row_count=nrows, columns=cols)
        return {
            "source": "local_csv",
            "path": BASE_CSV_PATH,
            "message": msg,
            "columns": cols,
            "row_count": nrows,
            "ok": True,
        }
    except FileNotFoundError:
        return {"source": "none", "ok": False, "error": f"Local CSV not found: {BASE_CSV_PATH}"}
    except Exception as e:
        return {"source": "none", "ok": False, "error": f"Local CSV load failed: {e}"}
