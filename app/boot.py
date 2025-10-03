# app/boot.py
from __future__ import annotations

import logging
from typing import Tuple

# Config: required + optional
from .config import BASE_DRIVE_FILE_ID, BASE_CSV_PATH  # always present
try:
    # Optional: set in .env or config.py; values: "auto" | "file" | "sheet"
    from .config import BASE_DRIVE_FILE_TYPE  # type: ignore
except Exception:  # pragma: no cover
    BASE_DRIVE_FILE_TYPE = "auto"  # default if not defined

from .services import db, drive
from .state import set_dataset_info

log = logging.getLogger(__name__)


def _load_from_drive(file_id: str, file_type: str = "auto") -> Tuple[str, list[str], int]:
    """
    Try to load CSV bytes from Google Drive and write them into SQLite 'data' table.

    Strategy:
      1) If OAuth helpers exist, try Drive API:
         - file_type == "file": download a regular file's bytes
         - file_type == "sheet": export a Google Sheet as CSV
         - file_type == "auto": try "file" then "sheet"
      2) Fallback to public 'uc?export=download' method (works if file is shared 'Anyone with link').
    Returns:
      (message, columns, row_count)
    """
    last_err: Exception | None = None

    # 1) OAuth API path (private files supported)
    has_api = hasattr(drive, "download_file_bytes_by_id_api") and hasattr(drive, "export_google_sheet_as_csv")
    if has_api:
        kinds = ["file", "sheet"] if file_type == "auto" else [file_type]
        for kind in kinds:
            try:
                if kind == "sheet":
                    blob = drive.export_google_sheet_as_csv(file_id)  # type: ignore[attr-defined]
                else:
                    blob = drive.download_file_bytes_by_id_api(file_id)  # type: ignore[attr-defined]
                # Write to DB; db.replace_data_table_from_csv_bytes returns (message, columns, row_count)
                return db.replace_data_table_from_csv_bytes(blob)
            except Exception as e:  # pragma: no cover
                last_err = e
                log.warning("Drive %s load failed via API: %s", kind, e)

    # 2) Public 'uc?export=download' fallback
    try:
        blob = drive.download_csv_by_id(file_id)
        return db.replace_data_table_from_csv_bytes(blob)
    except Exception as e:  # pragma: no cover
        last_err = e

    raise RuntimeError(last_err or Exception("Drive load failed"))


def autoload_initial_dataset() -> dict:
    """
    Attempt to load an initial dataset at startup.

    Priority:
      1) Google Drive ID from config/env (BASE_DRIVE_FILE_ID); type can be controlled
         with BASE_DRIVE_FILE_TYPE ("auto" | "file" | "sheet", default "auto")
      2) Local CSV fallback at BASE_CSV_PATH

    Returns a diagnostic dict:
      {source, message?, columns?, row_count?, path?, ok: bool, error?}
    """
    # 1) Try Drive first (if configured)
    if BASE_DRIVE_FILE_ID:
        try:
            msg, cols, nrows = _load_from_drive(BASE_DRIVE_FILE_ID, file_type=str(BASE_DRIVE_FILE_TYPE or "auto").lower())
            # Record global dataset state for the UI
            src_label = "drive(auto)" if BASE_DRIVE_FILE_TYPE in (None, "", "auto") else f"drive({BASE_DRIVE_FILE_TYPE})"
            set_dataset_info(source=src_label, label=BASE_DRIVE_FILE_ID, row_count=nrows, columns=cols)
            return {
                "source": "drive",
                "message": msg,
                "columns": cols,
                "row_count": nrows,
                "ok": True,
            }
        except Exception as e:  # pragma: no cover
            log.error("Autoload from Drive failed: %s", e)

    # 2) Fallback to local CSV (best-effort)
    try:
        with open(BASE_CSV_PATH, "rb") as f:
            msg, cols, nrows = db.replace_data_table_from_csv_bytes(f.read())
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
    except Exception as e:  # pragma: no cover
        return {"source": "none", "ok": False, "error": f"Local CSV load failed: {e}"}
