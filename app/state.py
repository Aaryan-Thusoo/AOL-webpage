# app/state.py
from __future__ import annotations
from typing import Any, Dict, List, Optional
from threading import RLock

_lock = RLock()
_current: Dict[str, Any] = {
    "source": None,
    "label": "",        # ID or path
    "file_name": "",    # human-readable name
    "row_count": None,
    "columns": [],
}

def set_dataset_info(*, source: str, label: str, row_count: Optional[int],
                     columns: List[str], file_name: str = ""):
    with _lock:
        _current["source"] = source
        _current["label"] = label
        _current["file_name"] = file_name
        _current["row_count"] = int(row_count) if row_count is not None else None
        _current["columns"] = list(columns or [])


def get_dataset_info() -> Dict[str, Any]:
    with _lock:
        return dict(_current)
