# app/services/drive.py
from __future__ import annotations
import requests
from typing import Optional


def download_csv_by_id(
    file_id: str,
    resource_key: Optional[str] = None,
    timeout: int = 180
) -> bytes:
    """Download a public Google Drive CSV file by ID (no OAuth)."""
    params = {"id": file_id, "export": "download"}
    if resource_key:
        params["resourcekey"] = resource_key

    endpoints = [
        "https://drive.google.com/uc",
        "https://drive.usercontent.google.com/download",
    ]

    with requests.Session() as s:
        for base in endpoints:
            r1 = s.get(base, params=params, timeout=timeout, allow_redirects=True)
            try:
                r1.raise_for_status()
            except Exception:
                continue

            ctype = (r1.headers.get("Content-Type") or "").lower()
            if "text/html" in ctype:
                # Handle large file confirm token
                token = None
                for k, v in r1.cookies.items():
                    if "download" in k and v:
                        token = v
                        break
                if (not token) and ("confirm=" in r1.url):
                    from urllib.parse import urlparse, parse_qs
                    q = parse_qs(urlparse(r1.url).query)
                    token = (q.get("confirm") or [None])[0]
                if token:
                    params2 = dict(params)
                    params2["confirm"] = token
                    r2 = s.get(base, params=params2, timeout=timeout, allow_redirects=True)
                    r2.raise_for_status()
                    if "text/html" in (r2.headers.get("Content-Type") or "").lower():
                        raise RuntimeError("Not a CSV: received HTML content (Drive confirm page)")
                    return r2.content
                raise RuntimeError("Not a CSV: received HTML content (Drive interstitial)")
            return r1.content

    raise requests.HTTPError(f"Public download failed for id={file_id}")
