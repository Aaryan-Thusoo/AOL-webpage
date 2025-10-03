# app/services/drive.py
# --- Public (no-auth) downloader for "anyone-with-link" CSVs ---
import re
import requests

def _drive_confirm_token_from_response(text: str):
    m = re.search(r'href="(/uc\?export=download[^"]*?confirm=([^"&]+)[^"]*?)"', text)
    if m:
        return m.group(2)
    m = re.search(r'confirm=([^&"]+)', text)
    return m.group(1) if m else None

def download_csv_by_id(file_id: str, timeout=60) -> bytes:
    """
    Public download (uc?export=download). Works for files shared 'Anyone with the link'.
    Hardened against the Google 'virus-scan' interstitial and HTML responses.
    """
    session = requests.Session()
    base = "https://drive.google.com/uc?export=download"
    params = {"id": file_id}

    r = session.get(base, params=params, timeout=timeout, allow_redirects=True)
    r.raise_for_status()
    content_type = r.headers.get("Content-Type", "").lower()

    # If we hit the virus-scan interstitial, re-request with the confirm token
    if "text/html" in content_type:
        try:
            token = _drive_confirm_token_from_response(r.text) or "t"
        except Exception:
            token = "t"
        params["confirm"] = token
        r = session.get(base, params=params, timeout=timeout, allow_redirects=True)
        r.raise_for_status()
        content_type = r.headers.get("Content-Type", "").lower()

    # Final guard: reject HTML masquerading as CSV
    head = r.content[:200].lstrip().lower()
    if "text/html" in content_type or head.startswith(b"<!doctype html") or head.startswith(b"<html"):
        raise RuntimeError("Drive returned HTML page instead of CSV (permission/scan page).")
    if not r.content:
        raise RuntimeError("Drive returned empty response.")
    return r.content


# --- OAuth (private files) helpers ---
import io, os
from typing import Optional, List, Dict, Any
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from ..config import GDRIVE_CREDENTIALS_FILE, GDRIVE_TOKEN_FILE, GDRIVE_SCOPES

def _get_drive_service():
    """
    Returns an authenticated Drive service.
    - First run: opens a browser and requests offline access (refresh token).
    - Later runs: silently refreshes expired access tokens using the refresh token.
    """
    creds: Optional[Credentials] = None

    # Load saved credentials if present
    if os.path.exists(GDRIVE_TOKEN_FILE):
        creds = Credentials.from_authorized_user_file(GDRIVE_TOKEN_FILE, GDRIVE_SCOPES)

    # If no valid creds, either refresh or run the local server flow
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            # silent refresh
            creds.refresh(Request())
        else:
            # force Google to issue a refresh token
            flow = InstalledAppFlow.from_client_secrets_file(GDRIVE_CREDENTIALS_FILE, GDRIVE_SCOPES)
            creds = flow.run_local_server(
                port=0,
                access_type="offline",  # IMPORTANT: get refresh token
                prompt="consent"        # IMPORTANT: force issuing refresh token
            )
        # persist the full token (includes refresh token)
        with open(GDRIVE_TOKEN_FILE, "w") as token:
            token.write(creds.to_json())

    return build("drive", "v3", credentials=creds)


def download_file_bytes_by_id_api(file_id: str) -> bytes:
    """
    Private file download (bytes) using Drive API (for CSV stored as a regular file).
    """
    service = _get_drive_service()
    request = service.files().get_media(fileId=file_id)
    buf = io.BytesIO()
    downloader = MediaIoBaseDownload(buf, request)
    done = False
    while not done:
        _, done = downloader.next_chunk()
    return buf.getvalue()

def export_google_sheet_as_csv(file_id: str) -> bytes:
    """
    Export a native Google Sheet as CSV using Drive API.
    """
    service = _get_drive_service()
    request = service.files().export_media(fileId=file_id, mimeType="text/csv")
    buf = io.BytesIO()
    downloader = MediaIoBaseDownload(buf, request)
    done = False
    while not done:
        _, done = downloader.next_chunk()
    return buf.getvalue()

def list_files_in_folder(folder_id: str) -> List[Dict[str, Any]]:
    """
    List files in a Drive folder (non-recursive).
    """
    service = _get_drive_service()
    q = f"'{folder_id}' in parents and trashed=false"
    fields = "nextPageToken, files(id, name, mimeType, size, modifiedTime)"
    page_token = None
    out: List[Dict[str, Any]] = []
    while True:
        resp = service.files().list(q=q, spaces="drive", fields=fields, pageToken=page_token).execute()
        out.extend(resp.get("files", []))
        page_token = resp.get("nextPageToken")
        if not page_token:
            break
    return out
