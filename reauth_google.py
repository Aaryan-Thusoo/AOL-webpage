from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
import os

SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]
CREDS_FILE = os.environ.get("GDRIVE_CREDENTIALS_FILE", "credentials.json")
TOKEN_FILE = os.environ.get("GDRIVE_TOKEN_FILE", "token.json")

def main():
    creds = None
    if os.path.exists(TOKEN_FILE):
        try:
            creds = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)
        except Exception:
            creds = None

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(CREDS_FILE, SCOPES)
            creds = flow.run_local_server(port=0, access_type="offline", prompt="consent")
        with open(TOKEN_FILE, "w") as f:
            f.write(creds.to_json())

    # quick sanity
    svc = build("drive","v3", credentials=creds)
    print(svc.files().list(pageSize=3, fields="files(id,name,size,modifiedTime)").execute())
    print("Reauth OK. token.json saved.")

if __name__ == "__main__":
    main()
