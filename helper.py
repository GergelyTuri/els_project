from google.colab import auth
import gspread
from google.auth import default
from google.colab import drive

def g_drive():
    """mounts Google Drive"""
    drive.mount('/content/drive')
    return

def auth():
    """authenticates user"""
    auth.authenticate_user()
    creds, _ = default()
    gc = gspread.authorize(creds)
    return