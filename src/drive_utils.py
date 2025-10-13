from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from PIL import Image
import io
import os
import pandas as pd
from functools import lru_cache

SCOPES = ['https://www.googleapis.com/auth/drive.readonly']

class DriveImageLoader:
    def __init__(self, credentials_path='credentials.json', folder_id=None):
        self.credentials_path = credentials_path
        self.folder_id = folder_id
        self.service = None
        self._file_cache = {}
        
    def authenticate(self):
        """Authenticate with Google Drive"""
        flow = InstalledAppFlow.from_client_secrets_file(self.credentials_path, SCOPES)
        creds = flow.run_local_server(port=0)
        self.service = build('drive', 'v3', credentials=creds)
        
    def get_drive_files(self):
        """Get list of files from Drive folder"""
        if not self.service:
            self.authenticate()
            
        results = self.service.files().list(
            q=f"'{self.folder_id}' in parents",
            fields='files(id,name,mimeType)'
        ).execute()
        
        files = results.get('files', [])
        for file in files:
            if file['mimeType'].startswith('image/'):
                self._file_cache[file['name']] = file['id']
        
        return self._file_cache
    
    @lru_cache(maxsize=1000)
    def load_image_from_drive(self, filename):
        """Load image directly from Google Drive"""
        if not self.service:
            self.authenticate()
            
        if filename not in self._file_cache:
            self.get_drive_files()
            
        if filename in self._file_cache:
            file_id = self._file_cache[filename]
            request = self.service.files().get_media(fileId=file_id)
            img_data = io.BytesIO(request.execute())
            return Image.open(img_data)
        
        return None

# Global drive loader instance
drive_loader = None

def init_drive_loader(credentials_path='credentials.json', folder_id='1ZXP3slTxtjvVaqTFrblfR8eR5lf07nNK'):
    """Initialize global drive loader"""
    global drive_loader
    drive_loader = DriveImageLoader(credentials_path, folder_id)
    return drive_loader

def get_image_from_drive(image_link):
    """Get image from drive using image link"""
    global drive_loader
    if not drive_loader:
        raise ValueError("Drive loader not initialized. Call init_drive_loader() first.")
    
    filename = os.path.basename(image_link)
    return drive_loader.load_image_from_drive(filename)