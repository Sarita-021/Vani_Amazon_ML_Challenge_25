from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from PIL import Image
import io
import concurrent.futures
import threading
from functools import lru_cache

SCOPES = ['https://www.googleapis.com/auth/drive.readonly']

class FastDriveImageLoader:
    def __init__(self, credentials_path='credentials.json', folder_id=None):
        self.credentials_path = credentials_path
        self.folder_id = folder_id
        self.service = None
        self._file_cache = {}
        self._lock = threading.Lock()
        
    def authenticate(self):
        flow = InstalledAppFlow.from_client_secrets_file(self.credentials_path, SCOPES)
        creds = flow.run_local_server(port=0)
        self.service = build('drive', 'v3', credentials=creds)
        
    def get_drive_files(self):
        if not self.service:
            self.authenticate()
            
        results = self.service.files().list(
            q=f"'{self.folder_id}' in parents",
            fields='files(id,name,mimeType)',
            pageSize=1000
        ).execute()
        
        files = results.get('files', [])
        for file in files:
            if file['mimeType'].startswith('image/'):
                self._file_cache[file['name']] = file['id']
        
        return self._file_cache
    
    def load_image_batch(self, filenames, max_workers=10):
        """Load multiple images concurrently"""
        if not self.service:
            self.authenticate()
            
        if not self._file_cache:
            self.get_drive_files()
        
        def load_single_image(filename):
            try:
                if filename in self._file_cache:
                    file_id = self._file_cache[filename]
                    request = self.service.files().get_media(fileId=file_id)
                    img_data = io.BytesIO(request.execute())
                    return filename, Image.open(img_data)
                return filename, None
            except:
                return filename, None
        
        results = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_filename = {executor.submit(load_single_image, fn): fn for fn in filenames}
            for future in concurrent.futures.as_completed(future_to_filename):
                filename, image = future.result()
                results[filename] = image
        
        return results

drive_loader = None

def init_fast_drive_loader(credentials_path='credentials.json', folder_id='1ZXP3slTxtjvVaqTFrblfR8eR5lf07nNK'):
    global drive_loader
    drive_loader = FastDriveImageLoader(credentials_path, folder_id)
    return drive_loader

def get_images_batch_from_drive(image_links, batch_size=50):
    global drive_loader
    if not drive_loader:
        raise ValueError("Drive loader not initialized")
    
    import os
    filenames = [os.path.basename(link) for link in image_links if link]
    
    # Process in batches
    all_images = {}
    for i in range(0, len(filenames), batch_size):
        batch = filenames[i:i+batch_size]
        batch_results = drive_loader.load_image_batch(batch)
        all_images.update(batch_results)
    
    return all_images