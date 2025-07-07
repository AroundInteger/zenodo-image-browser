import os
import requests
from typing import Dict, List, Optional
from dotenv import load_dotenv
import zipfile
import io

load_dotenv()

class ZenodoAPI:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("ZENODO_API_KEY")
        self.base_url = "https://zenodo.org/api"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}" if self.api_key else None
        }

    def search_datasets(self, query: str, size: int = 10) -> List[Dict]:
        """
        Search for datasets on Zenodo.
        
        Args:
            query (str): Search query
            size (int): Number of results to return
            
        Returns:
            List[Dict]: List of matching datasets
        """
        endpoint = f"{self.base_url}/records"
        params = {
            "q": query,
            "size": size,
            "type": "dataset"
        }
        
        try:
            response = requests.get(endpoint, params=params, headers=self.headers)
            response.raise_for_status()
            return response.json()["hits"]["hits"]
        except requests.exceptions.RequestException as e:
            print(f"Error searching datasets: {e}")
            return []

    def get_dataset(self, record_id: str) -> Optional[Dict]:
        """
        Get details for a specific dataset.
        
        Args:
            record_id (str): Zenodo record ID
            
        Returns:
            Optional[Dict]: Dataset details if found
        """
        endpoint = f"{self.base_url}/records/{record_id}"
        
        try:
            response = requests.get(endpoint, headers=self.headers)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error getting dataset: {e}")
            return None

    def get_files(self, record_id: str) -> List[Dict]:
        """
        Get files associated with a dataset, extracting ZIP archives in memory.
        Args:
            record_id (str): Zenodo record ID
        Returns:
            List[Dict]: List of files in the dataset, including extracted ZIP contents
        """
        dataset = self.get_dataset(record_id)
        files = []
        if dataset and "files" in dataset:
            for file_info in dataset["files"]:
                files.append(file_info)
                # Check for ZIP files by extension
                if file_info["key"].lower().endswith(".zip"):
                    zip_url = file_info["links"]["self"]
                    try:
                        response = requests.get(zip_url, headers=self.headers)
                        response.raise_for_status()
                        with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
                            for zipinfo in zf.infolist():
                                if zipinfo.is_dir():
                                    continue
                                # Only include supported file types (e.g., images, .mat, etc.)
                                # For now, include all files
                                files.append({
                                    "key": f"{file_info['key']}/{zipinfo.filename}",
                                    "from_zip": True,
                                    "zip_source": file_info["key"],
                                    "zip_inner_path": zipinfo.filename,
                                    "size": zipinfo.file_size,
                                    # No direct download link, but can be extracted on demand
                                })
                    except Exception as e:
                        print(f"Error extracting ZIP file {file_info['key']}: {e}")
        return files

    def download_file(self, file_url: str, save_path: str) -> bool:
        """
        Download a file from Zenodo.
        
        Args:
            file_url (str): URL of the file to download
            save_path (str): Local path to save the file
            
        Returns:
            bool: True if download successful
        """
        try:
            response = requests.get(file_url, headers=self.headers, stream=True)
            response.raise_for_status()
            
            with open(save_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            return True
        except requests.exceptions.RequestException as e:
            print(f"Error downloading file: {e}")
            return False 