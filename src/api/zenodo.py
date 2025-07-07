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

    def _extract_zip_recursively(self, zip_url: str, zip_source: str, zip_path: str = "") -> List[Dict]:
        """
        Recursively extract files from ZIP archives, handling nested ZIPs.
        """
        extracted_files = []
        try:
            response = requests.get(zip_url, headers=self.headers)
            response.raise_for_status()
            
            with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
                for zipinfo in zf.infolist():
                    if zipinfo.is_dir():
                        continue
                    
                    current_path = f"{zip_path}/{zipinfo.filename}" if zip_path else zipinfo.filename
                    full_key = f"{zip_source}/{current_path}"
                    
                    # Check if this is a nested ZIP file
                    if zipinfo.filename.lower().endswith('.zip'):
                        print(f"    Found nested ZIP: {current_path}")
                        # Extract the nested ZIP content
                        with zf.open(zipinfo.filename) as nested_zip_file:
                            nested_zip_content = nested_zip_file.read()
                            with zipfile.ZipFile(io.BytesIO(nested_zip_content)) as nested_zf:
                                for nested_info in nested_zf.infolist():
                                    if nested_info.is_dir():
                                        continue
                                    nested_path = f"{current_path}/{nested_info.filename}"
                                    nested_full_key = f"{zip_source}/{nested_path}"
                                    extracted_files.append({
                                        "key": nested_full_key,
                                        "from_zip": True,
                                        "zip_source": zip_source,
                                        "zip_inner_path": nested_path,
                                        "size": nested_info.file_size,
                                    })
                    else:
                        # Regular file
                        extracted_files.append({
                            "key": full_key,
                            "from_zip": True,
                            "zip_source": zip_source,
                            "zip_inner_path": current_path,
                            "size": zipinfo.file_size,
                        })
        except Exception as e:
            print(f"Error extracting ZIP file {zip_source}: {e}")
        
        return extracted_files

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
            zip_files = [f for f in dataset["files"] if f["key"].lower().endswith(".zip")]
            if zip_files:
                print(f"Found {len(zip_files)} ZIP files to extract...")
            
            for file_info in dataset["files"]:
                files.append(file_info)
                # Check for ZIP files by extension
                if file_info["key"].lower().endswith(".zip"):
                    print(f"Extracting {file_info['key']}...")
                    zip_url = file_info["links"]["self"]
                    extracted = self._extract_zip_recursively(zip_url, file_info["key"])
                    print(f"  Found {len(extracted)} files in {file_info['key']} (including nested)")
                    files.extend(extracted)
        
        print(f"Total files after extraction: {len(files)}")
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