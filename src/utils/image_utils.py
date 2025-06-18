from typing import List, Tuple, Optional
from PIL import Image
import io
import requests
import numpy as np

def is_valid_image_url(url: str) -> bool:
    """
    Check if a URL points to a valid image.
    
    Args:
        url (str): URL to check
        
    Returns:
        bool: True if URL points to a valid image
    """
    try:
        response = requests.head(url)
        content_type = response.headers.get("content-type", "")
        return content_type.startswith("image/")
    except requests.exceptions.RequestException:
        return False

def get_image_dimensions(url: str) -> Optional[Tuple[int, int]]:
    """
    Get dimensions of an image from URL.
    
    Args:
        url (str): URL of the image
        
    Returns:
        Optional[Tuple[int, int]]: Image dimensions (width, height) if successful
    """
    try:
        response = requests.get(url)
        img = Image.open(io.BytesIO(response.content))
        return img.size
    except (requests.exceptions.RequestException, IOError):
        return None

def resize_image(image: Image.Image, max_size: Tuple[int, int]) -> Image.Image:
    """
    Resize an image while maintaining aspect ratio.
    
    Args:
        image (Image.Image): Input image
        max_size (Tuple[int, int]): Maximum dimensions (width, height)
        
    Returns:
        Image.Image: Resized image
    """
    ratio = min(max_size[0] / image.size[0], max_size[1] / image.size[1])
    new_size = tuple(int(dim * ratio) for dim in image.size)
    return image.resize(new_size, Image.Resampling.LANCZOS)

def get_image_metadata(image: Image.Image) -> dict:
    """
    Extract metadata from an image.
    
    Args:
        image (Image.Image): Input image
        
    Returns:
        dict: Image metadata
    """
    return {
        "format": image.format,
        "mode": image.mode,
        "size": image.size,
        "info": image.info
    }

def is_supported_image_format(filename: str) -> bool:
    """
    Check if a file has a supported image format.
    
    Args:
        filename (str): Name of the file
        
    Returns:
        bool: True if format is supported
    """
    supported_formats = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".webp"}
    return any(filename.lower().endswith(fmt) for fmt in supported_formats)

def get_image_thumbnail(image: Image.Image, size: Tuple[int, int] = (200, 200)) -> Image.Image:
    """
    Create a thumbnail of an image.
    
    Args:
        image (Image.Image): Input image
        size (Tuple[int, int]): Thumbnail size
        
    Returns:
        Image.Image: Thumbnail image
    """
    image.thumbnail(size, Image.Resampling.LANCZOS)
    return image

def get_file_type(filename: str) -> str:
    """
    Determine the type of file based on its extension.
    
    Args:
        filename (str): Name of the file
        
    Returns:
        str: File type ('images', 'data', 'documents', 'videos', 'other')
    """
    filename_lower = filename.lower()
    
    # Image formats
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp', '.svg'}
    if any(filename_lower.endswith(ext) for ext in image_extensions):
        return 'images'
    
    # Data formats
    data_extensions = {'.csv', '.json', '.xml', '.xlsx', '.xls', '.txt', '.dat', '.h5', '.hdf5'}
    if any(filename_lower.endswith(ext) for ext in data_extensions):
        return 'data'
    
    # Document formats
    document_extensions = {'.pdf', '.doc', '.docx', '.ppt', '.pptx', '.md', '.rst'}
    if any(filename_lower.endswith(ext) for ext in document_extensions):
        return 'documents'
    
    # Video formats
    video_extensions = {'.mp4', '.avi', '.mov', '.wmv', '.flv', '.webm', '.mkv'}
    if any(filename_lower.endswith(ext) for ext in video_extensions):
        return 'videos'
    
    # Default to other
    return 'other'

def format_file_size(size_bytes: int) -> str:
    """
    Format file size in bytes to human-readable format.
    
    Args:
        size_bytes (int): Size in bytes
        
    Returns:
        str: Formatted file size (e.g., "1.5 MB", "2.3 GB")
    """
    if size_bytes == 0:
        return "0 B"
    
    size_names = ["B", "KB", "MB", "GB", "TB"]
    import math
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    
    return f"{s} {size_names[i]}" 