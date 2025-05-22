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