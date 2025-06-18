"""
Enhanced Image Analysis for Scientific Research
Advanced image processing capabilities for experimental images
"""

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageFilter, ImageEnhance
from skimage import measure, morphology, segmentation, filters
from skimage.feature import corner_harris, corner_peaks
from skimage.feature.peak import peak_local_max
from skimage.color import rgb2hsv, rgb2lab
from scipy import ndimage
from scipy.spatial.distance import cdist
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Any, Tuple, Optional
import json
import streamlit as st

class EnhancedImageAnalyzer:
    """Advanced image analysis for scientific research"""
    
    def __init__(self):
        self.analysis_methods = {
            'basic': ['dimensions', 'color_stats', 'histogram'],
            'morphology': ['particle_analysis', 'pore_analysis', 'grain_analysis'],
            'pattern': ['fractal_analysis', 'texture_analysis', 'orientation_analysis'],
            'advanced': ['object_detection', 'segmentation', 'feature_extraction'],
            'scientific': ['phase_analysis', 'defect_detection', 'crystal_analysis']
        }
    
    def analyze_image_comprehensive(self, image_path: str, analysis_types: List[str] = None) -> Dict[str, Any]:
        """Comprehensive image analysis with multiple techniques"""
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            return {"error": "Could not load image"}
        
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        results = {
            "image_info": self._get_basic_info(image_rgb),
            "analyses": {}
        }
        
        # Run requested analyses
        if analysis_types is None:
            analysis_types = ['basic', 'morphology', 'pattern']
        
        for analysis_type in analysis_types:
            if analysis_type == 'basic':
                results["analyses"]["basic"] = self._basic_analysis(image_rgb, image_gray)
            elif analysis_type == 'morphology':
                results["analyses"]["morphology"] = self._morphological_analysis(image_gray)
            elif analysis_type == 'pattern':
                results["analyses"]["pattern"] = self._pattern_analysis(image_gray)
            elif analysis_type == 'advanced':
                results["analyses"]["advanced"] = self._advanced_analysis(image_rgb, image_gray)
            elif analysis_type == 'scientific':
                results["analyses"]["scientific"] = self._scientific_analysis(image_rgb, image_gray)
        
        return results
    
    def _get_basic_info(self, image: np.ndarray) -> Dict[str, Any]:
        """Get basic image information"""
        height, width, channels = image.shape
        
        return {
            "dimensions": {"width": width, "height": height, "channels": channels},
            "aspect_ratio": width / height,
            "total_pixels": width * height,
            "memory_size_mb": (width * height * channels) / (1024 * 1024)
        }
    
    def _basic_analysis(self, image_rgb: np.ndarray, image_gray: np.ndarray) -> Dict[str, Any]:
        """Basic image analysis"""
        
        # Color analysis
        hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
        lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
        
        # Calculate color statistics
        color_stats = {
            "rgb": {
                "mean": [float(np.mean(image_rgb[:, :, i])) for i in range(3)],
                "std": [float(np.std(image_rgb[:, :, i])) for i in range(3)],
                "min": [float(np.min(image_rgb[:, :, i])) for i in range(3)],
                "max": [float(np.max(image_rgb[:, :, i])) for i in range(3)]
            },
            "hsv": {
                "mean": [float(np.mean(hsv[:, :, i])) for i in range(3)],
                "std": [float(np.std(hsv[:, :, i])) for i in range(3)]
            },
            "lab": {
                "mean": [float(np.mean(lab[:, :, i])) for i in range(3)],
                "std": [float(np.std(lab[:, :, i])) for i in range(3)]
            }
        }
        
        # Histogram analysis
        histograms = {
            "gray": cv2.calcHist([image_gray], [0], None, [256], [0, 256]).flatten().tolist(),
            "rgb": [cv2.calcHist([image_rgb], [i], None, [256], [0, 256]).flatten().tolist() for i in range(3)]
        }
        
        # Brightness and contrast
        brightness = float(np.mean(image_gray))
        contrast = float(np.std(image_gray))
        
        return {
            "color_statistics": color_stats,
            "histograms": histograms,
            "brightness": brightness,
            "contrast": contrast,
            "dynamic_range": float(np.max(image_gray) - np.min(image_gray))
        }
    
    def _morphological_analysis(self, image_gray: np.ndarray) -> Dict[str, Any]:
        """Morphological analysis for particle/grain detection"""
        
        # Apply thresholding
        _, binary = cv2.threshold(image_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Remove noise
        kernel = np.ones((3, 3), np.uint8)
        binary_cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary_cleaned = cv2.morphologyEx(binary_cleaned, cv2.MORPH_OPEN, kernel)
        
        # Label connected components
        labeled_image = measure.label(binary_cleaned)
        regions = measure.regionprops(labeled_image)
        
        # Analyze particles/grains
        particle_data = []
        for region in regions:
            if region.area > 50:  # Filter small noise
                particle_data.append({
                    "area": region.area,
                    "perimeter": region.perimeter,
                    "centroid": region.centroid,
                    "bbox": region.bbox,
                    "eccentricity": region.eccentricity,
                    "solidity": region.solidity,
                    "circularity": (4 * np.pi * region.area) / (region.perimeter ** 2) if region.perimeter > 0 else 0
                })
        
        # Calculate statistics
        if particle_data:
            areas = [p["area"] for p in particle_data]
            perimeters = [p["perimeter"] for p in particle_data]
            circularities = [p["circularity"] for p in particle_data]
            
            particle_stats = {
                "total_particles": len(particle_data),
                "area_stats": {
                    "mean": float(np.mean(areas)),
                    "std": float(np.std(areas)),
                    "min": float(np.min(areas)),
                    "max": float(np.max(areas)),
                    "median": float(np.median(areas))
                },
                "perimeter_stats": {
                    "mean": float(np.mean(perimeters)),
                    "std": float(np.std(perimeters))
                },
                "circularity_stats": {
                    "mean": float(np.mean(circularities)),
                    "std": float(np.std(circularities))
                }
            }
        else:
            particle_stats = {"total_particles": 0}
        
        return {
            "particle_analysis": particle_stats,
            "particle_details": particle_data,
            "porosity": 1 - (np.sum(binary_cleaned) / (binary_cleaned.shape[0] * binary_cleaned.shape[1] * 255))
        }
    
    def _pattern_analysis(self, image_gray: np.ndarray) -> Dict[str, Any]:
        """Pattern and texture analysis"""
        
        # Edge detection
        edges_sobel = filters.sobel(image_gray)
        edges_canny = cv2.Canny(image_gray, 50, 150)
        
        # Texture analysis using Local Binary Patterns (simplified)
        texture_features = self._calculate_texture_features(image_gray)
        
        # Orientation analysis
        orientation_map = self._calculate_orientation_map(image_gray)
        
        # Fractal analysis (simplified box-counting)
        fractal_dimension = self._estimate_fractal_dimension(image_gray)
        
        return {
            "edge_analysis": {
                "sobel_magnitude": float(np.mean(edges_sobel)),
                "canny_edge_density": float(np.sum(edges_canny > 0) / edges_canny.size)
            },
            "texture_features": texture_features,
            "orientation_analysis": orientation_map,
            "fractal_dimension": fractal_dimension
        }
    
    def _advanced_analysis(self, image_rgb: np.ndarray, image_gray: np.ndarray) -> Dict[str, Any]:
        """Advanced image analysis techniques"""
        
        # Feature detection
        corners = corner_peaks(corner_harris(image_gray), min_distance=5)
        
        # Blob detection
        blobs = self._detect_blobs(image_gray)
        
        # Watershed segmentation
        segmentation_result = self._watershed_segmentation(image_gray)
        
        return {
            "feature_detection": {
                "corners_detected": len(corners),
                "corner_positions": corners.tolist()
            },
            "blob_analysis": blobs,
            "segmentation": segmentation_result
        }
    
    def _scientific_analysis(self, image_rgb: np.ndarray, image_gray: np.ndarray) -> Dict[str, Any]:
        """Scientific domain-specific analysis"""
        
        # Phase analysis (for materials science)
        phase_analysis = self._analyze_phases(image_gray)
        
        # Defect detection
        defect_analysis = self._detect_defects(image_gray)
        
        # Crystal structure analysis
        crystal_analysis = self._analyze_crystal_structure(image_gray)
        
        return {
            "phase_analysis": phase_analysis,
            "defect_detection": defect_analysis,
            "crystal_analysis": crystal_analysis
        }
    
    def _calculate_texture_features(self, image: np.ndarray) -> Dict[str, float]:
        """Calculate texture features"""
        # Simplified texture analysis
        features = {}
        
        # Gray-level co-occurrence matrix features (simplified)
        features["contrast"] = float(np.std(image))
        features["homogeneity"] = float(1 / (1 + np.var(image)))
        features["energy"] = float(np.sum(image ** 2) / image.size)
        
        return features
    
    def _calculate_orientation_map(self, image: np.ndarray) -> Dict[str, Any]:
        """Calculate orientation map using gradient"""
        grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        
        orientation = np.arctan2(grad_y, grad_x)
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        return {
            "mean_orientation": float(np.mean(orientation)),
            "orientation_std": float(np.std(orientation)),
            "gradient_magnitude_mean": float(np.mean(magnitude))
        }
    
    def _estimate_fractal_dimension(self, image: np.ndarray) -> float:
        """Estimate fractal dimension using box-counting method"""
        # Simplified box-counting
        binary = image > np.mean(image)
        
        scales = [2, 4, 8, 16]
        counts = []
        
        for scale in scales:
            if scale < min(image.shape):
                scaled = binary[::scale, ::scale]
                counts.append(np.sum(scaled))
        
        if len(counts) > 1:
            # Linear fit to log-log plot
            log_scales = np.log(scales[:len(counts)])
            log_counts = np.log(counts)
            slope = np.polyfit(log_scales, log_counts, 1)[0]
            return float(-slope)
        
        return 2.0  # Default for non-fractal images
    
    def _detect_blobs(self, image: np.ndarray) -> Dict[str, Any]:
        """Detect blobs in the image"""
        # Simple blob detection using Laplacian of Gaussian
        blobs = peak_local_max(filters.gaussian(image, sigma=2), min_distance=10)
        
        return {
            "blobs_detected": len(blobs),
            "blob_positions": blobs.tolist()
        }
    
    def _watershed_segmentation(self, image: np.ndarray) -> Dict[str, Any]:
        """Perform watershed segmentation"""
        # Distance transform
        binary = image > np.mean(image)
        distance = ndimage.distance_transform_edt(binary)
        
        # Find peaks
        peaks = peak_local_max(distance, min_distance=20)
        
        # Watershed
        markers = np.zeros_like(image, dtype=int)
        markers[peaks[:, 0], peaks[:, 1]] = np.arange(1, len(peaks) + 1)
        
        labels = segmentation.watershed(-distance, markers, mask=binary)
        
        return {
            "segments": len(np.unique(labels)) - 1,  # Exclude background
            "segmentation_map": labels.tolist()
        }
    
    def _analyze_phases(self, image: np.ndarray) -> Dict[str, Any]:
        """Analyze different phases in the image"""
        # Simple phase analysis using clustering
        from sklearn.cluster import KMeans
        
        # Reshape for clustering
        pixels = image.reshape(-1, 1)
        
        # Find optimal number of clusters (phases)
        kmeans = KMeans(n_clusters=3, random_state=42)
        labels = kmeans.fit_predict(pixels)
        
        # Analyze each phase
        phases = {}
        for i in range(3):
            phase_pixels = pixels[labels == i]
            phases[f"phase_{i+1}"] = {
                "intensity_mean": float(np.mean(phase_pixels)),
                "intensity_std": float(np.std(phase_pixels)),
                "area_fraction": float(len(phase_pixels) / len(pixels))
            }
        
        return {
            "phases_detected": 3,
            "phase_properties": phases
        }
    
    def _detect_defects(self, image: np.ndarray) -> Dict[str, Any]:
        """Detect defects in the image"""
        # Simple defect detection using outlier detection
        mean_intensity = np.mean(image)
        std_intensity = np.std(image)
        
        # Defects are pixels significantly different from mean
        defects = np.abs(image - mean_intensity) > 2 * std_intensity
        
        defect_regions = measure.label(defects)
        defect_props = measure.regionprops(defect_regions)
        
        return {
            "defects_detected": len(defect_props),
            "defect_area_fraction": float(np.sum(defects) / defects.size),
            "defect_sizes": [prop.area for prop in defect_props]
        }
    
    def _analyze_crystal_structure(self, image: np.ndarray) -> Dict[str, Any]:
        """Analyze crystal structure patterns"""
        # FFT analysis for periodic patterns
        f_transform = np.fft.fft2(image)
        f_shift = np.fft.fftshift(f_transform)
        magnitude_spectrum = np.log(np.abs(f_shift) + 1)
        
        # Find dominant frequencies
        peaks = peak_local_max(magnitude_spectrum, min_distance=10)
        
        return {
            "dominant_frequencies": len(peaks),
            "periodicity_score": float(np.std(magnitude_spectrum)),
            "crystal_symmetry": "unknown"  # Would need more sophisticated analysis
        }

# MCP Tool Integration
def create_enhanced_image_analysis_tools() -> List[Dict[str, Any]]:
    """Create enhanced image analysis tools for MCP server"""
    
    tools = [
        {
            "name": "enhanced_image_analysis",
            "description": "Comprehensive scientific image analysis with multiple techniques",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "file_name": {"type": "string", "description": "Name of image file to analyze"},
                    "analysis_types": {
                        "type": "array",
                        "items": {"type": "string"},
                        "enum": ["basic", "morphology", "pattern", "advanced", "scientific"],
                        "description": "Types of analysis to perform"
                    },
                    "parameters": {
                        "type": "object",
                        "description": "Analysis-specific parameters"
                    }
                },
                "required": ["file_name"]
            }
        },
        {
            "name": "particle_analysis",
            "description": "Detailed particle/grain analysis for materials science",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "file_name": {"type": "string", "description": "Name of image file"},
                    "min_particle_size": {"type": "integer", "description": "Minimum particle size in pixels"},
                    "detection_method": {
                        "type": "string",
                        "enum": ["threshold", "watershed", "edge_based"],
                        "description": "Particle detection method"
                    }
                },
                "required": ["file_name"]
            }
        },
        {
            "name": "phase_analysis",
            "description": "Phase analysis for multi-phase materials",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "file_name": {"type": "string", "description": "Name of image file"},
                    "num_phases": {"type": "integer", "description": "Expected number of phases"},
                    "analysis_method": {
                        "type": "string",
                        "enum": ["clustering", "threshold", "machine_learning"],
                        "description": "Phase detection method"
                    }
                },
                "required": ["file_name"]
            }
        },
        {
            "name": "defect_detection",
            "description": "Detect and analyze defects in materials",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "file_name": {"type": "string", "description": "Name of image file"},
                    "defect_types": {
                        "type": "array",
                        "items": {"type": "string"},
                        "enum": ["cracks", "pores", "inclusions", "grain_boundaries"],
                        "description": "Types of defects to detect"
                    },
                    "sensitivity": {
                        "type": "number",
                        "description": "Detection sensitivity (0-1)"
                    }
                },
                "required": ["file_name"]
            }
        }
    ]
    
    return tools 