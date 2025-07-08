# src/ai/mcp_server.py
"""
Model Context Protocol (MCP) server implementation for secure AI data access
"""

import json
import asyncio
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import pandas as pd
import numpy as np
from pathlib import Path
import streamlit as st
from .enhanced_image_analysis import EnhancedImageAnalyzer, create_enhanced_image_analysis_tools

@dataclass
class MCPResource:
    """MCP Resource definition"""
    uri: str
    name: str
    description: str
    mimeType: str

@dataclass
class MCPTool:
    """MCP Tool definition"""
    name: str
    description: str
    inputSchema: Dict[str, Any]

class DataAnalysisMCPServer:
    """MCP Server for secure AI access to analysis data and tools"""
    
    def __init__(self, dataset_info: Dict[str, Any]):
        self.dataset_info = dataset_info
        self.resources = self._initialize_resources()
        self.tools = self._initialize_tools()
        self.image_analyzer = EnhancedImageAnalyzer()
    
    def _initialize_resources(self) -> List[MCPResource]:
        """Initialize available data resources"""
        resources = []
        
        # Dataset metadata resource
        resources.append(MCPResource(
            uri="dataset://metadata",
            name="Dataset Metadata",
            description="Complete dataset metadata including title, authors, files, and summary",
            mimeType="application/json"
        ))
        
        # Individual file resources
        for file_info in self.dataset_info.get('files', []):
            file_name = file_info.get('key', file_info.get('name', 'unknown'))
            file_type = file_info.get('type', 'unknown')
            file_size = file_info.get('size', 0)
            
            resources.append(MCPResource(
                uri=f"dataset://files/{file_name}",
                name=f"File: {file_name}",
                description=f"{file_type} file ({file_size} bytes)",
                mimeType=self._get_mime_type(file_type)
            ))
        
        return resources
    
    def _initialize_tools(self) -> List[MCPTool]:
        """Initialize available analysis tools"""
        tools = []
        
        # Data analysis tools
        tools.append(MCPTool(
            name="analyze_csv_data",
            description="Analyze CSV data with statistical operations, filtering, and visualization",
            inputSchema={
                "type": "object",
                "properties": {
                    "file_name": {"type": "string", "description": "Name of CSV file to analyze"},
                    "operation": {
                        "type": "string", 
                        "enum": ["summary", "correlation", "time_series", "filter"],
                        "description": "Type of analysis to perform"
                    },
                    "parameters": {
                        "type": "object",
                        "description": "Analysis-specific parameters"
                    }
                },
                "required": ["file_name", "operation"]
            }
        ))
        
        tools.append(MCPTool(
            name="analyze_image",
            description="Perform image analysis including measurements, feature detection, and pattern analysis",
            inputSchema={
                "type": "object",
                "properties": {
                    "file_name": {"type": "string", "description": "Name of image file to analyze"},
                    "analysis_type": {
                        "type": "string",
                        "enum": ["basic_info", "color_analysis", "edge_detection", "pattern_detection"],
                        "description": "Type of image analysis"
                    },
                    "parameters": {
                        "type": "object",
                        "description": "Analysis parameters (e.g., thresholds for edge detection)"
                    }
                },
                "required": ["file_name", "analysis_type"]
            }
        ))
        
        # Enhanced image analysis tools
        enhanced_tools = create_enhanced_image_analysis_tools()
        for tool_data in enhanced_tools:
            tools.append(MCPTool(
                name=tool_data["name"],
                description=tool_data["description"],
                inputSchema=tool_data["inputSchema"]
        ))
        
        tools.append(MCPTool(
            name="create_visualization",
            description="Create interactive plots and charts from data",
            inputSchema={
                "type": "object",
                "properties": {
                    "data_source": {"type": "string", "description": "Source file name"},
                    "plot_type": {
                        "type": "string",
                        "enum": ["line", "scatter", "histogram", "heatmap", "box"],
                        "description": "Type of plot to create"
                    },
                    "x_column": {"type": "string", "description": "Column for x-axis"},
                    "y_column": {"type": "string", "description": "Column for y-axis"},
                    "parameters": {
                        "type": "object",
                        "description": "Plot customization parameters"
                    }
                },
                "required": ["data_source", "plot_type"]
            }
        ))
        
        tools.append(MCPTool(
            name="compare_datasets",
            description="Compare multiple datasets or time periods within a dataset",
            inputSchema={
                "type": "object",
                "properties": {
                    "comparison_type": {
                        "type": "string",
                        "enum": ["statistical", "visual", "temporal"],
                        "description": "Type of comparison to perform"
                    },
                    "sources": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of data sources to compare"
                    },
                    "metrics": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Specific metrics to compare"
                    }
                },
                "required": ["comparison_type", "sources"]
            }
        ))
        
        return tools
    
    async def list_resources(self) -> List[Dict[str, Any]]:
        """List available resources for AI access"""
        return [
            {
                "uri": resource.uri,
                "name": resource.name,
                "description": resource.description,
                "mimeType": resource.mimeType
            }
            for resource in self.resources
        ]
    
    async def read_resource(self, uri: str) -> Dict[str, Any]:
        """Read a specific resource"""
        if uri == "dataset://metadata":
            return {
                "contents": [
                    {
                        "uri": uri,
                        "mimeType": "application/json",
                        "text": json.dumps(self.dataset_info, indent=2)
                    }
                ]
            }
        
        elif uri.startswith("dataset://files/"):
            file_name = uri.replace("dataset://files/", "")
            file_info = next((f for f in self.dataset_info['files'] if f['key'] == file_name), None)
            
            if not file_info:
                raise ValueError(f"File not found: {file_name}")
            
            # Return file metadata and sample content
            content = {
                "file_info": file_info,
                "sample_data": self._get_sample_data(file_info)
            }
            
            return {
                "contents": [
                    {
                        "uri": uri,
                        "mimeType": self._get_mime_type(file_info['type']),
                        "text": json.dumps(content, indent=2)
                    }
                ]
            }
        
        else:
            raise ValueError(f"Resource not found: {uri}")
    
    async def list_tools(self) -> List[Dict[str, Any]]:
        """List available tools for AI use"""
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "inputSchema": tool.inputSchema
            }
            for tool in self.tools
        ]
    
    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a tool with given arguments"""
        if name == "analyze_csv_data":
            return await self._analyze_csv_data(arguments)
        elif name == "analyze_image":
            return await self._analyze_image(arguments)
        elif name == "enhanced_image_analysis":
            return await self._enhanced_image_analysis(arguments)
        elif name == "particle_analysis":
            return await self._particle_analysis(arguments)
        elif name == "phase_analysis":
            return await self._phase_analysis(arguments)
        elif name == "defect_detection":
            return await self._defect_detection(arguments)
        elif name == "create_visualization":
            return await self._create_visualization(arguments)
        elif name == "compare_datasets":
            return await self._compare_datasets(arguments)
        else:
            raise ValueError(f"Unknown tool: {name}")
    
    async def _enhanced_image_analysis(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced image analysis using the new analyzer"""
        file_name = args['file_name']
        analysis_types = args.get('analysis_types', ['basic', 'morphology', 'pattern'])
        parameters = args.get('parameters', {})
        
        # Find the image file
        file_info = next((f for f in self.dataset_info['files'] 
                         if f.get('key', f.get('name', '')) == file_name), None)
        
        if not file_info:
            return {"content": [{"type": "text", "text": json.dumps({"error": f"Image file not found: {file_name}"})}]}
        
        # For demonstration, we'll simulate the analysis
        # In a real implementation, you would download the image and analyze it
        simulated_results = {
            "file_name": file_name,
            "analysis_types": analysis_types,
            "results": {
                "basic": {
                    "dimensions": {"width": 1024, "height": 768, "channels": 3},
                    "color_statistics": {
                        "rgb": {"mean": [128.5, 129.2, 127.8], "std": [45.2, 44.8, 46.1]},
                        "brightness": 128.5,
                        "contrast": 45.2
                    }
                },
                "morphology": {
                    "particle_analysis": {
                        "total_particles": 156,
                        "area_stats": {"mean": 245.6, "std": 89.3, "min": 23, "max": 892},
                        "circularity_stats": {"mean": 0.78, "std": 0.12}
                    },
                    "porosity": 0.23
                },
                "pattern": {
                    "edge_analysis": {"sobel_magnitude": 0.45, "canny_edge_density": 0.12},
                    "texture_features": {"contrast": 45.2, "homogeneity": 0.67, "energy": 0.89},
                    "fractal_dimension": 2.34
                }
            }
        }
        
        if 'advanced' in analysis_types:
            simulated_results["results"]["advanced"] = {
                "feature_detection": {"corners_detected": 89, "corner_positions": [[100, 200], [300, 400]]},
                "blob_analysis": {"blobs_detected": 45, "blob_positions": [[150, 250], [350, 450]]},
                "segmentation": {"segments": 12}
            }
        
        if 'scientific' in analysis_types:
            simulated_results["results"]["scientific"] = {
                "phase_analysis": {
                    "phases_detected": 3,
                    "phase_properties": {
                        "phase_1": {"intensity_mean": 85.2, "area_fraction": 0.45},
                        "phase_2": {"intensity_mean": 156.8, "area_fraction": 0.32},
                        "phase_3": {"intensity_mean": 234.1, "area_fraction": 0.23}
                    }
                },
                "defect_detection": {
                    "defects_detected": 23,
                    "defect_area_fraction": 0.08,
                    "defect_sizes": [15, 23, 8, 12, 19]
                },
                "crystal_analysis": {
                    "dominant_frequencies": 8,
                    "periodicity_score": 0.67,
                    "crystal_symmetry": "hexagonal"
                }
            }
        
        return {"content": [{"type": "text", "text": json.dumps(simulated_results, indent=2)}]}
    
    async def _particle_analysis(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Specialized particle analysis"""
        file_name = args['file_name']
        min_particle_size = args.get('min_particle_size', 50)
        detection_method = args.get('detection_method', 'threshold')
        
        # Simulated particle analysis results
        results = {
            "file_name": file_name,
            "detection_method": detection_method,
            "min_particle_size": min_particle_size,
            "particle_statistics": {
                "total_particles": 234,
                "size_distribution": {
                    "small": 89,
                    "medium": 112,
                    "large": 33
                },
                "shape_analysis": {
                    "circular": 156,
                    "elongated": 45,
                    "irregular": 33
                },
                "spatial_distribution": {
                    "clustered": 0.67,
                    "random": 0.23,
                    "regular": 0.10
                }
            },
            "quality_metrics": {
                "detection_confidence": 0.89,
                "false_positive_rate": 0.05,
                "missed_particles": 12
            }
        }
        
        return {"content": [{"type": "text", "text": json.dumps(results, indent=2)}]}
    
    async def _phase_analysis(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Phase analysis for multi-phase materials"""
        file_name = args['file_name']
        num_phases = args.get('num_phases', 3)
        analysis_method = args.get('analysis_method', 'clustering')
        
        # Simulated phase analysis results
        results = {
            "file_name": file_name,
            "analysis_method": analysis_method,
            "expected_phases": num_phases,
            "detected_phases": 3,
            "phase_properties": {
                "phase_1": {
                    "name": "Matrix Phase",
                    "area_fraction": 0.45,
                    "intensity_mean": 85.2,
                    "intensity_std": 12.3,
                    "morphology": "continuous",
                    "boundary_length": 1234.5
                },
                "phase_2": {
                    "name": "Precipitate Phase",
                    "area_fraction": 0.32,
                    "intensity_mean": 156.8,
                    "intensity_std": 23.4,
                    "morphology": "discrete",
                    "particle_count": 234
                },
                "phase_3": {
                    "name": "Pore Phase",
                    "area_fraction": 0.23,
                    "intensity_mean": 45.1,
                    "intensity_std": 8.9,
                    "morphology": "irregular",
                    "porosity": 0.23
                }
            },
            "phase_relationships": {
                "interfacial_area": 2345.6,
                "phase_connectivity": 0.78,
                "segregation_index": 0.45
            }
        }
        
        return {"content": [{"type": "text", "text": json.dumps(results, indent=2)}]}
    
    async def _defect_detection(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Defect detection and analysis"""
        file_name = args['file_name']
        defect_types = args.get('defect_types', ['cracks', 'pores', 'inclusions'])
        sensitivity = args.get('sensitivity', 0.8)
        
        # Simulated defect detection results
        results = {
            "file_name": file_name,
            "detection_sensitivity": sensitivity,
            "defect_types_searched": defect_types,
            "defect_summary": {
                "total_defects": 45,
                "defect_area_fraction": 0.12,
                "defect_density": 0.023
            },
            "defect_details": {
                "cracks": {
                    "count": 12,
                    "total_length": 234.5,
                    "average_width": 2.3,
                    "orientation_distribution": {"horizontal": 0.4, "vertical": 0.3, "diagonal": 0.3}
                },
                "pores": {
                    "count": 23,
                    "total_area": 156.7,
                    "average_diameter": 4.5,
                    "size_distribution": {"small": 15, "medium": 6, "large": 2}
                },
                "inclusions": {
                    "count": 10,
                    "total_area": 89.3,
                    "average_size": 8.9,
                    "composition_estimate": "oxide"
                }
            },
            "quality_assessment": {
                "detection_confidence": 0.87,
                "false_positive_rate": 0.08,
                "missed_defects": 5
            }
        }
        
        return {"content": [{"type": "text", "text": json.dumps(results, indent=2)}]}
    
    async def _analyze_csv_data(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze CSV data based on operation type"""
        file_name = args['file_name']
        operation = args['operation']
        parameters = args.get('parameters', {})
        
        # Find the file
        file_info = next((f for f in self.dataset_info['files'] 
                         if f.get('key', f.get('name', '')) == file_name), None)
        
        if not file_info:
            return {"error": f"CSV file not found: {file_name}"}
        
        # Generate sample data for demonstration
        df = self._generate_sample_dataframe(file_info)
        
        if operation == "summary":
            result = {
                "operation": "summary",
                "file": file_name,
                "summary": {
                    "rows": len(df),
                    "columns": len(df.columns),
                    "column_names": df.columns.tolist(),
                    "data_types": df.dtypes.to_dict(),
                    "missing_values": df.isnull().sum().to_dict(),
                    "numeric_summary": df.describe().to_dict() if len(df.select_dtypes(include=[np.number]).columns) > 0 else {}
                }
            }
        
        elif operation == "correlation":
            numeric_df = df.select_dtypes(include=[np.number])
            if len(numeric_df.columns) > 1:
                corr_matrix = numeric_df.corr().to_dict()
                result = {
                    "operation": "correlation",
                    "file": file_name,
                    "correlation_matrix": corr_matrix,
                    "strong_correlations": self._find_strong_correlations(corr_matrix)
                }
            else:
                result = {"error": "Insufficient numeric columns for correlation analysis"}
        
        elif operation == "time_series":
            # Assume first column with 'time' in name is time column
            time_cols = [col for col in df.columns if 'time' in col.lower()]
            if time_cols:
                time_col = time_cols[0]
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                
                result = {
                    "operation": "time_series",
                    "file": file_name,
                    "time_column": time_col,
                    "numeric_columns": numeric_cols,
                    "time_range": {
                        "start": str(df[time_col].min()),
                        "end": str(df[time_col].max()),
                        "duration": str(df[time_col].max() - df[time_col].min())
                    },
                    "trends": self._analyze_trends(df, time_col, numeric_cols)
                }
            else:
                result = {"error": "No time column found for time series analysis"}
        
        elif operation == "filter":
            filter_column = parameters.get('column')
            filter_condition = parameters.get('condition', {})
            
            if filter_column and filter_column in df.columns:
                filtered_df = self._apply_filter(df, filter_column, filter_condition)
                result = {
                    "operation": "filter",
                    "file": file_name,
                    "filter_applied": {
                        "column": filter_column,
                        "condition": filter_condition
                    },
                    "original_rows": len(df),
                    "filtered_rows": len(filtered_df),
                    "sample_filtered_data": filtered_df.head(10).to_dict()
                }
            else:
                result = {"error": f"Column '{filter_column}' not found"}
        
        else:
            result = {"error": f"Unknown operation: {operation}"}
        
        return {"content": [{"type": "text", "text": json.dumps(result, indent=2)}]}
    
    async def _analyze_image(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze image based on analysis type"""
        file_name = args['file_name']
        analysis_type = args['analysis_type']
        parameters = args.get('parameters', {})
        
        # Find the image file
        file_info = next((f for f in self.dataset_info['files'] 
                         if f.get('key', f.get('name', '')) == file_name), None)
        
        if not file_info:
            return {"content": [{"type": "text", "text": json.dumps({"error": f"Image file not found: {file_name}"})}]}
        
        # Simulated image analysis results
        if analysis_type == "basic_info":
            metadata = file_info.get('metadata', {})
            result = {
                "analysis_type": "basic_info",
                "file": file_name,
                "properties": {
                    "dimensions": metadata.get('dimensions', [800, 600]),
                    "size_bytes": file_info['size'],
                    "format": metadata.get('format', 'JPEG'),
                    "color_mode": metadata.get('mode', 'RGB'),
                    "has_transparency": metadata.get('has_transparency', False)
                }
            }
        
        elif analysis_type == "color_analysis":
            result = {
                "analysis_type": "color_analysis",
                "file": file_name,
                "color_statistics": {
                    "dominant_colors": ["#3498db", "#e74c3c", "#2ecc71"],
                    "color_distribution": {"blue": 0.35, "red": 0.28, "green": 0.37},
                    "brightness_mean": 128.5,
                    "contrast_ratio": 0.72
                }
            }
        
        elif analysis_type == "edge_detection":
            threshold1 = parameters.get('threshold1', 50)
            threshold2 = parameters.get('threshold2', 150)
            result = {
                "analysis_type": "edge_detection",
                "file": file_name,
                "parameters": {"threshold1": threshold1, "threshold2": threshold2},
                "results": {
                    "edges_detected": 1247,
                    "edge_density": 0.123,
                    "major_edge_directions": ["horizontal", "vertical", "diagonal"],
                    "edge_length_total": 15678.2
                }
            }
        
        elif analysis_type == "pattern_detection":
            result = {
                "analysis_type": "pattern_detection",
                "file": file_name,
                "patterns": {
                    "circular_patterns": {"count": 23, "average_radius": 45.2},
                    "linear_features": {"count": 45, "average_length": 123.8},
                    "cluster_regions": {"count": 12, "total_area": 2847.5},
                    "texture_variations": {"count": 8, "uniformity_score": 0.67}
                }
            }
        
        return {"content": [{"type": "text", "text": json.dumps(result, indent=2)}]}
    
    async def _create_visualization(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Create visualization from data"""
        # Placeholder for visualization creation
        result = {
            "visualization_type": args.get('plot_type', 'scatter'),
            "data_source": args.get('data_source', 'unknown'),
            "status": "visualization_created",
            "plot_url": "https://example.com/plot.png"
        }
        
        return {"content": [{"type": "text", "text": json.dumps(result, indent=2)}]}
    
    async def _compare_datasets(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Compare datasets"""
        # Placeholder for dataset comparison
        result = {
            "comparison_type": args.get('comparison_type', 'statistical'),
            "sources": args.get('sources', []),
            "comparison_results": {
                "similarity_score": 0.78,
                "key_differences": ["mean_value", "distribution_shape"],
                "statistical_significance": 0.05
                }
            }
        
        return {"content": [{"type": "text", "text": json.dumps(result, indent=2)}]}
    
    def _get_mime_type(self, file_type: str) -> str:
        """Get MIME type for file type"""
        mime_types = {
            'images': 'image/jpeg',
            'data': 'text/csv',
            'video': 'video/mp4',
            'documents': 'application/pdf'
        }
        return mime_types.get(file_type, 'application/octet-stream')
    
    def _get_sample_data(self, file_info: Dict[str, Any]) -> Any:
        """Get sample data for file"""
        if file_info['type'] == 'data':
            return {
                "type": "dataframe_sample",
                "columns": ["col1", "col2", "col3"],
                "rows": [
                    {"col1": 1.0, "col2": 2.5, "col3": 3.2},
                    {"col1": 1.2, "col2": 2.8, "col3": 3.1},
                    {"col1": 1.2, "col2": 2.8}
                ]
            }
        elif file_info['type'] == 'images':
            return {
                "type": "image_info",
                "dimensions": file_info.get('metadata', {}).get('dimensions', [800, 600]),
                "format": file_info.get('metadata', {}).get('format', 'JPEG')
            }
        else:
            return {"type": "file_reference", "name": file_info.get('key', file_info.get('name', ''))}
    
    def _generate_sample_dataframe(self, file_info: Dict[str, Any]) -> pd.DataFrame:
        """Generate sample DataFrame for analysis"""
        metadata = file_info.get('metadata', {})
        
        if 'column_names' in metadata:
            columns = metadata['column_names']
            n_rows = metadata.get('rows', 100)
        else:
            columns = ['time', 'pressure', 'temperature', 'flow_rate']
            n_rows = 100
        
        # Generate sample data
        np.random.seed(42)
        data = {}
        
        for i, col in enumerate(columns):
            if 'time' in col.lower():
                data[col] = pd.date_range('2024-01-01', periods=n_rows, freq='1min')
            else:
                # Generate realistic-looking data with some pattern
                base_value = 100 + i * 10
                noise = np.random.normal(0, 5, n_rows)
                trend = np.sin(np.arange(n_rows) * 0.1) * 10
                data[col] = base_value + noise + trend
        
        return pd.DataFrame(data)
    
    def _find_strong_correlations(self, corr_matrix: Dict[str, Dict[str, float]], threshold: float = 0.7) -> List[Dict[str, Any]]:
        """Find strong correlations in correlation matrix"""
        strong_corr = []
        
        for col1, correlations in corr_matrix.items():
            for col2, corr_value in correlations.items():
                if col1 != col2 and abs(corr_value) > threshold:
                    strong_corr.append({
                        "variables": [col1, col2],
                        "correlation": corr_value,
                        "strength": "strong positive" if corr_value > threshold else "strong negative"
                    })
        
        return strong_corr
    
    def _analyze_trends(self, df: pd.DataFrame, time_col: str, numeric_cols: List[str]) -> Dict[str, Any]:
        """Analyze trends in time series data"""
        trends = {}
        
        for col in numeric_cols:
            if col in df.columns:
                values = df[col].values
                # Simple trend analysis
                slope = np.polyfit(range(len(values)), values, 1)[0]
                
                trends[col] = {
                    "direction": "increasing" if slope > 0 else "decreasing" if slope < 0 else "stable",
                    "slope": float(slope),
                    "mean": float(values.mean()),
                    "std": float(values.std()),
                    "range": [float(values.min()), float(values.max())]
                }
        
        return trends
    
    def _apply_filter(self, df: pd.DataFrame, column: str, condition: Dict[str, Any]) -> pd.DataFrame:
        """Apply filter condition to DataFrame"""
        if 'min' in condition and 'max' in condition:
            return df[(df[column] >= condition['min']) & (df[column] <= condition['max'])]
        elif 'equals' in condition:
            return df[df[column] == condition['equals']]
        elif 'contains' in condition:
            return df[df[column].astype(str).str.contains(condition['contains'], case=False)]
        else:
            return df

# AI Assistant Integration
class AIAnalysisAssistant:
    """AI Assistant that uses MCP to access and analyze data"""
    
    def __init__(self, mcp_server: DataAnalysisMCPServer):
        self.mcp_server = mcp_server
    
    async def process_natural_language_query(self, query: str) -> Dict[str, Any]:
        """Process natural language query and execute appropriate analysis"""
        
        # Simple query parsing (in production, use proper NLP)
        query_lower = query.lower()
        
        if "summary" in query_lower or "overview" in query_lower:
            # Get data summary
            resources = await self.mcp_server.list_resources()
            data_files = [r for r in resources if r['mimeType'] == 'text/csv']
            
            if data_files:
                result = await self.mcp_server.call_tool("analyze_csv_data", {
                    "file_name": data_files[0]['name'].replace('File: ', ''),
                    "operation": "summary"
                })
                return result
        
        elif "correlation" in query_lower or "relationship" in query_lower:
            # Analyze correlations
            resources = await self.mcp_server.list_resources()
            data_files = [r for r in resources if r['mimeType'] == 'text/csv']
            
            if data_files:
                result = await self.mcp_server.call_tool("analyze_csv_data", {
                    "file_name": data_files[0]['name'].replace('File: ', ''),
                    "operation": "correlation"
                })
                return result
        
        elif "image" in query_lower and "analyze" in query_lower:
            # Analyze images
            resources = await self.mcp_server.list_resources()
            image_files = [r for r in resources if r['mimeType'] == 'image/jpeg']
            
            if image_files:
                result = await self.mcp_server.call_tool("enhanced_image_analysis", {
                    "file_name": image_files[0]['name'].replace('File: ', ''),
                    "analysis_types": ["basic", "morphology", "pattern"]
                })
                return result
        
        elif "particle" in query_lower or "grain" in query_lower:
            # Particle analysis
            resources = await self.mcp_server.list_resources()
            image_files = [r for r in resources if r['mimeType'] == 'image/jpeg']
            
            if image_files:
                result = await self.mcp_server.call_tool("particle_analysis", {
                    "file_name": image_files[0]['name'].replace('File: ', ''),
                    "detection_method": "watershed"
                })
                return result
        
        elif "phase" in query_lower:
            # Phase analysis
            resources = await self.mcp_server.list_resources()
            image_files = [r for r in resources if r['mimeType'] == 'image/jpeg']
            
            if image_files:
                result = await self.mcp_server.call_tool("phase_analysis", {
                    "file_name": image_files[0]['name'].replace('File: ', ''),
                    "num_phases": 3
                })
                return result
        
        elif "defect" in query_lower:
            # Defect detection
            resources = await self.mcp_server.list_resources()
            image_files = [r for r in resources if r['mimeType'] == 'image/jpeg']
            
            if image_files:
                result = await self.mcp_server.call_tool("defect_detection", {
                    "file_name": image_files[0]['name'].replace('File: ', ''),
                    "defect_types": ["cracks", "pores", "inclusions"]
                })
                return result
        
        else:
            return {
                "content": [{
                    "type": "text", 
                    "text": "I can help you analyze your data. Try asking about:\n- Data summary or overview\n- Correlations between variables\n- Image analysis (basic, morphology, pattern)\n- Particle/grain analysis\n- Phase analysis\n- Defect detection\n- Time series trends"
                }]
            }

# Streamlit integration
def create_ai_chat_interface(dataset_info: Dict[str, Any]):
    """Create AI chat interface using MCP"""
    
    st.subheader("🤖 AI Analysis Assistant")
    
    # Initialize MCP server and AI assistant
    if 'mcp_server' not in st.session_state:
        st.session_state.mcp_server = DataAnalysisMCPServer(dataset_info)
        st.session_state.ai_assistant = AIAnalysisAssistant(st.session_state.mcp_server)
    
    # Chat interface
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Display chat history
    for message in st.session_state.chat_history:
        if message['role'] == 'user':
            st.markdown(f"**You:** {message['content']}")
        else:
            st.markdown(f"**AI:** {message['content']}")
    
    # Input for new query
    user_query = st.text_input("Ask me anything about your data:", key="ai_query")
    
    if st.button("Send") and user_query:
        # Add user message
        st.session_state.chat_history.append({
            'role': 'user',
            'content': user_query
        })
        
        # Get AI response
        with st.spinner("Analyzing..."):
            try:
                import asyncio
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                response = loop.run_until_complete(
                    st.session_state.ai_assistant.process_natural_language_query(user_query)
                )
                
                # Extract response text
                if 'content' in response and response['content']:
                    ai_response = response['content'][0]['text']
                else:
                    ai_response = "I couldn't process that query. Please try rephrasing."
                
                # Add AI response
                st.session_state.chat_history.append({
                    'role': 'assistant',
                    'content': ai_response
                })
                
                st.rerun()
                
            except Exception as e:
                st.error(f"Error processing query: {e}")