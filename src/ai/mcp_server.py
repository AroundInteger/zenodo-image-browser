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
            resources.append(MCPResource(
                uri=f"dataset://files/{file_info['name']}",
                name=f"File: {file_info['name']}",
                description=f"{file_info['type']} file ({file_info['size']} bytes)",
                mimeType=self._get_mime_type(file_info['type'])
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
            file_info = next((f for f in self.dataset_info['files'] if f['name'] == file_name), None)
            
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
        elif name == "create_visualization":
            return await self._create_visualization(arguments)
        elif name == "compare_datasets":
            return await self._compare_datasets(arguments)
        else:
            raise ValueError(f"Unknown tool: {name}")
    
    async def _analyze_csv_data(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze CSV data based on operation type"""
        file_name = args['file_name']
        operation = args['operation']
        parameters = args.get('parameters', {})
        
        # Find the file
        file_info = next((f for f in self.dataset_info['files'] 
                         if f['name'] == file_name and f['type'] == 'data'), None)
        
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
                         if f['name'] == file_name and f['type'] == 'images'), None)
        
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
        """Get sample data for a file"""
        if file_info['type'] == 'data':
            # Return sample CSV data
            return {
                "type": "csv_sample",
                "columns": file_info.get('metadata', {}).get('column_names', ['col1', 'col2']),
                "sample_rows": [
                    {"col1": 1.0, "col2": 2.5},
                    {"col1": 1.1, "col2": 2.3},
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
            return {"type": "file_reference", "name": file_info['name']}
    
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
                result = await self.mcp_server.call_tool("analyze_image", {
                    "file_name": image_files[0]['name'].replace('File: ', ''),
                    "analysis_type": "basic_info"
                })
                return result
        
        else:
            return {
                "content": [{
                    "type": "text", 
                    "text": "I can help you analyze your data. Try asking about:\n- Data summary or overview\n- Correlations between variables\n- Image analysis\n- Time series trends"
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