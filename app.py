"""
Zenodo Image Browser - Main Application
A web application for browsing and analyzing experimental images from Zenodo datasets
"""

import streamlit as st
import os
from pathlib import Path
from src.api.zenodo import ZenodoAPI
from src.utils.image_utils import is_valid_image_url, get_image_dimensions, is_supported_image_format, get_file_type, format_file_size
from src.components.metadata_display import display_dataset_metadata, display_file_preview
from src.data.ingestion import DataIngestionPipeline
from src.analysis.interactive_tools import InteractiveAnalysisTools
from src.analysis.enhanced_image_interface import create_enhanced_image_analysis_interface
from src.ai.mcp_server import create_ai_chat_interface
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
from datetime import datetime

# Set page config
st.set_page_config(
    page_title="Zenodo Image Browser",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize components
zenodo_api = ZenodoAPI()
ingestion_pipeline = DataIngestionPipeline()
analysis_tools = InteractiveAnalysisTools()

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
    }
    .dataset-card {
        padding: 1rem;
        border: 1px solid #ddd;
        border-radius: 5px;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 5px;
        margin: 0.5rem 0;
    }
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .file-card {
        background-color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #e0e0e0;
        margin-bottom: 1rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f0f2f6;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1f77b4;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

# Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Home", "Browse Datasets", "Analysis", "Settings"]
)

# Main content
if page == "Home":
    st.title("Welcome to Zenodo Image Browser")
    st.markdown("""
    This application helps you explore and analyze experimental images from Zenodo datasets.
    
    ### Features
    - Browse and search Zenodo datasets
    - Preview images and videos
    - Interactive data visualization
    - AI-powered analysis and benchmarking
    - Upload and manage your own datasets
    
    ### Getting Started
    1. Use the sidebar to navigate between different sections
    2. Start by browsing available datasets or upload your own
    3. Select a dataset to view its contents
    4. Use the analysis tools to explore the data
    5. Ask AI questions about your data
    """)
    
    # Quick stats
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Datasets Available", "1000+", "via Zenodo")
    with col2:
        st.metric("Analysis Tools", "15+", "Interactive")
    with col3:
        st.metric("AI Features", "Active", "MCP Server")

elif page == "Browse Datasets":
    st.title("Browse Datasets")
    
    # Show current dataset info if available
    if 'current_dataset' in st.session_state:
        st.success(f"✅ Currently loaded: {st.session_state.current_dataset.get('metadata', {}).get('title', 'Unknown Dataset')}")
        if st.button("Clear Current Dataset"):
            del st.session_state.current_dataset
            st.rerun()
    
    # Tabs for different browse modes
    tab1, tab2 = st.tabs(["Zenodo Records", "Upload New Data"])
    
    with tab1:
        st.subheader("Access Zenodo Datasets")
        
        # Search functionality
        col1, col2 = st.columns([3, 1])
        with col1:
            search_query = st.text_input("Search datasets", placeholder="Enter keywords...")
        with col2:
            if st.button("Search"):
                if search_query:
                    with st.spinner("Searching..."):
                        results = zenodo_api.search_datasets(search_query)
                        if results:
                            st.session_state.search_results = results
                        else:
                            st.warning("No datasets found for your search.")
        
        # Direct record access
        st.subheader("Access Specific Record")
        record_id = st.text_input("Enter Zenodo Record ID", "7890690")
        
        if st.button("Load Record"):
            with st.spinner("Loading dataset and extracting ZIP files (this may take a few minutes for large datasets)..."):
                dataset = zenodo_api.get_dataset(record_id)
                if dataset:
                    # Get files with ZIP extraction
                    files = zenodo_api.get_files(record_id)
                    
                    # Count ZIP files for summary
                    zip_files = [f for f in files if f.get('from_zip', False)]
                    
                    # Update dataset with extracted files (store ALL files, not just first 50)
                    dataset['files'] = files
                    
                    # Use the enhanced metadata display
                    display_dataset_metadata(dataset)
                    st.session_state.current_dataset = dataset
                    
                    # Show files with enhanced preview
                    if files:
                        st.subheader(f"Files ({len(files)} total)")
                        
                        # Show summary of file types
                        if zip_files:
                            st.info(f"📦 Found {len(zip_files)} files extracted from ZIP archives")
                        
                        # For large datasets, show a warning and limit display in main view only
                        if len(files) > 10:
                            st.warning(f"⚠️ Large dataset detected ({len(files)} files). Only showing first 10 files in this view. Use the Analysis tools to browse all {len(files)} files.")
                            display_files = files[:10]
                        else:
                            display_files = files
                        
                        for file in display_files:
                            display_file_preview(file)
                else:
                    st.error("Failed to load dataset. Please check the record ID and try again.")
        
        # Display search results if available
        if 'search_results' in st.session_state:
            st.subheader("Search Results")
            for result in st.session_state.search_results[:5]:  # Show first 5 results
                with st.expander(f"{result.get('metadata', {}).get('title', 'Untitled')}"):
                    st.write(f"**Authors:** {', '.join(creator.get('name', '') for creator in result.get('metadata', {}).get('creators', []))}")
                    st.write(f"**Description:** {result.get('metadata', {}).get('description', 'No description')[:200]}...")
                    if st.button(f"Load Dataset", key=f"load_{result.get('id')}"):
                        st.session_state.current_dataset = result
                        st.rerun()
    
    with tab2:
        st.subheader("Upload Your Own Dataset")
        uploaded_dataset = ingestion_pipeline.upload_interface()
        if uploaded_dataset:
            st.session_state.current_dataset = uploaded_dataset
            st.success("Dataset uploaded successfully!")

elif page == "Analysis":
    st.title("Analysis Tools")
    
    if 'current_dataset' in st.session_state:
        # Use the enhanced analysis tools
        analysis_tools.create_analysis_interface(st.session_state.current_dataset)
        
        # Add AI chat interface
        st.markdown("---")
        st.subheader("🤖 AI Assistant")
        create_ai_chat_interface(st.session_state.current_dataset)
    else:
        st.info("Please load a dataset first to access analysis tools.")
        st.markdown("""
        ### Available Analysis Tools:
        - **Image Gallery**: Browse and filter images
        - **Image Analysis**: Measurements, filters, pattern detection
        - **Data Explorer**: Interactive data visualization
        - **Time Series Analysis**: Temporal data analysis
        - **Statistical Overview**: Comprehensive statistics
        - **AI Assistant**: Natural language queries about your data
        """)
        
        # Quick access to load a dataset
        st.markdown("---")
        st.subheader("Quick Load")
        record_id = st.text_input("Enter Zenodo Record ID", "4134841")
        if st.button("Load Dataset"):
            with st.spinner("Loading dataset..."):
                dataset = zenodo_api.get_dataset(record_id)
                if dataset:
                    files = zenodo_api.get_files(record_id)
                    dataset['files'] = files
                    st.session_state.current_dataset = dataset
                    st.success("Dataset loaded! You can now use the Analysis tools.")
                    st.rerun()
                else:
                    st.error("Failed to load dataset.")

elif page == "Settings":
    st.title("Settings")
    
    # API Configuration
    st.subheader("API Configuration")
    api_key = st.text_input("Zenodo API Key (optional)", type="password", help="For accessing private datasets")
    if api_key:
        st.session_state.zenodo_api = ZenodoAPI(api_key=api_key)
        st.success("API key configured!")
    
    # Application Settings
    st.subheader("Application Settings")
    col1, col2 = st.columns(2)
    
    with col1:
        st.checkbox("Enable AI features", value=True, key="ai_enabled")
        st.checkbox("Cache datasets locally", value=True, key="cache_enabled")
        st.checkbox("Auto-load last dataset", value=False, key="auto_load")
    
    with col2:
        st.selectbox("Default theme", ["Light", "Dark"], key="theme")
        st.selectbox("Language", ["English"], key="language")
        st.number_input("Max file size (MB)", min_value=1, max_value=1000, value=100, key="max_file_size")
    
    # Data Management
    st.subheader("Data Management")
    if st.button("Clear cached data"):
        # Clear session state
        for key in list(st.session_state.keys()):
            if key.startswith('cache_'):
                del st.session_state[key]
        st.success("Cached data cleared!")
    
    if st.button("Export settings"):
        st.info("Settings export feature coming soon!")

# Footer
st.markdown("---")
st.markdown("Built with Streamlit • [GitHub Repository](https://github.com/AroundInteger/zenodo-image-browser)")

def show_ai_page():
    """Display the AI assistant page"""
    st.header("🤖 AI Analysis Assistant")
    
    # Initialize with example dataset for demonstration
    if 'current_dataset' not in st.session_state:
        st.session_state.current_dataset = {
            "id": "7890690",
            "title": "Example Experimental Dataset",
            "files": [
                {"name": "sample_image.jpg", "type": "images", "size": 1024000},
                {"name": "experiment_data.csv", "type": "data", "size": 51200}
            ]
        }
    
    # Create AI chat interface
    create_ai_chat_interface(st.session_state.current_dataset)
