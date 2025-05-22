import streamlit as st
import os
from pathlib import Path
from src.api.zenodo import ZenodoAPI
from src.utils.image_utils import is_valid_image_url, get_image_dimensions, is_supported_image_format
import requests

# Set page config
st.set_page_config(
    page_title="Zenodo Image Browser",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize Zenodo API
zenodo_api = ZenodoAPI()

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
    
    ### Getting Started
    1. Use the sidebar to navigate between different sections
    2. Start by browsing available datasets
    3. Select a dataset to view its contents
    4. Use the analysis tools to explore the data
    """)

elif page == "Browse Datasets":
    st.title("Browse Datasets")
    
    # Add a section for direct record access
    st.subheader("Access Specific Record")
    record_id = st.text_input("Enter Zenodo Record ID", "7890690")
    
    if st.button("Load Record"):
        with st.spinner("Loading dataset..."):
            dataset = zenodo_api.get_dataset(record_id)
            if dataset:
                # Display dataset metadata
                st.markdown(f"""
                ### {dataset.get('metadata', {}).get('title', 'Untitled Dataset')}
                **Authors:** {', '.join(creator.get('name', '') for creator in dataset.get('metadata', {}).get('creators', []))}
                **Publication Date:** {dataset.get('metadata', {}).get('publication_date', 'N/A')}
                **Description:** {dataset.get('metadata', {}).get('description', 'No description available')}
                """)
                
                # Get and display files
                files = zenodo_api.get_files(record_id)
                if files:
                    st.subheader("Files")
                    for file in files:
                        file_name = file.get('key', '')
                        file_size = file.get('size', 0) / (1024 * 1024)  # Convert to MB
                        file_url = file.get('links', {}).get('self', '')
                        
                        col1, col2, col3 = st.columns([3, 1, 1])
                        with col1:
                            st.write(f"📄 {file_name}")
                        with col2:
                            st.write(f"{file_size:.2f} MB")
                        with col3:
                            if is_supported_image_format(file_name):
                                if st.button("Preview", key=file_name):
                                    st.image(file_url, caption=file_name)
                            else:
                                st.download_button(
                                    "Download",
                                    data=requests.get(file_url).content,
                                    file_name=file_name,
                                    key=f"download_{file_name}"
                                )
            else:
                st.error("Failed to load dataset. Please check the record ID and try again.")

elif page == "Analysis":
    st.title("Analysis Tools")
    st.write("Analysis tools will be implemented here.")
    
    # Placeholder for analysis options
    analysis_type = st.selectbox(
        "Select Analysis Type",
        ["Image Classification", "Feature Extraction", "Custom Analysis"]
    )
    
    if analysis_type:
        st.write(f"Selected analysis: {analysis_type}")

elif page == "Settings":
    st.title("Settings")
    st.write("Application settings will be implemented here.")
    
    # Placeholder for settings
    st.checkbox("Enable AI features")
    st.checkbox("Cache datasets locally")
    api_key = st.text_input("Zenodo API Key (optional)", type="password")
    if api_key:
        st.session_state.zenodo_api = ZenodoAPI(api_key=api_key)

# Footer
st.markdown("---")
st.markdown("Built with Streamlit • [GitHub Repository](https://github.com/yourusername/zenodo-image-browser)")
