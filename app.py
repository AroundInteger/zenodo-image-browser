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
# Try to import AI libraries, but make them optional
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("OpenAI module not available. OpenAI AI Assistant will be disabled.")

try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    print("Ollama module not available. Local AI Assistant will be disabled.")

# Import context analyzer
try:
    from src.ai.context_analyzer import DatasetContextAnalyzer
    CONTEXT_ANALYZER_AVAILABLE = True
except ImportError:
    CONTEXT_ANALYZER_AVAILABLE = False
    print("Context analyzer not available. AI Assistant will use basic context.")

# Import FutureHouse integration
try:
    from src.ai.futurehouse_client import create_futurehouse_client
    FUTUREHOUSE_AVAILABLE = True
except ImportError:
    FUTUREHOUSE_AVAILABLE = False
    print("FutureHouse integration not available. Install with: pip install futurehouse-client")

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
        
        # Show dataset context if available
        if CONTEXT_ANALYZER_AVAILABLE and 'current_dataset' in st.session_state:
            with st.expander("📊 Dataset Context (What the AI knows about your data)"):
                context_analyzer = DatasetContextAnalyzer()
                context = context_analyzer.analyze_dataset_context(st.session_state.current_dataset)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Metadata:**")
                    st.write(f"Title: {context['metadata']['title']}")
                    st.write(f"Authors: {', '.join(context['metadata']['authors'][:3])}")
                    st.write(f"Research Domains: {', '.join(context['scientific_context']['research_domains'])}")
                    
                    st.write("**File Structure:**")
                    st.write(f"Total Files: {context['file_structure']['total_files']}")
                    st.write(f"Total Size: {context['file_structure']['size_statistics']['total_size_mb']:.1f} MB")
                    st.write(f"ZIP Extracted: {context['file_structure']['zip_extracted_files']} files")
                
                with col2:
                    st.write("**Content Analysis:**")
                    st.write(context['content_analysis']['content_summary'])
                    st.write(f"Experiment Type: {context['scientific_context']['experiment_type']}")
                    st.write(f"Data Complexity: {context['scientific_context']['data_complexity']}")
                    
                    if context['content_analysis']['images']['image_categories']:
                        st.write("**Image Categories:**")
                        for category, count in context['content_analysis']['images']['image_categories'].items():
                            st.write(f"- {category}: {count} files")
        
        # Check if any AI providers are available
        if not OPENAI_AVAILABLE and not OLLAMA_AVAILABLE and not FUTUREHOUSE_AVAILABLE:
            st.warning("⚠️ No AI modules available. AI Assistant is disabled.")
            st.info("To enable AI Assistant, install: `pip install openai` or `pip install ollama` or `pip install futurehouse-client`")
        else:
            # AI Provider Selection
            ai_providers = []
            if OPENAI_AVAILABLE:
                ai_providers.append("OpenAI GPT")
            if OLLAMA_AVAILABLE:
                ai_providers.append("Local Ollama")
            if FUTUREHOUSE_AVAILABLE:
                ai_providers.append("FutureHouse Research")
            
            selected_provider = st.selectbox(
                "Choose AI Provider:", 
                ai_providers,
                help="OpenAI requires API key and billing. Ollama runs locally for free."
            )
            
            # OpenAI Configuration
            if selected_provider == "OpenAI GPT" and OPENAI_AVAILABLE:
                api_key = st.sidebar.text_input("OpenAI API Key", type="password", help="For OpenAI AI Assistant")
                if api_key:
                    user_input = st.text_input("Ask me anything about your data:")
                    if st.button("Ask OpenAI") and user_input:
                        # Prepare enhanced context using context analyzer
                        dataset = st.session_state.current_dataset
                        
                        if CONTEXT_ANALYZER_AVAILABLE:
                            context_analyzer = DatasetContextAnalyzer()
                            prompt = context_analyzer.create_ai_prompt_context(dataset, user_input)
                        else:
                            # Fallback to basic context
                            files = dataset.get('files', [])
                            file_summaries = [f"{f.get('key', '')} ({f.get('type', f.get('from_zip', ''))})" for f in files[:10]]
                            context = f"Dataset contains {len(files)} files. Example files: " + ", ".join(file_summaries)
                            prompt = f"You are a scientific data assistant. {context}\nUser question: {user_input}"
                        
                        with st.spinner("OpenAI is thinking..."):
                            try:
                                client = openai.OpenAI(api_key=api_key)
                                response = client.chat.completions.create(
                                    model="gpt-3.5-turbo",
                                    messages=[{"role": "user", "content": prompt}]
                                )
                                st.markdown(response.choices[0].message.content)
                            except Exception as e:
                                error_msg = str(e)
                                if "insufficient_quota" in error_msg or "429" in error_msg:
                                    st.error("🚫 OpenAI API quota exceeded. This usually means:")
                                    st.markdown("""
                                    - You've used up your free $5 credit
                                    - You need to add billing information to your OpenAI account
                                    - You've hit rate limits for your current plan
                                    
                                    **Solutions:**
                                    - Visit [OpenAI Platform](https://platform.openai.com/account/billing) to add billing info
                                    - Check your usage at [OpenAI Usage](https://platform.openai.com/usage)
                                    - Try switching to Local Ollama (free and offline)
                                    """)
                                else:
                                    st.error(f"OpenAI API error: {error_msg}")
                else:
                    st.info("Please enter your OpenAI API key in the sidebar to use OpenAI.")
            
            # Ollama Configuration
            elif selected_provider == "Local Ollama" and OLLAMA_AVAILABLE:
                # Model selection for Ollama
                available_models = ["llama2:7b", "llama2", "mistral", "codellama", "llama2:13b"]
                selected_model = st.selectbox(
                    "Choose Ollama Model:", 
                    available_models,
                    help="Larger models are more capable but slower. Llama2 7B is a good balance."
                )
                
                # Check if model is available
                try:
                    models = ollama.list()
                    installed_models = [model.model for model in models.models]
                    if selected_model not in installed_models:
                        st.warning(f"⚠️ Model '{selected_model}' not installed.")
                        if st.button(f"Install {selected_model}"):
                            with st.spinner(f"Installing {selected_model} (this may take several minutes)..."):
                                try:
                                    ollama.pull(selected_model)
                                    st.success(f"✅ {selected_model} installed successfully!")
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"Failed to install {selected_model}: {e}")
                    else:
                        st.success(f"✅ {selected_model} is ready to use!")
                except Exception as e:
                    st.error(f"Failed to check Ollama models: {e}")
                    st.info("Make sure Ollama is running. Install from: https://ollama.ai")
                
                # Chat interface for Ollama
                user_input = st.text_input("Ask me anything about your data:")
                if st.button("Ask Local AI") and user_input:
                    # Prepare enhanced context using context analyzer
                    dataset = st.session_state.current_dataset
                    
                    if CONTEXT_ANALYZER_AVAILABLE:
                        context_analyzer = DatasetContextAnalyzer()
                        prompt = context_analyzer.create_ai_prompt_context(dataset, user_input)
                    else:
                        # Fallback to basic context
                        files = dataset.get('files', [])
                        file_summaries = [f"{f.get('key', '')} ({f.get('type', f.get('from_zip', ''))})" for f in files[:10]]
                        context = f"Dataset contains {len(files)} files. Example files: " + ", ".join(file_summaries)
                        prompt = f"You are a scientific data assistant. {context}\nUser question: {user_input}"
                    with st.spinner("Local AI is thinking..."):
                        try:
                            response = ollama.chat(model=selected_model, messages=[{"role": "user", "content": prompt}])
                            st.markdown(response['message']['content'])
                        except Exception as e:
                            st.error(f"Ollama error: {e}")
                            st.info("Make sure Ollama is running and the model is installed.")
            
            # FutureHouse Configuration
            elif selected_provider == "FutureHouse Research" and FUTUREHOUSE_AVAILABLE:
                api_key = st.sidebar.text_input("FutureHouse API Key", type="password", help="For FutureHouse Research Assistant")
                
                if api_key:
                    # Initialize FutureHouse client
                    futurehouse_client = create_futurehouse_client(api_key)
                    
                    if not futurehouse_client:
                        st.error("Failed to initialize FutureHouse client. Please check your API key.")
                    else:
                        st.success("✅ FutureHouse Research Assistant ready!")
                        
                        # Define available jobs
                        jobs = {
                            'CROW': {
                                'name': 'CROW (Fast Search)',
                                'description': 'Ask questions of scientific data sources, get high-accuracy cited responses',
                                'best_for': 'Quick literature searches, methodology questions, recent findings'
                            },
                            'FALCON': {
                                'name': 'FALCON (Deep Search)', 
                                'description': 'Comprehensive research using multiple sources, detailed structured reports',
                                'best_for': 'In-depth literature reviews, comprehensive analysis, detailed research'
                            },
                            'OWL': {
                                'name': 'OWL (Precedent Search)',
                                'description': 'Find if anyone has done similar research or experiments',
                                'best_for': 'Checking for similar work, avoiding duplication, finding precedents'
                            },
                            'PHOENIX': {
                                'name': 'PHOENIX (Chemistry Tasks)',
                                'description': 'Chemistry-specific tasks, synthesis planning, molecular design',
                                'best_for': 'Chemical synthesis, molecular design, cheminformatics'
                            }
                        }
                        
                        # Job selection
                        st.subheader("🔍 Research Search Options")
                        job_options = {job['name']: job_id for job_id, job in jobs.items()}
                        selected_job = st.selectbox(
                            "Choose Research Tool:",
                            list(job_options.keys()),
                            help="Select the type of research search you want to perform"
                        )
                        
                        # Show job description
                        job_id = job_options[selected_job]
                        job_info = jobs[job_id]
                        st.info(f"**{job_info['name']}**: {job_info['description']}")
                        st.write(f"**Best for**: {job_info['best_for']}")
                        
                        # Show suggested queries if context analyzer is available
                        if CONTEXT_ANALYZER_AVAILABLE and 'current_dataset' in st.session_state:
                            st.subheader("💡 Suggested Research Questions")
                            context_analyzer = DatasetContextAnalyzer()
                            context = context_analyzer.analyze_dataset_context(st.session_state.current_dataset)
                            # Generate suggested queries based on dataset context
                            suggestions = []
                            
                            # Extract context information
                            research_domains = context.get('scientific_context', {}).get('research_domains', [])
                            experiment_type = context.get('scientific_context', {}).get('experiment_type', '')
                            content_summary = context.get('content_analysis', {}).get('content_summary', '')
                            
                            # General research questions
                            suggestions.append({
                                'question': 'What are the current state-of-the-art methods in this research area?',
                                'job_type': 'CROW',
                                'description': 'Find recent methodologies and techniques'
                            })
                            
                            suggestions.append({
                                'question': 'Has anyone conducted similar experiments or studies?',
                                'job_type': 'OWL', 
                                'description': 'Check for similar work and precedents'
                            })
                            
                            # Domain-specific suggestions
                            if 'chemistry' in research_domains or 'materials' in research_domains:
                                suggestions.append({
                                    'question': 'What synthesis methods are available for these materials?',
                                    'job_type': 'PHOENIX',
                                    'description': 'Chemistry-specific synthesis planning'
                                })
                            
                            if experiment_type == 'time_series':
                                suggestions.append({
                                    'question': 'What are the best practices for time series analysis in this field?',
                                    'job_type': 'CROW',
                                    'description': 'Time series analysis methodologies'
                                })
                            
                            if 'microscopy' in content_summary.lower():
                                suggestions.append({
                                    'question': 'What are the latest microscopy techniques for this type of analysis?',
                                    'job_type': 'FALCON',
                                    'description': 'Comprehensive microscopy review'
                                })
                            
                            # Add a comprehensive literature review suggestion
                            suggestions.append({
                                'question': 'Provide a comprehensive literature review of this research area',
                                'job_type': 'FALCON',
                                'description': 'Deep dive into the literature'
                            })
                            
                            for i, suggestion in enumerate(suggestions[:3]):  # Show top 3 suggestions
                                if st.button(f"Ask: {suggestion['question']}", key=f"suggest_{i}"):
                                    st.session_state.futurehouse_question = suggestion['question']
                                    st.session_state.futurehouse_job = suggestion['job_type']
                        
                        # User input
                        user_input = st.text_input(
                            "Ask a research question:",
                            value=st.session_state.get('futurehouse_question', ''),
                            help="Ask about related research, methodologies, or literature in your field"
                        )
                        
                        if st.button("Search Literature") and user_input:
                            with st.spinner(f"Searching scientific literature with {selected_job}..."):
                                try:
                                    # Get dataset context
                                    if CONTEXT_ANALYZER_AVAILABLE:
                                        context_analyzer = DatasetContextAnalyzer()
                                        context = context_analyzer.analyze_dataset_context(st.session_state.current_dataset)
                                    else:
                                        context = {'metadata': {'title': 'Unknown Dataset'}}
                                    
                                    # Perform search
                                    # Perform FutureHouse search
                                    try:
                                        # Create context string
                                        context_str = f"Dataset: {context.get('metadata', {}).get('title', '')} | "
                                        context_str += f"Research domains: {', '.join(context.get('scientific_context', {}).get('research_domains', []))} | "
                                        context_str += f"Content: {context.get('content_analysis', {}).get('content_summary', '')}"
                                        
                                        # Query FutureHouse
                                        api_result = futurehouse_client.query(job_id, user_input, context_str)
                                        
                                        # Format the response
                                        if api_result and 'answer' in api_result:
                                            result = {
                                                'success': True,
                                                'answer': api_result.get('answer', ''),
                                                'formatted_answer': api_result.get('formatted_answer', api_result.get('answer', '')),
                                                'has_successful_answer': True,
                                                'job_type': job_id,
                                                'query': user_input
                                            }
                                        else:
                                            result = {
                                                'success': False,
                                                'error': 'No response from FutureHouse API',
                                                'suggestion': 'Please try a different query or check your API key'
                                            }
                                    except Exception as e:
                                        result = {
                                            'success': False,
                                            'error': f"FutureHouse search failed: {str(e)}",
                                            'suggestion': 'Check your API key and internet connection'
                                        }
                                    
                                    if result['success']:
                                        st.success("✅ Research completed!")
                                        
                                        # Display results
                                        st.subheader("📚 Research Results")
                                        st.markdown(result['answer'])
                                        
                                        # Show query used
                                        with st.expander("🔍 Search Query Used"):
                                            st.code(result['query'])
                                        
                                        # Show job type info
                                        st.info(f"Search performed using: {job_info['name']}")
                                        
                                    else:
                                        st.error(f"❌ Research failed: {result['error']}")
                                        if 'suggestion' in result:
                                            st.info(f"💡 {result['suggestion']}")
                                            
                                except Exception as e:
                                    st.error(f"FutureHouse search error: {e}")
                                    st.info("Please check your API key and internet connection.")
                else:
                    st.info("Please enter your FutureHouse API key in the sidebar to use the Research Assistant.")
                    st.markdown("""
                    **Get FutureHouse API Key:**
                    1. Visit [FutureHouse Platform](https://futurehouse.ai)
                    2. Sign up for an account
                    3. Get your API key from the dashboard
                    4. The client is already included in this app
                    """)
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
