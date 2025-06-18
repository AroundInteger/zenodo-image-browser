# src/data/ingestion.py
import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import io
from typing import List, Dict, Any
import hashlib
import json
from datetime import datetime

class DataIngestionPipeline:
    """Handles upload, validation, and metadata extraction for new datasets"""
    
    def __init__(self):
        self.supported_formats = {
            'images': ['jpg', 'jpeg', 'png', 'gif', 'bmp', 'tiff', 'webp'],
            'data': ['csv', 'json', 'xlsx', 'txt', 'h5', 'hdf5'],
            'video': ['mp4', 'avi', 'mov', 'mkv'],
            'documents': ['pdf', 'docx', 'md']
        }
    
    def upload_interface(self) -> Dict[str, Any]:
        """Create the upload interface and return uploaded data info"""
        st.subheader("📤 Upload New Dataset")
        
        # Basic dataset info
        with st.form("dataset_upload"):
            col1, col2 = st.columns(2)
            
            with col1:
                title = st.text_input("Dataset Title*", placeholder="Enter a descriptive title")
                authors = st.text_area("Authors*", placeholder="Name 1, Affiliation 1\nName 2, Affiliation 2")
                keywords = st.text_input("Keywords", placeholder="keyword1, keyword2, keyword3")
            
            with col2:
                description = st.text_area("Description*", placeholder="Describe your dataset...")
                experiment_type = st.selectbox("Experiment Type", [
                    "Fluid Mechanics", "Pattern Formation", "Rheology", 
                    "Microscopy", "Time Series", "Other"
                ])
                access_level = st.selectbox("Access Level", ["Private", "Team", "Public"])
            
            # File upload
            st.markdown("### Upload Files")
            uploaded_files = st.file_uploader(
                "Choose files", 
                accept_multiple_files=True,
                help="Supported formats: Images, CSV, JSON, Excel, Videos, Documents"
            )
            
            # Submit button
            submitted = st.form_submit_button("Process Upload")
            
            if submitted and uploaded_files and title and authors and description:
                return self.process_upload(
                    uploaded_files, title, authors, description, 
                    keywords, experiment_type, access_level
                )
        
        return None
    
    def process_upload(self, files: List, title: str, authors: str, 
                      description: str, keywords: str, experiment_type: str, 
                      access_level: str) -> Dict[str, Any]:
        """Process uploaded files and extract metadata"""
        
        with st.spinner("Processing uploaded files..."):
            dataset_info = {
                'title': title,
                'authors': self.parse_authors(authors),
                'description': description,
                'keywords': [k.strip() for k in keywords.split(',') if k.strip()],
                'experiment_type': experiment_type,
                'access_level': access_level,
                'upload_date': datetime.now().isoformat(),
                'files': [],
                'summary': {}
            }
            
            # Process each file
            for uploaded_file in files:
                file_info = self.analyze_file(uploaded_file)
                dataset_info['files'].append(file_info)
            
            # Generate dataset summary
            dataset_info['summary'] = self.generate_dataset_summary(dataset_info['files'])
            
            # Display results
            self.display_upload_results(dataset_info)
            
            return dataset_info
    
    def analyze_file(self, uploaded_file) -> Dict[str, Any]:
        """Analyze individual file and extract metadata"""
        file_info = {
            'name': uploaded_file.name,
            'size': uploaded_file.size,
            'type': self.get_file_type(uploaded_file.name),
            'checksum': None,
            'metadata': {}
        }
        
        # Calculate checksum
        content = uploaded_file.read()
        uploaded_file.seek(0)  # Reset file pointer
        file_info['checksum'] = hashlib.md5(content).hexdigest()
        
        # Type-specific analysis
        file_ext = uploaded_file.name.split('.')[-1].lower()
        
        if file_ext in self.supported_formats['images']:
            file_info['metadata'] = self.analyze_image(uploaded_file)
        elif file_ext in ['csv']:
            file_info['metadata'] = self.analyze_csv(uploaded_file)
        elif file_ext in ['json']:
            file_info['metadata'] = self.analyze_json(uploaded_file)
        
        return file_info
    
    def analyze_image(self, uploaded_file) -> Dict[str, Any]:
        """Extract image metadata"""
        try:
            image = Image.open(uploaded_file)
            uploaded_file.seek(0)
            
            metadata = {
                'dimensions': image.size,
                'mode': image.mode,
                'format': image.format,
                'has_transparency': image.mode in ('RGBA', 'LA') or 'transparency' in image.info
            }
            
            # Extract EXIF data if available
            if hasattr(image, '_getexif') and image._getexif():
                metadata['exif'] = dict(image._getexif())
            
            return metadata
        except Exception as e:
            return {'error': str(e)}
    
    def analyze_csv(self, uploaded_file) -> Dict[str, Any]:
        """Analyze CSV file structure"""
        try:
            df = pd.read_csv(uploaded_file)
            uploaded_file.seek(0)
            
            metadata = {
                'rows': len(df),
                'columns': len(df.columns),
                'column_names': list(df.columns),
                'data_types': df.dtypes.to_dict(),
                'missing_values': df.isnull().sum().to_dict(),
                'sample_data': df.head(3).to_dict()
            }
            
            # Detect numeric columns for stats
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                metadata['statistics'] = df[numeric_cols].describe().to_dict()
            
            return metadata
        except Exception as e:
            return {'error': str(e)}
    
    def analyze_json(self, uploaded_file) -> Dict[str, Any]:
        """Analyze JSON file structure"""
        try:
            content = json.load(uploaded_file)
            uploaded_file.seek(0)
            
            metadata = {
                'type': type(content).__name__,
                'size': len(content) if isinstance(content, (list, dict)) else 1,
                'keys': list(content.keys()) if isinstance(content, dict) else None
            }
            
            return metadata
        except Exception as e:
            return {'error': str(e)}
    
    def get_file_type(self, filename: str) -> str:
        """Determine file category"""
        ext = filename.split('.')[-1].lower()
        for category, extensions in self.supported_formats.items():
            if ext in extensions:
                return category
        return 'unknown'
    
    def parse_authors(self, authors_text: str) -> List[Dict[str, str]]:
        """Parse authors text into structured format"""
        authors = []
        for line in authors_text.strip().split('\n'):
            if ',' in line:
                parts = line.split(',')
                name = parts[0].strip()
                affiliation = ','.join(parts[1:]).strip()
                authors.append({'name': name, 'affiliation': affiliation})
            else:
                authors.append({'name': line.strip(), 'affiliation': ''})
        return authors
    
    def generate_dataset_summary(self, files: List[Dict]) -> Dict[str, Any]:
        """Generate high-level dataset summary"""
        summary = {
            'total_files': len(files),
            'total_size_mb': sum(f['size'] for f in files) / (1024 * 1024),
            'file_types': {},
            'has_images': False,
            'has_data': False,
            'has_time_series': False
        }
        
        # Count file types
        for file_info in files:
            file_type = file_info['type']
            summary['file_types'][file_type] = summary['file_types'].get(file_type, 0) + 1
            
            if file_type == 'images':
                summary['has_images'] = True
            elif file_type == 'data':
                summary['has_data'] = True
                # Check if CSV might be time series
                if 'metadata' in file_info and 'column_names' in file_info['metadata']:
                    cols = [c.lower() for c in file_info['metadata']['column_names']]
                    if any(time_word in ' '.join(cols) for time_word in ['time', 'date', 'timestamp']):
                        summary['has_time_series'] = True
        
        return summary
    
    def display_upload_results(self, dataset_info: Dict[str, Any]):
        """Display upload processing results"""
        st.success("✅ Files processed successfully!")
        
        # Dataset overview
        st.subheader("Dataset Overview")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Files", dataset_info['summary']['total_files'])
        with col2:
            st.metric("Total Size", f"{dataset_info['summary']['total_size_mb']:.1f} MB")
        with col3:
            st.metric("File Types", len(dataset_info['summary']['file_types']))
        
        # File details
        with st.expander("📋 File Details"):
            for file_info in dataset_info['files']:
                st.markdown(f"**{file_info['name']}** ({file_info['type']})")
                st.markdown(f"- Size: {file_info['size'] / 1024:.1f} KB")
                st.markdown(f"- Checksum: `{file_info['checksum'][:8]}...`")
                
                if file_info['metadata']:
                    if file_info['type'] == 'images' and 'dimensions' in file_info['metadata']:
                        dims = file_info['metadata']['dimensions']
                        st.markdown(f"- Dimensions: {dims[0]} × {dims[1]}")
                    elif file_info['type'] == 'data' and 'rows' in file_info['metadata']:
                        rows = file_info['metadata']['rows']
                        cols = file_info['metadata']['columns']
                        st.markdown(f"- Data: {rows} rows × {cols} columns")
                
                st.markdown("---")
        
        # Next steps
        st.subheader("Next Steps")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("💾 Save Dataset"):
                # Here you would save to your storage system
                st.success("Dataset saved to workspace!")
        
        with col2:
            if st.button("🔍 Start Analysis"):
                # Navigate to analysis interface
                st.info("Ready for analysis!")

# Streamlit interface
def upload_page():
    """Main upload page interface"""
    st.title("📤 Data Upload & Ingestion")
    
    pipeline = DataIngestionPipeline()
    
    # Upload interface
    dataset_info = pipeline.upload_interface()
    
    if dataset_info:
        # Store in session state for further processing
        st.session_state.current_dataset = dataset_info