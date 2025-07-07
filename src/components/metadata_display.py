# src/components/metadata_display.py
import streamlit as st
from typing import Dict, Any
import pandas as pd
import requests

def display_dataset_metadata(dataset: Dict[Any, Any]) -> None:
    """Display comprehensive dataset metadata in organized sections"""
    metadata = dataset.get('metadata', {})
    
    # Main info section
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown(f"### {metadata.get('title', 'Untitled Dataset')}")
        
        # Authors with affiliations
        creators = metadata.get('creators', [])
        if creators:
            authors_text = []
            for creator in creators:
                name = creator.get('name', '')
                affiliation = creator.get('affiliation', '')
                if affiliation:
                    authors_text.append(f"{name} ({affiliation})")
                else:
                    authors_text.append(name)
            st.markdown(f"**Authors:** {'; '.join(authors_text)}")
        
        # Description
        description = metadata.get('description', '')
        if description:
            st.markdown("**Description:**")
            st.markdown(description, unsafe_allow_html=True)
    
    with col2:
        # Key metrics
        st.markdown("#### Dataset Info")
        
        # Publication date
        pub_date = metadata.get('publication_date', 'N/A')
        st.metric("Publication Date", pub_date)
        
        # DOI
        doi = metadata.get('doi', dataset.get('doi', 'N/A'))
        st.metric("DOI", doi)
        
        # Version
        version = metadata.get('version', 'N/A')
        st.metric("Version", version)
        
        # Access rights
        access_right = metadata.get('access_right', 'N/A')
        st.metric("Access", access_right.title())
    
    # Expandable sections
    with st.expander("📊 Technical Details"):
        col1, col2 = st.columns(2)
        
        with col1:
            # Keywords
            keywords = metadata.get('keywords', [])
            if keywords:
                st.markdown("**Keywords:**")
                for keyword in keywords:
                    st.markdown(f"- {keyword}")
        
        with col2:
            # Subjects
            subjects = metadata.get('subjects', [])
            if subjects:
                st.markdown("**Subjects:**")
                for subject in subjects:
                    identifier = subject.get('identifier', '')
                    term = subject.get('term', '')
                    st.markdown(f"- {term} ({identifier})")
    
    # Files summary
    files = dataset.get('files', [])
    if files:
        with st.expander("📁 Files Overview"):
            file_data = []
            total_size = 0
            
            for file in files:
                name = file.get('key', '')
                size_bytes = file.get('size', 0)
                size_mb = size_bytes / (1024 * 1024)
                total_size += size_bytes
                
                # Determine file type
                ext = name.split('.')[-1].lower() if '.' in name else 'unknown'
                file_data.append({
                    'Name': name,
                    'Type': ext,
                    'Size (MB)': f"{size_mb:.2f}"
                })
            
            # Display file summary
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Files", len(files))
            with col2:
                st.metric("Total Size", f"{total_size / (1024 * 1024):.1f} MB")
            with col3:
                # Most common file type
                types = [f['Type'] for f in file_data]
                most_common = max(set(types), key=types.count) if types else 'N/A'
                st.metric("Primary Type", most_common)
            
            # Files table
            if file_data:
                df = pd.DataFrame(file_data)
                st.dataframe(df, use_container_width=True)

def display_file_preview(file_info: Dict[Any, Any]) -> None:
    """Enhanced file preview with metadata and context-aware download actions"""
    file_name = file_info.get('key', '')
    file_size = file_info.get('size', 0)
    file_url = file_info.get('links', {}).get('self', '')
    from_zip = file_info.get('from_zip', False)
    zip_source = file_info.get('zip_source')
    zip_inner_path = file_info.get('zip_inner_path')
    
    # Add visual indicator for ZIP files
    if from_zip:
        st.markdown(f"#### 📦 {file_name} *(from ZIP: {zip_source})*")
    else:
        st.markdown(f"#### {file_name}")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # File type specific preview
        ext = file_name.split('.')[-1].lower() if '.' in file_name else ''
        
        if ext in ['jpg', 'jpeg', 'png', 'gif', 'bmp', 'tiff', 'webp']:
            try:
                if from_zip:
                    # For ZIP files, we need to extract and display
                    st.info("📦 Image from ZIP archive - use Analysis tools to view")
                else:
                    st.image(file_url, caption=file_name, use_container_width=True)
            except Exception as e:
                st.error(f"Could not load image: {e}")
        
        elif ext in ['txt', 'md', 'csv']:
            if st.button(f"Preview {file_name}", key=f"preview_{file_name}"):
                try:
                    if from_zip:
                        st.info("📦 File from ZIP archive - use Analysis tools to view")
                    else:
                        response = requests.get(file_url)
                        if response.status_code == 200:
                            if ext == 'csv':
                                # Show CSV preview
                                import io
                                df = pd.read_csv(io.StringIO(response.text))
                                st.dataframe(df.head(10))
                                st.caption(f"Showing first 10 rows of {len(df)} total rows")
                            else:
                                # Show text preview
                                content = response.text[:1000]  # First 1000 chars
                                st.text_area("File Preview", content, height=200)
                                if len(response.text) > 1000:
                                    st.caption("Showing first 1000 characters...")
                except Exception as e:
                    st.error(f"Could not preview file: {e}")
        
        else:
            if from_zip:
                st.info(f"📦 {ext.upper()} file from ZIP archive")
            else:
                st.info(f"Preview not available for .{ext} files")
    
    with col2:
        # File metadata
        st.metric("Size", f"{file_size / (1024 * 1024):.2f} MB")
        
        # Context-aware download actions
        if from_zip:
            # For files from ZIPs, provide two options
            st.markdown("**📦 ZIP Actions:**")
            
            # Download individual file from ZIP
            if st.button("📄 Download File", key=f"download_file_{file_name}"):
                try:
                    # We need the dataset context to find the parent ZIP file
                    # For now, show a message that this requires the full dataset context
                    st.info("📦 To download individual files from ZIP archives, please use the 'File Browser' tool in the Analysis section.")
                except Exception as e:
                    st.error(f"Download failed: {e}")
            
            # Download original ZIP
            if st.button("📦 Download ZIP", key=f"download_zip_{file_name}"):
                try:
                    # We need the dataset context to find the parent ZIP file
                    # For now, show a message that this requires the full dataset context
                    st.info("📦 To download ZIP archives, please use the 'File Browser' tool in the Analysis section.")
                except Exception as e:
                    st.error(f"ZIP download failed: {e}")
        else:
            # Standard download for regular files
            if st.button("📥 Download", key=f"download_{file_name}"):
                try:
                    response = requests.get(file_url)
                    st.download_button(
                        "Click to download",
                        data=response.content,
                        file_name=file_name,
                        key=f"dl_btn_{file_name}"
                    )
                except Exception as e:
                    st.error(f"Download failed: {e}")