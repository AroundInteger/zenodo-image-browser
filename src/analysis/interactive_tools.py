# src/analysis/interactive_tools.py
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from PIL import Image
import cv2
from typing import Dict, List, Any, Optional

class InteractiveAnalysisTools:
    """Interactive analysis tools for different data types"""
    
    def __init__(self):
        self.tools = {
            'images': ['Image Gallery', 'Image Analysis', 'Time-lapse Viewer', 'Pattern Detection'],
            'data': ['Data Explorer', 'Time Series Analysis', 'Statistical Overview', 'Correlation Analysis'],
            'mixed': ['Multi-modal Analysis', 'Custom Dashboard']
        }
    
    def create_analysis_interface(self, dataset_info: Dict[str, Any]):
        """Create analysis interface based on dataset content"""
        
        st.subheader("🔬 Interactive Analysis")
        
        # Determine available tools based on data types
        file_types = set(file['type'] for file in dataset_info.get('files', []))
        available_tools = []
        
        for file_type in file_types:
            if file_type in self.tools:
                available_tools.extend(self.tools[file_type])
        
        if len(file_types) > 1:
            available_tools.extend(self.tools['mixed'])
        
        # Tool selection
        selected_tool = st.selectbox("Select Analysis Tool", available_tools)
        
        # Execute selected tool
        if selected_tool == 'Image Gallery':
            self.image_gallery(dataset_info)
        elif selected_tool == 'Image Analysis':
            self.image_analysis_tool(dataset_info)
        elif selected_tool == 'Time-lapse Viewer':
            self.timelapse_viewer(dataset_info)
        elif selected_tool == 'Data Explorer':
            self.data_explorer(dataset_info)
        elif selected_tool == 'Time Series Analysis':
            self.time_series_analysis(dataset_info)
        elif selected_tool == 'Statistical Overview':
            self.statistical_overview(dataset_info)
        else:
            st.info(f"Tool '{selected_tool}' is under development")
    
    def image_gallery(self, dataset_info: Dict[str, Any]):
        """Interactive image gallery with filtering and sorting"""
        
        # Get image files
        image_files = [f for f in dataset_info['files'] if f['type'] == 'images']
        
        if not image_files:
            st.warning("No image files found in this dataset")
            return
        
        st.markdown(f"### 🖼️ Image Gallery ({len(image_files)} images)")
        
        # Gallery controls
        col1, col2, col3 = st.columns(3)
        
        with col1:
            images_per_row = st.selectbox("Images per row", [2, 3, 4, 5], index=1)
        
        with col2:
            sort_by = st.selectbox("Sort by", ["Name", "Size", "Upload order"])
        
        with col3:
            show_metadata = st.checkbox("Show metadata", value=False)
        
        # Sort images
        if sort_by == "Size":
            image_files.sort(key=lambda x: x['size'], reverse=True)
        elif sort_by == "Name":
            image_files.sort(key=lambda x: x['name'])
        
        # Display gallery
        for i in range(0, len(image_files), images_per_row):
            cols = st.columns(images_per_row)
            
            for j, col in enumerate(cols):
                if i + j < len(image_files):
                    file_info = image_files[i + j]
                    
                    with col:
                        # Display image (placeholder - in real implementation, load from storage)
                        st.image(
                            f"https://via.placeholder.com/300x200?text={file_info['name'][:10]}",
                            caption=file_info['name'],
                            use_column_width=True
                        )
                        
                        if show_metadata and 'metadata' in file_info:
                            metadata = file_info['metadata']
                            if 'dimensions' in metadata:
                                st.caption(f"📐 {metadata['dimensions'][0]}×{metadata['dimensions'][1]}")
                            if 'size' in file_info:
                                st.caption(f"💾 {file_info['size']/1024:.1f} KB")
                        
                        if st.button(f"Analyze", key=f"analyze_{file_info['name']}"):
                            st.session_state.selected_image = file_info
                            st.rerun()
    
    def image_analysis_tool(self, dataset_info: Dict[str, Any]):
        """Single image analysis with measurements and filters"""
        
        # Get image files
        image_files = [f for f in dataset_info['files'] if f['type'] == 'images']
        
        if not image_files:
            st.warning("No image files found in this dataset")
            return
        
        # Image selection
        selected_image_name = st.selectbox(
            "Select image to analyze",
            [f['name'] for f in image_files]
        )
        
        selected_image = next(f for f in image_files if f['name'] == selected_image_name)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("### 🔍 Image Analysis")
            
            # Analysis options
            analysis_type = st.selectbox(
                "Analysis Type",
                ["Basic Info", "Color Analysis", "Edge Detection", "Pattern Analysis"]
            )
            
            # Placeholder image display
            st.image(
                f"https://via.placeholder.com/600x400?text={selected_image['name'][:15]}",
                caption=f"Analyzing: {selected_image['name']}"
            )
            
            # Analysis results based on type
            if analysis_type == "Basic Info":
                self.display_basic_image_info(selected_image)
            elif analysis_type == "Color Analysis":
                self.display_color_analysis(selected_image)
            elif analysis_type == "Edge Detection":
                self.display_edge_detection(selected_image)
            elif analysis_type == "Pattern Analysis":
                self.display_pattern_analysis(selected_image)
        
        with col2:
            st.markdown("### ⚙️ Analysis Settings")
            
            # Tool-specific settings
            if analysis_type == "Edge Detection":
                threshold1 = st.slider("Lower threshold", 0, 255, 50)
                threshold2 = st.slider("Upper threshold", 0, 255, 150)
                st.session_state.edge_params = {'t1': threshold1, 't2': threshold2}
            
            elif analysis_type == "Color Analysis":
                color_space = st.selectbox("Color space", ["RGB", "HSV", "LAB"])
                st.session_state.color_space = color_space
            
            # Export options
            st.markdown("### 💾 Export")
            if st.button("Export Results"):
                st.success("Analysis results exported!")
            
            if st.button("Save to Report"):
                st.success("Added to analysis report!")
    
    def data_explorer(self, dataset_info: Dict[str, Any]):
        """Interactive data exploration for CSV/tabular data"""
        
        # Get data files
        data_files = [f for f in dataset_info['files'] if f['type'] == 'data']
        
        if not data_files:
            st.warning("No data files found in this dataset")
            return
        
        # File selection
        selected_file_name = st.selectbox(
            "Select data file",
            [f['name'] for f in data_files]
        )
        
        selected_file = next(f for f in data_files if f['name'] == selected_file_name)
        
        st.markdown(f"### 📊 Data Explorer: {selected_file['name']}")
        
        # Create sample data for demonstration
        if 'metadata' in selected_file and 'columns' in selected_file['metadata']:
            # Use actual metadata if available
            n_rows = selected_file['metadata'].get('rows', 100)
            columns = selected_file['metadata']['column_names']
        else:
            # Generate sample data
            n_rows = 100
            columns = ['time', 'pressure', 'temperature', 'flow_rate']
        
        # Generate sample DataFrame
        np.random.seed(42)
        data = {}
        for i, col in enumerate(columns):
            if 'time' in col.lower():
                data[col] = pd.date_range('2024-01-01', periods=n_rows, freq='1min')
            else:
                data[col] = np.random.normal(100 + i*10, 5, n_rows) + np.sin(np.arange(n_rows)*0.1)*5
        
        df = pd.DataFrame(data)
        
        # Data overview
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Rows", len(df))
        with col2:
            st.metric("Columns", len(df.columns))
        with col3:
            st.metric("Memory Usage", f"{df.memory_usage(deep=True).sum() / 1024:.1f} KB")
        with col4:
            st.metric("Missing Values", df.isnull().sum().sum())
        
        # Interactive plotting
        st.markdown("#### 📈 Interactive Plotting")
        
        plot_type = st.selectbox("Plot Type", ["Line Plot", "Scatter Plot", "Histogram", "Box Plot", "Correlation Heatmap"])
        
        if plot_type == "Line Plot":
            col1, col2 = st.columns(2)
            with col1:
                x_col = st.selectbox("X-axis", df.columns)
            with col2:
                y_cols = st.multiselect("Y-axis", [c for c in df.columns if c != x_col])
            
            if y_cols:
                fig = go.Figure()
                for y_col in y_cols:
                    fig.add_trace(go.Scatter(x=df[x_col], y=df[y_col], name=y_col, mode='lines'))
                
                fig.update_layout(title=f"{', '.join(y_cols)} vs {x_col}", xaxis_title=x_col, yaxis_title="Value")
                st.plotly_chart(fig, use_container_width=True)
        
        elif plot_type == "Scatter Plot":
            col1, col2, col3 = st.columns(3)
            with col1:
                x_col = st.selectbox("X-axis", df.select_dtypes(include=[np.number]).columns)
            with col2:
                y_col = st.selectbox("Y-axis", [c for c in df.select_dtypes(include=[np.number]).columns if c != x_col])
            with col3:
                color_col = st.selectbox("Color by", [None] + list(df.columns))
            
            if x_col and y_col:
                fig = px.scatter(df, x=x_col, y=y_col, color=color_col, title=f"{y_col} vs {x_col}")
                st.plotly_chart(fig, use_container_width=True)
        
        # Data table with filtering
        st.markdown("#### 📋 Data Table")
        
        # Filtering options
        with st.expander("Filter Data"):
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            if numeric_cols:
                filter_col = st.selectbox("Filter column", numeric_cols)
                min_val, max_val = float(df[filter_col].min()), float(df[filter_col].max())
                filter_range = st.slider(
                    f"Filter {filter_col}",
                    min_val, max_val, (min_val, max_val)
                )
                
                filtered_df = df[(df[filter_col] >= filter_range[0]) & (df[filter_col] <= filter_range[1])]
                st.dataframe(filtered_df, use_container_width=True)
            else:
                st.dataframe(df, use_container_width=True)
    
    def display_basic_image_info(self, image_info: Dict[str, Any]):
        """Display basic image information"""
        metadata = image_info.get('metadata', {})
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**File Information**")
            st.write(f"📁 **Name:** {image_info['name']}")
            st.write(f"💾 **Size:** {image_info['size']/1024:.1f} KB")
            if 'checksum' in image_info:
                st.write(f"🔑 **Checksum:** {image_info['checksum'][:8]}...")
        
        with col2:
            st.markdown("**Image Properties**")
            if 'dimensions' in metadata:
                dims = metadata['dimensions']
                st.write(f"📐 **Dimensions:** {dims[0]} × {dims[1]} pixels")
                st.write(f"📊 **Aspect Ratio:** {dims[0]/dims[1]:.2f}")
            if 'mode' in metadata:
                st.write(f"🎨 **Color Mode:** {metadata['mode']}")
            if 'format' in metadata:
                st.write(f"📋 **Format:** {metadata['format']}")
    
    def display_color_analysis(self, image_info: Dict[str, Any]):
        """Display color analysis results"""
        st.markdown("**Color Analysis Results**")
        
        # Simulated color histogram
        colors = ['Red', 'Green', 'Blue']
        values = np.random.randint(0, 256, 3)
        
        fig = go.Figure(data=[go.Bar(x=colors, y=values)])
        fig.update_layout(title="Color Channel Histogram", yaxis_title="Pixel Count")
        st.plotly_chart(fig, use_container_width=True)
        
        # Color statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Dominant Color", "Blue", delta="45%")
        with col2:
            st.metric("Brightness", "128", delta="12")
        with col3:
            st.metric("Contrast", "0.75", delta="0.1")
    
    def display_edge_detection(self, image_info: Dict[str, Any]):
        """Display edge detection results"""
        st.markdown("**Edge Detection Results**")
        
        # Parameters from session state
        params = st.session_state.get('edge_params', {'t1': 50, 't2': 150})
        
        st.write(f"Using thresholds: {params['t1']}, {params['t2']}")
        
        # Simulated edge detection metrics
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Edges Detected", "1,247", delta="23")
        with col2:
            st.metric("Edge Density", "12.3%", delta="2.1%")
        
        # Placeholder for edge image
        st.image(
            "https://via.placeholder.com/400x300?text=Edge+Detection+Result",
            caption="Edge Detection Result"
        )
    
    def display_pattern_analysis(self, image_info: Dict[str, Any]):
        """Display pattern analysis results"""
        st.markdown("**Pattern Analysis Results**")
        
        # Simulated pattern metrics
        patterns = {
            'Circular Patterns': 23,
            'Linear Features': 45,
            'Cluster Regions': 12,
            'Texture Variations': 8
        }
        
        col1, col2 = st.columns(2)
        
        with col1:
            for pattern, count in patterns.items():
                st.metric(pattern, count)
        
        with col2:
            # Pattern distribution chart
            fig = go.Figure(data=[go.Pie(labels=list(patterns.keys()), values=list(patterns.values()))])
            fig.update_layout(title="Pattern Distribution")
            st.plotly_chart(fig, use_container_width=True)

# Usage in main app
def analysis_page():
    """Main analysis page"""
    st.title("🔬 Interactive Analysis")
    
    # Check if we have a dataset to analyze
    if 'current_dataset' not in st.session_state:
        st.warning("No dataset selected. Please upload or browse a dataset first.")
        return
    
    dataset_info = st.session_state.current_dataset
    
    # Create analysis interface
    analysis_tools = InteractiveAnalysisTools()
    analysis_tools.create_analysis_interface(dataset_info)