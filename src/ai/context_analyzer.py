"""
Context Analyzer for AI Assistant
Provides rich context about Zenodo datasets to improve AI responses.
"""

import re
from collections import Counter
from typing import Dict, List, Any
import json


class DatasetContextAnalyzer:
    """Analyzes dataset context to provide rich information to AI Assistant."""
    
    def __init__(self):
        # Common scientific file patterns
        self.image_patterns = {
            'microscopy': r'(micro|scope|cell|tissue|stain|fluorescent|confocal|electron)',
            'graphs': r'(plot|chart|graph|figure|diagram|visualization)',
            'photos': r'(photo|image|picture|camera|photo)',
            'spectra': r'(spectrum|spectral|absorption|emission|raman|ir|uv)',
            'maps': r'(map|geographic|spatial|location|coordinates)',
            'timelapse': r'(time|sequence|series|animation|video)'
        }
        
        self.data_patterns = {
            'csv': r'\.csv$',
            'excel': r'\.(xlsx?|xls)$',
            'json': r'\.json$',
            'matlab': r'\.mat$',
            'python': r'\.(py|pkl|npy)$',
            'r_data': r'\.(r|rdata|rds)$',
            'text': r'\.(txt|md|log|dat)$'
        }
        
        self.experiment_patterns = {
            'control': r'(control|baseline|reference|standard)',
            'treatment': r'(treatment|experiment|test|sample|condition)',
            'replicate': r'(replicate|repeat|trial|run|batch)',
            'time_series': r'(time|hour|day|week|month|year)',
            'concentration': r'(conc|concentration|dose|molar|ppm|ppb)',
            'temperature': r'(temp|temperature|kelvin|celsius|fahrenheit)'
        }
    
    def analyze_dataset_context(self, dataset: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze dataset and return rich context for AI Assistant."""
        
        context = {
            'metadata': self._analyze_metadata(dataset),
            'file_structure': self._analyze_file_structure(dataset),
            'content_analysis': self._analyze_content(dataset),
            'scientific_context': self._analyze_scientific_context(dataset),
            'naming_patterns': self._analyze_naming_patterns(dataset),
            'summary': self._generate_summary(dataset)
        }
        
        return context
    
    def _analyze_metadata(self, dataset: Dict[str, Any]) -> Dict[str, Any]:
        """Extract and analyze Zenodo record metadata."""
        metadata = dataset.get('metadata', {})
        
        return {
            'title': metadata.get('title', 'Unknown'),
            'description': metadata.get('description', 'No description available'),
            'authors': [creator.get('name', '') for creator in metadata.get('creators', [])],
            'publication_date': metadata.get('publication_date', 'Unknown'),
            'keywords': metadata.get('keywords', []),
            'subjects': [subject.get('term', '') for subject in metadata.get('subjects', [])],
            'doi': metadata.get('doi', ''),
            'version': metadata.get('version', ''),
            'license': metadata.get('license', {}).get('id', 'Unknown'),
            'access_right': metadata.get('access_right', 'Unknown')
        }
    
    def _analyze_file_structure(self, dataset: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze file structure and organization."""
        files = dataset.get('files', [])
        
        # File type distribution
        file_types = Counter()
        file_sizes = []
        directories = set()
        
        for file in files:
            filename = file.get('key', '')
            size = file.get('size', 0)
            file_type = file.get('type', 'unknown')
            
            file_types[file_type] += 1
            file_sizes.append(size)
            
            # Extract directory structure
            if '/' in filename:
                dir_path = '/'.join(filename.split('/')[:-1])
                directories.add(dir_path)
        
        return {
            'total_files': len(files),
            'file_type_distribution': dict(file_types),
            'size_statistics': {
                'total_size_mb': sum(file_sizes) / (1024 * 1024),
                'average_size_mb': sum(file_sizes) / len(file_sizes) / (1024 * 1024) if file_sizes else 0,
                'largest_file_mb': max(file_sizes) / (1024 * 1024) if file_sizes else 0
            },
            'directory_structure': list(directories),
            'has_zip_files': any(f.get('from_zip', False) for f in files),
            'zip_extracted_files': len([f for f in files if f.get('from_zip', False)])
        }
    
    def _analyze_content(self, dataset: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze content types and patterns in files."""
        files = dataset.get('files', [])
        
        image_analysis = self._analyze_images(files)
        data_analysis = self._analyze_data_files(files)
        
        return {
            'images': image_analysis,
            'data_files': data_analysis,
            'content_summary': self._generate_content_summary(image_analysis, data_analysis)
        }
    
    def _analyze_images(self, files: List[Dict]) -> Dict[str, Any]:
        """Analyze image files and their characteristics."""
        image_files = [f for f in files if f.get('type') == 'images']
        
        image_types = Counter()
        image_patterns = Counter()
        
        for file in image_files:
            filename = file.get('key', '').lower()
            
            # Detect image type patterns
            for pattern_name, pattern in self.image_patterns.items():
                if re.search(pattern, filename, re.IGNORECASE):
                    image_patterns[pattern_name] += 1
            
            # File extension analysis
            ext = filename.split('.')[-1] if '.' in filename else 'unknown'
            image_types[ext] += 1
        
        return {
            'total_images': len(image_files),
            'image_formats': dict(image_types),
            'image_categories': dict(image_patterns),
            'sample_filenames': [f.get('key', '') for f in image_files[:5]]
        }
    
    def _analyze_data_files(self, files: List[Dict]) -> Dict[str, Any]:
        """Analyze data files and their characteristics."""
        data_files = [f for f in files if f.get('type') == 'data']
        
        data_types = Counter()
        experiment_patterns = Counter()
        
        for file in data_files:
            filename = file.get('key', '').lower()
            
            # Detect data file types
            for pattern_name, pattern in self.data_patterns.items():
                if re.search(pattern, filename, re.IGNORECASE):
                    data_types[pattern_name] += 1
            
            # Detect experiment patterns
            for pattern_name, pattern in self.experiment_patterns.items():
                if re.search(pattern, filename, re.IGNORECASE):
                    experiment_patterns[pattern_name] += 1
        
        return {
            'total_data_files': len(data_files),
            'data_formats': dict(data_types),
            'experiment_patterns': dict(experiment_patterns),
            'sample_filenames': [f.get('key', '') for f in data_files[:5]]
        }
    
    def _analyze_scientific_context(self, dataset: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze scientific context and research domain."""
        metadata = dataset.get('metadata', {})
        files = dataset.get('files', [])
        
        # Extract research domain from metadata
        subjects = [subject.get('term', '').lower() for subject in metadata.get('subjects', [])]
        keywords = [kw.lower() for kw in metadata.get('keywords', [])]
        description = metadata.get('description', '').lower()
        
        # Common research domains
        domains = {
            'biology': ['biology', 'biochemistry', 'microbiology', 'genetics', 'cell'],
            'chemistry': ['chemistry', 'chemical', 'molecular', 'synthesis', 'reaction'],
            'physics': ['physics', 'physical', 'mechanics', 'optics', 'quantum'],
            'materials': ['materials', 'polymer', 'nanomaterial', 'composite', 'crystal'],
            'engineering': ['engineering', 'mechanical', 'electrical', 'civil', 'chemical'],
            'environmental': ['environmental', 'ecology', 'climate', 'pollution', 'sustainability'],
            'medical': ['medical', 'clinical', 'health', 'disease', 'pharmaceutical']
        }
        
        detected_domains = []
        for domain, keywords_list in domains.items():
            if any(kw in ' '.join(subjects + keywords + [description]) for kw in keywords_list):
                detected_domains.append(domain)
        
        return {
            'research_domains': detected_domains,
            'subjects': subjects,
            'keywords': keywords,
            'experiment_type': self._infer_experiment_type(files),
            'data_complexity': self._assess_data_complexity(files)
        }
    
    def _analyze_naming_patterns(self, dataset: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze file naming patterns and conventions."""
        files = dataset.get('files', [])
        
        patterns = {
            'numbered_sequences': 0,
            'dated_files': 0,
            'experiment_codes': 0,
            'sample_identifiers': 0,
            'consistent_formatting': True
        }
        
        filenames = [f.get('key', '') for f in files]
        
        # Check for numbered sequences
        numbered_pattern = re.compile(r'\d{2,}')
        for filename in filenames:
            if numbered_pattern.search(filename):
                patterns['numbered_sequences'] += 1
        
        # Check for date patterns
        date_pattern = re.compile(r'\d{4}[-_]\d{2}[-_]\d{2}|\d{2}[-_]\d{2}[-_]\d{4}')
        for filename in filenames:
            if date_pattern.search(filename):
                patterns['dated_files'] += 1
        
        # Check for experiment codes (alphanumeric codes)
        code_pattern = re.compile(r'[A-Z]{2,}\d+|[A-Z]\d{2,}')
        for filename in filenames:
            if code_pattern.search(filename):
                patterns['experiment_codes'] += 1
        
        return patterns
    
    def _infer_experiment_type(self, files: List[Dict]) -> str:
        """Infer the type of experiment from file patterns."""
        filenames = [f.get('key', '').lower() for f in files]
        filename_text = ' '.join(filenames)
        
        experiment_indicators = {
            'time_series': ['time', 'hour', 'day', 'week', 'sequence', 'series'],
            'comparative': ['control', 'treatment', 'compare', 'vs', 'versus'],
            'calibration': ['calibration', 'cal', 'standard', 'reference'],
            'screening': ['screen', 'screening', 'library', 'array'],
            'characterization': ['characterization', 'analysis', 'measurement', 'test']
        }
        
        scores = {}
        for exp_type, indicators in experiment_indicators.items():
            scores[exp_type] = sum(1 for indicator in indicators if indicator in filename_text)
        
        if scores:
            return max(scores, key=scores.get)
        return 'unknown'
    
    def _assess_data_complexity(self, files: List[Dict]) -> str:
        """Assess the complexity of the dataset."""
        total_files = len(files)
        file_types = len(set(f.get('type', '') for f in files))
        has_zip = any(f.get('from_zip', False) for f in files)
        
        if total_files > 1000 or has_zip:
            return 'high'
        elif total_files > 100 or file_types > 5:
            return 'medium'
        else:
            return 'low'
    
    def _generate_content_summary(self, image_analysis: Dict, data_analysis: Dict) -> str:
        """Generate a human-readable content summary."""
        summary_parts = []
        
        if image_analysis['total_images'] > 0:
            summary_parts.append(f"{image_analysis['total_images']} image files")
            if image_analysis['image_categories']:
                categories = list(image_analysis['image_categories'].keys())
                summary_parts.append(f"including {', '.join(categories[:3])} images")
        
        if data_analysis['total_data_files'] > 0:
            summary_parts.append(f"{data_analysis['total_data_files']} data files")
            if data_analysis['data_formats']:
                formats = list(data_analysis['data_formats'].keys())
                summary_parts.append(f"in {', '.join(formats[:3])} format")
        
        return '; '.join(summary_parts) if summary_parts else "No content analysis available"
    
    def _generate_summary(self, dataset: Dict[str, Any]) -> str:
        """Generate a comprehensive summary for AI context."""
        metadata = self._analyze_metadata(dataset)
        file_structure = self._analyze_file_structure(dataset)
        content = self._analyze_content(dataset)
        scientific = self._analyze_scientific_context(dataset)
        
        summary = f"""
DATASET CONTEXT SUMMARY:

Title: {metadata['title']}
Authors: {', '.join(metadata['authors'][:3])}
Publication Date: {metadata['publication_date']}
Research Domains: {', '.join(scientific['research_domains'])}

Dataset Structure:
- Total Files: {file_structure['total_files']}
- File Types: {', '.join([f"{k}: {v}" for k, v in file_structure['file_type_distribution'].items()])}
- Total Size: {file_structure['size_statistics']['total_size_mb']:.1f} MB
- ZIP Files Extracted: {file_structure['zip_extracted_files']} files

Content Analysis:
- {content['content_summary']}
- Experiment Type: {scientific['experiment_type']}
- Data Complexity: {scientific['data_complexity']}

Key Features:
- Image Categories: {', '.join(content['images']['image_categories'].keys()) if content['images']['image_categories'] else 'None detected'}
- Data Formats: {', '.join(content['data_files']['data_formats'].keys()) if content['data_files']['data_formats'] else 'None detected'}
- Experiment Patterns: {', '.join(content['data_files']['experiment_patterns'].keys()) if content['data_files']['experiment_patterns'] else 'None detected'}
        """.strip()
        
        return summary
    
    def create_ai_prompt_context(self, dataset: Dict[str, Any], user_question: str) -> str:
        """Create a comprehensive context prompt for AI Assistant."""
        context = self.analyze_dataset_context(dataset)
        
        prompt = f"""
You are a scientific data assistant with expertise in analyzing research datasets. 

{context['summary']}

The user is asking: "{user_question}"

Based on the dataset context above, provide a helpful, accurate response. Consider:
1. The research domain and experimental context
2. The file structure and organization
3. The types of data and images available
4. Any patterns in naming conventions or experimental design

Be specific about what can be analyzed with the available data and suggest relevant analysis approaches.
        """.strip()
        
        return prompt 