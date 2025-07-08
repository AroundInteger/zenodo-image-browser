"""
FutureHouse Client Integration
Provides access to scientific literature search and research tools.
"""

import os
from typing import Dict, Any, Optional, List
import json

# Try to import FutureHouse client
try:
    from futurehouse_client import FutureHouseClient, JobNames
    FUTUREHOUSE_AVAILABLE = True
except ImportError:
    FUTUREHOUSE_AVAILABLE = False
    print("FutureHouse client not available. Install with: pip install futurehouse-client")


class FutureHouseIntegration:
    """Integration with FutureHouse platform for scientific literature search."""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize FutureHouse client."""
        self.api_key = api_key or os.getenv('FUTUREHOUSE_API_KEY')
        self.client = None
        
        if FUTUREHOUSE_AVAILABLE and self.api_key:
            try:
                self.client = FutureHouseClient(api_key=self.api_key)
                self.available = True
            except Exception as e:
                print(f"Failed to initialize FutureHouse client: {e}")
                self.available = False
        else:
            self.available = False
    
    def get_available_jobs(self) -> Dict[str, str]:
        """Get available FutureHouse jobs with descriptions."""
        return {
            'crow': {
                'name': 'CROW (Fast Search)',
                'description': 'Ask questions of scientific data sources, get high-accuracy cited responses',
                'best_for': 'Quick literature searches, methodology questions, recent findings'
            },
            'falcon': {
                'name': 'FALCON (Deep Search)', 
                'description': 'Comprehensive research using multiple sources, detailed structured reports',
                'best_for': 'In-depth literature reviews, comprehensive analysis, detailed research'
            },
            'owl': {
                'name': 'OWL (Precedent Search)',
                'description': 'Find if anyone has done similar research or experiments',
                'best_for': 'Checking for similar work, avoiding duplication, finding precedents'
            },
            'phoenix': {
                'name': 'PHOENIX (Chemistry Tasks)',
                'description': 'Chemistry-specific tasks, synthesis planning, molecular design',
                'best_for': 'Chemical synthesis, molecular design, cheminformatics'
            }
        }
    
    def create_research_query(self, dataset_context: Dict[str, Any], user_question: str, job_type: str) -> str:
        """Create a research query based on dataset context and user question."""
        
        # Extract key information from dataset context
        title = dataset_context.get('metadata', {}).get('title', '')
        research_domains = dataset_context.get('scientific_context', {}).get('research_domains', [])
        experiment_type = dataset_context.get('scientific_context', {}).get('experiment_type', '')
        content_summary = dataset_context.get('content_analysis', {}).get('content_summary', '')
        
        # Build context-aware query
        context_info = []
        if title:
            context_info.append(f"Dataset: {title}")
        if research_domains:
            context_info.append(f"Research domains: {', '.join(research_domains)}")
        if experiment_type and experiment_type != 'unknown':
            context_info.append(f"Experiment type: {experiment_type}")
        if content_summary:
            context_info.append(f"Content: {content_summary}")
        
        context_str = " | ".join(context_info)
        
        # Create job-specific queries
        if job_type == 'owl':
            # Precedent search - look for similar work
            return f"Has anyone conducted similar research or experiments? Context: {context_str}. Specific question: {user_question}"
        
        elif job_type == 'crow':
            # Fast search - methodology and recent findings
            return f"What are the current methods and recent findings in this research area? Context: {context_str}. Question: {user_question}"
        
        elif job_type == 'falcon':
            # Deep search - comprehensive analysis
            return f"Provide a comprehensive analysis of the literature in this research area. Context: {context_str}. Focus: {user_question}"
        
        elif job_type == 'phoenix':
            # Chemistry-specific tasks
            return f"What are the chemical synthesis methods and molecular design approaches relevant to this research? Context: {context_str}. Question: {user_question}"
        
        else:
            # Generic query
            return f"Research context: {context_str}. User question: {user_question}"
    
    def search_literature(self, dataset_context: Dict[str, Any], user_question: str, job_type: str = 'crow') -> Dict[str, Any]:
        """Search scientific literature using FutureHouse."""
        
        if not self.available or not self.client:
            return {
                'success': False,
                'error': 'FutureHouse client not available. Please check API key and installation.',
                'suggestion': 'Install with: pip install futurehouse-client'
            }
        
        try:
            # Create context-aware query
            query = self.create_research_query(dataset_context, user_question, job_type)
            
            # Map job type to FutureHouse job
            job_mapping = {
                'crow': JobNames.CROW,
                'falcon': JobNames.FALCON,
                'owl': JobNames.OWL,
                'phoenix': JobNames.PHOENIX
            }
            
            job_name = job_mapping.get(job_type, JobNames.CROW)
            
            # Submit task
            task_data = {
                "name": job_name,
                "query": query,
            }
            
            # Run task and wait for completion
            task_response = self.client.run_tasks_until_done(task_data)
            
            return {
                'success': True,
                'answer': task_response.answer,
                'formatted_answer': getattr(task_response, 'formatted_answer', task_response.answer),
                'has_successful_answer': getattr(task_response, 'has_successful_answer', True),
                'job_type': job_type,
                'query': query
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f"FutureHouse search failed: {str(e)}",
                'suggestion': 'Check your API key and internet connection'
            }
    
    def get_suggested_queries(self, dataset_context: Dict[str, Any]) -> List[Dict[str, str]]:
        """Generate suggested research queries based on dataset context."""
        
        research_domains = dataset_context.get('scientific_context', {}).get('research_domains', [])
        experiment_type = dataset_context.get('scientific_context', {}).get('experiment_type', '')
        content_summary = dataset_context.get('content_analysis', {}).get('content_summary', '')
        
        suggestions = []
        
        # General research questions
        suggestions.append({
            'question': 'What are the current state-of-the-art methods in this research area?',
            'job_type': 'crow',
            'description': 'Find recent methodologies and techniques'
        })
        
        suggestions.append({
            'question': 'Has anyone conducted similar experiments or studies?',
            'job_type': 'owl', 
            'description': 'Check for similar work and precedents'
        })
        
        # Domain-specific suggestions
        if 'chemistry' in research_domains or 'materials' in research_domains:
            suggestions.append({
                'question': 'What synthesis methods are available for these materials?',
                'job_type': 'phoenix',
                'description': 'Chemistry-specific synthesis planning'
            })
        
        if experiment_type == 'time_series':
            suggestions.append({
                'question': 'What are the best practices for time series analysis in this field?',
                'job_type': 'crow',
                'description': 'Time series analysis methodologies'
            })
        
        if 'microscopy' in content_summary.lower():
            suggestions.append({
                'question': 'What are the latest microscopy techniques for this type of analysis?',
                'job_type': 'falcon',
                'description': 'Comprehensive microscopy review'
            })
        
        # Add a comprehensive literature review suggestion
        suggestions.append({
            'question': 'Provide a comprehensive literature review of this research area',
            'job_type': 'falcon',
            'description': 'Deep dive into the literature'
        })
        
        return suggestions 