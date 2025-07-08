"""
FutureHouse Client Integration
Provides access to scientific literature search and research tools.
"""

import os
from typing import Dict, Any, Optional, List
import json
import requests
import logging

logger = logging.getLogger(__name__)

class FutureHouseClient:
    """
    Custom FutureHouse client that works with Python 3.8+
    Implements the core functionality without requiring the official client package
    """
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.futurehouse.ai"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
    
    def _make_request(self, endpoint: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Make a request to the FutureHouse API"""
        try:
            url = f"{self.base_url}{endpoint}"
            response = requests.post(url, headers=self.headers, json=data, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"FutureHouse API request failed: {e}")
            raise Exception(f"Failed to connect to FutureHouse API: {e}")
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse FutureHouse API response: {e}")
            raise Exception("Invalid response from FutureHouse API")
    
    def search_research_papers(self, query: str, context: str = "") -> Dict[str, Any]:
        """
        Search for related research papers using CROW job
        """
        prompt = f"""
        Context: {context}
        
        Query: {query}
        
        Please search for and return relevant research papers, publications, and scientific literature related to this query. 
        Focus on recent and authoritative sources that would be valuable for understanding the research context.
        """
        
        data = {
            "job": "CROW",
            "prompt": prompt,
            "parameters": {
                "max_results": 10,
                "include_abstracts": True,
                "sort_by": "relevance"
            }
        }
        
        return self._make_request("/v1/jobs", data)
    
    def extract_insights(self, query: str, context: str = "") -> Dict[str, Any]:
        """
        Extract key insights using FALCON job
        """
        prompt = f"""
        Context: {context}
        
        Query: {query}
        
        Please analyze this query and extract key insights, findings, and important information. 
        Focus on identifying patterns, trends, and significant discoveries in the research area.
        """
        
        data = {
            "job": "FALCON",
            "prompt": prompt,
            "parameters": {
                "extraction_mode": "comprehensive",
                "include_citations": True
            }
        }
        
        return self._make_request("/v1/jobs", data)
    
    def find_similar_datasets(self, query: str, context: str = "") -> Dict[str, Any]:
        """
        Find similar datasets using OWL job
        """
        prompt = f"""
        Context: {context}
        
        Query: {query}
        
        Please find similar datasets, repositories, and data sources that are related to this query.
        Focus on datasets that contain similar types of data, methodologies, or research domains.
        """
        
        data = {
            "job": "OWL",
            "prompt": prompt,
            "parameters": {
                "max_results": 10,
                "include_metadata": True,
                "similarity_threshold": 0.7
            }
        }
        
        return self._make_request("/v1/jobs", data)
    
    def generate_research_summary(self, query: str, context: str = "") -> Dict[str, Any]:
        """
        Generate research summary using PHOENIX job
        """
        prompt = f"""
        Context: {context}
        
        Query: {query}
        
        Please generate a comprehensive research summary that synthesizes the current state of knowledge
        in this area. Include key findings, methodologies, gaps in research, and future directions.
        """
        
        data = {
            "job": "PHOENIX",
            "prompt": prompt,
            "parameters": {
                "summary_length": "comprehensive",
                "include_recommendations": True,
                "focus_areas": ["methodology", "findings", "applications", "future_work"]
            }
        }
        
        return self._make_request("/v1/jobs", data)
    
    def query(self, job_type: str, query: str, context: str = "") -> Dict[str, Any]:
        """
        Generic query method that routes to the appropriate job type
        """
        job_methods = {
            "CROW": self.search_research_papers,
            "FALCON": self.extract_insights,
            "OWL": self.find_similar_datasets,
            "PHOENIX": self.generate_research_summary
        }
        
        if job_type not in job_methods:
            raise ValueError(f"Unsupported job type: {job_type}. Supported types: {list(job_methods.keys())}")
        
        return job_methods[job_type](query, context)
    
    def test_connection(self) -> bool:
        """
        Test if the API key is valid and connection works
        """
        try:
            # Simple test query
            data = {
                "job": "CROW",
                "prompt": "Test connection",
                "parameters": {"max_results": 1}
            }
            self._make_request("/v1/jobs", data)
            return True
        except Exception as e:
            logger.error(f"FutureHouse connection test failed: {e}")
            return False

def create_futurehouse_client(api_key: str) -> Optional[FutureHouseClient]:
    """
    Factory function to create a FutureHouse client with error handling
    """
    if not api_key or api_key.strip() == "":
        logger.warning("No FutureHouse API key provided")
        return None
    
    try:
        client = FutureHouseClient(api_key)
        # Test the connection
        if client.test_connection():
            logger.info("FutureHouse client initialized successfully")
            return client
        else:
            logger.error("FutureHouse API key validation failed")
            return None
    except Exception as e:
        logger.error(f"Failed to initialize FutureHouse client: {e}")
        return None 