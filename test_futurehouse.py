#!/usr/bin/env python3
"""
Test script for FutureHouse integration
"""

import os
from src.ai.futurehouse_client import FutureHouseIntegration

def test_futurehouse_integration():
    """Test FutureHouse integration functionality."""
    
    print("🧪 Testing FutureHouse Integration")
    print("=" * 50)
    
    # Test 1: Check if FutureHouse is available
    print("1. Checking FutureHouse availability...")
    if FutureHouseIntegration.FUTUREHOUSE_AVAILABLE:
        print("   ✅ FutureHouse client is available")
    else:
        print("   ❌ FutureHouse client not available")
        print("   💡 Install with: pip install futurehouse-client")
        return
    
    # Test 2: Initialize client
    print("\n2. Initializing FutureHouse client...")
    api_key = os.getenv('FUTUREHOUSE_API_KEY')
    if not api_key:
        print("   ⚠️  No FUTUREHOUSE_API_KEY found in environment")
        print("   💡 Set your API key: export FUTUREHOUSE_API_KEY='your_key_here'")
        return
    
    futurehouse = FutureHouseIntegration(api_key)
    if futurehouse.available:
        print("   ✅ FutureHouse client initialized successfully")
    else:
        print("   ❌ Failed to initialize FutureHouse client")
        return
    
    # Test 3: Get available jobs
    print("\n3. Getting available jobs...")
    jobs = futurehouse.get_available_jobs()
    print(f"   ✅ Found {len(jobs)} available jobs:")
    for job_id, job_info in jobs.items():
        print(f"      - {job_info['name']}: {job_info['description']}")
    
    # Test 4: Test query creation
    print("\n4. Testing query creation...")
    sample_context = {
        'metadata': {'title': 'Test Dataset'},
        'scientific_context': {
            'research_domains': ['chemistry', 'materials'],
            'experiment_type': 'characterization'
        },
        'content_analysis': {
            'content_summary': '100 microscopy images, 50 data files'
        }
    }
    
    test_question = "What are the latest synthesis methods?"
    query = futurehouse.create_research_query(sample_context, test_question, 'crow')
    print(f"   ✅ Generated query: {query[:100]}...")
    
    # Test 5: Test suggested queries
    print("\n5. Testing suggested queries...")
    suggestions = futurehouse.get_suggested_queries(sample_context)
    print(f"   ✅ Generated {len(suggestions)} suggestions:")
    for i, suggestion in enumerate(suggestions[:2]):
        print(f"      {i+1}. {suggestion['question']}")
    
    print("\n🎉 FutureHouse integration test completed!")
    print("\nTo use in the app:")
    print("1. Set your API key: export FUTUREHOUSE_API_KEY='your_key_here'")
    print("2. Run the Streamlit app: streamlit run app.py")
    print("3. Select 'FutureHouse Research' as AI provider")

if __name__ == "__main__":
    test_futurehouse_integration() 