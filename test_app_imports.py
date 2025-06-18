#!/usr/bin/env python3
"""
Test script to verify all app imports work correctly
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test all the imports used in the app"""
    
    print("🔍 Testing App Imports")
    print("=" * 40)
    
    # Test 1: Basic imports
    print("1. Testing basic imports...")
    try:
        import streamlit as st
        print("✅ streamlit imported successfully")
    except ImportError as e:
        print(f"❌ streamlit import failed: {e}")
        return False
    
    # Test 2: Image utils
    print("\n2. Testing image utils...")
    try:
        from src.utils.image_utils import (
            is_valid_image_url, 
            get_image_dimensions, 
            is_supported_image_format,
            get_file_type,
            format_file_size
        )
        print("✅ image_utils imported successfully")
    except ImportError as e:
        print(f"❌ image_utils import failed: {e}")
        return False
    
    # Test 3: Enhanced image analysis
    print("\n3. Testing enhanced image analysis...")
    try:
        from src.ai.enhanced_image_analysis import EnhancedImageAnalyzer, create_enhanced_image_analysis_tools
        print("✅ enhanced_image_analysis imported successfully")
    except ImportError as e:
        print(f"❌ enhanced_image_analysis import failed: {e}")
        return False
    
    # Test 4: MCP server
    print("\n4. Testing MCP server...")
    try:
        from src.ai.mcp_server import create_ai_chat_interface
        print("✅ mcp_server imported successfully")
    except ImportError as e:
        print(f"❌ mcp_server import failed: {e}")
        return False
    
    # Test 5: Analysis interfaces
    print("\n5. Testing analysis interfaces...")
    try:
        from src.analysis.enhanced_image_interface import create_enhanced_image_analysis_interface
        print("✅ enhanced_image_interface imported successfully")
    except ImportError as e:
        print(f"❌ enhanced_image_interface import failed: {e}")
        return False
    
    # Test 6: Test utility functions
    print("\n6. Testing utility functions...")
    try:
        # Test get_file_type
        file_type = get_file_type("test_image.jpg")
        print(f"✅ get_file_type works: 'test_image.jpg' -> {file_type}")
        
        # Test format_file_size
        formatted_size = format_file_size(1024000)
        print(f"✅ format_file_size works: 1024000 -> {formatted_size}")
        
    except Exception as e:
        print(f"❌ utility function test failed: {e}")
        return False
    
    print("\n" + "=" * 40)
    print("🎉 All imports and functions working correctly!")
    print("\nThe app should now run successfully with:")
    print("streamlit run app.py")
    
    return True

if __name__ == "__main__":
    success = test_imports()
    if success:
        print("\n✅ Ready to launch the application!")
    else:
        print("\n❌ Some imports failed. Please check the errors above.") 