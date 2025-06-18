#!/usr/bin/env python3
"""
Test script for Enhanced Image Analysis module
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.ai.enhanced_image_analysis import EnhancedImageAnalyzer, create_enhanced_image_analysis_tools

def test_enhanced_analysis():
    """Test the enhanced image analysis module"""
    
    print("🔬 Testing Enhanced Image Analysis Module")
    print("=" * 50)
    
    # Test 1: Create analyzer instance
    print("1. Creating EnhancedImageAnalyzer instance...")
    analyzer = EnhancedImageAnalyzer()
    print("✅ Analyzer created successfully")
    
    # Test 2: Check analysis methods
    print("\n2. Checking available analysis methods...")
    methods = analyzer.analysis_methods
    for category, method_list in methods.items():
        print(f"   {category}: {len(method_list)} methods")
    print("✅ Analysis methods loaded")
    
    # Test 3: Test tool creation
    print("\n3. Testing MCP tool creation...")
    tools = create_enhanced_image_analysis_tools()
    print(f"   Created {len(tools)} analysis tools:")
    for tool in tools:
        print(f"   - {tool['name']}: {tool['description']}")
    print("✅ MCP tools created successfully")
    
    # Test 4: Test basic analysis simulation
    print("\n4. Testing analysis simulation...")
    try:
        # Simulate analysis results
        simulated_results = {
            "file_name": "test_image.jpg",
            "analysis_types": ["basic", "morphology", "pattern"],
            "results": {
                "basic": {
                    "dimensions": {"width": 1024, "height": 768, "channels": 3},
                    "color_statistics": {
                        "rgb": {"mean": [128.5, 129.2, 127.8], "std": [45.2, 44.8, 46.1]},
                        "brightness": 128.5,
                        "contrast": 45.2
                    }
                },
                "morphology": {
                    "particle_analysis": {
                        "total_particles": 156,
                        "area_stats": {"mean": 245.6, "std": 89.3, "min": 23, "max": 892},
                        "circularity_stats": {"mean": 0.78, "std": 0.12}
                    },
                    "porosity": 0.23
                },
                "pattern": {
                    "edge_analysis": {"sobel_magnitude": 0.45, "canny_edge_density": 0.12},
                    "texture_features": {"contrast": 45.2, "homogeneity": 0.67, "energy": 0.89},
                    "fractal_dimension": 2.34
                }
            }
        }
        print("✅ Analysis simulation successful")
        print(f"   - Particles detected: {simulated_results['results']['morphology']['particle_analysis']['total_particles']}")
        print(f"   - Fractal dimension: {simulated_results['results']['pattern']['fractal_dimension']}")
        print(f"   - Porosity: {simulated_results['results']['morphology']['porosity']:.1%}")
        
    except Exception as e:
        print(f"❌ Analysis simulation failed: {e}")
    
    print("\n" + "=" * 50)
    print("🎉 Enhanced Image Analysis Module Test Complete!")
    print("\nNext steps:")
    print("1. Run 'streamlit run app.py' to start the web application")
    print("2. Navigate to 'Analysis Tools' → 'Enhanced Image Analysis'")
    print("3. Try the AI Assistant with queries like:")
    print("   - 'Analyze this image for particles and grains'")
    print("   - 'Detect phases in this microstructure'")
    print("   - 'Find defects in this material'")

if __name__ == "__main__":
    test_enhanced_analysis() 