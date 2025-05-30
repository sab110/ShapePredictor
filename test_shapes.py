#!/usr/bin/env python3
"""
Simple test script to verify shape generation and basic functionality
without requiring heavy ML dependencies.
"""
import os
import json
import numpy as np
from PIL import Image, ImageDraw
import cv2

def test_shape_generation():
    """Test that we can generate the 5 simple shapes."""
    print("Testing shape generation...")
    
    # Run the shape generation script
    import create_simple_shapes
    create_simple_shapes.create_simple_shapes()
    
    # Verify files were created
    expected_files = [
        'dataset/json/shape1.json',
        'dataset/json/shape2.json', 
        'dataset/json/shape3.json',
        'dataset/json/shape4.json',
        'dataset/json/shape5.json',
        'dataset/images/shape1.png',
        'dataset/images/shape2.png',
        'dataset/images/shape3.png', 
        'dataset/images/shape4.png',
        'dataset/images/shape5.png',
    ]
    
    all_exist = True
    for file_path in expected_files:
        if os.path.exists(file_path):
            print(f"✓ {file_path}")
        else:
            print(f"✗ {file_path} - MISSING")
            all_exist = False
    
    if all_exist:
        print("✅ All shape files generated successfully!")
    else:
        print("❌ Some files are missing")
        return False
    
    return True

def test_contour_module():
    """Test the contour processing without heavy ML dependencies."""
    print("\nTesting contour module...")
    
    try:
        import contour
        
        # Create a simple test mask (circle)
        mask = np.zeros((100, 100), dtype=np.uint8)
        cv2.circle(mask, (50, 50), 30, 255, -1)
        
        # Test mask_to_bezier_sequence
        vertices, bezier_curves = contour.mask_to_bezier_sequence(mask)
        
        print(f"✓ Contour extraction worked")
        print(f"  - Found {len(vertices)} vertices")
        print(f"  - Generated {len(bezier_curves)} Bezier curves")
        
        if len(bezier_curves) > 0:
            print(f"  - First curve: {bezier_curves[0]}")
        
        return True
        
    except Exception as e:
        print(f"❌ Contour module test failed: {e}")
        return False

def test_dataset_loading():
    """Test dataset loading logic without ML models."""
    print("\nTesting dataset structure...")
    
    try:
        # Check if we can load the JSON files
        shape_count = 0
        for i in range(1, 6):
            json_path = f"dataset/json/shape{i}.json"
            if os.path.exists(json_path):
                with open(json_path, 'r') as f:
                    data = json.load(f)
                    if 'scene' in data and len(data['scene']) >= 2:
                        shape_count += 1
                        print(f"✓ Shape {i}: {data['scene'][1]['description']}")
        
        if shape_count == 5:
            print(f"✅ All 5 shapes have valid JSON structure")
            return True
        else:
            print(f"❌ Only found {shape_count}/5 valid shapes")
            return False
            
    except Exception as e:
        print(f"❌ Dataset structure test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("=" * 50)
    print("SHAPE PREDICTOR - BASIC FUNCTIONALITY TEST")
    print("=" * 50)
    
    tests_passed = 0
    total_tests = 3
    
    if test_shape_generation():
        tests_passed += 1
    
    if test_contour_module():
        tests_passed += 1
        
    if test_dataset_loading():
        tests_passed += 1
    
    print("\n" + "=" * 50)
    print(f"RESULTS: {tests_passed}/{total_tests} tests passed")
    
    if tests_passed == total_tests:
        print("✅ Ready for ML training! Install PyTorch and run:")
        print("   pip install torch torchvision transformers")
        print("   python boundedShapePredSOTA.py")
    else:
        print("❌ Some basic functionality is not working")
    
    print("=" * 50)

if __name__ == "__main__":
    main() 