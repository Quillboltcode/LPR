"""
Test script for the crop_and_warp function in helper.py
"""
import cv2
import numpy as np
import torch
import json
import os
import sys

# Add parent directory to path to import helper
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from helper import crop_and_warp


def test_crop_and_warp():
    """
    Test the crop_and_warp function with real data from the dataset.
    """
    print("=" * 60)
    print("Testing crop_and_warp function")
    print("=" * 60)
    
    # Test with track_00001
    track_path = "data/train/Scenario-A/Brazilian/track_00001"
    
    # Check if the path exists
    if not os.path.exists(track_path):
        print(f"Error: Track path {track_path} does not exist!")
        return False
    
    # Load annotations
    annotations_file = os.path.join(track_path, 'annotations.json')
    with open(annotations_file, 'r') as f:
        annotations = json.load(f)
    
    print(f"\nLoaded annotations from {annotations_file}")
    print(f"Plate text: {annotations['plate_text']}")
    print(f"Plate layout: {annotations['plate_layout']}")
    
    # Test with both HR and LR images
    test_cases = [
        ('hr-001.png', 'High Resolution'),
        ('lr-001.png', 'Low Resolution')
    ]
    
    for img_file, description in test_cases:
        print(f"\n{'=' * 60}")
        print(f"Testing with {description} image: {img_file}")
        print(f"{'=' * 60}")
        
        # Load image
        img_path = os.path.join(track_path, img_file)
        image = cv2.imread(img_path)
        
        if image is None:
            print(f"Error: Could not load image {img_path}")
            continue
        
        print(f"Original image shape: {image.shape}")
        print(f"Original image dtype: {image.dtype}")
        
        # Get corner coordinates for this image
        if img_file not in annotations['corners']:
            print(f"Error: No corner coordinates found for {img_file}")
            continue
        
        json_data = annotations['corners'][img_file]
        print(f"\nCorner coordinates:")
        for corner, coords in json_data.items():
            print(f"  {corner}: {coords}")
        
        # Test with default target size
        print(f"\n{'-' * 60}")
        print("Test 1: Default target size (64, 320)")
        print(f"{'-' * 60}")
        
        try:
            result_tensor = crop_and_warp(image, json_data)
            
            print(f"✓ Function executed successfully")
            print(f"  Output tensor shape: {result_tensor.shape}")
            print(f"  Output tensor dtype: {result_tensor.dtype}")
            print(f"  Output tensor min value: {result_tensor.min().item():.4f}")
            print(f"  Output tensor max value: {result_tensor.max().item():.4f}")
            
            # Validate output
            expected_shape = (1, 3, 64, 320)
            if result_tensor.shape == expected_shape:
                print(f"  ✓ Output shape matches expected: {expected_shape}")
            else:
                print(f"  ✗ Output shape mismatch! Expected {expected_shape}, got {result_tensor.shape}")
            
            if result_tensor.dtype == torch.float32:
                print(f"  ✓ Output dtype is float32")
            else:
                print(f"  ✗ Output dtype is {result_tensor.dtype}, expected float32")
            
            if 0.0 <= result_tensor.min() <= 1.0 and 0.0 <= result_tensor.max() <= 1.0:
                print(f"  ✓ Values are normalized to [0, 1]")
            else:
                print(f"  ✗ Values are not properly normalized!")
            
        except Exception as e:
            print(f"✗ Error during execution: {e}")
            import traceback
            traceback.print_exc()
            continue
        
        # Test with custom target size
        print(f"\n{'-' * 60}")
        print("Test 2: Custom target size (32, 160)")
        print(f"{'-' * 60}")
        
        try:
            result_tensor = crop_and_warp(image, json_data, target_size=(32, 160))
            
            print(f"✓ Function executed successfully")
            print(f"  Output tensor shape: {result_tensor.shape}")
            
            expected_shape = (1, 3, 32, 160)
            if result_tensor.shape == expected_shape:
                print(f"  ✓ Output shape matches expected: {expected_shape}")
            else:
                print(f"  ✗ Output shape mismatch! Expected {expected_shape}, got {result_tensor.shape}")
            
        except Exception as e:
            print(f"✗ Error during execution: {e}")
            import traceback
            traceback.print_exc()
        
        # Save visualization
        print(f"\n{'-' * 60}")
        print("Saving visualization...")
        print(f"{'-' * 60}")
        
        try:
            # Convert tensor back to numpy for visualization
            tensor_np = result_tensor.squeeze(0).permute(1, 2, 0).numpy()
            tensor_np = (tensor_np * 255).astype(np.uint8)
            tensor_np = cv2.cvtColor(tensor_np, cv2.COLOR_RGB2BGR)
            
            # Save the result
            output_path = f"tests/test_output_{img_file.replace('.png', '')}_warped.png"
            cv2.imwrite(output_path, tensor_np)
            print(f"✓ Saved warped image to {output_path}")
            
            # Also save original with corners marked
            original_with_corners = image.copy()
            corners = [
                json_data['top-left'],
                json_data['top-right'],
                json_data['bottom-right'],
                json_data['bottom-left'],
                json_data['top-left']  # Close the polygon
            ]
            corners = np.array(corners, dtype=np.int32)
            cv2.polylines(original_with_corners, [corners], True, (0, 255, 0), 2)
            
            original_path = f"tests/test_output_{img_file.replace('.png', '')}_original.png"
            cv2.imwrite(original_path, original_with_corners)
            print(f"✓ Saved original with corners to {original_path}")
            
        except Exception as e:
            print(f"✗ Error saving visualization: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'=' * 60}")
    print("Testing complete!")
    print(f"{'=' * 60}")
    return True


def test_edge_cases():
    """
    Test edge cases and error handling.
    """
    print(f"\n{'=' * 60}")
    print("Testing Edge Cases")
    print(f"{'=' * 60}")
    
    # Create a simple test image
    test_image = np.ones((100, 200, 3), dtype=np.uint8) * 255
    
    # Test case 1: Perfect rectangle
    print(f"\nTest 1: Perfect rectangle corners")
    json_data = {
        'top-left': [10, 10],
        'top-right': [90, 10],
        'bottom-right': [90, 40],
        'bottom-left': [10, 40]
    }
    
    try:
        result = crop_and_warp(test_image, json_data)
        print(f"✓ Perfect rectangle: Output shape {result.shape}")
    except Exception as e:
        print(f"✗ Error: {e}")
    
    # Test case 2: Skewed quadrilateral
    print(f"\nTest 2: Skewed quadrilateral")
    json_data = {
        'top-left': [10, 10],
        'top-right': [95, 15],
        'bottom-right': [85, 45],
        'bottom-left': [15, 40]
    }
    
    try:
        result = crop_and_warp(test_image, json_data)
        print(f"✓ Skewed quadrilateral: Output shape {result.shape}")
    except Exception as e:
        print(f"✗ Error: {e}")
    
    # Test case 3: Very small plate
    print(f"\nTest 3: Very small plate (5x5 pixels)")
    json_data = {
        'top-left': [50, 50],
        'top-right': [54, 50],
        'bottom-right': [54, 54],
        'bottom-left': [50, 54]
    }
    
    try:
        result = crop_and_warp(test_image, json_data)
        print(f"✓ Small plate: Output shape {result.shape}")
    except Exception as e:
        print(f"✗ Error: {e}")
    
    # Test case 4: Large plate
    print(f"\nTest 4: Large plate (180x80 pixels)")
    json_data = {
        'top-left': [10, 10],
        'top-right': [189, 10],
        'bottom-right': [189, 89],
        'bottom-left': [10, 89]
    }
    
    try:
        result = crop_and_warp(test_image, json_data)
        print(f"✓ Large plate: Output shape {result.shape}")
    except Exception as e:
        print(f"✗ Error: {e}")


if __name__ == "__main__":
    # Create tests directory if it doesn't exist
    os.makedirs("tests", exist_ok=True)
    
    # Run main test
    test_crop_and_warp()
    
    # Run edge case tests
    test_edge_cases()
    
    print(f"\n{'=' * 60}")
    print("All tests completed!")
    print(f"{'=' * 60}")
