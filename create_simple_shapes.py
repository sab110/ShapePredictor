#!/usr/bin/env python3
"""
Script to create 5 simple shapes for training the shape prediction model.
Each shape will have clear, distinct characteristics to help the model learn diversity.
"""
import os
import json
import numpy as np
from PIL import Image, ImageDraw
import cv2

def create_simple_shapes():
    """Create 5 simple shapes with their corresponding JSON metadata."""
    
    # Create directories
    os.makedirs("dataset/json", exist_ok=True)
    os.makedirs("dataset/images", exist_ok=True)
    for i in range(5):
        os.makedirs(f"dataset/masks/shape{i+1}", exist_ok=True)
    
    # Define the 5 shapes
    shapes = [
        {
            "name": "shape1",
            "description": "A red circle",
            "draw_func": lambda draw, size: draw.ellipse([size//4, size//4, 3*size//4, 3*size//4], fill="red")
        },
        {
            "name": "shape2", 
            "description": "A blue square",
            "draw_func": lambda draw, size: draw.rectangle([size//4, size//4, 3*size//4, 3*size//4], fill="blue")
        },
        {
            "name": "shape3",
            "description": "A green triangle",
            "draw_func": lambda draw, size: draw.polygon([
                (size//2, size//4),      # top
                (size//4, 3*size//4),    # bottom left
                (3*size//4, 3*size//4)   # bottom right
            ], fill="green")
        },
        {
            "name": "shape4",
            "description": "A yellow diamond",
            "draw_func": lambda draw, size: draw.polygon([
                (size//2, size//6),      # top
                (5*size//6, size//2),    # right
                (size//2, 5*size//6),    # bottom
                (size//6, size//2)       # left
            ], fill="yellow")
        },
        {
            "name": "shape5",
            "description": "A purple pentagon",
            "draw_func": lambda draw, size: draw.polygon([
                (size//2, size//5),                    # top
                (4*size//5, 2*size//5),               # top right
                (3*size//4, 4*size//5),               # bottom right
                (size//4, 4*size//5),                 # bottom left
                (size//5, 2*size//5)                  # top left
            ], fill="purple")
        }
    ]
    
    image_size = 256
    
    for i, shape_info in enumerate(shapes):
        shape_name = shape_info["name"]
        
        # Create background image
        img = Image.new('RGB', (image_size, image_size), color='white')
        draw = ImageDraw.Draw(img)
        
        # Draw the shape
        shape_info["draw_func"](draw, image_size)
        
        # Save the image
        img.save(f"dataset/images/{shape_name}.png")
        
        # Create mask for the shape
        # Convert to grayscale and threshold to create mask
        img_gray = img.convert('L')
        img_array = np.array(img_gray)
        
        # Create mask where non-white pixels are the shape
        mask = np.where(img_array < 250, 255, 0).astype(np.uint8)
        
        # Save the mask
        mask_img = Image.fromarray(mask)
        mask_img.save(f"dataset/masks/{shape_name}/mask_1.png")
        
        # Create background mask (entire image)
        bg_mask = np.ones((image_size, image_size), dtype=np.uint8) * 255
        bg_mask_img = Image.fromarray(bg_mask)
        bg_mask_img.save(f"dataset/masks/{shape_name}/mask_0.png")
        
        # Create JSON metadata
        json_data = {
            "global_style": {
                "visual_style": "simple geometric",
                "color_mode": "solid colors",
                "line_weight": "none",
                "lighting": "uniform",
                "camera": "orthographic",
                "mood": "neutral"
            },
            "scene": [
                {
                    "id": 0,
                    "description": f"A white canvas containing a {shape_info['description'].lower()}",
                    "mask_path": "mask_0.png",
                    "parent": -1
                },
                {
                    "id": 1,
                    "description": shape_info["description"],
                    "mask_path": "mask_1.png", 
                    "parent": 0
                }
            ]
        }
        
        # Save JSON
        with open(f"dataset/json/{shape_name}.json", 'w') as f:
            json.dump(json_data, f, indent=2)
        
        print(f"Created {shape_name}: {shape_info['description']}")

if __name__ == "__main__":
    create_simple_shapes()
    print("Successfully created 5 simple shapes for training!") 