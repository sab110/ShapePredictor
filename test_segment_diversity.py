#!/usr/bin/env python3
"""
Quick test to verify that the fixed ShapePredictor produces different segments within each shape.
"""
import torch
import numpy as np
from boundedShapePredSOTA import PolygonPredictor, PolygonConfig, AugmentedDataset

def test_segment_diversity():
    """Test if the model produces different segments within the same shape"""
    print("=== Testing Segment Diversity ===")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create model with fixed architecture
    cfg = PolygonConfig(
        d_model=256,
        n_head=8,
        num_dec_layers_seq=6,
        dim_feedforward=1024,
        max_segments=30,
        dropout=0.1
    )
    
    model = PolygonPredictor(cfg=cfg).to(device)
    model.eval()
    
    # Create dummy inputs
    batch_size = 2
    child_embs = torch.randn(batch_size, 512, 768, device=device)  # T5 embeddings
    parent_embs = torch.randn(batch_size, 512, 768, device=device)
    parent_bezier = torch.randn(batch_size, 30, 6, device=device) * 0.5 + 0.5  # [0,1] range
    
    # Forward pass
    with torch.no_grad():
        outputs = model(child_embs, parent_embs, parent_bezier)
    
    pred_segments = outputs['segments']
    pred_types = outputs['type_logits'].argmax(dim=-1)
    
    print(f"Output shape: {pred_segments.shape}")
    
    # Test each sample in the batch
    for sample_idx in range(batch_size):
        print(f"\n--- Sample {sample_idx} ---")
        sample_segments = pred_segments[sample_idx]
        sample_types = pred_types[sample_idx]
        
        # Check first 5 segments
        print("First 5 predicted segments:")
        for i in range(5):
            seg = sample_segments[i].cpu().numpy()
            seg_type = sample_types[i].item()
            print(f"Segment {i}: {seg} (type: {seg_type})")
        
        # Test for diversity
        first_5_segments = sample_segments[:5]
        
        # Check if all segments are identical
        first_segment = first_5_segments[0]
        all_identical = True
        for i in range(1, 5):
            if not torch.allclose(first_segment, first_5_segments[i], atol=1e-4):
                all_identical = False
                break
        
        if all_identical:
            print("❌ ISSUE: All segments are identical!")
            print(f"First segment: {first_segment.cpu().numpy()}")
        else:
            print("✅ SUCCESS: Segments are different!")
            
            # Calculate variance as diversity measure
            coords = first_5_segments.cpu().numpy()
            variance = np.var(coords.flatten())
            print(f"Coordinate variance: {variance:.6f}")
            
            # Show differences between consecutive segments
            for i in range(4):
                diff = torch.norm(first_5_segments[i+1] - first_5_segments[i]).item()
                print(f"Difference between segment {i} and {i+1}: {diff:.4f}")

if __name__ == "__main__":
    test_segment_diversity() 