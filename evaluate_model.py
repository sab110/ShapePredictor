#!/usr/bin/env python3
"""
Evaluation script for the shape prediction model.
Tests the model on the 5 simple shapes and visualizes predictions vs ground truth.
"""
import os
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import json
from boundedShapePredSOTA import PolygonPredictor, PolygonConfig, AugmentedDataset, collate_fn
from torch.utils.data import DataLoader

def load_trained_model(checkpoint_path, device):
    """Load the trained model from checkpoint."""
    print(f"Loading model from {checkpoint_path}...")
    
    # Create model
    cfg = PolygonConfig()
    model = PolygonPredictor(cfg=cfg).to(device)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Handle potential compiled model wrapper
    model_to_load = model
    if hasattr(model, "_orig_mod"):
        model_to_load = model._orig_mod
    
    missing_keys, unexpected_keys = model_to_load.load_state_dict(checkpoint["model_state_dict"], strict=False)
    
    if missing_keys:
        print(f"Missing keys: {missing_keys}")
    if unexpected_keys:
        print(f"Unexpected keys: {unexpected_keys}")
    
    model.eval()
    print(f"Model loaded successfully! Best loss was: {checkpoint.get('best_loss', 'N/A')}")
    return model

def render_bezier_curve(start_pt, curve_params, curve_type, steps=50):
    """Render a Bezier curve given start point, parameters, and type."""
    start_pt = np.array(start_pt)
    
    if curve_type == 0:  # Line segment
        end_pt = curve_params[4:6]
        return np.array([start_pt, end_pt])
    elif curve_type == 1:  # Quadratic Bezier
        ctrl_pt = curve_params[0:2]
        end_pt = curve_params[4:6]
        t = np.linspace(0, 1, steps)[:, None]
        curve = (1-t)**2 * start_pt + 2*(1-t)*t * ctrl_pt + t**2 * end_pt
        return curve
    else:  # Cubic Bezier
        ctrl1_pt = curve_params[0:2]
        ctrl2_pt = curve_params[2:4]
        end_pt = curve_params[4:6]
        t = np.linspace(0, 1, steps)[:, None]
        curve = ((1-t)**3 * start_pt + 
                3*(1-t)**2*t * ctrl1_pt + 
                3*(1-t)*t**2 * ctrl2_pt + 
                t**3 * end_pt)
        return curve

def calculate_curve_metrics(pred_curves, gt_curves, pred_types, gt_types):
    """Calculate evaluation metrics for curve prediction."""
    metrics = {}
    
    # Convert to numpy for easier processing
    pred_curves_np = pred_curves.detach().cpu().numpy()
    gt_curves_np = gt_curves.detach().cpu().numpy()
    pred_types_np = pred_types.detach().cpu().numpy()
    gt_types_np = gt_types.detach().cpu().numpy()
    
    # 1. Curve parameter L1 error (only for valid segments)
    valid_mask = (gt_curves_np != -1).any(axis=-1)  # Valid segments
    if valid_mask.any():
        pred_valid = pred_curves_np[valid_mask]
        gt_valid = gt_curves_np[valid_mask]
        curve_error = np.abs(pred_valid - gt_valid).mean()
        metrics['curve_l1_error'] = curve_error
    else:
        metrics['curve_l1_error'] = 0.0
    
    # 2. Type prediction accuracy
    valid_types = gt_types_np[valid_mask] if valid_mask.any() else []
    pred_types_valid = pred_types_np[valid_mask] if valid_mask.any() else []
    
    if len(valid_types) > 0:
        type_accuracy = (pred_types_valid == valid_types).mean()
        metrics['type_accuracy'] = type_accuracy
    else:
        metrics['type_accuracy'] = 0.0
    
    # 3. Endpoint prediction error (most important for shape)
    if valid_mask.any():
        pred_endpoints = pred_curves_np[valid_mask, 4:6]  # Last 2 coordinates are endpoints
        gt_endpoints = gt_curves_np[valid_mask, 4:6]
        endpoint_error = np.linalg.norm(pred_endpoints - gt_endpoints, axis=1).mean()
        metrics['endpoint_error'] = endpoint_error
    else:
        metrics['endpoint_error'] = 0.0
    
    return metrics

def evaluate_single_sample(model, sample, device, shape_name):
    """Evaluate model on a single sample and return results."""
    # Prepare input (add batch dimension)
    child_embs = sample['child_embs'].unsqueeze(0).to(device)
    parent_embs = sample['parent_embs'].unsqueeze(0).to(device)
    parent_bezier = sample['parent_bezier_segs'].unsqueeze(0).to(device)
    gt_curves = sample['gt_curves'].unsqueeze(0)
    
    # Get ground truth types
    gt_c1 = gt_curves[:, :, :2]
    gt_c2 = gt_curves[:, :, 2:4]
    gt_line_mask = (gt_c1 < 0).all(dim=-1) & (gt_c2 < 0).all(dim=-1)
    gt_quad_mask = (gt_c1 < 0).all(dim=-1) & ~(gt_c2 < 0).all(dim=-1)
    gt_cubic_mask = ~(gt_line_mask | gt_quad_mask)
    
    gt_types = torch.zeros_like(gt_line_mask, dtype=torch.long)
    gt_types[gt_quad_mask] = 1
    gt_types[gt_cubic_mask] = 2
    
    # Model prediction
    with torch.no_grad():
        outputs = model(child_embs, parent_embs, parent_bezier)
    
    pred_segments = outputs['segments'][0]  # Remove batch dimension
    pred_type_logits = outputs['type_logits'][0]
    pred_types = torch.argmax(pred_type_logits, dim=-1)
    pred_stops = outputs['stops'][0]
    
    # Calculate metrics
    metrics = calculate_curve_metrics(
        pred_segments, gt_curves[0], pred_types, gt_types[0]
    )
    
    return {
        'shape_name': shape_name,
        'pred_segments': pred_segments.cpu().numpy(),
        'gt_curves': gt_curves[0].cpu().numpy(),
        'pred_types': pred_types.cpu().numpy(),
        'gt_types': gt_types[0].cpu().numpy(),
        'pred_stops': pred_stops.cpu().item(),
        'metrics': metrics
    }

def visualize_predictions(results, output_dir="evaluation_results"):
    """Create visualizations comparing predictions vs ground truth."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a summary plot with all shapes
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    fig.suptitle('Shape Prediction Results: Ground Truth (Green) vs Prediction (Red)', fontsize=16)
    
    for i, result in enumerate(results):
        ax = axes[i]
        shape_name = result['shape_name']
        pred_segs = result['pred_segments']
        gt_segs = result['gt_curves']
        pred_types = result['pred_types']
        gt_types = result['gt_types']
        
        # Plot ground truth in green
        gt_endpoints = gt_segs[:, 4:6]
        gt_starts = np.roll(gt_endpoints, 1, axis=0)
        gt_starts[0] = gt_endpoints[-1] if len(gt_endpoints) > 0 else [0, 0]
        
        for j, (seg, start, gt_type) in enumerate(zip(gt_segs, gt_starts, gt_types)):
            if (seg == -1).all():  # Skip invalid segments
                continue
            try:
                curve = render_bezier_curve(start, seg, gt_type, steps=50)
                ax.plot(curve[:, 0], curve[:, 1], '-', color='green', linewidth=3, alpha=0.7)
            except:
                continue
        
        # Plot predictions in red
        pred_endpoints = pred_segs[:, 4:6]
        pred_starts = np.roll(pred_endpoints, 1, axis=0)
        pred_starts[0] = pred_endpoints[-1] if len(pred_endpoints) > 0 else [0, 0]
        
        for j, (seg, start, pred_type) in enumerate(zip(pred_segs, pred_starts, pred_types)):
            if j > result['pred_stops']:  # Respect stop prediction
                break
            try:
                curve = render_bezier_curve(start, seg, pred_type, steps=50)
                ax.plot(curve[:, 0], curve[:, 1], '--', color='red', linewidth=2)
            except:
                continue
        
        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(-0.1, 1.1)
        ax.set_aspect('equal')
        ax.set_title(f'{shape_name}\\nEndpoint Error: {result["metrics"]["endpoint_error"]:.4f}')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'all_shapes_comparison.png'), dpi=150, bbox_inches='tight')
    plt.show()
    
    # Create individual detailed plots
    for result in results:
        fig, ax = plt.subplots(figsize=(8, 8))
        shape_name = result['shape_name']
        
        # Same plotting logic but for individual shapes
        pred_segs = result['pred_segments']
        gt_segs = result['gt_curves']
        pred_types = result['pred_types']
        gt_types = result['gt_types']
        
        # Ground truth
        gt_endpoints = gt_segs[:, 4:6]
        gt_starts = np.roll(gt_endpoints, 1, axis=0)
        gt_starts[0] = gt_endpoints[-1] if len(gt_endpoints) > 0 else [0, 0]
        
        for j, (seg, start, gt_type) in enumerate(zip(gt_segs, gt_starts, gt_types)):
            if (seg == -1).all():
                continue
            try:
                curve = render_bezier_curve(start, seg, gt_type, steps=100)
                ax.plot(curve[:, 0], curve[:, 1], '-', color='green', linewidth=4, 
                       label='Ground Truth' if j == 0 else "", alpha=0.8)
            except:
                continue
        
        # Predictions
        pred_endpoints = pred_segs[:, 4:6]
        pred_starts = np.roll(pred_endpoints, 1, axis=0)
        pred_starts[0] = pred_endpoints[-1] if len(pred_endpoints) > 0 else [0, 0]
        
        for j, (seg, start, pred_type) in enumerate(zip(pred_segs, pred_starts, pred_types)):
            if j > result['pred_stops']:
                break
            try:
                curve = render_bezier_curve(start, seg, pred_type, steps=100)
                ax.plot(curve[:, 0], curve[:, 1], '--', color='red', linewidth=3,
                       label='Prediction' if j == 0 else "", alpha=0.8)
            except:
                continue
        
        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(-0.1, 1.1)
        ax.set_aspect('equal')
        ax.set_title(f'{shape_name} - Detailed Comparison\\n'
                    f'Endpoint Error: {result["metrics"]["endpoint_error"]:.4f}, '
                    f'Type Accuracy: {result["metrics"]["type_accuracy"]:.2f}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.savefig(os.path.join(output_dir, f'{shape_name}_detailed.png'), dpi=150, bbox_inches='tight')
        plt.close()

def check_diversity(results):
    """Check if the model produces diverse outputs for different inputs."""
    print("\\n" + "="*50)
    print("DIVERSITY ANALYSIS")
    print("="*50)
    
    # Compare predictions across different shapes
    all_pred_endpoints = []
    all_pred_types = []
    
    for result in results:
        pred_segs = result['pred_segments']
        pred_types = result['pred_types']
        
        # Get first few predicted endpoints and types
        valid_segs = pred_segs[:3]  # First 3 segments
        valid_types = pred_types[:3]
        
        all_pred_endpoints.append(valid_segs[:, 4:6])  # Endpoints
        all_pred_types.append(valid_types)
    
    # Calculate variance in predictions
    all_endpoints = np.concatenate(all_pred_endpoints, axis=0)
    endpoint_variance = np.var(all_endpoints, axis=0).mean()
    
    all_types = np.concatenate(all_pred_types, axis=0)
    unique_types = len(np.unique(all_types))
    
    print(f"Endpoint coordinate variance: {endpoint_variance:.6f}")
    print(f"Number of unique curve types predicted: {unique_types}/3")
    
    # Compare each shape's predictions
    print("\\nPer-shape first endpoint predictions:")
    for i, result in enumerate(results):
        shape_name = result['shape_name']
        first_endpoint = result['pred_segments'][0, 4:6]
        first_type = result['pred_types'][0]
        print(f"  {shape_name}: endpoint=({first_endpoint[0]:.3f}, {first_endpoint[1]:.3f}), type={first_type}")
    
    # Check if all predictions are identical (the original problem)
    first_pred = results[0]['pred_segments']
    all_identical = True
    for result in results[1:]:
        if not np.allclose(first_pred, result['pred_segments'], atol=1e-3):
            all_identical = False
            break
    
    if all_identical:
        print("\\n❌ WARNING: All predictions are nearly identical!")
        print("   The original problem may still exist.")
    else:
        print("\\n✅ SUCCESS: Predictions are diverse across different shapes!")
        print("   The model is generating different outputs for different inputs.")
    
    return {
        'endpoint_variance': endpoint_variance,
        'unique_types': unique_types,
        'all_identical': all_identical
    }

def main():
    print("="*60)
    print("SHAPE PREDICTION MODEL EVALUATION")
    print("="*60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load trained model
    checkpoint_path = "bezier_checkpoints/best_model.pth"
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        print("Make sure you've trained the model first!")
        return
    
    model = load_trained_model(checkpoint_path, device)
    
    # Load dataset
    print("\\nLoading dataset...")
    dataset = AugmentedDataset(root_dir="dataset", max_samples=5)
    print(f"Loaded {len(dataset)} samples")
    
    # Evaluate each shape
    print("\\nEvaluating model on each shape...")
    results = []
    shape_names = ['shape1', 'shape2', 'shape3', 'shape4', 'shape5']
    
    for i in range(len(dataset)):
        sample = dataset[i]
        shape_name = shape_names[i] if i < len(shape_names) else f'shape_{i+1}'
        
        result = evaluate_single_sample(model, sample, device, shape_name)
        results.append(result)
        
        metrics = result['metrics']
        print(f"  {shape_name}:")
        print(f"    Curve L1 Error: {metrics['curve_l1_error']:.6f}")
        print(f"    Type Accuracy: {metrics['type_accuracy']:.3f}")
        print(f"    Endpoint Error: {metrics['endpoint_error']:.6f}")
    
    # Calculate overall metrics
    print("\\n" + "="*50)
    print("OVERALL METRICS")
    print("="*50)
    
    avg_curve_error = np.mean([r['metrics']['curve_l1_error'] for r in results])
    avg_type_accuracy = np.mean([r['metrics']['type_accuracy'] for r in results])
    avg_endpoint_error = np.mean([r['metrics']['endpoint_error'] for r in results])
    
    print(f"Average Curve L1 Error: {avg_curve_error:.6f}")
    print(f"Average Type Accuracy: {avg_type_accuracy:.3f}")
    print(f"Average Endpoint Error: {avg_endpoint_error:.6f}")
    
    # Check diversity (the main issue we were trying to fix)
    diversity_stats = check_diversity(results)
    
    # Create visualizations
    print("\\n" + "="*50)
    print("GENERATING VISUALIZATIONS")
    print("="*50)
    
    visualize_predictions(results)
    print("Visualizations saved to 'evaluation_results/' directory")
    
    # Final assessment
    print("\\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    
    if diversity_stats['all_identical']:
        print("❌ ISSUE DETECTED: Model still produces identical outputs")
        print("   Consider training longer or adjusting model architecture")
    else:
        print("✅ SUCCESS: Model produces diverse outputs for different shapes")
        
        if avg_endpoint_error < 0.1:
            print("✅ EXCELLENT: Shape accuracy is very good")
        elif avg_endpoint_error < 0.2:
            print("✅ GOOD: Shape accuracy is acceptable")
        else:
            print("⚠️  FAIR: Shape accuracy could be improved")
        
        if avg_type_accuracy > 0.8:
            print("✅ EXCELLENT: Curve type classification is very good")
        elif avg_type_accuracy > 0.6:
            print("✅ GOOD: Curve type classification is acceptable")
        else:
            print("⚠️  FAIR: Curve type classification needs improvement")
    
    print("\\nEvaluation complete! Check the 'evaluation_results/' folder for visualizations.")

if __name__ == "__main__":
    main() 