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
import logging
from datetime import datetime
from boundedShapePredSOTA import PolygonPredictor, PolygonConfig, AugmentedDataset, collate_fn
from torch.utils.data import DataLoader

def setup_logging(output_dir):
    """Setup logging configuration"""
    log_dir = os.path.join(output_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'evaluation_{timestamp}.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def load_trained_model(checkpoint_path, device):
    """Load the trained model from checkpoint."""
    logger = logging.getLogger(__name__)
    logger.info(f"Loading model from {checkpoint_path}...")
    
    # Create model with correct dimensions to match checkpoint
    cfg = PolygonConfig()
    cfg.d_model = 256  # Update model dimension to match checkpoint (was previously 128)
    cfg.dim_feedforward = 1024  # Update feedforward dimension to match checkpoint (was previously 512)
    model = PolygonPredictor(cfg=cfg).to(device)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Handle potential compiled model wrapper
    model_to_load = model
    if hasattr(model, "_orig_mod"):
        model_to_load = model._orig_mod
    
    missing_keys, unexpected_keys = model_to_load.load_state_dict(checkpoint["model_state_dict"], strict=False)
    
    if missing_keys:
        logger.warning(f"Missing keys: {missing_keys}")
    if unexpected_keys:
        logger.warning(f"Unexpected keys: {unexpected_keys}")
    
    model.eval()
    logger.info(f"Model loaded successfully! Best loss was: {checkpoint.get('best_loss', 'N/A')}")
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
    logger = logging.getLogger(__name__)
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
        logger.debug(f"Curve L1 error: {curve_error:.6f}")
    else:
        metrics['curve_l1_error'] = 0.0
        logger.warning("No valid segments found for curve error calculation")
    
    # 2. Type prediction accuracy
    valid_types = gt_types_np[valid_mask] if valid_mask.any() else []
    pred_types_valid = pred_types_np[valid_mask] if valid_mask.any() else []
    
    if len(valid_types) > 0:
        type_accuracy = (pred_types_valid == valid_types).mean()
        metrics['type_accuracy'] = type_accuracy
        logger.debug(f"Type accuracy: {type_accuracy:.3f}")
    else:
        metrics['type_accuracy'] = 0.0
        logger.warning("No valid types found for accuracy calculation")
    
    # 3. Endpoint prediction error (most important for shape)
    if valid_mask.any():
        pred_endpoints = pred_curves_np[valid_mask, 4:6]  # Last 2 coordinates are endpoints
        gt_endpoints = gt_curves_np[valid_mask, 4:6]
        endpoint_error = np.linalg.norm(pred_endpoints - gt_endpoints, axis=1).mean()
        metrics['endpoint_error'] = endpoint_error
        logger.debug(f"Endpoint error: {endpoint_error:.6f}")
    else:
        metrics['endpoint_error'] = 0.0
        logger.warning("No valid endpoints found for error calculation")
    
    return metrics

def evaluate_single_sample(model, sample, device, shape_name):
    """Evaluate model on a single sample and return results."""
    logger = logging.getLogger(__name__)
    logger.info(f"Evaluating shape: {shape_name}")
    
    # Prepare input (add batch dimension)
    child_embs = sample['child_embs'].unsqueeze(0).to(device)
    parent_embs = sample['parent_embs'].unsqueeze(0).to(device)
    parent_bezier = sample['parent_bezier_segs'].unsqueeze(0).to(device)
    gt_curves = sample['gt_curves']  # This is [T_gt, 6] for individual samples
    
    # Get ground truth types - fix tensor indexing for 2D tensor
    gt_c1 = gt_curves[:, :2]  # [T_gt, 2] instead of [:, :, :2]
    gt_c2 = gt_curves[:, 2:4]  # [T_gt, 2] instead of [:, :, 2:4]
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
    
    # Fix: Use the correct key names from the model output
    pred_stops = outputs['stop_index'][0]  # Changed from 'stops' to 'stop_index'
    pred_stop_scores = outputs.get('stop_scores', [None])[0] if 'stop_scores' in outputs else None
    
    # === ENHANCED DEBUGGING OUTPUT ===
    logger.info(f"\n=== DETAILED DEBUG INFO FOR {shape_name.upper()} ===")
    
    # 1. STOP PREDICTIONS
    logger.info(f"STOP PREDICTIONS:")
    logger.info(f"  Stop Index: {pred_stops.item()}")
    if pred_stop_scores is not None:
        stop_scores_np = pred_stop_scores.cpu().numpy()
        logger.info(f"  Stop Scores (first 10): {stop_scores_np[:10]}")
        logger.info(f"  Stop Probabilities (first 10): {stop_scores_np[:10]}")
        logger.info(f"  Max Stop Score: {stop_scores_np.max():.4f} at position {stop_scores_np.argmax()}")
    
    # 2. PREDICTED SEGMENTS (ALL VALID ONES)
    valid_segments = min(pred_stops.item() + 5, pred_segments.shape[0])  # Show a few extra
    logger.info(f"\nPREDICTED SEGMENTS (showing first {valid_segments}):")
    for i in range(valid_segments):
        seg = pred_segments[i].cpu().numpy()
        seg_type = pred_types[i].cpu().numpy().item()  # Extract scalar value
        type_names = {0: "Line", 1: "Quad", 2: "Cubic"}
        
        # Check if segment is valid (not all -1)
        is_valid = not (seg == -1).all()
        validity = "VALID" if is_valid else "MASKED"
        
        logger.info(f"  Segment {i:2d}: [{seg[0]:7.3f}, {seg[1]:7.3f}, {seg[2]:7.3f}, {seg[3]:7.3f}, {seg[4]:7.3f}, {seg[5]:7.3f}] | Type: {type_names[seg_type]} ({seg_type}) | {validity}")
        
        if i == pred_stops.item():
            logger.info(f"  ^^^^^^^^^^ STOP INDEX {pred_stops.item()} ^^^^^^^^^^")
    
    # 3. TYPE PREDICTIONS ANALYSIS
    logger.info(f"\nTYPE PREDICTIONS ANALYSIS:")
    type_counts = {0: 0, 1: 0, 2: 0}
    valid_type_counts = {0: 0, 1: 0, 2: 0}
    
    for i in range(valid_segments):
        seg = pred_segments[i].cpu().numpy()
        seg_type = pred_types[i].cpu().numpy().item()  # Extract scalar value
        type_counts[seg_type] += 1
        
        if not (seg == -1).all():  # Only count valid segments
            valid_type_counts[seg_type] += 1
    
    logger.info(f"  Total Type Distribution: Line={type_counts[0]}, Quad={type_counts[1]}, Cubic={type_counts[2]}")
    logger.info(f"  Valid Type Distribution: Line={valid_type_counts[0]}, Quad={valid_type_counts[1]}, Cubic={valid_type_counts[2]}")
    
    # 4. TYPE PREDICTION CONFIDENCE
    logger.info(f"\nTYPE PREDICTION CONFIDENCE (first {min(10, valid_segments)}):")
    for i in range(min(10, valid_segments)):
        type_probs = torch.softmax(pred_type_logits[i], dim=0).cpu().numpy()
        pred_type = pred_types[i].cpu().numpy().item()  # Extract scalar value
        confidence = type_probs[pred_type]
        logger.info(f"  Segment {i}: Predicted={pred_type} | Confidence={confidence:.3f} | Probs=[{type_probs[0]:.3f}, {type_probs[1]:.3f}, {type_probs[2]:.3f}]")
    
    # 5. GROUND TRUTH COMPARISON
    logger.info(f"\nGROUND TRUTH COMPARISON:")
    gt_type_counts = {0: 0, 1: 0, 2: 0}
    valid_gt_segments = (gt_curves != -1).any(dim=-1).sum().item()
    
    for i in range(min(gt_types.shape[0], 10)):
        gt_seg = gt_curves[i].cpu().numpy()
        gt_type = gt_types[i].cpu().numpy().item()  # Extract scalar value
        is_valid = not (gt_seg == -1).all()
        
        if is_valid:
            gt_type_counts[gt_type] += 1
            
        logger.info(f"  GT Seg {i:2d}: [{gt_seg[0]:7.3f}, {gt_seg[1]:7.3f}, {gt_seg[2]:7.3f}, {gt_seg[3]:7.3f}, {gt_seg[4]:7.3f}, {gt_seg[5]:7.3f}] | Type: {gt_type}")
    
    logger.info(f"  GT Type Distribution: Line={gt_type_counts[0]}, Quad={gt_type_counts[1]}, Cubic={gt_type_counts[2]}")
    logger.info(f"  GT Valid Segments: {valid_gt_segments}")
    
    logger.info(f"=== END DEBUG INFO FOR {shape_name.upper()} ===\n")
    
    # === EXISTING CHECKS ===
    # DEBUG: Check if segments are still identical
    logger.info(f"DEBUG - {shape_name} first 5 predicted segments:")
    for i in range(min(5, pred_segments.shape[0])):
        seg = pred_segments[i].cpu().numpy()
        logger.info(f"  Segment {i}: [{seg[0]:.3f}, {seg[1]:.3f}, {seg[2]:.3f}, {seg[3]:.3f}, {seg[4]:.3f}, {seg[5]:.3f}]")
    
    # Check if all segments are identical
    first_seg = pred_segments[0]
    identical_count = 0
    for i in range(1, min(10, pred_segments.shape[0])):  # Check first 10 segments
        if torch.allclose(first_seg, pred_segments[i], atol=1e-4):
            identical_count += 1
    
    if identical_count > 0:
        logger.warning(f"  {shape_name}: {identical_count}/9 segments are identical to the first!")
    else:
        logger.info(f"  {shape_name}: All segments are unique - PROBLEM FIXED!")
    
    # Calculate metrics - pass gt_curves directly since it's already 2D
    metrics = calculate_curve_metrics(
        pred_segments, gt_curves, pred_types, gt_types
    )
    
    logger.info(f"Shape {shape_name} metrics:")
    logger.info(f"  Curve L1 Error: {metrics['curve_l1_error']:.6f}")
    logger.info(f"  Type Accuracy: {metrics['type_accuracy']:.3f}")
    logger.info(f"  Endpoint Error: {metrics['endpoint_error']:.6f}")
    
    return {
        'shape_name': shape_name,
        'pred_segments': pred_segments.cpu().numpy(),
        'gt_curves': gt_curves.cpu().numpy(),  # gt_curves is already [T_gt, 6]
        'pred_types': pred_types.cpu().numpy(),
        'gt_types': gt_types.cpu().numpy(),
        'pred_stops': pred_stops.cpu().numpy(),
        'pred_stop_scores': pred_stop_scores.cpu().numpy() if pred_stop_scores is not None else None,
        'metrics': metrics
    }

def visualize_predictions(results, output_dir="evaluation_results"):
    """Create visualizations comparing predictions vs ground truth."""
    logger = logging.getLogger(__name__)
    logger.info("Generating visualizations...")
    
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
            except Exception as e:
                logger.warning(f"Error rendering ground truth curve for {shape_name}, segment {j}: {e}")
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
            except Exception as e:
                logger.warning(f"Error rendering prediction curve for {shape_name}, segment {j}: {e}")
                continue
        
        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(-0.1, 1.1)
        ax.set_aspect('equal')
        ax.set_title(f'{shape_name}\nEndpoint Error: {result["metrics"]["endpoint_error"]:.4f}')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    summary_plot_path = os.path.join(output_dir, 'all_shapes_comparison.png')
    plt.savefig(summary_plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved summary plot to {summary_plot_path}")
    
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
            except Exception as e:
                logger.warning(f"Error rendering ground truth curve for {shape_name}, segment {j}: {e}")
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
            except Exception as e:
                logger.warning(f"Error rendering prediction curve for {shape_name}, segment {j}: {e}")
                continue
        
        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(-0.1, 1.1)
        ax.set_aspect('equal')
        ax.set_title(f'{shape_name} - Detailed Comparison\n'
                    f'Endpoint Error: {result["metrics"]["endpoint_error"]:.4f}, '
                    f'Type Accuracy: {result["metrics"]["type_accuracy"]:.2f}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        detailed_plot_path = os.path.join(output_dir, f'{shape_name}_detailed.png')
        plt.savefig(detailed_plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved detailed plot for {shape_name} to {detailed_plot_path}")

def check_diversity(results):
    """Check if the model produces diverse outputs for different inputs."""
    logger = logging.getLogger(__name__)
    logger.info("\n" + "="*50)
    logger.info("DIVERSITY ANALYSIS")
    logger.info("="*50)
    
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
    
    logger.info(f"Endpoint coordinate variance: {endpoint_variance:.6f}")
    logger.info(f"Number of unique curve types predicted: {unique_types}/3")
    
    # Compare each shape's predictions
    logger.info("\nPer-shape first endpoint predictions:")
    for i, result in enumerate(results):
        shape_name = result['shape_name']
        first_endpoint = result['pred_segments'][0, 4:6]
        first_type = result['pred_types'][0]
        logger.info(f"  {shape_name}: endpoint=({first_endpoint[0]:.3f}, {first_endpoint[1]:.3f}), type={first_type}")
    
    # Check if all predictions are identical (the original problem)
    first_pred = results[0]['pred_segments']
    all_identical = True
    for result in results[1:]:
        if not np.allclose(first_pred, result['pred_segments'], atol=1e-3):
            all_identical = False
            break
    
    if all_identical:
        logger.warning("\n[WARNING] All predictions are nearly identical!")
        logger.warning("   The original problem may still exist.")
    else:
        logger.info("\n[SUCCESS] Predictions are diverse across different shapes!")
        logger.info("   The model is generating different outputs for different inputs.")
    
    return {
        'endpoint_variance': endpoint_variance,
        'unique_types': unique_types,
        'all_identical': all_identical
    }

def main():
    # Setup logging
    logger = setup_logging("evaluation_results")
    logger.info("="*60)
    logger.info("SHAPE PREDICTION MODEL EVALUATION")
    logger.info("="*60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load trained model
    checkpoint_path = "bezier_checkpoints/best_model.pth"
    if not os.path.exists(checkpoint_path):
        logger.error(f"❌ Checkpoint not found: {checkpoint_path}")
        logger.error("Make sure you've trained the model first!")
        return
    
    model = load_trained_model(checkpoint_path, device)
    
    # Load dataset
    logger.info("\nLoading dataset...")
    dataset = AugmentedDataset(root_dir="dataset", max_samples=5)
    logger.info(f"Loaded {len(dataset)} samples")
    
    # Evaluate each shape
    logger.info("\nEvaluating model on each shape...")
    results = []
    shape_names = ['shape1', 'shape2', 'shape3', 'shape4', 'shape5']
    
    for i in range(len(dataset)):
        sample = dataset[i]
        shape_name = shape_names[i] if i < len(shape_names) else f'shape_{i+1}'
        
        result = evaluate_single_sample(model, sample, device, shape_name)
        results.append(result)
        
        metrics = result['metrics']
        logger.info(f"  {shape_name}:")
        logger.info(f"    Curve L1 Error: {metrics['curve_l1_error']:.6f}")
        logger.info(f"    Type Accuracy: {metrics['type_accuracy']:.3f}")
        logger.info(f"    Endpoint Error: {metrics['endpoint_error']:.6f}")
    
    # Calculate overall metrics
    logger.info("\n" + "="*50)
    logger.info("OVERALL METRICS")
    logger.info("="*50)
    
    avg_curve_error = np.mean([r['metrics']['curve_l1_error'] for r in results])
    avg_type_accuracy = np.mean([r['metrics']['type_accuracy'] for r in results])
    avg_endpoint_error = np.mean([r['metrics']['endpoint_error'] for r in results])
    
    logger.info(f"Average Curve L1 Error: {avg_curve_error:.6f}")
    logger.info(f"Average Type Accuracy: {avg_type_accuracy:.3f}")
    logger.info(f"Average Endpoint Error: {avg_endpoint_error:.6f}")
    
    # Check diversity (the main issue we were trying to fix)
    diversity_stats = check_diversity(results)
    
    # === ADD COMPREHENSIVE DEBUGGING SUMMARY ===
    logger.info("\n" + "="*60)
    logger.info("COMPREHENSIVE DEBUGGING SUMMARY")
    logger.info("="*60)
    
    # 1. STOP PREDICTION SUMMARY
    logger.info("\n1. STOP PREDICTION ANALYSIS:")
    stop_indices = [r['pred_stops'].item() if hasattr(r['pred_stops'], 'item') else r['pred_stops'] for r in results]
    logger.info(f"   Stop indices: {stop_indices}")
    logger.info(f"   Average stop index: {np.mean(stop_indices):.1f}")
    logger.info(f"   Stop index range: {min(stop_indices)} - {max(stop_indices)}")
    
    # 2. SEGMENT COORDINATE ANALYSIS
    logger.info("\n2. SEGMENT COORDINATE ANALYSIS:")
    all_first_segs = [r['pred_segments'][0] for r in results]
    all_coords = np.array(all_first_segs)
    
    coord_means = np.mean(all_coords, axis=0)
    coord_vars = np.var(all_coords, axis=0)
    
    logger.info(f"   First segment coordinate means: [{coord_means[0]:.3f}, {coord_means[1]:.3f}, {coord_means[2]:.3f}, {coord_means[3]:.3f}, {coord_means[4]:.3f}, {coord_means[5]:.3f}]")
    logger.info(f"   First segment coordinate vars:  [{coord_vars[0]:.4f}, {coord_vars[1]:.4f}, {coord_vars[2]:.4f}, {coord_vars[3]:.4f}, {coord_vars[4]:.4f}, {coord_vars[5]:.4f}]")
    logger.info(f"   Average coordinate variance: {np.mean(coord_vars):.4f}")
    
    if np.mean(coord_vars) < 0.001:
        logger.warning("   [WARNING] Very low coordinate variance - segments may be too similar!")
    else:
        logger.info("   [SUCCESS] Good coordinate variance - segments are diverse!")
    
    # 3. TYPE PREDICTION SUMMARY
    logger.info("\n3. TYPE PREDICTION ANALYSIS:")
    all_type_distributions = {}
    
    for i, result in enumerate(results):
        shape_name = result['shape_name']
        pred_types = result['pred_types']
        valid_length = min(stop_indices[i] + 1, len(pred_types))
        
        type_counts = {0: 0, 1: 0, 2: 0}
        for j in range(valid_length):
            pred_type_scalar = pred_types[j].item() if hasattr(pred_types[j], 'item') else pred_types[j]  # Handle both tensor and numpy
            type_counts[pred_type_scalar] += 1
        
        all_type_distributions[shape_name] = type_counts
        logger.info(f"   {shape_name}: Line={type_counts[0]}, Quad={type_counts[1]}, Cubic={type_counts[2]}")
    
    # Overall type statistics
    total_types = {0: 0, 1: 0, 2: 0}
    for dist in all_type_distributions.values():
        for t, count in dist.items():
            total_types[t] += count
    
    total_segs = sum(total_types.values())
    type_percentages = {t: (count/total_segs)*100 if total_segs > 0 else 0 for t, count in total_types.items()}
    
    logger.info(f"   Overall type distribution: Line={type_percentages[0]:.1f}%, Quad={type_percentages[1]:.1f}%, Cubic={type_percentages[2]:.1f}%")
    
    if type_percentages[2] > 90:
        logger.warning("   [WARNING] Model heavily biased toward cubic curves!")
    elif min(type_percentages.values()) < 5:
        logger.warning("   [WARNING] Some curve types are rarely predicted!")
    else:
        logger.info("   [SUCCESS] Good type diversity across predictions!")
    
    # 4. COORDINATE RANGE ANALYSIS
    logger.info("\n4. COORDINATE RANGE ANALYSIS:")
    all_valid_coords = []
    for result in results:
        segs = result['pred_segments']
        valid_segs = segs[~(segs == -1).all(axis=1)]  # Remove masked segments
        if len(valid_segs) > 0:
            all_valid_coords.extend(valid_segs.flatten())
    
    if all_valid_coords:
        coord_min = min(all_valid_coords)
        coord_max = max(all_valid_coords)
        coord_range = coord_max - coord_min
        
        logger.info(f"   Coordinate range: [{coord_min:.3f}, {coord_max:.3f}] (span: {coord_range:.3f})")
        
        if coord_range < 0.1:
            logger.warning("   [WARNING] Very narrow coordinate range - shapes may be too constrained!")
        elif coord_range > 0.8:
            logger.info("   [SUCCESS] Good coordinate range - shapes can be diverse!")
        else:
            logger.info("   [MODERATE] Reasonable coordinate range.")
    
    # 5. IDENTICAL SEGMENTS CHECK
    logger.info("\n5. IDENTICAL SEGMENTS CHECK:")
    for result in results:
        shape_name = result['shape_name']
        segs = result['pred_segments']
        stop_idx = result['pred_stops'].item() if hasattr(result['pred_stops'], 'item') else result['pred_stops']
        
        # Check first few segments for similarity
        identical_pairs = 0
        total_pairs = 0
        
        for i in range(min(5, len(segs))):
            for j in range(i+1, min(5, len(segs))):
                total_pairs += 1
                if np.allclose(segs[i], segs[j], atol=1e-3):
                    identical_pairs += 1
        
        similarity_pct = (identical_pairs / total_pairs * 100) if total_pairs > 0 else 0
        logger.info(f"   {shape_name}: {identical_pairs}/{total_pairs} segment pairs are identical ({similarity_pct:.1f}%)")
    
    logger.info("="*60)
    
    # Create visualizations
    logger.info("\n" + "="*50)
    logger.info("GENERATING VISUALIZATIONS")
    logger.info("="*50)
    
    visualize_predictions(results)
    logger.info("Visualizations saved to 'evaluation_results/' directory")
    
    # Final assessment
    logger.info("\n" + "="*60)
    logger.info("EVALUATION SUMMARY")
    logger.info("="*60)
    
    if diversity_stats['all_identical']:
        logger.warning("[ISSUE DETECTED] Model still produces identical outputs")
        logger.warning("   Consider training longer or adjusting model architecture")
    else:
        logger.info("[SUCCESS] Model produces diverse outputs for different shapes")
        
        if avg_endpoint_error < 0.1:
            logger.info("[EXCELLENT] Shape accuracy is very good")
        elif avg_endpoint_error < 0.2:
            logger.info("[GOOD] Shape accuracy is acceptable")
        else:
            logger.warning("[FAIR] Shape accuracy could be improved")
        
        if avg_type_accuracy > 0.8:
            logger.info("[EXCELLENT] Curve type classification is very good")
        elif avg_type_accuracy > 0.6:
            logger.info("[GOOD] Curve type classification is acceptable")
        else:
            logger.warning("[FAIR] Curve type classification needs improvement")
    
    logger.info("\nEvaluation complete! Check the 'evaluation_results/' folder for visualizations.")

if __name__ == "__main__":
    main() 