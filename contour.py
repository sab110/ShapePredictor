"""
Minimal contour module to convert masks to Bezier sequences and vertex sequences.
"""
import numpy as np
import cv2
import math

# Try to import scipy, but fall back to basic implementation if not available
try:
    from scipy.optimize import minimize
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("Warning: scipy not available, using basic Bezier fitting")

def mask_to_vertex_sequence(mask, epsilon_ratio=0.01):
    """Convert a binary mask to a simplified vertex sequence."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return np.array([]), np.array([])
    
    # Get the largest contour
    contour = max(contours, key=cv2.contourArea)
    contour = contour.squeeze(1)  # Remove redundant dimension
    
    # Simplify the contour
    epsilon = epsilon_ratio * cv2.arcLength(contour, True)
    simplified = cv2.approxPolyDP(contour, epsilon, True)
    simplified = simplified.squeeze(1)
    
    return simplified, simplified

def fit_quadratic_bezier_basic(points):
    """Basic quadratic Bezier fitting without scipy optimization."""
    if len(points) < 3:
        # Not enough points for quadratic fit, return line
        p0, p1 = points[0], points[-1]
        return [p0[0], p0[1], -1, -1, p1[0], p1[1]]
    
    p0, p2 = points[0], points[-1]
    
    # Simple approximation: use the middle point as control point
    mid_idx = len(points) // 2
    p1 = points[mid_idx]
    
    # Return format: [cx1, cy1, cx2, cy2, ex, ey]
    # For quadratic: cx1, cy1 = control point, cx2, cy2 = -1 (unused), ex, ey = endpoint
    return [p1[0], p1[1], -1, -1, p2[0], p2[1]]

def fit_quadratic_bezier(points):
    """Fit a quadratic Bezier curve to a set of points."""
    if not HAS_SCIPY:
        return fit_quadratic_bezier_basic(points)
        
    if len(points) < 3:
        # Not enough points for quadratic fit, return line
        p0, p1 = points[0], points[-1]
        return [p0[0], p0[1], -1, -1, p1[0], p1[1]]
    
    p0, p2 = points[0], points[-1]
    
    def bezier_point(t, P0, P1, P2):
        """Calculate point on quadratic Bezier curve."""
        return (1-t)**2 * P0 + 2*(1-t)*t * P1 + t**2 * P2
    
    def objective(p1_flat):
        """Objective function to minimize for Bezier fitting."""
        p1 = np.array([p1_flat[0], p1_flat[1]])
        error = 0
        n = len(points)
        for i, point in enumerate(points):
            t = i / (n - 1) if n > 1 else 0
            bezier_pt = bezier_point(t, p0, p1, p2)
            error += np.sum((point - bezier_pt)**2)
        return error
    
    # Initial guess for control point
    mid_idx = len(points) // 2
    initial_p1 = points[mid_idx]
    
    # Optimize
    result = minimize(objective, initial_p1, method='BFGS')
    optimal_p1 = result.x
    
    # Return format: [cx1, cy1, cx2, cy2, ex, ey]
    # For quadratic: cx1, cy1 = control point, cx2, cy2 = -1 (unused), ex, ey = endpoint
    return [optimal_p1[0], optimal_p1[1], -1, -1, p2[0], p2[1]]

def fit_cubic_bezier_basic(points):
    """Basic cubic Bezier fitting without scipy optimization."""
    if len(points) < 4:
        return fit_quadratic_bezier_basic(points)
    
    p0, p3 = points[0], points[-1]
    
    # Simple approximation: use 1/3 and 2/3 points as control points
    third_idx = len(points) // 3
    two_third_idx = 2 * len(points) // 3
    p1 = points[third_idx]
    p2 = points[two_third_idx]
    
    # Return format: [cx1, cy1, cx2, cy2, ex, ey]
    return [p1[0], p1[1], p2[0], p2[1], p3[0], p3[1]]

def fit_cubic_bezier(points):
    """Fit a cubic Bezier curve to a set of points."""
    if not HAS_SCIPY:
        return fit_cubic_bezier_basic(points)
        
    if len(points) < 4:
        return fit_quadratic_bezier(points)
    
    p0, p3 = points[0], points[-1]
    
    def bezier_point(t, P0, P1, P2, P3):
        """Calculate point on cubic Bezier curve."""
        return (1-t)**3 * P0 + 3*(1-t)**2*t * P1 + 3*(1-t)*t**2 * P2 + t**3 * P3
    
    def objective(params):
        """Objective function for cubic Bezier fitting."""
        p1 = np.array([params[0], params[1]])
        p2 = np.array([params[2], params[3]])
        error = 0
        n = len(points)
        for i, point in enumerate(points):
            t = i / (n - 1) if n > 1 else 0
            bezier_pt = bezier_point(t, p0, p1, p2, p3)
            error += np.sum((point - bezier_pt)**2)
        return error
    
    # Initial guess for control points
    third_idx = len(points) // 3
    two_third_idx = 2 * len(points) // 3
    initial_p1 = points[third_idx]
    initial_p2 = points[two_third_idx]
    initial_params = [initial_p1[0], initial_p1[1], initial_p2[0], initial_p2[1]]
    
    # Optimize
    result = minimize(objective, initial_params, method='BFGS')
    optimal_params = result.x
    
    # Return format: [cx1, cy1, cx2, cy2, ex, ey]
    return [optimal_params[0], optimal_params[1], optimal_params[2], optimal_params[3], p3[0], p3[1]]

def segment_complexity(points):
    """Calculate the complexity of a point segment."""
    if len(points) < 3:
        return 0
    
    # Simple complexity measure based on curvature variation
    total_curvature = 0
    for i in range(1, len(points) - 1):
        p1, p2, p3 = points[i-1], points[i], points[i+1]
        v1 = p2 - p1
        v2 = p3 - p2
        
        # Calculate angle between vectors
        if np.linalg.norm(v1) > 0 and np.linalg.norm(v2) > 0:
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            cos_angle = np.clip(cos_angle, -1, 1)
            angle = np.arccos(cos_angle)
            total_curvature += angle
    
    return total_curvature

def split_into_segments(contour, max_segments=30):
    """Split a contour into segments for Bezier fitting."""
    if len(contour) <= 3:
        return [contour]
    
    # Simple splitting strategy: divide based on complexity
    n_points = len(contour)
    segment_size = max(3, n_points // max_segments)
    
    segments = []
    for i in range(0, n_points, segment_size):
        end_idx = min(i + segment_size + 1, n_points)  # +1 for overlap
        segment = contour[i:end_idx]
        if len(segment) >= 2:
            segments.append(segment)
    
    # Ensure we don't exceed max_segments
    if len(segments) > max_segments:
        segments = segments[:max_segments]
    
    return segments

def mask_to_bezier_sequence(mask, max_ctrl=2, dev_thresh=0.5, epsilon_ratio=0.01, 
                           merge_thresh=0.01, angle_thresh_deg=5):
    """
    Convert a binary mask to a sequence of Bezier curves.
    
    Args:
        mask: Binary mask (numpy array)
        max_ctrl: Maximum control points (1=quadratic, 2=cubic)
        dev_thresh: Deviation threshold for splitting
        epsilon_ratio: Contour simplification ratio
        merge_thresh: Threshold for merging close points
        angle_thresh_deg: Angle threshold for corner detection
    
    Returns:
        tuple: (vertices, bezier_segments)
            vertices: List of simplified contour vertices
            bezier_segments: List of Bezier parameters [cx1, cy1, cx2, cy2, ex, ey]
    """
    
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return [], []
    
    # Get the largest contour
    contour = max(contours, key=cv2.contourArea)
    contour = contour.squeeze(1)  # Remove redundant dimension
    
    if len(contour) < 3:
        return contour.tolist(), []
    
    # Simplify the contour
    epsilon = epsilon_ratio * cv2.arcLength(contour, True)
    simplified = cv2.approxPolyDP(contour, epsilon, True)
    simplified = simplified.squeeze(1)
    
    # Split into segments
    segments = split_into_segments(simplified, max_segments=30)
    
    bezier_curves = []
    all_vertices = []
    
    for segment in segments:
        if len(segment) < 2:
            continue
            
        all_vertices.extend(segment[:-1])  # Avoid duplicate endpoints
        
        # Fit Bezier curve based on max_ctrl
        if max_ctrl >= 2 and len(segment) >= 4:
            # Cubic Bezier
            bezier_params = fit_cubic_bezier(segment)
        elif len(segment) >= 3:
            # Quadratic Bezier
            bezier_params = fit_quadratic_bezier(segment)
        else:
            # Line segment
            p0, p1 = segment[0], segment[-1]
            bezier_params = [-1, -1, -1, -1, p1[0], p1[1]]
        
        bezier_curves.append(bezier_params)
    
    # Add the last vertex if needed
    if len(simplified) > 0:
        all_vertices.append(simplified[-1])
    
    return all_vertices, bezier_curves 