import os
import json
import cv2
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision.transforms.functional as TF
import random
import time
from tqdm import tqdm
import logging
from datetime import datetime
from contour import mask_to_bezier_sequence, mask_to_vertex_sequence
import matplotlib.path as mpath
from torch.nn.utils.rnn import pad_sequence
from transformers import CLIPProcessor, CLIPModel, AutoTokenizer, T5EncoderModel
import functools
import torch.nn.functional as F
from matplotlib.patches import Polygon as MplPolygon
from scipy.interpolate import splprep, splev
from shapely.geometry import Point, Polygon as ShapelyPolygon
from typing import Optional, List, Dict, Tuple, Union
import traceback
import math

NUM_INTERIOR_POINTS = 256
CURVE_PARAMS = 6  # [cx1, cy1, cx2, cy2, ex, ey] - Model Output Format

# -----------------------------------------------------------------------------  
# Helper Functions for Context-Point Sampling (unchanged)  
# -----------------------------------------------------------------------------  
def generate_deterministic_context_points(
    parent_verts_scaled: Union[np.ndarray, None],
    parent_bin:          np.ndarray,
    N_total:             int,
    N_boundary:          int,
    W:                   int,
    H:                   int,
    sx:                  float,
    sy:                  float
) -> np.ndarray:
    """
    1) Take up to N_boundary points from the polygon boundary
       (first vertices, then evenly spaced along edges).
    2) If still need more (N_total - len(pts)), uniformly subsample
       that many points from the interior mask pixels.
    3) Never pad with [0,0].
    """
    pts = []

    # --- 1) Boundary sampling ---
    if N_boundary > 0 and parent_verts_scaled is not None and len(parent_verts_scaled) >= 2:
        verts = np.asarray(parent_verts_scaled, dtype=np.float32)
        # a) original vertices
        take_verts = min(len(verts), N_boundary)
        pts.extend(verts[:take_verts].tolist())

        # b) evenly‐spaced extras along edges
        extra = N_boundary - take_verts
        if extra > 0:
            # build closed loop
            edges = np.vstack([verts, verts[0]])
            vecs  = edges[1:] - edges[:-1]
            lens  = np.hypot(vecs[:,0], vecs[:,1])
            perim = lens.sum()
            dists = np.linspace(0, perim, extra, endpoint=False)
            cum   = np.concatenate([[0], np.cumsum(lens)])
            for d in dists:
                i = np.searchsorted(cum, d, side='right') - 1
                t = (d - cum[i]) / (lens[i] + 1e-12)
                p = verts[i] + t * vecs[i]
                pts.append([float(p[0]), float(p[1])])

    # --- 2) Interior sampling if still short ---
    needed = N_total - len(pts)
    if needed > 0 and parent_bin is not None:
        ys, xs = np.where(parent_bin > 0)
        M       = xs.shape[0]
        if M > 0:
            # uniformly subsample 'needed' indices from the sorted mask list
            idxs = np.linspace(0, M - 1, needed, dtype=int)
            sel  = np.stack([xs[idxs], ys[idxs]], axis=1).astype(np.float32)

            # normalize & scale
            sel[:,0] = (sel[:,0] / (W - 1 + 1e-9)) * sx
            sel[:,1] = (sel[:,1] / (H - 1 + 1e-9)) * sy

            pts.extend(sel.tolist())
        else:
            # no interior pixels: fall back to boundary verts if any
            if parent_verts_scaled is not None and len(parent_verts_scaled) > 0:
                pool = np.asarray(parent_verts_scaled, dtype=np.float32)
                idxs = np.arange(needed) % pool.shape[0]
                fallback = pool[idxs]
                pts.extend(fallback.tolist())
            else:
                # last resort: uniform grid over entire canvas
                side = int(np.ceil(np.sqrt(needed)))
                xs_lin = np.linspace(0, W-1, side)
                ys_lin = np.linspace(0, H-1, side)
                xx, yy = np.meshgrid(xs_lin, ys_lin, indexing='xy')
                grid   = np.stack([xx.ravel(), yy.ravel()], axis=1)[:needed]
                grid[:,0] = (grid[:,0] / (W - 1 + 1e-9)) * sx
                grid[:,1] = (grid[:,1] / (H - 1 + 1e-9)) * sy
                pts.extend(grid.tolist())

    # --- 3) Truncate if overshot ---
    if len(pts) > N_total:
        pts = pts[:N_total]

    return np.asarray(pts, dtype=np.float32)

def generate_deterministic_context_points_gpu(
    parent_verts_scaled: Union[torch.Tensor, None],
    parent_bin:          torch.Tensor,
    N_total:             int,
    N_boundary:          int,
    N_interior:          int
) -> torch.Tensor:
    device = parent_bin.device
    dtype = torch.float32
    pts = []
    # 1) Boundary sampling: include each polygon vertex first
    if N_boundary > 0 and parent_verts_scaled is not None and parent_verts_scaled.size(0) >= 1:
        V = parent_verts_scaled.to(device=device, dtype=dtype)
        # Add each vertex
        pts.extend(V.cpu().tolist())
        extra = N_boundary - V.size(0)
        if extra > 0:
            # Distribute extra points along edges
            edges = torch.cat([V, V[0:1]], dim=0)
            vecs = edges[1:] - edges[:-1]
            lens = vecs.norm(dim=1)
            perim = lens.sum()
            dists = torch.linspace(0.0, perim, steps=extra, device=device, dtype=dtype)
            cum = torch.cat([torch.zeros(1,device=device,dtype=dtype), lens.cumsum(dim=0)])
            idx = (torch.bucketize(dists, cum) - 1).clamp(0, lens.size(0)-1)
            t = (dists - cum[idx]) / (lens[idx] + 1e-12)
            extra_pts = (V[idx] + vecs[idx] * t.unsqueeze(1)).cpu().tolist()
            pts.extend(extra_pts)
    else:
        pts.extend([[0.0, 0.0]] * N_boundary)

    # 2) Interior via uniform grid
    if N_interior > 0 and parent_bin is not None:
        H, W = parent_bin.shape
        g = int(np.ceil(np.sqrt(N_interior)))
        xs = torch.linspace(0, W-1, steps=g, device=device)
        ys = torch.linspace(0, H-1, steps=g, device=device)
        xx, yy = torch.meshgrid(xs, ys, indexing='xy')
        cand = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=1)
        ix, jx = cand[:,1].long(), cand[:,0].long()
        inside = parent_bin[ix, jx] > 0
        valid = cand[inside]
        Mv = valid.size(0)
        if Mv >= N_interior:
            sel = valid[:N_interior]
        else:
            pad = valid[-1:].expand(N_interior - Mv, 2)
            sel = torch.cat([valid, pad], dim=0)
        sel = sel.cpu().numpy()
        sel[:,0] = sel[:,0] / (W - 1 + 1e-9)
        sel[:,1] = sel[:,1] / (H - 1 + 1e-9)
        pts_i = sel.tolist()
    else:
        pts_i = [[0.0, 0.0]] * N_interior
    pts.extend(pts_i)

    # 3) Pad/truncate
    if len(pts) < N_total:
        pts.extend([pts[-1]] * (N_total - len(pts)))
    pts = pts[:N_total]
    return torch.tensor(pts, dtype=torch.float32, device=device)

# -----------------------------------------------------------------------------  
# Mask & Contour Utilities (unchanged)  
# -----------------------------------------------------------------------------  
def load_mask_np(mask_path):
    img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    return binary

def extract_full_contour(mask_np):
    contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contour = max(contours, key=cv2.contourArea)
    return contour.squeeze(1)

def simplify_contour(contour, epsilon_ratio):
    epsilon = epsilon_ratio * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    return approx.squeeze(1)

def mask_to_spline_points(mask_np, W, H, num_samples, smoothing):
    """
    Fits a periodic smoothing spline to the detailed full contour of the mask,
    then samples `num_samples` points equidistant in parameter space.
    Normalizes coordinates to [0,1].
    """
    contour = extract_full_contour(mask_np)
    x = contour[:, 0].astype(np.float32)
    y = contour[:, 1].astype(np.float32)
    try:
        tck, _ = splprep([x, y], s=smoothing, per=True)
        u_new = np.linspace(0, 1, num_samples)
        x_new, y_new = splev(u_new, tck)
    except Exception:
        total = len(x)
        idx = np.linspace(0, total - 1, num_samples).astype(int)
        x_new = x[idx]
        y_new = y[idx]
    pts = np.vstack([x_new / W, y_new / H]).T
    return pts

from scipy.optimize import minimize

def fit_quadratic_bezier(points):
    p0, p2 = points[0], points[-1]
    def loss(p1_flat):
        p1 = p1_flat.reshape(2)
        t = np.linspace(0, 1, len(points))[:, None]
        curve = (1 - t)**2 * p0 + 2*(1 - t)*t*p1 + t**2*p2
        return np.sum(np.linalg.norm(curve - points, axis=1))
    init = (p0 + p2) / 2
    res = minimize(loss, init, method='Powell')
    return p0, res.x, p2

def measure_complexity(segment):
    chord = np.linalg.norm(segment[-1] - segment[0])
    arc   = np.sum(np.linalg.norm(np.diff(segment, axis=0), axis=1))
    return arc / chord if chord>1e-8 else 1.0

def find_max_deviation_point(segment):
    p0, p1 = segment[0], segment[-1]
    chord = p1 - p0
    L = np.linalg.norm(chord)
    if L<1e-8: return None, 0.0
    unit = chord / L
    normal = np.array([-unit[1], unit[0]])
    devs = [abs(np.dot((pt-p0), normal)) for pt in segment]
    idx = int(np.argmax(devs))
    return idx, devs[idx]

def split_and_fit(segment, threshold_complex=1.01, threshold_dev=.5):
    """Return a list of (p0, ctrl, p2, type_flag) for this segment."""
    comp = measure_complexity(segment)
    if comp <= threshold_complex:
        p0, p2 = segment[0], segment[-1]
        return [ (p0, p0, p2, 0) ]
    idx, dev = find_max_deviation_point(segment)
    if dev <= threshold_dev or idx in (0, len(segment)-1):
        a,b,c = fit_quadratic_bezier(segment)
        return [ (a, b, c, 1) ]
    first  = split_and_fit(segment[:idx+1], threshold_complex, threshold_dev)
    second = split_and_fit(segment[idx:],   threshold_complex, threshold_dev)
    return first + second

# -----------------------------------------------------------------------------  
# Context-Point Validity & Visualization (unchanged)  
# -----------------------------------------------------------------------------  
def check_context_point_validity(context_points, parent_bin, sx, sy):
    """
    Checks which context points fall inside the binary parent mask.
    """
    H, W = parent_bin.shape
    points = context_points.clone().detach().cpu().numpy()
    xs = np.clip((points[:, 0] / sx * W).astype(int), 0, W - 1)
    ys = np.clip((points[:, 1] / sy * H).astype(int), 0, H - 1)
    mask_values = parent_bin[ys, xs] > 0
    return mask_values

def visualize_context_with_mask(context_points, parent_bin, sx, sy):
    validity = check_context_point_validity(context_points, parent_bin, sx, sy)
    points = context_points.clone().detach().cpu().numpy()

    plt.imshow(parent_bin, cmap='gray')
    plt.scatter(points[:,0] / sx * parent_bin.shape[1],
                points[:,1] / sy * parent_bin.shape[0],
                c=['green' if v else 'red' for v in validity], s=4)
    plt.title("Green = Inside, Red = Outside")
    plt.show()

def save_visualization(pts, mismatches, mask_path, out_path="mask_sampling_debug.png"):
    wrong_pts = np.array([pt for (_i, pt, pix, poly) in mismatches if pix != poly],
                         dtype=np.float32)

    fig, ax = plt.subplots(figsize=(6,6))
    ax.set_title(f"Sampling on {mask_path}", fontsize=10)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.scatter(pts[:,0], pts[:,1],
               s=5, c="lightgray", label="all points")

    if wrong_pts.size:
        ax.scatter(wrong_pts[:,0], wrong_pts[:,1],
                   s=30, c="red", marker="x", label="pixel≠poly")

    ax.legend(markerscale=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved visualization to {out_path}")

# -----------------------------------------------------------------------------  
# Dataset Definition (unchanged except for minor formatting)  
# -----------------------------------------------------------------------------  
class AugmentedDataset(Dataset):
    """
    Loads scenes, computes:
      - FLAN-T5 sequence embeddings for child & parent (fixed [512, hidden])
      - parent_bbox corners [4,2]
      - normalized parent Bézier segments
      - normalized child Bézier GT curves
    """
    def __init__(
        self,
        root_dir: str = "dataset",
        json_dir: str = "json",
        masks_dir: str = "masks",
        images_dir: str = "images",
        max_samples: int = None,
        poly_epsilon_ratio: float = 0.01,
        text_max_length: int = 512,
    ):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
        self.encoder = T5EncoderModel.from_pretrained("google/flan-t5-base").encoder.eval()
        for p in self.encoder.parameters(): p.requires_grad = False
        self.text_max_length = text_max_length
        self.hidden_size = self.encoder.config.hidden_size
        self.epsilon = float(poly_epsilon_ratio)

        self.json_dir = os.path.join(root_dir, json_dir)
        self.masks_dir = os.path.join(root_dir, masks_dir)
        self.images_dir = os.path.join(root_dir, images_dir)

        scene_to_shapes = {}
        for fn in os.listdir(self.json_dir):
            if not fn.endswith('.json'): continue
            data = json.load(open(os.path.join(self.json_dir, fn)))
            scene = fn[:-5]
            if isinstance(data.get('scene'), list):
                scene_to_shapes[scene] = data['scene']

        raw = []
        available = {f[:-4] for f in os.listdir(self.images_dir) if f.endswith('.png')}
        for scene, shapes in scene_to_shapes.items():
            if scene not in available: continue
            id_map = {s['id']: s for s in shapes if isinstance(s, dict) and 'id' in s}
            for s in shapes:
                if not isinstance(s, dict): continue
                pid, mname = s.get('parent'), s.get('mask_path')
                if pid is None or pid < 0 or not mname: continue
                child_mask = os.path.join(self.masks_dir, scene, mname)
                if not os.path.exists(child_mask): continue
                parent_s = id_map.get(pid, {})
                pm = parent_s.get('mask_path')
                if pm and os.path.exists(os.path.join(self.masks_dir, scene, pm)):
                    parent_mask = os.path.join(self.masks_dir, scene, pm)
                    parent_is_image = False
                else:
                    parent_mask = os.path.join(self.images_dir, f"{scene}.png")
                    parent_is_image = True
                raw.append({
                    'scene': scene,
                    'child_mask': child_mask,
                    'parent_mask': parent_mask,
                    'parent_is_image': parent_is_image,
                    'child_desc': s.get('description', ''),
                    'parent_desc': parent_s.get('description', '')
                })
                if max_samples and len(raw) >= max_samples:
                    break
            if max_samples and len(raw) >= max_samples:
                break

        self.samples = []
        for info in raw:
            img = Image.open(os.path.join(self.images_dir, f"{info['scene']}.png"))
            W, H = img.size; canvas = max(W, H)
            child_bin = cv2.resize(
                cv2.imread(info['child_mask'], cv2.IMREAD_GRAYSCALE),
                (W, H), interpolation=cv2.INTER_NEAREST
            )
            if not info['parent_is_image']:
                pm = cv2.resize(
                    cv2.imread(info['parent_mask'], cv2.IMREAD_GRAYSCALE),
                    (W, H), interpolation=cv2.INTER_NEAREST
                )
            else:
                pm = np.ones((H, W), np.uint8) * 255

            mask_sq = np.zeros((canvas, canvas), np.uint8)
            mask_sq[:H, :W] = pm
            ys, xs = np.where(mask_sq > 0)
            if xs.size>0:
                x_min, x_max = xs.min(), xs.max()
                y_min, y_max = ys.min(), ys.max()
            else:
                x_min, y_min, x_max, y_max = 0,0,canvas-1,canvas-1
            bbox = torch.tensor([
                [x_min,y_min], [x_min,y_max],
                [x_max,y_min], [x_max,y_max]
            ], dtype=torch.float32) / float(canvas - 1)

            _, p_segs_raw = mask_to_bezier_sequence(
                mask_sq, max_ctrl=2, dev_thresh=0,
                epsilon_ratio=self.epsilon, merge_thresh=0.01, angle_thresh_deg=1
            )
            p_segs = torch.tensor(p_segs_raw, dtype=torch.float32)
            p_segs[p_segs>=0] /= float(canvas - 1)
            pad_len = 30 - p_segs.size(0)
            pad_tensor = -1 * torch.ones((pad_len, 6), dtype=p_segs.dtype, device=p_segs.device)
            p_segs= torch.cat([p_segs, pad_tensor], dim=0)

            mask_sq[:,:] = 0; mask_sq[:H,:W] = child_bin
            _, c_segs_raw = mask_to_bezier_sequence(
                mask_sq, max_ctrl=2, dev_thresh=0,
                epsilon_ratio=self.epsilon, merge_thresh=0.01, angle_thresh_deg=1
            )
            gt = torch.tensor(c_segs_raw, dtype=torch.float32)
            gt[gt>=0] /= float(canvas - 1)
            lengths = gt.size(0)
            pad_len = 30 - gt.size(0)
            pad_tensor = -1 * torch.ones((pad_len, 6), dtype=gt.dtype, device=gt.device)
            gt = torch.cat([gt, pad_tensor], dim=0)

            toks_c = self.tokenizer(
                [info['child_desc']],
                padding='max_length', truncation=True,
                max_length=self.text_max_length,
                return_tensors='pt'
            )
            toks_p = self.tokenizer(
                [info['parent_desc']],
                padding='max_length', truncation=True,
                max_length=self.text_max_length,
                return_tensors='pt'
            )
            with torch.no_grad():
                enc_c = self.encoder(
                    input_ids=toks_c.input_ids,
                    attention_mask=toks_c.attention_mask
                )
                enc_p = self.encoder(
                    input_ids=toks_p.input_ids,
                    attention_mask=toks_p.attention_mask
                )
            seq_c = enc_c.last_hidden_state.squeeze(0)
            seq_p = enc_p.last_hidden_state.squeeze(0)
            mask_c = toks_c.attention_mask.squeeze(0).bool()
            mask_p = toks_p.attention_mask.squeeze(0).bool()

            self.samples.append({
                'child_embs': seq_c,
                'child_mask': mask_c,
                'parent_embs': seq_p,
                'parent_mask': mask_p,
                'parent_bbox': bbox,
                'parent_bezier': p_segs,
                'gt_curves': gt,
                'lengths': lengths
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        p_segs = sample['parent_bezier']
        pad_mask = torch.zeros(p_segs.size(0), dtype=torch.bool)
        return {
            'child_embs':        sample['child_embs'],
            'child_mask':        sample['child_mask'],
            'parent_embs':       sample['parent_embs'],
            'parent_mask':       sample['parent_mask'],
            'parent_bbox':       sample['parent_bbox'],
            'parent_bezier':     sample['parent_bezier'],
            'parent_bezier_segs':sample['parent_bezier'],
            'padding_mask':      pad_mask,
            'gt_curves':         sample['gt_curves'],
            'lengths':           sample['lengths'],
        }

def collate_fn(batch):
    child_embs    = torch.stack([b['child_embs']    for b in batch], dim=0)
    child_mask    = torch.stack([b['child_mask']    for b in batch], dim=0)
    parent_embs   = torch.stack([b['parent_embs']   for b in batch], dim=0)
    parent_mask   = torch.stack([b['parent_mask']   for b in batch], dim=0)
    bbox          = torch.stack([b['parent_bbox']   for b in batch], dim=0)

    pb              = [b['parent_bezier'] for b in batch]
    parent_bezier   = torch.nn.utils.rnn.pad_sequence(pb, batch_first=True, padding_value=-1.0)
    padding_mask    = parent_bezier[:,:,0] < 0

    gc           = [b['gt_curves'] for b in batch]
    gt_curves    = torch.nn.utils.rnn.pad_sequence(gc, batch_first=True, padding_value=0.0)

    lengths      = torch.tensor([b['lengths'] for b in batch], dtype=torch.long)

    return {
        'child_embs':        child_embs,
        'child_mask':        child_mask,
        'parent_embs':       parent_embs,
        'parent_mask':       parent_mask,
        'parent_bbox':       bbox,
        'parent_bezier':     parent_bezier,
        'parent_bezier_segs':parent_bezier,
        'padding_mask':      padding_mask,
        'gt_curves':         gt_curves,
        'lengths':           lengths,
    }

# -----------------------------------------------------------------------------  
# Model Configuration & Components  
# -----------------------------------------------------------------------------  
from transformers import T5Tokenizer, T5EncoderModel
from dataclasses import dataclass

@dataclass
class PolygonConfig:
    d_model: int = 128
    n_head: int = 8
    num_dec_layers_seq: int = 6
    dim_feedforward: int = 512
    max_segments: int = 30
    dropout: float = 0.1
    num_fusion_layers: int = 2
    t5_model_name: str = "google/flan-t5-base"
    max_text_length: int = 512

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=30):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)

def generate_square_subsequent_mask(sz: int) -> torch.Tensor:
    mask = torch.triu(torch.ones(sz, sz, dtype=torch.bool), diagonal=1)
    return mask.float().masked_fill(mask, float('-inf')).masked_fill(~mask, 0.0)

class ShapePredictor(nn.Module):
    def __init__(self,
                 d_model: int,
                 num_heads: int,
                 num_decoder_layers: int,
                 num_segments: int,
                 dim_feedforward: int,
                 dropout: float,
                 out_dim: int = 6):
        super().__init__()
        self.num_segments = num_segments
        self.d_model = d_model

        # Learnable segment queries with STRONG positional differentiation
        self.segment_queries = nn.Parameter(torch.randn(num_segments, d_model) * 0.2)

        # CRITICAL: Strong positional embeddings to differentiate segments
        self.positional_embeddings = nn.Parameter(torch.randn(num_segments, d_model) * 0.3)

        # Transformer decoder with cross-attention to memory
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation='relu'
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=num_decoder_layers,
            norm=nn.LayerNorm(d_model)
        )

        # Coordinate head with segment-specific processing
        self.coord_head = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(d_model, out_dim),
        )

        # Type head
        self.type_head = nn.Linear(d_model, 3)

        # Stop head - ENHANCED with positional awareness
        self.stop_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1)
        )

        # Add positional encodings for stop prediction
        self.stop_position_encoding = nn.Parameter(
            torch.randn(num_segments, d_model) * 0.1
        )

        # Initialize with diversity in mind
        self._init_weights()

    def _init_weights(self):
        """Initialize with VERY strong segment differentiation"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=1.2)
                if m.bias is not None:
                    nn.init.uniform_(m.bias, -0.1, 0.1)

        # CRITICAL: Initialize stop head to FAVOR CONTINUATION over early stopping
        with torch.no_grad():
            final_stop_layer = self.stop_head[-1]
            final_stop_layer.bias.fill_(-2.0)
            nn.init.xavier_uniform_(final_stop_layer.weight, gain=0.8)

        # Ensure segment queries start VERY different with strong patterns
        with torch.no_grad():
            for i in range(self.num_segments):
                phase = (i + 1) * 2.0 * np.pi / self.num_segments
                pattern1 = torch.sin(torch.arange(self.d_model, dtype=torch.float) * phase * 0.1) * 0.3
                pattern2 = torch.cos(torch.arange(self.d_model, dtype=torch.float) * phase * 0.05) * 0.2
                pattern3 = torch.sin(torch.arange(self.d_model, dtype=torch.float) * phase * 0.2) * 0.1

                self.segment_queries[i] += pattern1 + pattern2 + pattern3

                pos_pattern = torch.sin(torch.arange(self.d_model, dtype=torch.float) * (i + 1) * 0.3) * 0.4
                self.positional_embeddings[i] += pos_pattern

                self.segment_queries[i] += torch.randn(self.d_model) * 0.1
                self.positional_embeddings[i] += torch.randn(self.d_model) * 0.15

    def forward(self, H_memory: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B = H_memory.size(0)

        # Create segment-specific memory by weighted aggregation
        memory_per_segment = []
        for i in range(self.num_segments):
            arange = torch.arange(H_memory.size(1), device=H_memory.device).float()
            segment_weights = torch.softmax(
                torch.sin(arange * (i * 0.7 + 1.3)) * 2.0 +
                torch.cos(arange * (i * 0.5 + 0.8)) * 1.5 +
                torch.sin(torch.tensor(i * 1.7, device=H_memory.device)) * 3.0,
                dim=0
            )
            segment_memory = H_memory * segment_weights.unsqueeze(0).unsqueeze(2)
            memory_per_segment.append(segment_memory.mean(dim=1))
        segment_memories = torch.stack(memory_per_segment, dim=1)

        # Build initial queries by hashing each segment's memory
        queries = []
        for i in range(self.num_segments):
            prime_list = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29,
                          31, 37, 41, 43, 47, 53, 59, 61, 67, 71,
                          73, 79, 83, 89, 97, 101, 103, 107, 109, 113]
            prime = prime_list[i % 30]

            seg_mem_i = segment_memories[:, i]
            segment_fingerprint = (
                torch.sin(seg_mem_i * prime * 0.1) +
                torch.cos(seg_mem_i * prime * 0.07) * 0.8 +
                torch.tanh(seg_mem_i * prime * 0.13) * 0.6
            )
            i_encoding = (
                torch.sin(torch.tensor(i * 2.1, device=H_memory.device)) * seg_mem_i * 0.7 +
                torch.cos(torch.tensor(i * 3.3, device=H_memory.device)) * seg_mem_i * 0.5 +
                torch.sin(torch.tensor(i * 5.7, device=H_memory.device)) * 0.4
            )
            segment_query = segment_fingerprint + i_encoding
            queries.append(segment_query)

        queries = torch.stack(queries, dim=1)
        queries = queries + 0.05 * self.positional_embeddings.unsqueeze(0)

        # Decode each segment independently with its own “mixed memory”
        all_decoded = []
        for i in range(self.num_segments):
            segment_query = queries[:, i:i+1]  # (B, 1, d_model)
            # Combine segment-specific memory with a global-context “mean‐pooled” memory
            segment_memory = segment_memories[:, i:i+1]  # (B, 1, d_model)
            broader_context = H_memory.mean(dim=1, keepdim=True) * 0.2  # (B, 1, d_model)
            mixed_memory = torch.cat([segment_memory, broader_context], dim=1)  # (B, 2, d_model)

            decoded_segment = self.decoder(tgt=segment_query, memory=mixed_memory)
            all_decoded.append(decoded_segment.squeeze(1))

        decoded_features = torch.stack(all_decoded, dim=1)  # (B, num_segments, d_model)

        # Produce normalized coordinates per segment, with per‐segment sinusoidal scaling & offset
        coords_list = []
        for i in range(self.num_segments):
            segment_features = decoded_features[:, i]  # (B, d_model)
            segment_coords = self.coord_head(segment_features)  # (B, 6)

            # --- APPLY THE FIX: use a scalar perturbation rather than a d_model‐sized vector ---
            # original code used: input_perturbation = torch.sin(mem.mean(dim=1)[:, None] * (i + 1) * 0.3)
            # which has shape (B, 1, d_model). Instead, we sum over d_model to get a scalar per batch:
            scalar_input = torch.sin((H_memory.mean(dim=1).sum(dim=1) * (i + 1) * 0.3))
            # shape of scalar_input is (B,), so unsqueeze to (B,1) and broadcast to (B,6):
            scalar_input = scalar_input.unsqueeze(1)

            i_tensor = torch.tensor(i, dtype=torch.float, device=H_memory.device)
            segment_scale = 0.8 + 0.4 * torch.sin(i_tensor * 1.7)
            segment_offset = 0.1 * torch.cos(i_tensor * 2.3)
            seg_norm = torch.tanh(segment_coords) * segment_scale + segment_offset
            seg_norm = seg_norm + scalar_input * 0.15
            seg_norm = torch.sigmoid(seg_norm)

            coords_list.append(seg_norm)

        coords_final = torch.stack(coords_list, dim=1)  # (B, num_segments, 6)

        # Type logits with per‐segment sinusoidal bias
        type_logits_list = []
        for i in range(self.num_segments):
            segment_features = decoded_features[:, i]
            type_bias = torch.tensor([
                0.3 * torch.sin(torch.tensor(i * 2.1)),
                0.3 * torch.cos(torch.tensor(i * 1.8)),
                0.3 * torch.sin(torch.tensor(i * 3.2))
            ], device=H_memory.device)
            segment_types = self.type_head(segment_features) + type_bias.unsqueeze(0)
            type_logits_list.append(segment_types)
        type_logits = torch.stack(type_logits_list, dim=1)  # (B, num_segments, 3)

        # Stop logits with sinusoidal positional encoding & linearly increasing bias
        stop_logits_list = []
        for i in range(self.num_segments):
            segment_features = decoded_features[:, i]
            pos_encoding = self.stop_position_encoding[i:i+1].expand(B, -1)
            stop_features = segment_features + pos_encoding
            position_bias = -2.0 + (i * 0.4)
            segment_stop = self.stop_head(stop_features).squeeze(-1) + position_bias
            stop_logits_list.append(segment_stop)
        stop_logits = torch.stack(stop_logits_list, dim=1)  # (B, num_segments)

        return coords_final, type_logits, stop_logits


class SimpleShapeEncoder(nn.Module):
    def __init__(self, in_dim: int = 6, dim: int = 128, seq_len: int = 50):
        super().__init__()
        self.seq_len = seq_len
        self.pos_emb = nn.Parameter(torch.randn(seq_len, dim))
        self.input_proj = nn.Linear(in_dim, dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )
        self.to_latent = nn.Linear(dim, dim)

    def forward(self, shape_pts: torch.Tensor) -> torch.Tensor:
        B, S_parent_actual, C_in = shape_pts.shape
        if C_in != self.input_proj.in_features:
            raise ValueError(f"SimpleShapeEncoder expects {self.input_proj.in_features} features, got {C_in}")
        if S_parent_actual > self.seq_len:
            shape_pts = shape_pts[:, :self.seq_len, :]
            S_parent_actual = self.seq_len

        x = self.input_proj(shape_pts)  # (B, S, dim)
        x = x + self.pos_emb[:S_parent_actual, :].unsqueeze(0)  # (B, S, dim)

        # Determine padding per batch
        is_padding_segment = (shape_pts == -1).all(dim=2)  # (B, S)
        actual_lengths = torch.full((B,), S_parent_actual, dtype=torch.long, device=shape_pts.device)
        for i in range(B):
            first_pad_idx = torch.where(is_padding_segment[i])[0]
            if len(first_pad_idx) > 0:
                actual_lengths[i] = first_pad_idx[0]

        s_indices = torch.arange(S_parent_actual, device=x.device).expand(B, S_parent_actual)
        valid_segment_mask_2D = (s_indices < actual_lengths.unsqueeze(1))  # (B, S)

        x = x * valid_segment_mask_2D.float().unsqueeze(-1)  # zero out padding
        x = self.mlp(x)  # (B, S, dim)
        x = x * valid_segment_mask_2D.float().unsqueeze(-1)

        summed_features = x.sum(dim=1)  # (B, dim)
        num_active_elements = actual_lengths.float().unsqueeze(-1).clamp(min=1.0)
        x_pooled = summed_features / num_active_elements  # (B, dim)

        return self.to_latent(x_pooled)  # (B, dim)


class PolygonPredictor(nn.Module):
    def __init__(self, cfg: PolygonConfig):
        super().__init__()
        self.cfg = cfg

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(cfg.t5_model_name)
            full_t5_model = T5EncoderModel.from_pretrained(cfg.t5_model_name)
            self.encoder = full_t5_model.encoder
            d_text = full_t5_model.config.hidden_size
            for p in self.encoder.parameters():
                p.requires_grad = False
        except ImportError:
            print(f"Warning: transformers library not found. Mocking T5 components.")
            self.tokenizer = None; self.encoder = nn.Identity(); d_text = 768
        except Exception as e:
            print(f"Error loading T5 model '{cfg.t5_model_name}': {e}. Mocking T5 components.")
            self.tokenizer = None; self.encoder = nn.Identity(); d_text = 768

        self.text_proj = nn.Linear(d_text, cfg.d_model)

        self.shape_encoder = SimpleShapeEncoder(
            in_dim=6,
            dim=cfg.d_model,
            seq_len=cfg.max_segments
        )

        self.modality_type_embeddings = nn.Parameter(torch.randn(3, cfg.d_model))

        self.text_enhancement = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model),
            nn.GELU(),
            nn.Linear(cfg.d_model, cfg.d_model)
        )

        self.dynamic_pos_weights = nn.Parameter(torch.randn(3, cfg.d_model) * 0.1)

        fusion_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model, nhead=cfg.n_head,
            dim_feedforward=cfg.dim_feedforward, dropout=cfg.dropout,
            batch_first=True, activation=F.gelu
        )
        self.fusion_enc = nn.TransformerEncoder(
            fusion_layer, num_layers=cfg.num_fusion_layers,
            norm=nn.LayerNorm(cfg.d_model)
        )

        self.coord_decoder = ShapePredictor(
            d_model=cfg.d_model,
            num_heads=cfg.n_head,
            num_decoder_layers=cfg.num_dec_layers_seq,
            num_segments=cfg.max_segments,
            dim_feedforward=cfg.dim_feedforward,
            dropout=cfg.dropout,
            out_dim=6
        )

        self.coordinate_head = nn.Sequential(
            nn.Linear(cfg.d_model, 2 * cfg.d_model),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(2 * cfg.d_model, cfg.d_model),
            nn.ReLU(),
            nn.Linear(cfg.d_model, 6)
        )

        self.type_head = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model // 2),
            nn.ReLU(),
            nn.Linear(cfg.d_model // 2, 3)
        )
        self.stop_head = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.max_segments),
            nn.Sigmoid()
        )

    def encode_text_embeddings(self, text_embeddings: torch.Tensor) -> torch.Tensor:
        if text_embeddings.dim() == 3:
            B, S, D = text_embeddings.shape

            content_scores = torch.sum(text_embeddings ** 2, dim=-1)
            attention_weights = torch.softmax(content_scores, dim=-1)
            attention_pooled = torch.sum(text_embeddings * attention_weights.unsqueeze(-1), dim=1)

            pos_weights = torch.exp(-0.1 * torch.abs(
                torch.arange(S, device=text_embeddings.device, dtype=torch.float) - S/2
            ))
            pos_weights = pos_weights / pos_weights.sum()
            pos_pooled = torch.sum(text_embeddings * pos_weights.unsqueeze(0).unsqueeze(-1), dim=1)

            variance_weights = torch.var(text_embeddings, dim=-1) + 1e-8
            variance_weights = torch.softmax(variance_weights, dim=-1)
            variance_pooled = torch.sum(text_embeddings * variance_weights.unsqueeze(-1), dim=1)

            max_pooled, _ = torch.max(text_embeddings, dim=1)
            min_pooled, _ = torch.min(text_embeddings, dim=1)

            chunk_size = max(1, S // 4)
            chunks = []
            for i in range(0, S, chunk_size):
                chunk = text_embeddings[:, i:i+chunk_size]
                if chunk.size(1) > 0:
                    chunks.append(chunk.mean(dim=1))
            while len(chunks) < 4:
                chunks.append(chunks[-1] if chunks else torch.zeros_like(attention_pooled))

            combined_features = torch.cat([
                attention_pooled,
                pos_pooled,
                variance_pooled,
                max_pooled,
                min_pooled,
                chunks[0],
                chunks[1],
                chunks[2],
                chunks[3]
            ], dim=-1)  # [B, 9D]

            proj1 = F.gelu(self.text_proj(combined_features[:, :D]))
            proj2 = F.gelu(self.text_proj(combined_features[:, D:2*D]))
            proj3 = F.gelu(self.text_proj(combined_features[:, 2*D:3*D]))

            intermediate = proj1 + 0.5 * proj2 + 0.3 * proj3

        elif text_embeddings.dim() == 2:
            B, D = text_embeddings.shape
            view1 = text_embeddings
            view2 = torch.roll(text_embeddings, shifts=1, dims=-1)
            view3 = text_embeddings * torch.sin(
                torch.arange(D, device=text_embeddings.device).float()
            )
            combined = torch.cat([view1, view2, view3], dim=-1)
            intermediate = F.gelu(self.text_proj(combined[:, :D]))
        else:
            raise ValueError(f"Unexpected text_embeddings dim: {text_embeddings.shape}")

        enhanced = self.text_enhancement(intermediate)

        input_signature = torch.sum(text_embeddings.view(text_embeddings.size(0), -1), dim=-1, keepdim=True)

        hash1 = torch.sin(input_signature * 7.3) * torch.arange(1, self.cfg.d_model + 1, device=enhanced.device).float()
        hash2 = torch.cos(input_signature * 11.7) * torch.arange(1, self.cfg.d_model + 1, device=enhanced.device).float() * 0.5
        hash3 = torch.tanh(input_signature * 3.1) * torch.arange(1, self.cfg.d_model + 1, device=enhanced.device).float() * 0.3

        final_features = enhanced + 0.8 * hash1 + 0.6 * hash2 + 0.4 * hash3

        return final_features

    def encode_parent_shape(self, parent_bezier_data: torch.Tensor) -> torch.Tensor:
        base_encoding = self.shape_encoder(parent_bezier_data)

        B = parent_bezier_data.size(0)
        valid_coords = parent_bezier_data[parent_bezier_data != -1]
        if valid_coords.numel() > 0:
            coord_variance = torch.var(valid_coords)
            coord_mean = torch.mean(valid_coords)
            coord_range = torch.max(valid_coords) - torch.min(valid_coords)
            shape_stats = torch.stack([coord_variance, coord_mean, coord_range]).unsqueeze(0).expand(B, -1)
        else:
            shape_stats = torch.zeros(B, 3, device=parent_bezier_data.device)

        shape_signatures = []
        for b in range(B):
            shape_data = parent_bezier_data[b]
            valid_mask = (shape_data != -1).any(dim=-1)
            if valid_mask.any():
                shape_sum = torch.sum(shape_data[valid_mask])
                shape_count = valid_mask.sum().float()
                shape_avg = shape_sum / (shape_count + 1e-8)
            else:
                shape_avg = torch.tensor(0.0, device=parent_bezier_data.device)
            shape_signatures.append(shape_avg)
        shape_signature_tensor = torch.stack(shape_signatures).unsqueeze(-1)

        hash_base = shape_signature_tensor * 100.0
        d_model = self.cfg.d_model
        shape_hash1 = torch.sin(hash_base * 13.7) * torch.arange(1, d_model + 1, device=base_encoding.device).float()
        shape_hash2 = torch.cos(hash_base * 19.3) * torch.arange(1, d_model + 1, device=base_encoding.device).float() * 0.7
        shape_hash3 = torch.tanh(hash_base * 7.1) * torch.arange(1, d_model + 1, device=base_encoding.device).float() * 0.5

        stats_expansion = torch.sin(
            shape_stats.unsqueeze(-1) * torch.arange(1, d_model + 1, device=base_encoding.device).float()
        ) * 0.3
        stats_features = stats_expansion.mean(dim=1)

        enhanced_shape_encoding = (
            base_encoding +
            0.9 * shape_hash1 +
            0.7 * shape_hash2 +
            0.5 * shape_hash3 +
            0.4 * stats_features
        )
        return enhanced_shape_encoding

    def fuse_modalities(self, c_feat, p_feat, s_feat) -> torch.Tensor:
        B = c_feat.size(0)

        if self.training:
            noise_scale = 0.1
            c_feat = c_feat + torch.randn_like(c_feat) * noise_scale
            p_feat = p_feat + torch.randn_like(p_feat) * noise_scale
            s_feat = s_feat + torch.randn_like(s_feat) * noise_scale

        pos_c = torch.tanh(c_feat.mean(dim=-1, keepdim=True)) * 0.5
        pos_p = torch.tanh(p_feat.mean(dim=-1, keepdim=True)) * 0.5
        pos_s = torch.tanh(s_feat.mean(dim=-1, keepdim=True)) * 0.5

        mods = torch.stack([c_feat, p_feat, s_feat], dim=1)
        positions = torch.stack([pos_c, pos_p, pos_s], dim=1)

        fixed_pos = self.modality_type_embeddings.unsqueeze(0)
        dynamic_pos = positions * self.dynamic_pos_weights.unsqueeze(0)

        mods = mods + fixed_pos + dynamic_pos

        fused_memory = self.fusion_enc(mods)
        fused_memory = fused_memory + mods * 0.2

        return fused_memory

    def forward(self,
                child_embs: torch.Tensor,
                parent_embs: torch.Tensor,
                parent_bezier: torch.Tensor,
               ) -> Dict[str, torch.Tensor]:

        device = parent_bezier.device
        child_embs = child_embs.to(device)
        parent_embs = parent_embs.to(device)

        c_feat = self.encode_text_embeddings(child_embs)
        p_feat = self.encode_text_embeddings(parent_embs)
        s_feat = self.encode_parent_shape(parent_bezier)

        mem = self.fuse_modalities(c_feat, p_feat, s_feat)

        # First, run coord_decoder(mem). It might return either:
        #   (coords, type_logits, stop_logits)   <-- if ShapePredictor.forward
        # or a dictionary <-- if someone replaced coord_decoder with a custom method
        decoder_out = self.coord_decoder(mem)

        if isinstance(decoder_out, dict):
            coords_logits = decoder_out["coords_normalized"]
            type_logits   = decoder_out["types_logits"]
            stop_logits   = decoder_out["stop_scores"]
        else:
            # assume a length‐3 tuple
            coords_logits, type_logits, stop_logits = decoder_out

        # Now normalize + scale coords_logits exactly as before,
        # but with the corrected scalar perturbation.
        B, T_seg, _ = coords_logits.shape
        coords_scaled = []
        for i in range(T_seg):
            segment_feats = coords_logits[:, i]  # (B, 6)

            # Compute a scalar perturbation (instead of a full d_model–sized vector)
            scalar_input = torch.sin((mem.mean(dim=1).sum(dim=1) * (i + 1) * 0.3))  # (B,)
            scalar_input = scalar_input.unsqueeze(1)  # (B, 1), will broadcast to (B, 6)

            i_tensor = torch.tensor(i, dtype=torch.float, device=device)
            segment_scale = 0.8 + 0.4 * torch.sin(i_tensor * 1.7)
            segment_offset = 0.1 * torch.cos(i_tensor * 2.3)

            seg_norm = torch.tanh(segment_feats) * segment_scale + segment_offset
            seg_norm = seg_norm + scalar_input * 0.15
            seg_norm = torch.sigmoid(seg_norm)  # (B, 6)

            coords_scaled.append(seg_norm)

        coords_scaled = torch.stack(coords_scaled, dim=1)  # (B, T_seg, 6)

        # Stop scores by sigmoid
        stop_scores = torch.sigmoid(stop_logits)
        position_weights = torch.arange(stop_scores.size(1), device=stop_scores.device, dtype=torch.float)
        position_bias = position_weights * 0.1
        adjusted_stop_scores = stop_scores + position_bias.unsqueeze(0)

        exceed_threshold = (adjusted_stop_scores > 0.6).float()
        stop_index = torch.where(
            exceed_threshold.sum(dim=1) > 0,
            torch.argmax(exceed_threshold, dim=1),
            torch.argmax(adjusted_stop_scores, dim=1)
        )

        return {
            "segments": coords_scaled,
            "type_logits": type_logits,
            "stop_scores": stop_scores,
            "adjusted_stop_scores": adjusted_stop_scores,
            "stop_index": stop_index,
        }

# -----------------------------------------------------------------------------  
# Compute Loss: Unified with Repulsion, Mask-Penalty, Type Rebalance  
# -----------------------------------------------------------------------------  
def compute_loss(
    pred_coords:   torch.Tensor,    # (B, S, 6)
    pred_type_logits: torch.Tensor, # (B, S, 3)
    pred_stop_logits: torch.Tensor, # (B, S) ← post‐sigmoid “scores”
    gt_coords:     torch.Tensor,    # (B, S, 6)
    gt_type_labels: torch.Tensor,   # (B, S), values in {0,1,2}; masked segments set to 0
    gt_stop_indices: torch.Tensor,  # (B,)
    lambda_curve: float = 1.0,
    lambda_type:  float = 0.5,    # reduced from 1.0 → 0.5 to de‐bias “always‐cubic”
    lambda_stop:  float = 0.1,
    lambda_uniqueness: float = 200.0,
    lambda_repulsion: float = 300.0,
    lambda_mask: float = 500.0,
    repulsion_delta: float = 0.20,  # desired min L∞ distance between two valid segments
):
    """
    Returns a single scalar `loss` that combines:
      1. curve L1 loss on all valid control‐points,
      2. cross‐entropy loss on type predictions,
      3. cross‐entropy loss on stop predictions (we’ll treat post‐sigmoid scores as “logits”),
      4. “uniqueness” penalty (to discourage exact copy‐paste across adjacent segments),
      5. “repulsion” penalty (to push ANY two valid segments at least repulsion_delta apart),
      6. “mask‐coordinate” penalty (to force extra coords→–1 when GT type != cubic).
    """

    B, S, _ = pred_coords.shape
    device = pred_coords.device

    # 1) Binary mask for “valid” vs “masked” segments:
    #    A segment is “valid” if its gt_coords != all –1’s. We assume masked segments are exactly (-1,…,-1).
    with torch.no_grad():
        is_masked = (gt_coords == -1.0).all(dim=-1)      # True = masked
        is_valid  = ~is_masked                           # True = valid

    ##### (1) Curve‐coordinate L1 loss on valid segments only #####
    coord_diff = torch.abs(pred_coords - gt_coords)      # (B, S, 6)
    curve_l1 = (coord_diff * is_valid.unsqueeze(-1).float()).sum() \
               / (is_valid.sum().clamp(min=1) * 6.0)     # average per‐coordinate L1

    ##### (2) Type‐classification cross‐entropy #####
    flat_logits = pred_type_logits.view(B*S, 3)          # (B*S, 3)
    flat_labels = gt_type_labels.view(B*S)               # (B*S,)
    flat_valid_mask = is_valid.view(-1)                  # (B*S,)
    if flat_valid_mask.sum() > 0:
        ce_loss = F.cross_entropy(
            flat_logits[flat_valid_mask],
            flat_labels[flat_valid_mask]
        )
    else:
        ce_loss = torch.tensor(0.0, device=device)

    ##### (3) Stop‐prediction cross‐entropy #####
    stop_loss = 0.0
    for b in range(B):
        true_stop = gt_stop_indices[b].clamp(min=0, max=S-1)
        stop_loss += F.cross_entropy(
            pred_stop_logits[b].unsqueeze(0),      # shape (1, S)
            true_stop.unsqueeze(0)                # shape (1,)
        )
    stop_loss = stop_loss / B

    ##### (4) Adjacency “uniqueness” penalty #####
    eps = 1e-6
    uniq_loss = 0.0
    count_pairs = 0
    for b in range(B):
        for i in range(S-1):
            if is_valid[b, i] and is_valid[b, i+1]:
                diff = pred_coords[b, i] - pred_coords[b, i+1]        # (6,)
                dist2 = (diff**2).sum() + eps
                uniq_loss += 1.0 / dist2
                count_pairs += 1
    if count_pairs > 0:
        uniq_loss = uniq_loss / count_pairs
    else:
        uniq_loss = torch.tensor(0.0, device=device)

    ##### (5) Pairwise repulsion among ANY two valid segments #####
    repel_loss = 0.0
    count_repels = 0
    for b in range(B):
        valid_indices = torch.nonzero(is_valid[b], as_tuple=False).view(-1)
        for idx_i in range(len(valid_indices)):
            i = valid_indices[idx_i].item()
            for idx_j in range(idx_i+1, len(valid_indices)):
                j = valid_indices[idx_j].item()
                coord_i = pred_coords[b, i]
                coord_j = pred_coords[b, j]
                linf = torch.max(torch.abs(coord_i - coord_j))
                gap = F.relu(repulsion_delta - linf)
                repel_loss += gap * gap
                count_repels += 1
    if count_repels > 0:
        repel_loss = repel_loss / count_repels
    else:
        repel_loss = torch.tensor(0.0, device=device)

    ##### (6) “Mask‐coordinate” penalty #####
    mask_coord_loss = 0.0
    count_masks = 0
    for b in range(B):
        for s in range(S):
            gt_t = gt_type_labels[b, s].item()
            if is_valid[b, s] and gt_t != 2:  # if ground‐truth says “not cubic”
                diff = pred_coords[b, s] - (-1.0)
                mask_coord_loss += diff.abs().sum()
                count_masks += 1
    if count_masks > 0:
        mask_coord_loss = mask_coord_loss / (count_masks * 6.0)
    else:
        mask_coord_loss = torch.tensor(0.0, device=device)

    ##### Combine everything #####
    loss = (
        lambda_curve       * curve_l1
      + lambda_type        * ce_loss
      + lambda_stop        * stop_loss
      + lambda_uniqueness  * uniq_loss
      + lambda_repulsion   * repel_loss
      + lambda_mask        * mask_coord_loss
    )
    return loss, {
        "curve_l1"       : curve_l1.item(),
        "type_ce"        : ce_loss.item(),
        "stop_ce"        : stop_loss.item(),
        "uniq_penalty"   : uniq_loss.item(),
        "repel_penalty"  : repel_loss.item(),
        "mask_penalty"   : mask_coord_loss.item(),
    }

# -----------------------------------------------------------------------------  
# Model Training Utilities  
# -----------------------------------------------------------------------------  
def setup_logging(output_dir):
    """Setup logging configuration"""
    log_dir = os.path.join(output_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'training_{timestamp}.log')

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def save_checkpoint(model, optimizer, epoch, current_best_loss, checkpoint_path):
    unwrapped_model = model
    if hasattr(model, "_orig_mod"):
        unwrapped_model = model._orig_mod

    full_state_dict = unwrapped_model.state_dict()

    filtered_state_dict = {
        k: v for k, v in full_state_dict.items()
        if not k.startswith("encoder.")
           and not k.startswith("_orig_mod.encoder.")
    }

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": filtered_state_dict,
        'optimizer_state_dict': optimizer.state_dict(),
        'best_loss': current_best_loss,
    }

    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}")
    return checkpoint

def post_training_test(model, dataset, device, num_samples=5):
    """
    Comprehensive post-training test to verify model behavior.
    Tests coordinate ranges, segment types, and output diversity.
    """
    model.eval()
    results = {
        'coordinate_ranges': [],
        'segment_types': [],
        'stop_indices': [],
        'diversity_scores': [],
        'type_distribution': [],
        'coordinate_validity': [],
        'segment_continuity': [],
        'coordinate_precision': []
    }

    print("\n=== Post-Training Test Results ===")

    indices = random.sample(range(len(dataset)), min(len(dataset), num_samples))

    for i in indices:
        sample = dataset[i]
        ce = sample['child_embs'].unsqueeze(0).to(device)
        pe = sample['parent_embs'].unsqueeze(0).to(device)
        pseg = sample['parent_bezier_segs'].unsqueeze(0).to(device)
        gt   = sample['gt_curves']

        with torch.no_grad():
            out = model(ce, pe, pseg)

        pred_segs  = out['segments'][0].cpu().numpy()
        pred_types = out['type_logits'][0].cpu().numpy().argmax(-1)
        stop_idx = int(out['stop_index'][0].item())

        valid_coords = pred_segs[pred_segs != -1]
        if len(valid_coords) > 0:
            coord_min = float(valid_coords.min())
            coord_max = float(valid_coords.max())
            results['coordinate_ranges'].append((coord_min, coord_max))
            print(f"\nSample {i} - Coordinate Range: [{coord_min:.3f}, {coord_max:.3f}]")

            if coord_min < 0 or coord_max > 1:
                print(f"WARNING: Coordinates outside [0,1] range!")

        type_counts = np.bincount(pred_types[:stop_idx+1], minlength=3)
        results['segment_types'].append(type_counts.tolist())
        print(f"Segment Types: Line={type_counts[0]}, Quad={type_counts[1]}, Cubic={type_counts[2]}")

        results['stop_indices'].append(int(stop_idx))
        print(f"Stop Index: {stop_idx}")

        if len(valid_coords) > 0:
            coord_variance = float(np.var(valid_coords))
            results['diversity_scores'].append(coord_variance)
            print(f"Diversity Score (variance): {coord_variance:.3f}")

        print("\nCoordinate Structure Analysis:")
        gt_segs = gt.cpu().numpy()
        gt_ends   = gt_segs[:,4:6]
        gt_starts = np.roll(gt_ends, 1, axis=0);  gt_starts[0]  = gt_ends[-1]
        pr_ends = pred_segs[:,4:6]
        pr_starts = np.roll(pr_ends, 1, axis=0); pr_starts[0] = pr_ends[-1]

        valid_segments = 0
        prev_end = None
        continuity_errors = []

        for j, seg in enumerate(pred_segs[:stop_idx+1]):
            if (seg == -1).all():
                continue
            valid_segments += 1
            print(f"Segment {j}:")
            print(f"  Type: {pred_types[j]}")
            print(f"  Coords: {seg.tolist()}")

            if prev_end is not None:
                current_start = seg[:2]
                if not np.allclose(prev_end, current_start, atol=1e-4):
                    continuity_errors.append(j)
            prev_end = seg[-2:]

            if pred_types[j] == 0:
                assert (seg[:4] == -1).all(), f"Line segment {j} should have -1 for control points"
            elif pred_types[j] == 1:
                assert (seg[:2] == -1).all(), f"Quad segment {j} should have -1 for first control point"
                assert (seg[2:4] != -1).any(), f"Quad segment {j} should have valid second control point"
            elif pred_types[j] == 2:
                assert (seg[:6] != -1).any(), f"Cubic segment {j} should have valid control points"

        results['segment_continuity'].append({
            'valid_segments': valid_segments,
            'continuity_errors': continuity_errors
        })

        results['coordinate_validity'].append({
            'all_valid': bool(np.all((valid_coords >= 0) & (valid_coords <= 1))) if len(valid_coords)>0 else True,
            'has_negative': bool(np.any(valid_coords < 0)) if len(valid_coords)>0 else False,
            'has_above_one': bool(np.any(valid_coords > 1)) if len(valid_coords)>0 else False
        })

        precision_stats = {}
        if len(valid_coords) > 0:
            precision_stats = {
                'min_precision': float(np.min(np.abs(valid_coords))),
                'max_precision': float(np.max(np.abs(valid_coords))),
                'mean_precision': float(np.mean(np.abs(valid_coords)))
            }
        else:
            precision_stats = {'min_precision': 0.0, 'max_precision':0.0, 'mean_precision':0.0}
        results['coordinate_precision'].append(precision_stats)

    print("\n=== Aggregate Test Results ===")
    if results['coordinate_ranges']:
        all_mins, all_maxs = zip(*results['coordinate_ranges'])
        print(f"Overall Coordinate Range: [{min(all_mins):.3f}, {max(all_maxs):.3f}]")

    if results['segment_types']:
        avg_types = np.mean(results['segment_types'], axis=0)
        print(f"Average Segment Type Distribution: Line={avg_types[0]:.1f}, Quad={avg_types[1]:.1f}, Cubic={avg_types[2]:.1f}")

    if results['diversity_scores']:
        print(f"Average Diversity Score: {np.mean(results['diversity_scores']):.3f}")

    validity_issues = [r for r in results['coordinate_validity'] if not r['all_valid']]
    if validity_issues:
        print("\nWARNING: Found coordinate validity issues!")
        for i, issue in enumerate(validity_issues):
            print(f"Sample {i}:")
            if issue['has_negative']:
                print("  - Contains negative coordinates")
            if issue['has_above_one']:
                print("  - Contains coordinates > 1")

    continuity_issues = [r for r in results['segment_continuity'] if r['continuity_errors']]
    if continuity_issues:
        print("\nWARNING: Found segment continuity issues!")
        for i, issue in enumerate(continuity_issues):
            if issue['continuity_errors']:
                print(f"Sample {i}:")
                print(f"  - Discontinuities at segments: {issue['continuity_errors']}")

    return results

def visualize_predictions(model, dataset, device, output_dir, epoch, max_vis=20):
    """Visualize predicted vs ground-truth shapes."""
    def render_segment(start_pt, seg, seg_type, steps=50):
        if seg_type == 0:
            end_pt = seg[-2:]
            return np.vstack([start_pt, end_pt])
        elif seg_type == 1:
            ctrl_pt, end_pt = seg[-4:-2], seg[-2:]
            t = np.linspace(0,1,steps)[:,None]
            B0 = (1-t)**2; B1 = 2*(1-t)*t; B2 = t**2
            return B0*start_pt + B1*ctrl_pt + B2*end_pt
        else:
            p1, p2, p3 = seg[:2], seg[2:4], seg[4:6]
            t = np.linspace(0,1,steps)[:,None]
            B0 = (1-t)**3; B1 = 3*(1-t)**2*t
            B2 = 3*(1-t)*t**2; B3 = t**3
            return B0*start_pt + B1*p1 + B2*p2 + B3*p3

    vis_dir = os.path.join(output_dir, f"vis_epoch_{epoch}")
    os.makedirs(vis_dir, exist_ok=True)
    model.to(device).eval()

    indices = random.sample(range(len(dataset)), min(len(dataset), max_vis))
    for i in indices:
        sample = dataset[i]
        try:
            ce = sample['child_embs'].unsqueeze(0).to(device)
            pe = sample['parent_embs'].unsqueeze(0).to(device)
            bb = sample['parent_bbox'].unsqueeze(0).to(device)
            pseg = sample['parent_bezier_segs'].unsqueeze(0).to(device)
            gt   = sample['gt_curves']

            with torch.no_grad():
                out = model(ce, pe, pseg)

            pred_segs  = out['segments'][0].cpu().numpy()
            pred_types = out['type_logits'][0].cpu().numpy().argmax(-1)
            gt_segs    = gt.cpu().numpy()

            gt_ends   = gt_segs[:,4:6]; pred_ends = pred_segs[:,4:6]
            gt_starts = np.roll(gt_ends, 1, axis=0);  gt_starts[0]  = gt_ends[-1]
            pr_starts = np.roll(pred_ends, 1, axis=0); pr_starts[0] = pred_ends[-1]

            fig, ax = plt.subplots(figsize=(6,6))
            ax.set_aspect('equal','box')
            ax.set_xlim(-0.05,1.05); ax.set_ylim(-0.05,1.05)
            ax.set_title(f"Sample {i}")

            # Plot GT in green
            for idx, (seg, st) in enumerate(zip(gt_segs, gt_starts)):
                negs = int((seg[:4]==-1).sum())
                stype = 0 if negs==4 else 1 if negs==2 else 2
                curve = render_segment(st, seg, stype, steps=100)
                ax.plot(curve[:,0], curve[:,1], '-', color='green', lw=2, label='GT' if idx==0 else "")

            # Plot Pred in red
            for idx, (seg, st, stype) in enumerate(zip(pred_segs, pr_starts, pred_types)):
                curve = render_segment(st, seg, stype, steps=100)
                ax.plot(curve[:,0], curve[:,1], '--', color='red', lw=2, label='Pred' if idx==0 else "")

            ax.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(vis_dir, f"sample_{i}.png"), dpi=120)
            plt.close(fig)

        except Exception as e:
            print(f"Error visualizing sample {i}: {e}")
            traceback.print_exc()

    model.train()

# -----------------------------------------------------------------------------  
# Sanity-Check Functions for Debugging (unchanged from previous update)  
# -----------------------------------------------------------------------------  
def check_coordinate_variation(model, dataset, device, num_samples=20):
    """
    Run the model on `num_samples` random inputs from `dataset` and verify:
      1. The coordinate outputs are not identical across samples.
      2. All coordinates live in [0,1].
      3. The variance of coordinates is not vanishingly small.
    """
    model.eval()
    coords_list = []
    with torch.no_grad():
        for i in np.random.choice(len(dataset), size=min(len(dataset), num_samples), replace=False):
            sample = dataset[i]
            ce = sample['child_embs'].unsqueeze(0).to(device)
            pe = sample['parent_embs'].unsqueeze(0).to(device)
            pseg = sample['parent_bezier_segs'].unsqueeze(0).to(device)
            out = model(ce, pe, pseg)
            pred_segs = out['segments'][0].cpu().numpy()  # shape (S, 6)
            coords_list.append(pred_segs.flatten())

    coords_array = np.stack(coords_list, axis=0)  # (num_samples, S*6)

    # 1) Check if all outputs are exactly the same
    all_identical = np.allclose(coords_array, coords_array[0], atol=1e-6)
    print(f"[Sanity] Are all coordinate vectors identical? {all_identical}")

    # 2) Check coordinate range
    coord_min, coord_max = coords_array.min(), coords_array.max()
    print(f"[Sanity] Min coordinate across samples: {coord_min:.5f}, Max: {coord_max:.5f}")
    if coord_min < 0 or coord_max > 1:
        print("  → WARNING: Some coordinates lie outside [0,1].")

    # 3) Check overall coordinate variance
    total_var = coords_array.var(axis=0).mean()
    print(f"[Sanity] Mean per‐coordinate variance across samples: {total_var:.8f}")
    if total_var < 1e-4:
        print("  → WARNING: Coordinate variance is extremely low. Model may be predicting nearly constant outputs.")

    return {
        "all_identical": bool(all_identical),
        "coord_min": float(coord_min),
        "coord_max": float(coord_max),
        "mean_variance": float(total_var),
    }

def check_type_diversity(model, dataset, device, num_samples=20):
    """
    Run the model on `num_samples` random inputs from `dataset` and tally:
      - How often each segment‐type (Line/Quad/Cubic) is predicted,
      - Whether some inputs always collapse to the same type distribution.
    """
    model.eval()
    from collections import Counter
    type_counts = Counter()
    per_sample_distributions = []

    with torch.no_grad():
        for i in np.random.choice(len(dataset), size=min(len(dataset), num_samples), replace=False):
            sample = dataset[i]
            ce = sample['child_embs'].unsqueeze(0).to(device)
            pe = sample['parent_embs'].unsqueeze(0).to(device)
            pseg = sample['parent_bezier_segs'].unsqueeze(0).to(device)
            out = model(ce, pe, pseg)
            preds = out['type_logits'][0].cpu().numpy().argmax(-1)  # (S,)
            stop_idx = int(out['stop_index'][0].item())
            valid_types = preds[: stop_idx + 1]
            counts = Counter(valid_types.tolist())
            per_sample_distributions.append(counts)
            for t, c in counts.items():
                type_counts[t] += c

    total_segments = sum(type_counts.values())
    dist = {
        "Line": type_counts[0]/total_segments if total_segments>0 else 0.0,
        "Quad": type_counts[1]/total_segments if total_segments>0 else 0.0,
        "Cubic": type_counts[2]/total_segments if total_segments>0 else 0.0
    }
    print(f"[Sanity] Overall predicted‐type distribution (aggregated): {dist}")

    unique_patterns = { tuple(sorted(d.items())) for d in per_sample_distributions }
    print(f"[Sanity] How many unique per‐input type‐histograms? {len(unique_patterns)} "
          f"(out of {len(per_sample_distributions)} sampled)")

    if len(unique_patterns) == 1:
        print("  → WARNING: Every input gives exactly the same type‐histogram.")

    return {
        "aggregate_distribution": dist,
        "unique_sample_histograms": len(unique_patterns),
    }

def check_stop_index_distribution(model, dataset, device, num_samples=20):
    """
    Verify that the predicted stop indices are not always the same.
    """
    model.eval()
    stops = []
    with torch.no_grad():
        for i in np.random.choice(len(dataset), size=min(len(dataset), num_samples), replace=False):
            sample = dataset[i]
            ce = sample['child_embs'].unsqueeze(0).to(device)
            pe = sample['parent_embs'].unsqueeze(0).to(device)
            pseg = sample['parent_bezier_segs'].unsqueeze(0).to(device)
            out = model(ce, pe, pseg)
            stops.append(int(out['stop_index'][0].item()))

    stops = np.array(stops)
    print(f"[Sanity] Predicted stop‐indices (sampled): {stops.tolist()}")
    if np.all(stops == stops[0]):
        print("  → WARNING: All stop indices are identical.")
    return {
        "unique_stop_indices": int(len(np.unique(stops))),
        "all_same": bool(np.all(stops == stops[0])),
    }

def sanity_check_full(
    model: torch.nn.Module,
    dataset,
    device: torch.device,
    num_samples: int = 20
):
    """
    Run all sanity checks together and return a summary dict.
    """
    print("=== Running full model‐sanity checks ===")
    coord_stats = check_coordinate_variation(model, dataset, device, num_samples=num_samples)
    type_stats  = check_type_diversity(model, dataset, device, num_samples=num_samples)
    stop_stats  = check_stop_index_distribution(model, dataset, device, num_samples=num_samples)

    summary = {
        "coordinate_stats": coord_stats,
        "type_stats": type_stats,
        "stop_stats": stop_stats,
    }
    return summary

# -----------------------------------------------------------------------------  
# Training Function: Batched  
# -----------------------------------------------------------------------------  
def train_model_batched(
    dataset_path,
    model_name=None,
    output_dir="bezier_checkpoints_overfit",
    num_epochs=100,
    learning_rate=5e-4,
    batch_size=None,
    max_samples=None,
    run_visualization=False,
    run_post_training_test=True
):
    logger = setup_logging(output_dir)
    logger.info("Starting training process")
    logger.info(f"Configuration: epochs={num_epochs}, lr={learning_rate}, batch_size={batch_size}, max_samples={max_samples}")

    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    ds = AugmentedDataset(root_dir=dataset_path, max_samples=max_samples)
    if len(ds) == 0:
        logger.error("Empty dataset!")
        raise RuntimeError("Empty dataset!")

    N = len(ds)
    actual_batch_size = N if batch_size is None else batch_size
    logger.info(f"Dataset size: {N}, Batch size: {actual_batch_size}")

    loader = DataLoader(
        ds,
        batch_size=actual_batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
        collate_fn=collate_fn
    )

    cfg = PolygonConfig(
        d_model=256,
        n_head=8,
        num_dec_layers_seq=6,
        dim_feedforward=1024,
        max_segments=30,
        dropout=0.1,
        num_fusion_layers=3
    )

    logger.info(f"Model configuration: {cfg}")
    model = PolygonPredictor(cfg=cfg).to(device)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate,
        betas=(0.9, 0.999),
        weight_decay=0.01
    )
    logger.info(f"Optimizer: AdamW with lr={learning_rate}, weight_decay=0.01")

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5
    )

    start_epoch = 1
    best_loss = float('inf')
    patience = 10
    no_improve_epochs = 0

    if model_name and os.path.exists(model_name):
        logger.info(f"Resuming training from checkpoint: {model_name}")
        ckpt = torch.load(model_name, map_location=device)

        model_to_load = model
        if hasattr(model, "_orig_mod"):
            model_to_load = model._orig_mod

        missing_keys, unexpected_keys = model_to_load.load_state_dict(ckpt["model_state_dict"], strict=False)

        if missing_keys:
            logger.warning(f"Missing keys: {missing_keys}")
        if unexpected_keys:
            logger.warning(f"Unexpected keys: {unexpected_keys}")

        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            for state in optimizer.state.values():
                for k_opt, v_opt in state.items():
                    if isinstance(v_opt, torch.Tensor):
                        state[k_opt] = v_opt.to(device)

        if "epoch" in ckpt:
            start_epoch = ckpt['epoch'] + 1

        if "best_loss" in ckpt:
            best_loss = ckpt['best_loss']

        logger.info(f"Loaded checkpoint successfully. Best loss was: {best_loss}")
    else:
        logger.info("Training from scratch.")

    best_model_path = os.path.join(output_dir, "best_model.pth")

    logger.info(f"Starting training from epoch {start_epoch} up to {num_epochs}")
    logger.info(f"Batch size: {actual_batch_size}. Training on {N} samples.")

    for epoch in range(start_epoch, num_epochs + 1):
        model.train()
        epoch_total_loss = 0.0
        epoch_start_time = time.time()

        for batch_idx, batch_data in enumerate(loader):
            loss_value = train_batch(
                model, batch_data, optimizer, device, batch_idx,
                lambda_curve=600.0,
                lambda_stop=5.0,
                lambda_type=300.0,
                lambda_uniqueness=200.0,
                lambda_repulsion=300.0,
                lambda_mask=500.0,
                repulsion_delta=0.20,
                debug_mode=True
            )
            epoch_total_loss += loss_value

        avg_epoch_loss = epoch_total_loss / len(loader) if len(loader) > 0 else float('nan')
        epoch_time = time.time() - epoch_start_time

        logger.info(f"Epoch {epoch:3d}/{num_epochs} - Loss: {avg_epoch_loss:.6f} - Time: {epoch_time:.2f}s")

        scheduler.step(avg_epoch_loss)
        current_lr = optimizer.param_groups[0]['lr']
        logger.info(f"Current learning rate: {current_lr:.6f}")

        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            no_improve_epochs = 0
            logger.info(f"New best model at epoch {epoch}, loss {best_loss:.6f}")
            save_checkpoint(model, optimizer, epoch, best_loss, best_model_path)
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= patience:
                logger.info(f"Early stopping triggered after {patience} epochs without improvement")
                break

        if avg_epoch_loss < 1e-6:
            logger.info(f"Perfect overfit achieved at epoch {epoch}!")
            break

        if run_visualization and epoch % 5 == 0:
            visualize_predictions(model, ds, device, output_dir, epoch)

    final_path = os.path.join(output_dir, "final_model.pth")
    save_checkpoint(model, optimizer, num_epochs, best_loss, final_path)
    logger.info(f"Saved final model to {final_path}")

    if run_post_training_test and os.path.exists(best_model_path):
        logger.info("Running post-training tests...")
        best_ckpt = torch.load(best_model_path, map_location=device)
        test_model = PolygonPredictor(cfg=cfg).to(device)
        test_model.load_state_dict(best_ckpt["model_state_dict"], strict=False)
        test_results = post_training_test(test_model, ds, device)

        test_results_path = os.path.join(output_dir, "post_training_test_results.json")
        with open(test_results_path, 'w') as f:
            json.dump(test_results, f, indent=2)
        logger.info(f"Post-training test results saved to {test_results_path}")

    return model

def train_batch(
    model,
    batch,
    optimizer,
    device,
    batch_idx,
    lambda_curve: float = 500.0,
    lambda_stop: float = 10.0,
    lambda_type: float = 200.0,
    lambda_uniqueness: float = 200.0,
    lambda_repulsion: float = 300.0,
    lambda_mask: float = 500.0,
    repulsion_delta: float = 0.20,
    debug_mode: bool = True,
):
    logger = logging.getLogger(__name__)

    child_embs = batch['child_embs'].to(device)
    parent_embs = batch['parent_embs'].to(device)
    parent_bbox = batch['parent_bbox'].to(device)
    parent_segs = batch['parent_bezier_segs'].to(device)
    gt_curves = batch['gt_curves'].to(device)
    lengths = batch['lengths'].to(device).float()

    B = gt_curves.size(0)
    if B == 0:
        logger.warning("Empty batch encountered")
        return 0.0

    out = model(child_embs, parent_embs, parent_segs)
    coords_pred    = out['segments']       # (B, S, 6)
    type_logits    = out['type_logits']    # (B, S, 3)
    stop_logits    = out['stop_scores']    # (B, S) ← post‐sigmoid “scores”

    # Build ground-truth type labels based on gt_curves:
    S = coords_pred.size(1)
    T_gt_dim = gt_curves.size(1)
    gt_types = torch.zeros((B, S), dtype=torch.long, device=device)
    for b in range(B):
        for s in range(min(S, T_gt_dim)):
            seg = gt_curves[b, s]
            c1 = seg[:2]; c2 = seg[2:4]
            if (c1 < 0).all() and (c2 < 0).all():
                gt_types[b, s] = 0  # line
            elif (c1 < 0).all() and not (c2 < 0).all():
                gt_types[b, s] = 1  # quad
            else:
                gt_types[b, s] = 2  # cubic

    gt_stop_indices = (lengths - 1).clamp(min=0).long()

    loss, loss_dict = compute_loss(
        pred_coords=coords_pred,
        pred_type_logits=type_logits,
        pred_stop_logits=stop_logits,
        gt_coords=gt_curves,
        gt_type_labels=gt_types,
        gt_stop_indices=gt_stop_indices,
        lambda_curve=lambda_curve,
        lambda_type=lambda_type,
        lambda_stop=lambda_stop,
        lambda_uniqueness=lambda_uniqueness,
        lambda_repulsion=lambda_repulsion,
        lambda_mask=lambda_mask,
        repulsion_delta=repulsion_delta
    )

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    if debug_mode and (batch_idx % 10 == 0 or batch_idx < 5):
        logger.info(
            f"[Batch {batch_idx}] "
            f"Total={loss.item():.3f} | "
            f"curve={loss_dict['curve_l1']:.3f} type={loss_dict['type_ce']:.3f} "
            f"stop={loss_dict['stop_ce']:.3f} uniq={loss_dict['uniq_penalty']:.3f} "
            f"repel={loss_dict['repel_penalty']:.3f} mask={loss_dict['mask_penalty']:.3f}"
        )

    return loss.item()

# -----------------------------------------------------------------------------  
# Main Entrypoint  
# -----------------------------------------------------------------------------  
if __name__ == "__main__":
    create_dummy = False
    if create_dummy:
        print("Setting up dummy dataset structure for testing...")
        dummy_root = "dummy_bezier_dataset"
        os.makedirs(os.path.join(dummy_root, "json"), exist_ok=True)
        os.makedirs(os.path.join(dummy_root, "masks/scene1"), exist_ok=True)
        os.makedirs(os.path.join(dummy_root, "images"), exist_ok=True)
        dummy_json = {
            "scene": [{"id": 0, "description": "background", "mask_path": None, "parent": -1},
                      {"id": 1, "description": "a wiggly shape", "mask_path": "wiggly_mask.png", "parent": 0}]
        }
        with open(os.path.join(dummy_root, "json/scene1.json"), 'w') as f: json.dump(dummy_json, f)
        img_w, img_h = 100, 100
        dummy_image = Image.new('RGB', (img_w, img_h), color='white')
        dummy_image.save(os.path.join(dummy_root, "images/scene1.png"))
        mask = np.zeros((img_h, img_w), dtype=np.uint8)
        pts = np.array([[10, 50], [30, 20], [50, 50], [70, 80], [90, 50]], np.int32)
        cv2.polylines(mask, [pts], isClosed=False, color=255, thickness=10)
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if cnts: cv2.drawContours(mask, cnts, -1, 255, -1)
        Image.fromarray(mask).save(os.path.join(dummy_root, "masks/scene1/wiggly_mask.png"))
        print("Dummy data created.")
        dataset_to_train = dummy_root
    else:
        dataset_to_train = "dataset"

    print(f"Using dataset: {dataset_to_train}")
    trained_model = train_model_batched(
        dataset_path=dataset_to_train,
        model_name="",
        output_dir="bezier_checkpoints",
        num_epochs=50,
        learning_rate=1e-3,
        batch_size=2,
        max_samples=5,
        run_visualization=True,
        run_post_training_test=True
    )

    # Run sanity checks on the final trained model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = AugmentedDataset(root_dir=dataset_to_train, max_samples=20)
    print("\n=== Sanity Checks on Trained Model ===")
    sanity_results = sanity_check_full(trained_model, dataset, device, num_samples=20)
    print("\nSanity Check Summary:", sanity_results)
