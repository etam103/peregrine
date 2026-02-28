#!/usr/bin/env python3
"""Visualize MUSt3R 3D reconstruction — interactive colored point cloud."""
import struct
import numpy as np
from PIL import Image

def load_pointmaps(path):
    with open(path, 'rb') as f:
        H = struct.unpack('<I', f.read(4))[0]
        W = struct.unpack('<I', f.read(4))[0]
        n = H * W
        pts1 = np.frombuffer(f.read(n * 3 * 4), dtype=np.float32).reshape(n, 3)
        conf1 = np.frombuffer(f.read(n * 4), dtype=np.float32)
        pts2 = np.frombuffer(f.read(n * 3 * 4), dtype=np.float32).reshape(n, 3)
        conf2 = np.frombuffer(f.read(n * 4), dtype=np.float32)
    return H, W, pts1, conf1, pts2, conf2

def load_ppm(path):
    with open(path, 'rb') as f:
        magic = f.readline().strip()
        assert magic == b'P6'
        line = f.readline()
        while line.startswith(b'#'):
            line = f.readline()
        w, h = map(int, line.split())
        _ = f.readline()
        data = f.read()
    return np.frombuffer(data, dtype=np.uint8).reshape(h, w, 3)

H, W, pts1, conf1, pts2, conf2 = load_pointmaps('must3r_output.bin')

# Try to find matching input images
import sys, os
cli_ppm1 = sys.argv[1] if len(sys.argv) > 1 else None
cli_ppm2 = sys.argv[2] if len(sys.argv) > 2 else None
ppm_paths = []
if cli_ppm1:
    ppm_paths.append((cli_ppm1, cli_ppm2 or cli_ppm1))
ppm_paths += [
    ('vids/view1.ppm', 'vids/view2.ppm'),
    (f'weights/real_view1_{W}.ppm', f'weights/real_view2_{W}.ppm'),
    ('weights/real_view1_512.ppm', 'weights/real_view2_512.ppm'),
    ('weights/real_view1.ppm', 'weights/real_view2.ppm'),
]
img1 = img2 = None
for p1, p2 in ppm_paths:
    if os.path.exists(p1):
        img1 = load_ppm(p1)
        img2 = load_ppm(p2) if os.path.exists(p2) else img1
        print(f"Using colors from: {p1}, {p2}")
        break
if img1 is None:
    print("No PPM images found, using white")
    img1 = np.full((H, W, 3), 200, dtype=np.uint8)
    img2 = img1

img1_resized = np.array(Image.fromarray(img1).resize((W, H), Image.NEAREST))
colors1 = img1_resized.reshape(-1, 3)
img2_resized = np.array(Image.fromarray(img2).resize((W, H), Image.NEAREST))
colors2 = img2_resized.reshape(-1, 3)

print(f"Pointmap: {H}x{W} = {H*W} points")
print(f"View 1 Z: [{pts1[:,2].min():.3f}, {pts1[:,2].max():.3f}]")
print(f"View 1 conf: [{conf1.min():.2f}, {conf1.max():.2f}], mean={conf1.mean():.2f}")

# Depth-based foreground separation
z1 = pts1[:, 2]
z_bg = np.percentile(z1, 75)  # background wall at far depth
foreground = z1 < z_bg

# High confidence filter
conf_hi = np.percentile(conf1, 25)

import plotly.graph_objects as go

def rgb_strings(colors):
    return [f'rgb({r},{g},{b})' for r, g, b in colors]

# Subsample if too many points for smooth rendering
max_points = 150000
mask = (conf1 > conf_hi) & foreground
idx = np.where(mask)[0]
if len(idx) > max_points:
    idx = idx[np.random.choice(len(idx), max_points, replace=False)]

print(f"Displaying {len(idx)} foreground points (conf > {conf_hi:.2f}, z < {z_bg:.2f})")

# View 2 filtering
z2 = pts2[:, 2]
z_bg2 = np.percentile(z2, 75)
conf_hi2 = np.percentile(conf2, 25)
mask2 = (conf2 > conf_hi2) & (z2 < z_bg2)
idx2 = np.where(mask2)[0]
if len(idx2) > max_points:
    idx2 = idx2[np.random.choice(len(idx2), max_points, replace=False)]
print(f"View 2: {len(idx2)} foreground points (conf > {conf_hi2:.2f}, z < {z_bg2:.2f})")

fig = go.Figure()

fig.add_trace(go.Scatter3d(
    x=pts1[idx, 0], y=pts1[idx, 1], z=pts1[idx, 2],
    mode='markers',
    marker=dict(size=1.2, color=rgb_strings(colors1[idx]), opacity=0.95),
    name='View 1',
))

fig.add_trace(go.Scatter3d(
    x=pts2[idx2, 0], y=pts2[idx2, 1], z=pts2[idx2, 2],
    mode='markers',
    marker=dict(size=1.2, color=rgb_strings(colors2[idx2]), opacity=0.95),
    name='View 2',
))

fig.update_layout(
    title=f'MUSt3R 3D Reconstruction — {H}x{W} (Peregrine, pure Rust)',
    scene=dict(
        xaxis_title='X', yaxis_title='Y', zaxis_title='Z (depth)',
        aspectmode='data',
        camera=dict(eye=dict(x=0, y=-0.5, z=-2.0)),  # front view
    ),
    width=1200, height=800,
)

out_path = sys.argv[3] if len(sys.argv) > 3 else 'must3r_3d_viewer.html'
fig.write_html(out_path)
print(f"\nSaved to {out_path}")
