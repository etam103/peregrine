#!/usr/bin/env python3
"""Multi-view MUSt3R reconstruction from video.

Extracts N frames, runs pairwise MUSt3R, aligns into a common coordinate
system via global pose optimization, and produces an interactive 3D mesh.

Usage:
    python3 reconstruct_video.py vids/out.mp4 --frames 12 --resolution 224
    python3 reconstruct_video.py vids/rgb.mp4 --frames 12 --resolution 224 --pairs dense
    python3 reconstruct_video.py vids/rgb.mp4 --frames 12 --resolution 224 --pairs all
"""
import argparse
import struct
import subprocess
import sys
import tempfile
import os
import shutil

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def extract_frames(video_path, n_frames, resolution, out_dir):
    """Extract N evenly-spaced frames from video as PPM files."""
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=nb_frames",
         "-of", "csv=p=0", video_path],
        capture_output=True, text=True,
    )
    total_str = probe.stdout.strip()
    if total_str and total_str != "N/A":
        total_frames = int(total_str)
    else:
        probe2 = subprocess.run(
            ["ffprobe", "-v", "error", "-count_frames",
             "-select_streams", "v:0",
             "-show_entries", "stream=nb_read_frames",
             "-of", "csv=p=0", video_path],
            capture_output=True, text=True, check=True,
        )
        total_frames = int(probe2.stdout.strip())

    indices = np.linspace(0, total_frames - 1, n_frames, dtype=int)
    select_expr = "+".join(f"eq(n\\,{idx})" for idx in indices)

    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error",
         "-i", video_path,
         "-vf", f"select='{select_expr}',scale={resolution}:{resolution}:flags=lanczos",
         "-vsync", "vfr", "-pix_fmt", "rgb24",
         os.path.join(out_dir, "frame_%03d.ppm")],
        check=True,
    )

    ppm_paths = [os.path.join(out_dir, f"frame_{i+1:03d}.ppm") for i in range(n_frames)]
    ppm_paths = [p for p in ppm_paths if os.path.exists(p)]
    print(f"Extracted {len(ppm_paths)} frames from {video_path}")
    return ppm_paths


def load_pointmaps(path):
    """Read MUSt3R binary output: H, W, pts1, conf1, pts2, conf2."""
    with open(path, "rb") as f:
        H = struct.unpack("<I", f.read(4))[0]
        W = struct.unpack("<I", f.read(4))[0]
        n = H * W
        pts1 = np.frombuffer(f.read(n * 3 * 4), dtype=np.float32).reshape(n, 3).copy()
        conf1 = np.frombuffer(f.read(n * 4), dtype=np.float32).copy()
        pts2 = np.frombuffer(f.read(n * 3 * 4), dtype=np.float32).reshape(n, 3).copy()
        conf2 = np.frombuffer(f.read(n * 4), dtype=np.float32).copy()
    return H, W, pts1, conf1, pts2, conf2


def load_ppm(path):
    """Load a P6 PPM image as an (H, W, 3) uint8 array."""
    with open(path, "rb") as f:
        magic = f.readline().strip()
        assert magic == b"P6", f"Expected P6, got {magic}"
        line = f.readline()
        while line.startswith(b"#"):
            line = f.readline()
        w, h = map(int, line.split())
        _ = f.readline()
        data = f.read()
    return np.frombuffer(data, dtype=np.uint8).reshape(h, w, 3)


# ---------------------------------------------------------------------------
# Pair generation
# ---------------------------------------------------------------------------

def generate_pairs(n_frames, mode):
    """Generate (i, j) pairs. Every view appears as first image in >= 1 pair."""
    if mode == 'consecutive':
        return [(i, i + 1) for i in range(n_frames - 1)]

    pairs = set()
    # Always include consecutive
    for i in range(n_frames - 1):
        pairs.add((i, i + 1))

    if mode == 'dense':
        # Skip-1
        for i in range(n_frames - 2):
            pairs.add((i, i + 2))
        # Skip-2
        for i in range(n_frames - 3):
            pairs.add((i, i + 3))
    elif mode == 'all':
        for i in range(n_frames):
            for j in range(i + 1, n_frames):
                pairs.add((i, j))

    # Ensure every view appears as first image in at least one pair
    first_views = {p[0] for p in pairs}
    for v in range(n_frames):
        if v not in first_views:
            target = 0 if v != 0 else 1
            pairs.add((v, target))

    return sorted(pairs)


# ---------------------------------------------------------------------------
# Procrustes alignment
# ---------------------------------------------------------------------------

def procrustes_rigid(src, dst, weights=None):
    """Weighted SVD Procrustes: find (R, t, s) so dst ~ s*(src @ R.T) + t."""
    if weights is None:
        weights = np.ones(len(src), dtype=np.float64)
    weights = weights / weights.sum()

    src = src.astype(np.float64)
    dst = dst.astype(np.float64)

    c_src = (weights[:, None] * src).sum(axis=0)
    c_dst = (weights[:, None] * dst).sum(axis=0)

    src_c = src - c_src
    dst_c = dst - c_dst

    H = (src_c * weights[:, None]).T @ dst_c

    U, S, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    sign = np.array([1, 1, np.sign(d)])
    R = Vt.T @ np.diag(sign) @ U.T

    var_src = (weights[:, None] * src_c ** 2).sum()
    s = (S * sign).sum() / var_src if var_src > 1e-12 else 1.0

    t = c_dst - s * (R @ c_src)
    return R, t, s


def apply_transform(pts, R, t, s):
    """Apply rigid transform: s * (pts @ R.T) + t."""
    return s * (pts @ R.T) + t


# ---------------------------------------------------------------------------
# Global pose optimization
# ---------------------------------------------------------------------------

def unpack_poses(params, n_views):
    """Unpack flat parameter vector to list of (R, t, s) per view.
    View 0 is fixed at identity and not in params."""
    poses = [(np.eye(3), np.zeros(3), 1.0)]
    for i in range(1, n_views):
        off = (i - 1) * 7
        R = Rotation.from_rotvec(params[off:off + 3]).as_matrix()
        t = params[off + 3:off + 6].copy()
        s = np.exp(params[off + 6])
        poses.append((R, t, s))
    return poses


def pack_poses(poses):
    """Pack per-view (R, t, s) to flat parameter vector (skip view 0)."""
    params = []
    for i in range(1, len(poses)):
        R, t, s = poses[i]
        rv = Rotation.from_matrix(R).as_rotvec()
        params.extend(rv)
        params.extend(t)
        params.append(np.log(max(s, 1e-12)))
    return np.array(params, dtype=np.float64)


def select_canonical_pairs(n_views, pairs, pair_results):
    """For each view, pick the pair where it's first image with highest mean conf1."""
    canon = {}
    for v in range(n_views):
        best_pair = None
        best_conf = -1
        for (i, j) in pairs:
            if i == v:
                conf1 = pair_results[(i, j)][1]
                mc = conf1.mean()
                if mc > best_conf:
                    best_conf = mc
                    best_pair = (i, j)
        canon[v] = best_pair
    return canon


def build_residual_data(n_views, pairs, pair_results, canon_pairs, n_subsample=500):
    """Precompute fixed data for the residual function.

    For each pair (i, j), the constraint is:
        T_i(pts2_{ij}) ~ T_j(canon_pts1_j)
    where canon_pts1_j is pts1 from the canonical pair for view j.
    """
    rng = np.random.RandomState(42)
    constraints = []

    for (i, j) in pairs:
        canon_j = canon_pairs[j]
        if canon_j is None:
            continue

        pts2_ij = pair_results[(i, j)][2]     # view j's pixels in view i's frame
        conf2_ij = pair_results[(i, j)][3]

        canon_pts1_j = pair_results[canon_j][0]  # view j's pixels in view j's frame
        canon_conf1_j = pair_results[canon_j][1]

        n_pts = len(pts2_ij)
        if n_pts > n_subsample:
            idx = rng.choice(n_pts, n_subsample, replace=False)
        else:
            idx = np.arange(n_pts)

        w = np.minimum(conf2_ij[idx], canon_conf1_j[idx])
        w = np.sqrt(np.maximum(w, 0.0))

        constraints.append((
            i, j,
            pts2_ij[idx].astype(np.float64),
            canon_pts1_j[idx].astype(np.float64),
            w.astype(np.float64),
        ))

    return constraints


def global_pose_residuals(params, n_views, constraints):
    """Residuals: for each constraint, weighted difference between
    T_i(pts2) and T_j(canon_pts1)."""
    poses = unpack_poses(params, n_views)
    residuals = []

    for (i, j, pts2_sub, canon_sub, w) in constraints:
        R_i, t_i, s_i = poses[i]
        R_j, t_j, s_j = poses[j]

        pred_j = s_i * (pts2_sub @ R_i.T) + t_i
        canon_j_global = s_j * (canon_sub @ R_j.T) + t_j

        diff = pred_j - canon_j_global
        weighted = w[:, None] * diff
        residuals.append(weighted.ravel())

    return np.concatenate(residuals)


def optimize_global_poses(n_views, pairs, pair_results, init_poses, verbose=True):
    """Run global pose optimization via scipy least_squares."""
    canon_pairs = select_canonical_pairs(n_views, pairs, pair_results)

    if verbose:
        for v in range(n_views):
            print(f"  View {v} canonical pair: {canon_pairs[v]}")

    constraints = build_residual_data(n_views, pairs, pair_results, canon_pairs)

    total_residuals = sum(len(c[2]) * 3 for c in constraints)
    n_params = (n_views - 1) * 7
    if verbose:
        print(f"  {len(constraints)} constraints, {total_residuals:,} residuals, {n_params} params")

    x0 = pack_poses(init_poses)

    result = least_squares(
        global_pose_residuals,
        x0,
        args=(n_views, constraints),
        method='trf',
        loss='soft_l1',
        jac='2-point',
        verbose=2 if verbose else 0,
        max_nfev=200,
    )

    if verbose:
        print(f"  Optimizer: {result.message}")
        print(f"  Cost: {result.cost:.4f}, nfev: {result.nfev}")

    return unpack_poses(result.x, n_views), canon_pairs


# ---------------------------------------------------------------------------
# Point fusion
# ---------------------------------------------------------------------------

def fuse_pointmaps(n_views, pairs, pair_results, poses, H, W):
    """Fuse all predictions of each view's pixels into one pointmap per view.

    For view v, collects:
      - From pair (i, v): T_i(pts2_{iv}) -- view v predicted from pair i's frame
      - From pair (v, k): T_v(pts1_{vk}) -- view v predicted from its own frame
    Weighted average by confidence.
    """
    n_pts = H * W
    fused = []

    for v in range(n_views):
        sum_pts = np.zeros((n_pts, 3), dtype=np.float64)
        sum_w = np.zeros(n_pts, dtype=np.float64)

        for (i, j) in pairs:
            if j == v:
                # pair (i, v): pts2 predicts view v in view i's frame
                R_i, t_i, s_i = poses[i]
                pts2 = pair_results[(i, j)][2].astype(np.float64)
                conf2 = pair_results[(i, j)][3].astype(np.float64)
                pts_global = s_i * (pts2 @ R_i.T) + t_i
                w = np.maximum(conf2, 0.0)
                sum_pts += w[:, None] * pts_global
                sum_w += w

            if i == v:
                # pair (v, k): pts1 predicts view v in view v's frame
                R_v, t_v, s_v = poses[v]
                pts1 = pair_results[(i, j)][0].astype(np.float64)
                conf1 = pair_results[(i, j)][1].astype(np.float64)
                pts_global = s_v * (pts1 @ R_v.T) + t_v
                w = np.maximum(conf1, 0.0)
                sum_pts += w[:, None] * pts_global
                sum_w += w

        safe_w = np.maximum(sum_w, 1e-12)
        fused.append((sum_pts / safe_w[:, None]).astype(np.float32))

    return fused


# ---------------------------------------------------------------------------
# Mesh building
# ---------------------------------------------------------------------------

def build_grid_faces(H, W):
    """Build triangle faces for an H*W grid. Returns (N_faces, 3) int array."""
    r = np.arange(H - 1)
    c = np.arange(W - 1)
    rr, cc = np.meshgrid(r, c, indexing="ij")
    rr = rr.ravel()
    cc = cc.ravel()

    tl = rr * W + cc
    tr = rr * W + cc + 1
    bl = (rr + 1) * W + cc
    br = (rr + 1) * W + cc + 1

    faces = np.column_stack([
        np.concatenate([tl, tr]),
        np.concatenate([tr, br]),
        np.concatenate([bl, bl]),
    ])
    return faces


def filter_faces_by_edge_length(pts, faces, max_edge_factor=5.0):
    """Remove triangles with any edge longer than max_edge_factor * median edge."""
    v0, v1, v2 = pts[faces[:, 0]], pts[faces[:, 1]], pts[faces[:, 2]]
    e0 = np.linalg.norm(v1 - v0, axis=1)
    e1 = np.linalg.norm(v2 - v1, axis=1)
    e2 = np.linalg.norm(v0 - v2, axis=1)
    max_e = np.maximum(np.maximum(e0, e1), e2)
    median_e = np.median(max_e)
    if median_e < 1e-12:
        return faces
    keep = max_e < max_edge_factor * median_e
    return faces[keep]


def filter_faces_by_confidence(conf, faces, conf_thresh):
    """Remove triangles where any vertex has low confidence."""
    c0, c1, c2 = conf[faces[:, 0]], conf[faces[:, 1]], conf[faces[:, 2]]
    min_c = np.minimum(np.minimum(c0, c1), c2)
    return faces[min_c > conf_thresh]


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pair(binary, weights_path, ppm1, ppm2, resolution, output_bin):
    """Run MUSt3R on one pair of frames."""
    cmd = [binary, weights_path, ppm1, ppm2, str(resolution)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"ERROR running MUSt3R:\n{result.stderr}", file=sys.stderr)
        sys.exit(1)
    for line in result.stdout.strip().split("\n"):
        if "time" in line.lower() or "ms" in line.lower() or "bbox" in line.lower():
            print(f"  {line.strip()}")
    if os.path.exists("must3r_output.bin"):
        shutil.move("must3r_output.bin", output_bin)
    else:
        print("ERROR: must3r_output.bin not found after running binary", file=sys.stderr)
        sys.exit(1)


def init_poses_from_procrustes(n_views, pairs, pair_results):
    """Initialize per-view poses from Procrustes chain on consecutive pairs."""
    poses = [(np.eye(3), np.zeros(3), 1.0)]  # view 0 = identity

    # Chain consecutive pairs: pair (k, k+1) gives view k+1's transform
    prev_pts2_global = apply_transform(
        pair_results[(0, 1)][2],  # pts2 from pair (0,1)
        np.eye(3), np.zeros(3), 1.0,
    )

    for k in range(1, n_views - 1):
        pair_key = (k, k + 1)
        pts1_k = pair_results[pair_key][0]
        conf1_k = pair_results[pair_key][1]
        prev_conf2 = pair_results[(k - 1, k)][3]

        weights = np.minimum(prev_conf2, conf1_k)
        R_k, t_k, s_k = procrustes_rigid(pts1_k, prev_pts2_global, weights)
        poses.append((R_k, t_k, s_k))

        print(f"  View {k}: scale={s_k:.4f}")

        pts2_k = pair_results[pair_key][2]
        prev_pts2_global = apply_transform(pts2_k, R_k, t_k, s_k)

    # View N-1: find a pair where N-1 is first image, Procrustes to global
    v_last = n_views - 1
    last_pair = None
    for (i, j) in pairs:
        if i == v_last:
            last_pair = (i, j)
            break

    if last_pair is not None:
        pts1_last = pair_results[last_pair][0]
        conf1_last = pair_results[last_pair][1]
        prev_conf2 = pair_results[(v_last - 1, v_last)][3]
        weights = np.minimum(prev_conf2, conf1_last)
        R_last, t_last, s_last = procrustes_rigid(pts1_last, prev_pts2_global, weights)
        poses.append((R_last, t_last, s_last))
        print(f"  View {v_last}: scale={s_last:.4f}")
    else:
        # Fallback: use identity (shouldn't happen if pairs are generated correctly)
        poses.append((np.eye(3), np.zeros(3), 1.0))
        print(f"  View {v_last}: identity (no canonical pair)")

    return poses


def main():
    parser = argparse.ArgumentParser(description="Multi-view MUSt3R video reconstruction")
    parser.add_argument("video", help="Input video file")
    parser.add_argument("--frames", type=int, default=12, help="Number of frames to extract (default: 12)")
    parser.add_argument("--resolution", type=int, default=224, help="Image resolution (default: 224)")
    parser.add_argument("--weights", default=None, help="Path to weights .bin")
    parser.add_argument("--binary", default="target/release/examples/must3r", help="Path to MUSt3R binary")
    parser.add_argument("--output", default=None, help="Output HTML path")
    parser.add_argument("--conf-percentile", type=float, default=25, help="Confidence filter percentile (default: 25)")
    parser.add_argument("--max-faces", type=int, default=500000, help="Max triangle faces (default: 500K)")
    parser.add_argument("--points", action="store_true", help="Render as point cloud instead of mesh")
    parser.add_argument("--pairs", choices=["consecutive", "dense", "all"],
                        default="consecutive",
                        help="Pair selection mode (default: consecutive)")
    args = parser.parse_args()

    weights_path = args.weights or f"weights/must3r_{args.resolution}.bin"
    if not os.path.exists(weights_path):
        print(f"Weights not found: {weights_path}", file=sys.stderr)
        sys.exit(1)
    if not os.path.exists(args.binary):
        print(f"Binary not found: {args.binary}  (run: cargo build --example must3r --release)", file=sys.stderr)
        sys.exit(1)

    video_stem = os.path.splitext(os.path.basename(args.video))[0]
    output_html = args.output or f"vids/{video_stem}_multiview.html"

    # 1. Extract frames
    tmp_dir = tempfile.mkdtemp(prefix="must3r_frames_")
    try:
        ppm_paths = extract_frames(args.video, args.frames, args.resolution, tmp_dir)
        n_views = len(ppm_paths)

        # 2. Generate pairs
        pairs = generate_pairs(n_views, args.pairs)
        print(f"\nRunning {len(pairs)} pairwise reconstructions "
              f"({args.pairs} mode) at {args.resolution}x{args.resolution}...")

        # 3. Run MUSt3R on all pairs
        pair_results = {}   # (i, j) -> (pts1, conf1, pts2, conf2, H, W)
        view_colors = {}    # view_idx -> (H*W, 3) uint8
        H = W = None

        for pi, (i, j) in enumerate(pairs):
            print(f"\nPair {pi+1}/{len(pairs)}: frame {i} + frame {j}")
            out_bin = os.path.join(tmp_dir, f"pair_{i}_{j}.bin")
            run_pair(args.binary, weights_path, ppm_paths[i], ppm_paths[j],
                     args.resolution, out_bin)
            pH, pW, pts1, conf1, pts2, conf2 = load_pointmaps(out_bin)
            H, W = pH, pW
            pair_results[(i, j)] = (pts1, conf1, pts2, conf2, H, W)
            if i not in view_colors:
                view_colors[i] = load_ppm(ppm_paths[i]).reshape(-1, 3)
            if j not in view_colors:
                view_colors[j] = load_ppm(ppm_paths[j]).reshape(-1, 3)

        # 4. Alignment
        if args.pairs == 'consecutive':
            # --- Original Procrustes chain (no optimization) ---
            n_pairs = n_views - 1
            print(f"\nAligning {n_pairs} pairs via Procrustes...")

            transforms = [(np.eye(3), np.zeros(3), 1.0)]
            prev_pts2_global = pair_results[(0, 1)][2].copy()

            for k in range(1, n_pairs):
                pts1_k = pair_results[(k, k + 1)][0]
                conf1_k = pair_results[(k, k + 1)][1]
                prev_conf2 = pair_results[(k - 1, k)][3]

                weights = np.minimum(prev_conf2, conf1_k)
                R_k, t_k, s_k = procrustes_rigid(pts1_k, prev_pts2_global, weights)
                transforms.append((R_k, t_k, s_k))
                print(f"  Pair {k}: scale={s_k:.4f}")

                pts2_k = pair_results[(k, k + 1)][2]
                prev_pts2_global = apply_transform(pts2_k, R_k, t_k, s_k)

            # Build per-view data: view 0 from pair 0 pts1, then each pair's pts2
            print(f"\nBuilding meshes...")
            views = []
            # View 0
            R0, t0, s0 = transforms[0]
            pts1_0 = pair_results[(0, 1)][0]
            views.append((
                apply_transform(pts1_0, R0, t0, s0),
                view_colors[0],
                pair_results[(0, 1)][1],
                H, W,
            ))
            # Views 1..N-1: from each consecutive pair's pts2
            for k in range(n_pairs):
                R, t, s = transforms[k]
                pts2_k = pair_results[(k, k + 1)][2]
                views.append((
                    apply_transform(pts2_k, R, t, s),
                    view_colors[k + 1],
                    pair_results[(k, k + 1)][3],
                    H, W,
                ))

        else:
            # --- Global pose optimization ---
            print(f"\nInitializing poses from Procrustes chain...")
            init_poses = init_poses_from_procrustes(n_views, pairs, pair_results)

            print(f"\nRunning global pose optimization...")
            poses, canon_pairs = optimize_global_poses(
                n_views, pairs, pair_results, init_poses,
            )

            print(f"\nFusing pointmaps...")
            fused = fuse_pointmaps(n_views, pairs, pair_results, poses, H, W)

            # Build views from fused pointmaps
            print(f"\nBuilding meshes...")
            views = []
            for v in range(n_views):
                # Use canonical pair's conf1 as the confidence for this view
                canon = canon_pairs[v]
                conf = pair_results[canon][1] if canon is not None else np.ones(H * W)
                views.append((fused[v], view_colors[v], conf, H, W))

        print(f"  {len(views)} views")

        # 5. Build mesh from views
        # Compute global confidence threshold
        all_confs = np.concatenate([v[2] for v in views])
        conf_thresh = np.percentile(all_confs, args.conf_percentile)
        print(f"  Confidence threshold (p{args.conf_percentile:.0f}): {conf_thresh:.2f}")

        base_faces = None
        all_mesh_pts = []
        all_mesh_colors = []
        all_mesh_faces = []
        vertex_offset = 0

        for vi, (pts, colors, conf, vH, vW) in enumerate(views):
            if base_faces is None or base_faces.shape != build_grid_faces(vH, vW).shape:
                base_faces = build_grid_faces(vH, vW)

            faces = base_faces.copy()
            faces = filter_faces_by_confidence(conf, faces, conf_thresh)

            if len(faces) > 0:
                faces = filter_faces_by_edge_length(pts, faces, max_edge_factor=10.0)

            if len(faces) == 0:
                continue

            used = np.unique(faces)
            remap = np.full(vH * vW, -1, dtype=np.int64)
            remap[used] = np.arange(len(used)) + vertex_offset

            all_mesh_pts.append(pts[used])
            all_mesh_colors.append(colors[used])
            all_mesh_faces.append(remap[faces])
            vertex_offset += len(used)

        all_pts = np.concatenate(all_mesh_pts, axis=0)
        all_colors = np.concatenate(all_mesh_colors, axis=0)
        all_faces = np.concatenate(all_mesh_faces, axis=0)
        print(f"  {len(all_pts):,} vertices, {len(all_faces):,} faces")

        # Remove outliers (3-sigma per axis)
        for axis in range(3):
            mu = np.mean(all_pts[:, axis])
            sigma = np.std(all_pts[:, axis])
            inlier = np.abs(all_pts[:, axis] - mu) < 3 * sigma
            outlier_set = set(np.where(~inlier)[0])
            if outlier_set:
                face_ok = np.array([
                    all_faces[fi, 0] not in outlier_set and
                    all_faces[fi, 1] not in outlier_set and
                    all_faces[fi, 2] not in outlier_set
                    for fi in range(len(all_faces))
                ])
                all_faces = all_faces[face_ok]
        print(f"  After outlier removal: {len(all_faces):,} faces")

        # Subsample faces if too many
        if len(all_faces) > args.max_faces:
            idx = np.random.choice(len(all_faces), args.max_faces, replace=False)
            all_faces = all_faces[idx]
            print(f"  Subsampled to {args.max_faces:,} faces")

        # 6. Visualize
        print(f"\nGenerating interactive 3D viewer...")
        import plotly.graph_objects as go

        fig = go.Figure()

        if args.points:
            used_verts = np.unique(all_faces)
            pts_show = all_pts[used_verts]
            cols_show = all_colors[used_verts]
            if len(pts_show) > 300000:
                idx = np.random.choice(len(pts_show), 300000, replace=False)
                pts_show = pts_show[idx]
                cols_show = cols_show[idx]
            rgb_strs = [f"rgb({r},{g},{b})" for r, g, b in cols_show]
            fig.add_trace(go.Scatter3d(
                x=pts_show[:, 0], y=pts_show[:, 1], z=pts_show[:, 2],
                mode="markers",
                marker=dict(size=1.0, color=rgb_strs, opacity=0.9),
                name="Points",
            ))
        else:
            vc = [f"rgb({r},{g},{b})" for r, g, b in all_colors]
            fig.add_trace(go.Mesh3d(
                x=all_pts[:, 0], y=all_pts[:, 1], z=all_pts[:, 2],
                i=all_faces[:, 0], j=all_faces[:, 1], k=all_faces[:, 2],
                vertexcolor=vc,
                flatshading=True,
                name="Mesh",
            ))

        fig.update_layout(
            title=f"MUSt3R Multi-View Reconstruction — {args.frames} frames, "
                  f"{args.resolution}x{args.resolution}, {args.pairs} pairs "
                  f"(Peregrine, pure Rust)",
            scene=dict(
                xaxis_title="X", yaxis_title="Y", zaxis_title="Z",
                aspectmode="data",
                camera=dict(eye=dict(x=0, y=-0.5, z=-2.0)),
            ),
            width=1200, height=800,
        )

        fig.write_html(output_html, config={"scrollZoom": True})
        print(f"\nSaved to {output_html}")

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
