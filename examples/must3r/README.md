# MUSt3R Example

423M parameter ViT-L encoder + ViT-B decoder for 3D reconstruction from image pairs.

## Usage

```bash
# Single pair inference
cargo run --example must3r --release -- weights/must3r_224.bin img1.ppm img2.ppm

# Higher resolution
cargo run --example must3r --release -- weights/must3r_512.bin img1.ppm img2.ppm 512x384

# Server mode (load weights once, process pairs over stdin/stdout)
echo -e "img1.ppm\timg2.ppm\t224\t224" | cargo run --example must3r --release -- weights/must3r_224.bin --server

# Metal GPU
cargo run --example must3r --release --features metal -- weights/must3r_224.bin img1.ppm img2.ppm --gpu

# GPU + heterogeneous pipeline (overlaps GPU and CPU decoder views)
cargo run --example must3r --release --features metal -- weights/must3r_224.bin img1.ppm img2.ppm --gpu --pipeline
```

## Multi-View Reconstruction Pipeline

`scripts/reconstruct_video.py` extracts frames from video, runs all-pairs MUSt3R inference via Peregrine, then jointly optimizes camera poses and fuses pointmaps into a coherent 3D reconstruction. Supports server mode for persistent weight loading, parallel workers for multi-process inference, and optional Metal GPU acceleration.

```
python3 scripts/reconstruct_video.py vids/rgb.mp4 --frames 12 --resolution 512 --pairs all --workers 4
```

| Mode | Pairs | Inference (12 frames) |
|------|------:|----------------------:|
| consecutive | 11 | ~7s (224) |
| dense | 31 | ~21s (224) |
| all | 67 | ~45s (224), ~3min (512) |

Server mode (`--server` flag on the Rust binary) loads weights once and processes pairs over stdin/stdout, eliminating ~0.5s overhead per pair. Parallel workers (`--workers N`) distribute pairs across N server processes for near-linear wall-clock scaling.

## Performance

| Resolution | CPU | GPU | GPU+Pipeline | PyTorch CPU |
|-----------|----:|----:|-------------:|------------:|
| 224x224 | 0.59s | 0.53s | **0.54s** | 0.67s |
| 512x384 | 1.95s | 1.55s | **1.44s** | 2.26s |
| Weight load | **0.6s** | 0.6s | 0.6s | 1.6s |

GPU mode (`--gpu`) keeps the entire attention pipeline on Metal — QKV reshape, 2D RoPE, scaled dot-product attention, and output reshape all run as GPU kernels with no CPU round-trips. Pipeline mode (`--pipeline`) overlaps feat1 (GPU) and feat2 (CPU/AMX) decoder processing via `MTLSharedEvent` signaling.

## Weight Conversion

```bash
python scripts/convert_must3r.py weights/MUSt3R_224_cvpr.pth weights/must3r_224.bin
python scripts/convert_must3r.py weights/MUSt3R_512_cvpr.pth weights/must3r_512.bin
```
