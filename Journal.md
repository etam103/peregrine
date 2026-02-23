# Peregrine Performance Journal

Model: YoloNet (54,344 params), batch_size=1, 20 epochs.
Hardware: Apple Silicon, macOS, Apple Accelerate BLAS.

## Baseline

Naive implementation: scalar loops for all ops, no BLAS, no parallelism.

| Metric | Value |
|---|---|
| Epoch time | ~12s |
| Final avg loss | ~4.07 |

## Round 1: BLAS + Rayon fundamentals

**Changes:**
- Replaced scalar matmul/conv2d with Apple Accelerate `cblas_sgemm` (1x1 conv as matrix multiply)
- Added rayon parallelism for elementwise ops (add, mul, relu, sigmoid, scale, etc.) gated by `PAR_THRESHOLD = 10_000`
- Eliminated unnecessary `.clone()` calls in hot paths

| Metric | Value |
|---|---|
| Epoch time | ~1.2s |
| Final avg loss | ~4.07 |
| Speedup vs baseline | ~10x |

## Round 2: Kernel optimizations (2025-02-20)

**Changes:**
1. **Parallelized MaxPool2d forward** — replaced 6-nested scalar loops with `par_chunks_mut` over (batch, channel) planes. Inlined index computation with precomputed base offsets instead of calling `idx4` per element.
2. **Parallelized MaxPool2d backward** — same `par_chunks_mut` approach; no write conflicts within non-overlapping 2x2 pool planes.
3. **Fused conv2d + relu + pool forward** — new `Op::Conv2dReluPool` that runs sgemm+bias then fused relu+maxpool in one parallel pass. Eliminates 2 intermediate `Vec` allocations and 2 extra data passes per ConvBlock.
4. **Fused conv2d + relu + pool backward** — single backward arm: pool scatter through `max_indices`, relu mask by `pre_relu_data > 0`, conv backward via BLAS.
5. **Parallelized `accumulate_grad`** — rayon `par_iter_mut` for large gradient tensors.

| Metric | Value |
|---|---|
| Epoch time | ~0.52s |
| Final avg loss | ~4.07 |
| Speedup vs Round 1 | ~2.3x |
| Speedup vs baseline | ~23x |

### Per-epoch times (Round 2)

| Epoch | Time (s) | Avg Loss |
|---|---|---|
| 0 | 0.55 | 4.8729 |
| 1 | 0.52 | 4.6618 |
| 2 | 0.52 | 4.5014 |
| 3 | 0.52 | 4.3846 |
| 4 | 0.53 | 4.3004 |
| 5 | 0.52 | 4.2414 |
| 6 | 0.52 | 4.1976 |
| 7 | 0.52 | 4.1671 |
| 8 | 0.52 | 4.1376 |
| 9 | 0.52 | 4.1279 |
| 10 | 0.52 | 4.1194 |
| 11 | 0.52 | 4.1120 |
| 12 | 0.51 | 4.1057 |
| 13 | 0.52 | 4.0998 |
| 14 | 0.52 | 4.0948 |
| 15 | 0.52 | 4.0898 |
| 16 | 0.52 | 4.0856 |
| 17 | 0.52 | 4.0816 |
| 18 | 0.52 | 4.0777 |
| 19 | 0.52 | 4.0740 |

### Inference

No detections above 0.5 confidence threshold yet (expected — tiny model, few epochs, small dataset).

## COCO val2017 full dataset run (2025-02-20)

**Changes:**
- Added `CocoDataset` loader (parses `instances_*.json` directly)
- Auto-detects COCO val2017 images, falls back to small VOC dataset

**Dataset:** COCO val2017 — 2,960 images containing car/person/dog (out of 5,000 total).

| Metric | Value |
|---|---|
| Epoch time | ~51.5s |
| Per-image time | ~17.4ms |
| Total training (20 epochs) | ~17.2 min |
| Final avg loss | 3.2757 |
| Min loss | 0.9327 |

### Per-epoch times (COCO val2017)

| Epoch | Time (s) | Avg Loss | Min Loss |
|---|---|---|---|
| 0 | 51.02 | 3.7214 | 1.5693 |
| 1 | 51.73 | 3.5086 | 1.2264 |
| 2 | 51.39 | 3.4069 | 1.0845 |
| 3 | 51.70 | 3.3545 | 1.0148 |
| 4 | 51.98 | 3.3299 | 0.9898 |
| 5 | 50.63 | 3.3180 | 0.9831 |
| 6 | 51.94 | 3.3113 | 0.9662 |
| 7 | 50.66 | 3.3067 | 0.9578 |
| 8 | 52.15 | 3.2858 | 0.9537 |
| 9 | 52.71 | 3.2839 | 0.9466 |
| 10 | 51.12 | 3.2827 | 0.9423 |
| 11 | 54.65 | 3.2817 | 0.9395 |
| 12 | 52.94 | 3.2809 | 0.9375 |
| 13 | 53.24 | 3.2801 | 0.9360 |
| 14 | 52.00 | 3.2794 | 0.9350 |
| 15 | 53.26 | 3.2787 | 0.9343 |
| 16 | 53.54 | 3.2780 | 0.9339 |
| 17 | 50.64 | 3.2772 | 0.9333 |
| 18 | 50.41 | 3.2765 | 0.9330 |
| 19 | 51.85 | 3.2757 | 0.9327 |

### Inference

Model now produces detections above 0.5 confidence (many false positives — expected with tiny 54k-param model). Loss decreased from 3.72 to 3.28, min per-sample loss reached 0.93.

### Scaling analysis

| Dataset | Images | Epoch time | Per-image |
|---|---|---|---|
| VOC small | 30 | 0.52s | 17.3ms |
| COCO val2017 | 2,960 | 51.5s | 17.4ms |

Per-image time is consistent (~17.4ms) regardless of dataset size, confirming no overhead from dataset scaling. Total time scales linearly with image count.

## GPU Assessment (2025-02-20)

**Recommendation: Don't add Metal GPU support.** At 54k params and these matrix sizes, GPU kernel launch + CPU-GPU data transfer overhead exceeds the computation itself. Apple Accelerate on CPU is already optimal at this scale. GPU becomes worthwhile at millions of parameters with large batch sizes.

## RT-DETR: Transformer-based detection (2026-02-20)

Replaced the YOLO anchor-based detector with RT-DETR, a transformer-based end-to-end object detector. Architecture: ResNet backbone → multi-scale 1x1 channel projections → transformer encoder (self-attention) → transformer decoder (cross-attention with learned object queries) → classification + bbox heads. Training uses Hungarian matching + set-based loss (BCE classification + L1 bbox regression).

**Model:** RT-DETR (11.2M params), embed_dim=64, 4 heads, 1 encoder + 1 decoder layer, 20 queries.

### Problem: memory explosion

With INPUT_SIZE=416, the backbone produces large spatial maps (s2: 208x208, s3: 104x104, s4: 52x52) concatenated into a 56,784-token sequence. Self-attention is O(n^2), requiring a [56784, 56784] attention matrix per head — 12.9 GB per head, 51.6 GB for 4 heads. Exceeds 64 GB RAM.

### Fix: reduce input + pool features

1. **INPUT_SIZE 416 → 256** — backbone spatial dims become s2: 128x128, s3: 64x64, s4: 32x32
2. **Pool projected features to 32x32** before the encoder — `max_pool2d` brings all scales to the same resolution. Sequence: 3 × 32×32 = 3,072 tokens
3. **Eliminated double forward pass** — reuse training outputs for detection overlay logging

| Component | Before | After |
|---|---|---|
| Encoder sequence | 56,784 tokens | 3,072 tokens |
| Attention (4 heads) | ~52 GB | ~150 MB |
| Peak RSS | OOM | 630 MB |

### Problem: NaN loss after 6 steps

Initial loss was 36,280 and exploded to NaN by step 6. Root cause: `randn` used a fixed `std=0.1` for all weight tensors regardless of layer dimensions. For deep backbone conv layers with large fan_in (e.g., [512, 512, 3, 3] → fan_in=4608), `std=0.1` is ~7x too large vs proper Xavier (`sqrt(1/4608)=0.015`). This caused activations to grow through the backbone, producing encoder outputs with std≈50, which propagated through cross-attention to class logits with std≈25 — giving a massive initial loss whose gradients exploded through 6+ transformer/backbone layers.

### Fix: Xavier init + gradient clipping

1. **Xavier fan-in initialization**: `std = sqrt(1/fan_in)`, computed per-tensor from shape (2D linear: fan_in=shape[1], 4D conv: fan_in=shape[1]*kH*kW)
2. **Global gradient clipping**: compute gradient norm across all parameters, scale effective LR if norm > 1.0

### Results (COCO val2017, 2960 images)

| Metric | Before fixes | After fixes |
|---|---|---|
| Initial loss | 36,280 → NaN | 1.60 |
| Epoch 0 avg loss | N/A (crashed) | 1.2680 |
| Epoch 0 min loss | N/A | 0.2053 |
| Epoch 0 time | N/A | 1165s (~19.4 min) |
| Peak RSS | OOM (>52 GB) | 630 MB |

Loss decreasing across epochs confirms the model is learning. Epoch 1 showed lower early losses (~0.18) compared to epoch 0's start (~1.6).

## Multi-framework wall-clock benchmarks (2026-02-22)

Built a complete benchmarking suite to compare Peregrine against PyTorch 2.10.0, MLX 0.30.6, TensorFlow 2.20.0, and tinygrad 0.12.0. All benchmarks run on CPU with `nice -n 10` to keep resource usage under 80%.

14 operations benchmarked across all 5 frameworks: matmul (128/256/512), elementwise add/mul/exp (100K/500K), relu (100K), softmax (8x128/8x512), MLP forward pass, and full training step (fwd + backward + Adam).

Methodology: 5 warmup iterations discarded, 50 timed iterations (20 for heavy ops), median reported in microseconds.

### Results (all times in microseconds)

| Operation | Peregrine (us) | PyTorch (us) | MLX (us) | TensorFlow (us) | tinygrad (us) | Best |
|-----------|---------------:|-------------:|---------:|----------------:|--------------:|------|
| matmul 128x128 | 6.0 | 5.7 | 20.9 | 93.8 | 459.8 | PyTorch |
| matmul 256x256 | 32.2 | 30.7 | 47.6 | 194.0 | 435.8 | PyTorch |
| matmul 512x512 | **162.3** | 165.2 | 173.7 | 675.9 | 434.1 | Peregrine |
| add 100k | 121.3 | 46.6 | **31.3** | 53.1 | 193.7 | MLX |
| add 500k | 159.8 | **73.7** | 81.1 | 86.3 | 194.7 | PyTorch |
| mul 100k | 116.8 | 43.9 | **29.3** | 44.7 | 199.8 | MLX |
| mul 500k | 124.5 | **65.9** | 85.3 | 76.3 | 190.8 | PyTorch |
| exp 100k | 133.2 | 71.7 | **60.4** | 66.8 | 228.4 | MLX |
| exp 500k | 254.3 | 161.1 | 226.2 | **107.5** | 230.9 | TensorFlow |
| relu 100k | 87.4 | 44.9 | **28.8** | 39.3 | 347.1 | MLX |
| softmax 8x128 | **3.9** | 39.7 | 17.0 | 10.2 | 699.7 | Peregrine |
| softmax 8x512 | 15.3 | 40.1 | 19.1 | **12.9** | 628.6 | TensorFlow |
| MLP fwd 64x784 | 28.5 | **28.4** | 52.8 | 250.4 | 1830.8 | PyTorch |
| train step 64 | 1030.5 | 1462.1 | **782.4** | 8414.2 | 24801.1 | MLX |

**Geometric mean (Peregrine / framework):**
- vs PyTorch: **1.12x** (Peregrine slightly slower)
- vs MLX: **1.16x** (Peregrine slightly slower)
- vs TensorFlow: **0.66x** (Peregrine 1.5x faster)
- vs tinygrad: **0.14x** (Peregrine 7x faster)

**Wins by framework:** PyTorch 5/14, MLX 5/14, Peregrine 2/14, TensorFlow 2/14

### Analysis

**Where Peregrine wins:**
- Softmax 8x128: 3.9 us — 10x faster than PyTorch (39.7 us), 4x faster than MLX (17 us). Our fused log-sum-exp implementation avoids framework dispatch overhead.
- Matmul 512x512: 162.3 us — edged out PyTorch (165.2) and MLX (173.7). Direct Accelerate/sgemm dispatch with minimal wrapper overhead.
- MLP forward: 28.5 us — tied with PyTorch, 2x faster than MLX. Matmul advantage carries through.
- Training step: 1031 us — 1.4x faster than PyTorch (1462), but 1.3x slower than MLX (782).

**Where Peregrine loses:**
- Elementwise ops are 2-4x slower than both PyTorch and MLX at 100K elements.
- Root cause: allocation overhead. Every op allocates a fresh `Vec<f32>`. At 100K elements (400KB), malloc+memset cost is a significant fraction of total time. PyTorch and MLX both use memory pool allocators.
- Secondary: PyTorch and MLX use hand-tuned NEON SIMD intrinsics. Peregrine relies on rustc autovectorization.

### Next steps

Created two milestones in Linear:
1. **Close Elementwise & Matmul Performance Gap vs PyTorch** — buffer pool, NEON intrinsics, Rayon tuning
2. **Beat MLX on CPU Wall-Clock Performance** — same fundamentals, plus training step optimization

Target: geometric mean < 0.85x vs both PyTorch and MLX.
