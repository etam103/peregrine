# Rustorch Performance Journal

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
