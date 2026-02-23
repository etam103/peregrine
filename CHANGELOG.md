# Changelog

All notable changes to Peregrine are documented here.
Benchmark numbers included for performance-related changes.

**Hardware:** Apple Silicon (M-series), macOS, f32 precision

---

## [0.8.0] - 2026-02-22

### Added — NEON intrinsics & elementwise dominance

**Hand-tuned NEON kernels** (`src/simd_kernels.rs` — new, ~530 lines)
- 14 NEON intrinsic kernels processing 4 f32s per iteration via `float32x4_t`
- Forward: `vec_add_f32`, `vec_sub_f32`, `vec_mul_f32`, `vec_div_f32`, `vec_neg_f32`, `vec_abs_f32`, `vec_scale_f32`, `vec_relu_f32`, `vec_add_inplace_f32`
- Transcendental: `vec_exp_f32` (Cephes-style polynomial, ~1.2e-7 relative error), `vec_sigmoid_f32`, `vec_tanh_f32`, `vec_gelu_f32`
- Backward: `vec_relu_backward_f32`, `vec_abs_backward_f32`, `vec_tanh_backward_f32`, `vec_sigmoid_backward_f32`
- All gated with `#[cfg(target_arch = "aarch64")]`; non-ARM targets use scalar fallback

**NEON Adam optimizer** (`src/simd_kernels.rs`)
- `adam_step_f32` vectorizes the 10+ FLOPs/element Adam inner loop
- Uses `vrsqrteq_f32` + Newton step for fast approximate sqrt
- Integrated in `src/optim.rs` via cfg-gated dispatch

**Pool bypass for small tensors** (`src/cpu_pool.rs`)
- `MIN_POOL_SIZE = 1024`: tensors < 1024 elements skip the HashMap pool
- Eliminates ~18 HashMap lookups per MLP forward pass where pool overhead exceeds malloc savings

**Integration** (`src/tensor.rs` — ~25 call sites)
- All forward ops (add, sub, mul, div, neg, abs, scale, relu, exp, sigmoid, tanh, gelu) dispatch to NEON kernels in the single-threaded path
- All backward ops (mul, relu, sigmoid, exp, scale, sub/neg, abs, tanh) dispatch to NEON kernels
- `accumulate_grad` uses `vec_add_inplace_f32`

### Benchmark Results (CPU, Apple Silicon, all times in microseconds)

Before (v0.7.0) → After (v0.8.0):

| Operation | Before (us) | After (us) | Speedup |
|-----------|------------:|-----------:|--------:|
| add 100k | 73.2 | 12.5 | 5.9x |
| mul 100k | 73.8 | 12.5 | 5.9x |
| relu 100k | 41.0 | 8.8 | 4.7x |
| MLP fwd | 37.1 | 32.6 | 1.1x |
| train step | 1135.7 | 809.1 | 1.4x |

**Geometric mean (Peregrine / framework):**
- vs PyTorch: 1.01x → **0.70x** (Peregrine 1.4x faster)
- vs MLX: 0.97x → **0.57x** (Peregrine 1.8x faster)
- vs TensorFlow: 0.61x → **0.40x**
- vs tinygrad: 0.12x → **0.08x**
- vs JAX: 0.49x → **0.39x**

**Wins:** Peregrine 8/14, PyTorch 3/14, JAX 2/14, TensorFlow 1/14

---

## [0.7.0] - 2026-02-22

### Added — CPU performance optimizations (close gap vs PyTorch & MLX)

**CPU buffer pool** (`src/cpu_pool.rs`)
- Thread-local, size-bucketed `HashMap<usize, Vec<Vec<f32>>>` with power-of-2 keys
- `pool_get(len)` / `pool_recycle(buf)` — eliminates malloc on every forward & backward op
- `Drop` impl on `TensorInner` auto-recycles `data` and `grad` buffers
- Converted ~20 forward ops and ~15 backward ops from `.collect()` to pool-allocated loops

**SIMD auto-vectorization** (`.cargo/config.toml`)
- `target-cpu=apple-m1` enables full NEON/ASIMD instruction set
- Simple loops like `data[i] = a[i] + b[i]` now emit vectorized `fadd v0.4s`

**Rayon threshold tuning**
- Dual thresholds: `PAR_THRESHOLD_CHEAP = 500_000` (add, mul, relu, etc.), `PAR_THRESHOLD_EXPENSIVE = 100_000` (exp, log, sqrt, etc.)
- Avoids ~15us Rayon spawn overhead on ops that complete in <10us single-threaded

**Adam/SGD gradient borrow fix** (`src/optim.rs`)
- Borrow `grad` and `data` fields in-place via split `&mut` instead of cloning entire gradient Vec

**JAX benchmark** (`scripts/bench_jax.py`)
- Added JAX 0.9.0.1 to the 6-framework comparison suite

### Benchmark Results (CPU, Apple Silicon, all times in microseconds)

Before → After (Peregrine medians):

| Operation | Before (us) | After (us) | Speedup |
|-----------|------------:|-----------:|--------:|
| relu 100k | 87.4 | 41.0 | 2.13x |
| add 100k | 121.3 | 73.2 | 1.66x |
| mul 100k | 116.8 | 73.8 | 1.58x |
| mul 500k | 124.5 | 109.3 | 1.14x |
| add 500k | 159.8 | 118.9 | 1.34x |
| train step | 1030.5 | 1135.7 | 0.91x |

**Geometric mean (Peregrine / framework):**
- vs PyTorch: 1.12x → **1.01x** (parity)
- vs MLX: 1.16x → **0.97x** (Peregrine now faster)
- vs TensorFlow: 0.66x → **0.61x**
- vs tinygrad: 0.14x → **0.12x**
- vs JAX: **0.49x** (Peregrine 2x faster)

**Wins:** MLX 5/14, Peregrine 3/14, PyTorch 3/14, TensorFlow 2/14, JAX 1/14

---

## [0.6.0] - 2026-02-22

### Added — Multi-framework wall-clock benchmark suite
- `benches/wallclock.rs` — Peregrine benchmark (standalone binary, JSON output)
- `scripts/bench_pytorch.py` — PyTorch 2.10.0 benchmark
- `scripts/bench_mlx.py` — MLX 0.30.6 benchmark
- `scripts/bench_tensorflow.py` — TensorFlow 2.20.0 benchmark
- `scripts/bench_tinygrad.py` — tinygrad 0.12.0 benchmark
- `scripts/compare_bench.py` — multi-framework comparison table (markdown output)
- `scripts/bench_compare.sh` — orchestrator: builds, runs all 5 frameworks sequentially with `nice -n 10`

14 operations benchmarked: matmul (128/256/512), add, mul, exp (100K/500K), relu, softmax (128/512), MLP forward, training step.

### Benchmark Results (CPU, Apple Silicon, all times in microseconds)

| Operation | Peregrine (us) | PyTorch (us) | MLX (us) | TensorFlow (us) | tinygrad (us) |
|-----------|---------------:|-------------:|---------:|----------------:|--------------:|
| matmul 512x512 | **162** | 165 | 174 | 676 | 434 |
| softmax 8x128 | **3.9** | 39.7 | 17.0 | 10.2 | 700 |
| MLP fwd 64x784 | **28.5** | 28.4 | 52.8 | 250 | 1831 |
| train step 64 | **1031** | 1462 | 782 | 8414 | 24801 |

**Geometric mean (Peregrine / framework):**
- vs PyTorch: 1.12x (near parity)
- vs MLX: 1.16x (near parity)
- vs TensorFlow: 0.66x (Peregrine 1.5x faster)
- vs tinygrad: 0.14x (Peregrine 7x faster)

**Wins:** PyTorch 5/14, MLX 5/14, Peregrine 2/14, TensorFlow 2/14

---

## [0.5.0] - 2026-02-22

### Added — Metal GPU Backend (`--features metal`)
- objc2-metal FFI foundation with safe Rust wrappers (GpuContext, GpuBuffer)
- 21 Metal compute shaders: add, sub, mul, div, neg, exp, log, sqrt, relu, sigmoid, tanh, sin, cos, abs, scale, matmul (fused bias+relu), sum, max, min, softmax, transpose, layernorm
- 35 GPU tests (12 basics + 23 CPU vs Metal parity)

| Operation | CPU | GPU | Speedup |
|-----------|-----|-----|---------|
| matmul 1024x1024 | 1.07 ms | 3.49 ms | CPU 3.3x (BLAS) |
| add 1M elements | 1.04 ms | 306 µs | GPU 3.4x |
| mul 1M elements | 1.17 ms | 294 µs | GPU 4.0x |
| exp 1M elements | 1.32 ms | 284 µs | GPU 4.6x |

### Added — Numerically stable log_softmax
- `x - max - log(sum(exp(x - max)))` with backward pass
- Fixes NaN crash in cross_entropy_loss after many epochs

### Added — MNIST end-to-end example
- MLP (784→128→64→10), Adam optimizer, 10 epochs, 97.5% test accuracy

### Added — PyTorch numerical parity (23 tests)
- Cross-validates matmul, softmax, log_softmax, layernorm, cross_entropy, 14 element-wise ops, Adam step, and full 10-step MLP training
- All within 1e-4 to 1e-7 absolute error

### Added — Criterion benchmark suite
- `cargo bench` / `cargo bench --features metal`
- Covers matmul, element-wise, softmax, MLP forward, training step

| Operation | Time |
|-----------|------|
| MLP forward batch=64 | 186 µs |
| Training step (fwd+bwd+Adam) batch=64 | ~3 ms |

## [0.4.0] - 2026-02-20

### Added — Core tensor ops and training infrastructure
- 11 element-wise ops with autograd: sub, div, neg, exp, log, sqrt, abs, pow, sin, cos, tanh
- Reduction/shape ops: mean, squeeze, unsqueeze, max, min, argmax, argmin
- Creation ops: ones, full, arange, linspace, eye
- NumPy-style broadcasting for add, sub, mul, div
- Neural network layers: Linear, Embedding, CrossEntropyLoss, MSELoss
- Optimizers: SGD (momentum, Nesterov, weight decay), Adam, AdamW
- LR schedulers: StepLR, CosineAnnealing, Warmup
- Gradient clipping (by norm and by value)

### Changed
- Restructured as `peregrine` library crate. RT-DETR moved to examples.
- Dropped dead YOLO code

## [0.3.0] - 2026-02-20

### Added
- RT-DETR architecture: multi-head attention, transformer encoder/decoder, ResNet backbone, learned object queries
- Hungarian matching and set-based loss for end-to-end detection training
- Global gradient clipping (max norm = 1.0)

### Changed
- Xavier fan-in weight initialization — fixes NaN loss in deep networks
- Multi-scale feature pooling (3,072 tokens vs 56,784), reducing attention memory from ~52 GB to ~150 MB

## [0.2.0] - 2026-02-20

### Performance
- BLAS acceleration via Apple Accelerate (matmul, conv2d 1x1)
- Rayon parallelism for element-wise ops (>10k threshold)
- Clone elimination in backward pass

## [0.1.0] - 2026-02-18

### Added
- Tensor with N-dimensional storage and shared ownership
- Reverse-mode autograd engine
- Forward/backward: add, mul, matmul, relu, sigmoid, sum, scale, add_bias
- SGD optimizer
- Object detection demo with ASCII visualization
