# Changelog

All notable changes to Peregrine are documented here.
Benchmark numbers included for performance-related changes.

**Hardware:** Apple Silicon (M-series), macOS, f32 precision

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

### Benchmark Results (CPU, Apple Silicon)

| Operation | Peregrine | PyTorch | MLX | TensorFlow | tinygrad |
|-----------|----------:|--------:|----:|-----------:|---------:|
| matmul 512x512 | **162 us** | 165 | 174 | 676 | 434 |
| softmax 8x128 | **3.9 us** | 39.7 | 17.0 | 10.2 | 700 |
| MLP fwd 64x784 | **28.5 us** | 28.4 | 52.8 | 250 | 1831 |
| train step 64 | **1031 us** | 1462 | 782 | 8414 | 24801 |

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
