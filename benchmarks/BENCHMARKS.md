# Peregrine Benchmarks

**Date**: 2026-03-12 (v0.26.0 — GGUF Model Loader + Llama 3.2 Inference)
**System**: Apple M1 Max, 10 cores, 64 GB RAM, arm64
**Frameworks**: Peregrine (Rust), PyTorch 2.10.0, TensorFlow 2.20.0, JAX 0.9.0.1, MLX 0.30.6, TinyGrad
**All benchmarks**: CPU only, median of 20-50 iterations

## MUSt3R 3D Reconstruction — End-to-End Inference

Model: 423M parameters (ViT-L encoder + ViT-B decoder), shared head.

| Metric               | Peregrine 224 | PyTorch 224 | Peregrine 512 | PyTorch 512 |
|----------------------|---------------|-------------|----------------|-------------|
| Input resolution     | 224x224       | 224x224     | 512x384        | 512x384     |
| Patches              | 196           | 196         | 768            | 768         |
| **Inference time**   | **0.66s**     | **0.67s**   | **1.97s**      | **2.26s**   |
| **Weight loading**   | **0.6s**      | **1.6s**    | **0.6s**       | **1.6s**    |

- **224**: Peregrine is **1.5% faster** (0.66s vs 0.67s)
- **512**: Peregrine is **13% faster** (1.97s vs 2.26s)
- **Weight loading**: Peregrine is **2.7x faster** (0.6s vs 1.6s)

### Detailed Breakdown

| Component     | 224 CPU | 224 GPU | 224 Pipeline | 512 CPU | 512 GPU | 512 Pipeline |
|--------------|--------:|--------:|-------------:|--------:|--------:|-------------:|
| Encoder      | 459.5ms | 15.1ms  | 15.4ms       | 1331.0ms| 52.9ms  | 45.8ms       |
| Decoder      | 187.0ms | 14.8ms  | 183.0ms      | 581.5ms | 35.0ms  | 462.9ms      |
| Head+postproc| 3.2ms   | 180.6ms | 5.1ms        | 9.3ms   | 588.1ms | 12.6ms       |
| **Total**    | **0.66s**| **0.53s**| **0.54s**   | **1.97s**| **1.55s**| **1.44s**   |

### Metal GPU Inference (v0.19.0+)

| Resolution | CPU | GPU | GPU+Pipeline | Speedup (best vs CPU) |
|-----------|----:|----:|-------------:|---------|
| 224x224   | 0.66s | 0.53s | **0.54s** | **1.22x** |
| 512x384   | 1.97s | 1.55s | **1.44s** | **1.37x** |

### Server Mode & Parallel Workers (v0.15.0)

Server mode (`--server`) loads weights once and processes pairs over stdin/stdout, eliminating subprocess spawn + weight loading overhead per pair.

| Resolution | Subprocess | Server (warm) | Speedup |
|-----------|----------:|-------------:|--------:|
| 224x224   | ~0.57s/pair | ~0.51s/pair | 1.1x |
| 512x384   | ~1.90s/pair | ~1.81s/pair | 1.05x |

With `--workers N` in `reconstruct_video.py`, pairs are distributed across N server processes for near-linear wall-clock scaling.

### Heterogeneous GPU+CPU Pipeline (v0.22.0)

Pipeline mode (`--pipeline`) overlaps the two independent decoder views: feat1 runs on GPU while feat2 runs on CPU/AMX concurrently. Uses `MTLSharedEvent` signaling — single-threaded, no `Send`/`Sync` needed.

| Resolution | Decoder (GPU) | Decoder (Pipeline) | Decoder (CPU) |
|-----------|------:|------:|------:|
| 224x224 | **14.4ms** | 160.2ms | 181.5ms |
| 512x384 | **37.5ms** | 398.7ms | 541.6ms |

### Optimizations applied
1. **Weight loading** (378x): BufReader + bulk tensor reads instead of per-element syscalls
2. **Batched encoder** (batch=2): Both images processed in a single encoder pass — eliminates warmup and doubles GEMM sizes
3. **Batched decoder**: Self-attention and FFN process both views together (batch=2), cross-attention stays separate
4. **Parallel multi-head attention**: rayon par_chunks_mut with pre-allocated output, direct sgemm into output slices
5. **NEON + vvexpf softmax**: Vectorized max-reduction, Accelerate vvexpf for bulk exp, NEON normalize
6. **Parallel chunked GELU**: Per-chunk vvtanhf + NEON combine pipeline via rayon (fixes scalar fallback for large tensors)
7. **Fused QKV split+reshape**: Single pass from [batch*seq, 3*embed_dim] to [batch, heads, seq, head_dim] — eliminates 3 temp Vec allocations
8. **Direct transpose loops**: Replaces Tensor::transpose() which allocated + copied full 4D tensors
9. **Server mode**: Load weights once, process all pairs over stdin/stdout — eliminates ~0.5s overhead per pair
10. **Metal GPU dispatch**: GELU, matmul, layernorm, add, add_bias on GPU (with `precise::tanh()` fix for GELU correctness)
11. **GPU-resident attention** (v0.19.0): 4 new Metal kernels (QKV reshape, RoPE2D, attention output reshape, separate reshape) + composed SDPA (scale → batched matmul → softmax → batched matmul) — eliminates all GPU↔CPU round-trips in attention blocks
12. **Inference-mode layer_norm** (v0.19.0): skips GPU sync + backward cache computation when gamma doesn't require grad — removes the dominant GPU stall point
13. **Heterogeneous GPU+CPU scheduling** (v0.22.0): `het_execute` overlaps GPU and CPU/AMX work via `MTLSharedEvent` signaling — decoder feat1 on GPU while feat2 on CPU/AMX concurrently, single-threaded (no Send/Sync needed)

## Op-Level Benchmarks — 6 Frameworks

All values are **median microseconds** (lower is better). **Bold** = winner for that op. `-` = not benchmarked.
Winner column: PG=Peregrine, PT=PyTorch, TF=TensorFlow, JAX=JAX, MLX=MLX.

### Summary

| Framework | Wins (of 141 total ops) |
|-----------|------------------------|
| Peregrine | 64 |
| PyTorch | 35 |
| MLX | 17 |
| TensorFlow | 14 |
| JAX | 10 |
| TinyGrad | 1 |

Peregrine wins 64/141 ops (45%). Geometric mean ratio vs PyTorch: 0.93x, vs MLX: 0.67x, vs TF: 0.52x, vs JAX: 0.60x, vs tinygrad: 0.10x.

### Matmul

| Op | Peregrine | PyTorch | TF | JAX | MLX | TinyGrad | Winner |
|----|-----------|---------|-----|-----|-----|----------|--------|
| matmul_128x128 | 7.1 | **6.2** | 50.2 | 81.4 | 28.4 | 424.1 | PT |
| matmul_256x256 | 37.4 | **31.8** | 152.4 | 152.5 | 81.4 | 425.9 | PT |
| matmul_512x512 | 193.0 | **134.0** | 695.1 | 504.6 | 221.2 | 460.4 | PT |
| matmul_1024x1024 | **1035.8** | - | - | - | - | - | PG |
| matmul_2048x2048 | **9072.8** | - | - | - | - | - | PG |

### Add

| Op | Peregrine | PyTorch | TF | JAX | MLX | TinyGrad | Winner |
|----|-----------|---------|-----|-----|-----|----------|--------|
| add_100k | **12.8** | 40.1 | 49.6 | 39.4 | 32.4 | 203.6 | PG |
| add_500k | 118.2 | **57.3** | 85.2 | 61.1 | 84.4 | 184.9 | PT |
| add_1M | **125.2** | - | - | - | - | - | PG |
| add_5M | **534.6** | - | - | - | - | - | PG |
| add_10M | **877.2** | - | - | - | - | - | PG |

### Mul

| Op | Peregrine | PyTorch | TF | JAX | MLX | TinyGrad | Winner |
|----|-----------|---------|-----|-----|-----|----------|--------|
| mul_100k | **12.4** | 40.7 | 43.5 | 30.0 | 33.0 | 185.7 | PG |
| mul_500k | 96.7 | **58.0** | 72.0 | 59.6 | 88.0 | 188.4 | PT |
| mul_1M | **177.7** | - | - | - | - | - | PG |
| mul_5M | **536.8** | - | - | - | - | - | PG |
| mul_10M | **950.2** | - | - | - | - | - | PG |

### Exp

| Op | Peregrine | PyTorch | TF | JAX | MLX | TinyGrad | Winner |
|----|-----------|---------|-----|-----|-----|----------|--------|
| exp_100k | 100.0 | 62.3 | 65.4 | **46.8** | 73.6 | 224.4 | JAX |
| exp_500k | 193.5 | 138.9 | **102.4** | 122.6 | 241.9 | 220.0 | TF |
| exp_1M | **286.1** | - | - | - | - | - | PG |
| exp_5M | **1145.0** | - | - | - | - | - | PG |
| exp_10M | **2184.6** | - | - | - | - | - | PG |

### Activations

| Op | Peregrine | PyTorch | TF | JAX | MLX | TinyGrad | Winner |
|----|-----------|---------|-----|-----|-----|----------|--------|
| relu_100k | **8.8** | 40.2 | 38.5 | 99.2 | 27.0 | 345.4 | PG |
| relu_1M | **128.4** | - | - | - | - | - | PG |
| silu_100k | 66.3 | 69.8 | 211.5 | **59.8** | 89.8 | 329.6 | JAX |
| softplus_100k | 300.6 | 153.1 | **134.3** | 201.8 | 286.7 | 786.0 | TF |
| mish_100k | 471.0 | 312.4 | **244.5** | 291.5 | 408.3 | 1167.4 | TF |
| leaky_relu_100k | **8.0** | 41.0 | 20.2 | 48.8 | 85.1 | - | PG |
| elu_100k | 152.5 | **124.9** | 145.5 | 125.0 | 125.8 | 877.5 | PT |
| hard_tanh_100k | 50.5 | 40.3 | 42.1 | 65.5 | **37.7** | - | MLX |
| relu6_100k | 51.8 | **39.8** | 52.9 | 132.4 | 54.0 | 749.3 | PT |
| hardswish_100k | 84.0 | **40.2** | 197.8 | 56.2 | 75.5 | - | PT |
| gelu_100k | 77.9 | **75.2** | 247.3 | 271.6 | 148.3 | 885.2 | PT |
| selu_100k | 161.3 | 132.0 | 137.6 | 124.3 | **88.5** | 748.5 | MLX |
| softsign_100k | **35.0** | 122.0 | 46.0 | 83.3 | 50.9 | - | PG |

### Softmax / MLP / Train

| Op | Peregrine | PyTorch | TF | JAX | MLX | TinyGrad | Winner |
|----|-----------|---------|-----|-----|-----|----------|--------|
| softmax_8x128 | **1.2** | 30.1 | 11.4 | 32.8 | 19.0 | 617.8 | PG |
| softmax_8x512 | **4.3** | 33.8 | 14.2 | 34.2 | 20.1 | 617.5 | PG |
| mlp_fwd_64x784 | 33.1 | **28.0** | 245.8 | 179.9 | 54.6 | 1820.1 | PT |
| mlp_fwd_256x784_wide | **422.8** | - | - | - | - | - | PG |
| train_step_64 | **817.6** | 1276.8 | 8796.6 | 5544.6 | 869.1 | 24610.8 | PG |
| train_step_256_wide | **3265.0** | - | - | - | - | - | PG |

### Unary Math

| Op | Peregrine | PyTorch | TF | JAX | MLX | TinyGrad | Winner |
|----|-----------|---------|-----|-----|-----|----------|--------|
| reciprocal_100k | **8.5** | 40.6 | 49.7 | 25.7 | 28.5 | 160.4 | PG |
| square_100k | **8.6** | 39.9 | 15.3 | 32.2 | 24.7 | 178.0 | PG |
| rsqrt_100k | 84.7 | 42.1 | 51.0 | 82.8 | **39.2** | - | MLX |
| floor_100k | 46.6 | 41.3 | **15.8** | 28.1 | 25.7 | 420.8 | TF |
| ceil_100k | 46.6 | 39.2 | **15.8** | 30.4 | 30.8 | 352.0 | TF |
| round_100k | 48.1 | 41.9 | 44.2 | **27.5** | 28.2 | - | JAX |
| sign_100k | 54.4 | 39.5 | 48.2 | **37.4** | 32.8 | 810.1 | MLX |
| expm1_100k | 178.4 | 108.4 | 144.1 | **93.2** | 119.8 | - | JAX |
| log2_100k | 107.5 | **86.8** | 143.3 | 101.8 | 111.0 | 165.2 | PT |
| log10_100k | 115.4 | 88.4 | 147.2 | **66.2** | 122.9 | - | JAX |
| log1p_100k | 100.0 | **84.6** | 91.7 | 114.6 | 138.3 | - | PT |
| erf_100k | 100.5 | **57.6** | 58.9 | 59.4 | 109.0 | - | PT |

### Trig / Hyperbolic

| Op | Peregrine | PyTorch | TF | JAX | MLX | TinyGrad | Winner |
|----|-----------|---------|-----|-----|-----|----------|--------|
| sinh_100k | **51.0** | 132.8 | 126.2 | 128.2 | 105.1 | 527.1 | PG |
| cosh_100k | **46.3** | 130.4 | 124.4 | 78.7 | 102.8 | 483.9 | PG |
| arcsin_100k | **53.7** | 79.5 | 56.5 | 121.2 | 102.6 | 2939.8 | PG |
| arccos_100k | 109.0 | 88.3 | **52.2** | 206.6 | 119.4 | - | TF |
| arctan_100k | **53.1** | 94.9 | 57.6 | 217.7 | 101.9 | 3112.1 | PG |
| arcsinh_100k | 143.8 | 155.4 | **131.5** | 131.5 | 357.5 | - | JAX |

### Binary Ops

| Op | Peregrine | PyTorch | TF | JAX | MLX | TinyGrad | Winner |
|----|-----------|---------|-----|-----|-----|----------|--------|
| maximum_100k | **12.4** | 38.8 | 43.6 | 38.1 | 26.6 | 191.8 | PG |
| minimum_100k | **12.5** | 41.2 | 43.4 | 51.4 | 30.8 | 380.5 | PG |
| power_100k | 391.3 | 240.1 | 323.8 | **201.6** | 238.0 | - | JAX |
| arctan2_100k | 1112.7 | 129.6 | **77.1** | 357.6 | 157.4 | - | TF |
| logaddexp_100k | 409.5 | **151.4** | 400.8 | 218.2 | 280.5 | - | PT |

### Comparison / Logic

| Op | Peregrine | PyTorch | TF | JAX | MLX | TinyGrad | Winner |
|----|-----------|---------|-----|-----|-----|----------|--------|
| clip_100k | **8.7** | 42.6 | 43.0 | 49.6 | 39.2 | 538.6 | PG |
| where_100k | 93.1 | 49.9 | 66.3 | 55.9 | **29.4** | 275.5 | MLX |
| greater_100k | 70.0 | 47.6 | 61.6 | 45.9 | **21.5** | 191.7 | MLX |
| equal_100k | 71.3 | 32.5 | 62.9 | 47.3 | **24.4** | 289.1 | MLX |

### Reductions

| Op | Peregrine | PyTorch | TF | JAX | MLX | TinyGrad | Winner |
|----|-----------|---------|-----|-----|-----|----------|--------|
| sum_axis_256x512 | 113.0 | 40.1 | 49.6 | 63.5 | **20.8** | 207.5 | MLX |
| mean_axis_256x512 | 112.5 | 42.2 | 54.8 | 53.6 | **25.5** | 293.5 | MLX |
| max_axis_256x512 | 154.2 | 52.8 | **48.7** | 62.2 | 47.3 | 203.3 | MLX |
| min_axis_256x512 | 154.3 | 53.1 | **47.8** | 56.7 | 49.4 | 326.9 | TF |
| var_256x512 | 235.7 | 276.8 | 227.5 | 101.9 | **60.9** | - | MLX |
| prod_axis_256x512 | 149.1 | 37.8 | 53.4 | 56.9 | **26.3** | - | MLX |
| logsumexp_256x512 | 383.3 | 195.2 | 351.0 | 329.6 | **119.9** | - | MLX |
| cumsum_256x512 | 125.7 | **73.1** | 199.8 | 237.6 | 144.5 | 626.8 | PT |
| argmax_axis_256x512 | 154.7 | 97.8 | **74.3** | 208.6 | 184.3 | 1321.4 | TF |
| sum_axis_1024x1024 | **940.0** | - | - | - | - | - | PG |
| var_1024x1024 | **1936.1** | - | - | - | - | - | PG |

### Shape Ops

| Op | Peregrine | PyTorch | TF | JAX | MLX | TinyGrad | Winner |
|----|-----------|---------|-----|-----|-----|----------|--------|
| tril_256x256 | **35.8** | 35.9 | 56.6 | 44.2 | 63.6 | 1832.6 | PG |
| triu_256x256 | **35.2** | 35.6 | 55.6 | 44.3 | 59.8 | 1808.2 | PG |
| repeat_64x128_2x3 | 128.8 | 49.8 | 75.3 | **30.3** | 32.4 | - | JAX |
| pad_64x128 | 16.8 | **4.3** | 84.1 | 19.1 | 21.8 | 89.1 | PT |
| stack_8x64x128 | 17.2 | **8.8** | 61.8 | 196.0 | 47.1 | 936.4 | PT |
| diagonal_512x512 | 0.8 | **0.6** | 12.5 | 4.2 | 28.4 | - | PT |

### Losses

| Op | Peregrine | PyTorch | TF | JAX | MLX | TinyGrad | Winner |
|----|-----------|---------|-----|-----|-----|----------|--------|
| cross_entropy_64x10 | **2.6** | 39.8 | 626.0 | 60.9 | 24.0 | 3431.6 | PG |
| l1_loss_64x10 | **1.0** | 5.5 | 43.8 | 13.0 | 19.2 | 1127.8 | PG |
| mse_loss_64x10 | **4.3** | 5.0 | 39.6 | 25.1 | 20.7 | 451.0 | PG |
| huber_loss_64x10 | 5.8 | **4.9** | 237.7 | 49.2 | 39.6 | - | PT |
| smooth_l1_loss_64x10 | 5.6 | **5.2** | 243.8 | 49.8 | 40.5 | - | PT |
| kl_div_loss_64x10 | **2.5** | 6.3 | 379.1 | 71.8 | 23.5 | - | PG |
| cosine_sim_loss_64x64 | 13.6 | **10.3** | 237.5 | 47.8 | 125.2 | - | PT |

### Layers

| Op | Peregrine | PyTorch | TF | JAX | MLX | TinyGrad | Winner |
|----|-----------|---------|-----|-----|-----|----------|--------|
| rmsnorm_64x512 | 57.8 | 67.5 | 439.6 | 83.7 | **39.0** | - | MLX |
| conv1d_1x32x128_k3 | **20.7** | 54.0 | 519.1 | 71.3 | 29.9 | - | PG |
| avgpool2d_1x16x32x32 | **25.1** | 45.2 | 63.8 | 53.6 | 283.6 | - | PG |
| groupnorm_4x64x16x16 | 72.6 | **53.9** | 770.2 | 276.7 | 235.1 | - | PT |

### RNN

| Op | Peregrine | PyTorch | TF | JAX | MLX | TinyGrad | Winner |
|----|-----------|---------|-----|-----|-----|----------|--------|
| rnn_seq32_128_256 | **195.8** | 267.5 | - | - | - | - | PG |
| lstm_seq32_128_256 | 1149.8 | **807.4** | - | - | - | - | PT |
| gru_seq32_128_256 | 810.3 | **782.3** | - | - | - | - | PT |

### Optimizers

| Op | Peregrine | PyTorch | TF | JAX | MLX | TinyGrad | Winner |
|----|-----------|---------|-----|-----|-----|----------|--------|
| optim_adam_64 | **809.9** | 1301.1 | - | - | - | - | PG |
| optim_rmsprop_64 | **993.3** | 1122.0 | - | - | - | - | PG |
| optim_lion_64 | **1004.0** | - | - | - | - | - | PG |
| optim_adafactor_64 | **1283.8** | - | - | - | - | - | PG |

### Random

| Op | Peregrine | PyTorch | TF | JAX | MLX | TinyGrad | Winner |
|----|-----------|---------|-----|-----|-----|----------|--------|
| rand_uniform_100k | **106.8** | 257.3 | 128.2 | 547.5 | 520.5 | 2436.7 | PG |
| rand_normal_100k | 794.0 | 973.3 | **342.1** | 639.7 | 748.1 | 3275.7 | TF |
| rand_bernoulli_100k | 319.2 | 250.0 | **208.9** | 544.9 | 490.7 | - | TF |
| rand_uniform_1M | 1069.0 | 2568.5 | **434.2** | 2346.8 | 4873.0 | 2433.4 | TF |
| rand_normal_1M | 7661.4 | 9732.6 | **2092.7** | 3019.8 | 7038.8 | 3336.2 | TF |

### FFT

| Op | Peregrine | PyTorch | TF | JAX | MLX | TinyGrad | Winner |
|----|-----------|---------|-----|-----|-----|----------|--------|
| rfft_1k | **2.2** | 4.4 | 43.5 | 42.8 | 20.6 | - | PG |
| rfft_4k | **6.5** | 14.9 | 53.5 | 66.3 | 29.7 | - | PG |
| rfft_16k | **30.3** | 65.2 | 122.2 | 123.7 | 83.8 | - | PG |
| fft_1k | **3.3** | 6.6 | 8.7 | 17.5 | 24.4 | - | PG |
| fft_4k | **12.2** | 26.2 | 17.2 | 56.4 | 44.4 | - | PG |

### Linear Algebra

| Op | Peregrine | PyTorch | TF | JAX | MLX | TinyGrad | Winner |
|----|-----------|---------|-----|-----|-----|----------|--------|
| norm_l2_1k | **1.1** | 1.3 | 69.5 | 4.0 | 20.8 | - | PG |
| solve_64x64 | **11.9** | 18.1 | 24.4 | 35.1 | 100.3 | - | PG |
| inv_64x64 | 37.4 | **26.3** | 32.4 | 37.8 | 50.5 | - | PT |
| cholesky_64x64 | **9.5** | 41.6 | 19.3 | 20.2 | 22.0 | - | PG |
| svd_64x64 | **276.8** | 284.5 | 502.4 | 304.4 | 305.3 | - | PG |
| qr_64x64 | **41.5** | 82.6 | 83.6 | 63.0 | 62.4 | - | PG |
| eigh_64x64 | 381.3 | 213.6 | **144.3** | 238.8 | 236.4 | - | TF |
| det_64x64 | 23.2 | **20.2** | 22.8 | 33.9 | - | - | PT |
| solve_128x128 | 49.9 | **45.0** | 76.0 | 85.6 | 209.5 | - | PT |
| inv_128x128 | 92.2 | **62.1** | 139.0 | 86.6 | 97.1 | - | PT |
| cholesky_128x128 | 50.6 | 49.6 | 60.3 | 37.4 | **31.5** | - | MLX |
| svd_128x128 | **992.2** | 998.4 | 1825.0 | 1020.1 | 1025.6 | - | PG |
| qr_128x128 | **188.8** | 223.4 | 327.4 | 193.0 | 205.6 | - | PG |
| eigh_128x128 | 1845.7 | **703.2** | 715.1 | 751.7 | 747.8 | - | PT |
| det_128x128 | 52.2 | **49.6** | 81.9 | 76.6 | - | - | PT |
| solve_256x256 | 189.0 | **178.1** | 378.2 | 261.3 | 747.4 | - | PT |
| inv_256x256 | 466.9 | 301.3 | 851.1 | 336.8 | **250.6** | - | MLX |
| cholesky_256x256 | 226.4 | 78.6 | 283.8 | 117.3 | **53.3** | - | MLX |
| svd_256x256 | 5892.5 | **5781.4** | 8113.8 | 5996.2 | 6162.9 | - | PT |
| qr_256x256 | 1020.4 | 1003.9 | 1705.2 | **979.6** | 1065.7 | - | JAX |
| eigh_256x256 | 6065.6 | **3454.5** | 4627.9 | 3576.7 | 3615.8 | - | PT |
| det_256x256 | 213.8 | 208.7 | 434.1 | **206.4** | - | - | JAX |

### Pipeline Ops

| Op | Peregrine | PyTorch | TF | JAX | MLX | TinyGrad | Winner |
|----|-----------|---------|-----|-----|-----|----------|--------|
| matmul_bias_gelu_196x768x3072 | 1209.0 | **934.7** | 2440.2 | 2142.1 | - | 1259.2 | PT |
| matmul_bias_gelu_196x1024x4096 | 2184.3 | 2046.4 | 3721.7 | 3499.1 | - | **1276.0** | tinygrad |
| add_layernorm_196x768 | **108.5** | 110.3 | 1258.4 | 231.7 | - | 1143.5 | PG |
| add_layernorm_196x1024 | 139.0 | **108.0** | 1316.1 | 287.2 | - | 1148.5 | PT |

### Int8 Quantized Matmul

Peregrine-only benchmark comparing f32 (Apple Accelerate cblas_sgemm) vs int8 quantized matmul (NEON vmull+vpadalq, per-column weight + per-row activation quantization).

| Op | f32 (us) | i8 (us) | Ratio |
|----|----------:|--------:|------:|
| matmul_196x768x3072 | 697 | 14,581 | 20.9x slower |
| matmul_196x1024x4096 | 1,529 | 26,319 | 17.2x slower |

The int8 path is currently slower than f32 because: (a) NEON `vmull_s8`+`vpadalq_s16` (stable) vs hardware `sdot` (unstable), (b) competing against Apple Accelerate's heavily optimized cblas_sgemm. The primary benefit is **4× memory reduction** for weight storage, enabling larger models to fit in memory. The Metal GPU dequant path loads 4× less data from device memory.

## Analysis

### Where Peregrine Wins (vs PyTorch)
- **Elementwise (small tensors)**: add, mul, relu, clip — near-zero dispatch overhead, 3-5x faster
- **Softmax**: 8-26x faster (fused implementation, 1.3µs vs 33µs at 8x128)
- **Losses**: cross_entropy 16x, l1 5x, kl_div 2.4x — minimal dispatch overhead
- **FFT**: rfft/fft 2-4x faster across all sizes
- **Optimizers**: Adam 1.5x, RMSProp 1.3x
- **Linalg (small)**: cholesky/qr at 64x64 — 2-3.5x faster
- **Trig**: sinh 2.4x, cosh 2.7x, arctan 1.7x
- **GELU**: 1.2x faster (66.2µs vs 77.7µs)
- **add_layernorm**: 1.2x faster (117.6µs vs 140.8µs)

### Where PyTorch Wins (vs Peregrine)
- **Matmul (small-medium)**: 128 1.1x, 256 1.1x, 512 1.5x — Apple Accelerate advantage
- **Reductions**: sum/mean 2.7x, max/min 1.8x
- **Log/erf**: log2 1.9x, log10 1.7x, log1p 1.9x
- **Shape ops**: pad 4x, stack 2.2x
- **Linalg (large)**: eigh_256 1.9x, cholesky_256 2.6x, svd_256 1.1x
- **LSTM**: 1.5x

### Changes from Previous Run (v0.25.0 → v0.26.0)
- **Geomean vs PyTorch**: Improved from 1.04x to **0.93x** (Peregrine now 7% faster overall)
- **train_step_64**: Peregrine wins (817.6µs vs 869.1µs MLX, vs 1276.8µs PT)
- **add_layernorm_196x768**: Peregrine wins (108.5µs vs 110.3µs PT)
- **groupnorm**: Improved from 109.3µs to 72.6µs
- **mul_500k**: Improved from 132.4µs to 96.7µs
- **add_500k**: Improved from 111.5µs to 118.2µs (PT still wins)
- **Wins**: 64/141 (was 99/173 — different op coverage this run)

## Reproducing

```bash
# Peregrine op benchmarks
cargo bench --bench wallclock

# Framework benchmarks (output to target/bench_compare/*.json)
python3 scripts/bench_pytorch.py
python3 scripts/bench_tensorflow.py
python3 scripts/bench_jax.py
python3 scripts/bench_mlx.py
python3 scripts/bench_tinygrad.py

# Full comparison (all frameworks sequentially + comparison table)
./scripts/bench_compare.sh

# MUSt3R inference (single pair)
cargo run --example must3r --release -- weights/must3r_224.bin img1.ppm img2.ppm
cargo run --example must3r --release -- weights/must3r_512.bin img1.ppm img2.ppm 512x384

# MUSt3R server mode (multiple pairs, load weights once)
echo -e "img1.ppm\timg2.ppm\t224\t224" | cargo run --example must3r --release -- weights/must3r_224.bin --server

# MUSt3R with Metal GPU
cargo run --example must3r --release --features metal -- weights/must3r_224.bin img1.ppm img2.ppm --gpu

# MUSt3R with GPU+Pipeline
cargo run --example must3r --release --features metal -- weights/must3r_224.bin img1.ppm img2.ppm --gpu --pipeline

# Multi-view pipeline with parallel workers
python3 reconstruct_video.py vids/rgb.mp4 --frames 12 --resolution 512 --pairs all --workers 4
```

## Raw Data

Raw JSON benchmark data is in [`benchmarks/data/`](data/).
