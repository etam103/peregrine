# Peregrine Benchmarks

**Date**: 2026-03-12 (v0.23.0 — GQA + Speculative Decoding + Sparse Attention)
**System**: Apple M1 Max, 10 cores, 64 GB RAM, arm64
**Frameworks**: Peregrine (Rust), PyTorch 2.10.0, TensorFlow 2.20.0, JAX 0.9.0.1, MLX 0.30.6, TinyGrad
**All benchmarks**: CPU only, median of 20-50 iterations

## MUSt3R 3D Reconstruction — End-to-End Inference

Model: 423M parameters (ViT-L encoder + ViT-B decoder), shared head.

| Metric               | Peregrine 224 | PyTorch 224 | Peregrine 512 | PyTorch 512 |
|----------------------|---------------|-------------|----------------|-------------|
| Input resolution     | 224x224       | 224x224     | 512x384        | 512x384     |
| Patches              | 196           | 196         | 768            | 768         |
| **Inference time**   | **0.65s**     | **0.67s**   | **1.89s**      | **2.26s**   |
| **Weight loading**   | **0.6s**      | **1.6s**    | **0.6s**       | **1.6s**    |

- **224**: Peregrine is **3% faster** (0.65s vs 0.67s)
- **512**: Peregrine is **16% faster** (1.89s vs 2.26s)
- **Weight loading**: Peregrine is **2.7x faster** (0.6s vs 1.6s)

### Detailed Breakdown

| Component     | 224 CPU | 224 GPU | 224 Pipeline | 512 CPU | 512 GPU | 512 Pipeline |
|--------------|--------:|--------:|-------------:|--------:|--------:|-------------:|
| Encoder      | 451.4ms | 14.2ms  | 14.1ms       | 1294.8ms| 43.7ms  | 43.3ms       |
| Decoder      | 181.5ms | 14.4ms  | 160.2ms      | 541.6ms | 37.5ms  | 398.7ms      |
| Head+postproc| 3.0ms   | 180.4ms | 3.8ms        | 8.7ms   | 596.5ms | 14.6ms       |
| **Total**    | **0.65s**| **0.53s**| **0.50s**   | **1.89s**| **1.55s**| **1.36s**   |

### Metal GPU Inference (v0.19.0+)

| Resolution | CPU | GPU | GPU+Pipeline | Speedup (best vs CPU) |
|-----------|----:|----:|-------------:|---------|
| 224x224   | 0.65s | 0.53s | **0.50s** | **1.30x** |
| 512x384   | 1.89s | 1.55s | **1.36s** | **1.39x** |

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
| PyTorch | 31 |
| MLX | 16 |
| JAX | 16 |
| TensorFlow | 13 |
| TinyGrad | 1 |

Peregrine wins 64/141 ops. Geometric mean ratio vs PyTorch: 0.94x (Peregrine faster), vs MLX: 0.68x, vs TF: 0.51x, vs JAX: 0.65x, vs tinygrad: 0.10x.

### Matmul

| Op | Peregrine | PyTorch | TF | JAX | MLX | TinyGrad | Winner |
|----|-----------|---------|-----|-----|-----|----------|--------|
| matmul_128x128 | **5.7** | 5.9 | 96.3 | 80.3 | 28.4 | 428.9 | PG |
| matmul_256x256 | 69.2 | **31.7** | 195.0 | 158.3 | 81.4 | 443.4 | PT |
| matmul_512x512 | 219.9 | **142.5** | 660.2 | 505.4 | 221.2 | 449.7 | PT |
| matmul_1024x1024 | **1013.6** | - | - | - | - | - | PG |
| matmul_2048x2048 | **9444.2** | - | - | - | - | - | PG |

### Add

| Op | Peregrine | PyTorch | TF | JAX | MLX | TinyGrad | Winner |
|----|-----------|---------|-----|-----|-----|----------|--------|
| add_100k | **12.7** | 40.3 | 56.0 | 33.8 | 32.4 | 193.4 | PG |
| add_500k | 93.4 | 61.2 | 80.8 | **60.4** | 84.4 | 195.1 | JAX |
| add_1M | **119.7** | - | - | - | - | - | PG |
| add_5M | **575.3** | - | - | - | - | - | PG |
| add_10M | **946.1** | - | - | - | - | - | PG |

### Mul

| Op | Peregrine | PyTorch | TF | JAX | MLX | TinyGrad | Winner |
|----|-----------|---------|-----|-----|-----|----------|--------|
| mul_100k | **12.5** | 39.8 | 49.7 | 35.1 | 33.0 | 192.5 | PG |
| mul_500k | 152.5 | **59.6** | 77.1 | 63.7 | 88.0 | 194.7 | PT |
| mul_1M | **177.6** | - | - | - | - | - | PG |
| mul_5M | **618.3** | - | - | - | - | - | PG |
| mul_10M | **831.1** | - | - | - | - | - | PG |

### Exp

| Op | Peregrine | PyTorch | TF | JAX | MLX | TinyGrad | Winner |
|----|-----------|---------|-----|-----|-----|----------|--------|
| exp_100k | 95.2 | 64.3 | 72.8 | **46.2** | 73.6 | 224.4 | JAX |
| exp_500k | 208.7 | 138.2 | **117.9** | 118.0 | 241.9 | 219.5 | TF |
| exp_1M | **307.3** | - | - | - | - | - | PG |
| exp_5M | **1105.3** | - | - | - | - | - | PG |
| exp_10M | **2167.3** | - | - | - | - | - | PG |

### Activations

| Op | Peregrine | PyTorch | TF | JAX | MLX | TinyGrad | Winner |
|----|-----------|---------|-----|-----|-----|----------|--------|
| relu_100k | **8.8** | 38.6 | 40.3 | 99.2 | 27.0 | 349.2 | PG |
| relu_1M | **134.3** | - | - | - | - | - | PG |
| silu_100k | 65.3 | 77.0 | 251.4 | **52.6** | 89.8 | 340.3 | JAX |
| softplus_100k | 261.2 | 155.9 | **137.2** | 155.3 | 286.7 | 791.2 | TF |
| mish_100k | 536.1 | 311.7 | 254.3 | **242.0** | 408.3 | 1174.0 | JAX |
| leaky_relu_100k | **8.1** | 41.3 | 19.7 | 30.5 | 85.1 | - | PG |
| elu_100k | 140.8 | 123.2 | 139.5 | **77.5** | 125.8 | 874.8 | JAX |
| hard_tanh_100k | 51.5 | 40.3 | 44.1 | 38.8 | **37.7** | - | MLX |
| relu6_100k | 51.5 | **41.4** | 50.8 | 110.6 | 54.0 | 748.1 | PT |
| hardswish_100k | 85.2 | 40.5 | 206.9 | **26.8** | 75.5 | - | JAX |
| gelu_100k | 80.8 | **71.2** | 252.7 | 221.3 | 148.3 | 858.6 | PT |
| selu_100k | 168.0 | 130.3 | 141.7 | **82.5** | 88.5 | 754.4 | JAX |
| softsign_100k | **34.9** | 125.8 | 47.9 | 59.0 | 50.9 | - | PG |

### Softmax / MLP / Train

| Op | Peregrine | PyTorch | TF | JAX | MLX | TinyGrad | Winner |
|----|-----------|---------|-----|-----|-----|----------|--------|
| softmax_8x128 | **1.2** | 32.1 | 11.7 | 31.6 | 19.0 | 631.8 | PG |
| softmax_8x512 | **4.3** | 35.0 | 14.5 | 34.0 | 20.1 | 635.2 | PG |
| mlp_fwd_64x784 | 32.6 | **27.7** | 273.3 | 186.8 | 54.6 | 1817.5 | PT |
| mlp_fwd_256x784_wide | **423.1** | - | - | - | - | - | PG |
| train_step_64 | **819.3** | 1284.8 | 8635.0 | 5173.8 | 869.1 | 24367.6 | PG |
| train_step_256_wide | **3354.9** | - | - | - | - | - | PG |

### Unary Math

| Op | Peregrine | PyTorch | TF | JAX | MLX | TinyGrad | Winner |
|----|-----------|---------|-----|-----|-----|----------|--------|
| reciprocal_100k | **8.8** | 42.0 | 48.6 | 33.2 | 28.5 | 162.9 | PG |
| square_100k | **8.8** | 37.4 | 15.8 | 29.2 | 24.7 | 175.2 | PG |
| rsqrt_100k | 86.8 | 40.9 | 57.5 | 92.5 | **39.2** | - | MLX |
| floor_100k | 46.7 | 39.4 | **17.7** | 34.3 | 25.7 | 409.9 | TF |
| ceil_100k | 46.6 | 42.0 | **17.7** | 34.1 | 30.8 | 358.6 | TF |
| round_100k | 46.6 | 43.3 | 47.2 | 30.9 | **28.2** | - | MLX |
| sign_100k | 54.4 | 39.7 | 47.7 | 36.4 | **32.8** | 801.1 | MLX |
| expm1_100k | 165.5 | 111.7 | 145.1 | **98.9** | 119.8 | - | JAX |
| log2_100k | 111.8 | 85.6 | 149.4 | **64.0** | 111.0 | 165.2 | JAX |
| log10_100k | 104.4 | 87.2 | 151.5 | **56.4** | 122.9 | - | JAX |
| log1p_100k | 130.8 | **82.4** | 91.6 | 104.6 | 138.3 | - | PT |
| erf_100k | 115.9 | 57.0 | 57.5 | **42.9** | 109.0 | - | JAX |

### Trig / Hyperbolic

| Op | Peregrine | PyTorch | TF | JAX | MLX | TinyGrad | Winner |
|----|-----------|---------|-----|-----|-----|----------|--------|
| sinh_100k | **52.0** | 133.3 | 132.7 | 114.5 | 105.1 | 544.2 | PG |
| cosh_100k | **47.2** | 131.6 | 133.7 | 71.9 | 102.8 | 465.4 | PG |
| arcsin_100k | **53.1** | 74.1 | 54.2 | 114.2 | 102.6 | 2961.6 | PG |
| arccos_100k | 111.7 | 89.1 | **53.5** | 202.9 | 119.4 | - | TF |
| arctan_100k | **54.2** | 93.4 | 58.6 | 214.2 | 101.9 | 3086.2 | PG |
| arcsinh_100k | 132.1 | 148.9 | 143.4 | **121.1** | 357.5 | - | JAX |

### Binary Ops

| Op | Peregrine | PyTorch | TF | JAX | MLX | TinyGrad | Winner |
|----|-----------|---------|-----|-----|-----|----------|--------|
| maximum_100k | **13.4** | 38.9 | 42.2 | 30.5 | 26.6 | 191.0 | PG |
| minimum_100k | **13.4** | 42.2 | 44.2 | 32.9 | 30.8 | 384.5 | PG |
| power_100k | 392.5 | 230.9 | 277.1 | **146.6** | 238.0 | - | JAX |
| arctan2_100k | 1124.9 | 135.4 | **75.4** | 319.6 | 157.4 | - | TF |
| logaddexp_100k | 416.6 | 153.5 | 364.0 | **152.0** | 280.5 | - | JAX |

### Comparison / Logic

| Op | Peregrine | PyTorch | TF | JAX | MLX | TinyGrad | Winner |
|----|-----------|---------|-----|-----|-----|----------|--------|
| clip_100k | **9.0** | 41.3 | 43.1 | 35.6 | 39.2 | 541.0 | PG |
| where_100k | 94.9 | 51.0 | 66.5 | 34.8 | **29.4** | 281.6 | MLX |
| greater_100k | 71.4 | 49.1 | 49.1 | 26.7 | **21.5** | 191.7 | MLX |
| equal_100k | 71.4 | 30.2 | 56.4 | 29.4 | **24.4** | 291.1 | MLX |

### Reductions

| Op | Peregrine | PyTorch | TF | JAX | MLX | TinyGrad | Winner |
|----|-----------|---------|-----|-----|-----|----------|--------|
| sum_axis_256x512 | 114.8 | 38.6 | 51.2 | 52.3 | **20.8** | 211.0 | MLX |
| mean_axis_256x512 | 112.6 | 43.5 | 50.7 | 47.5 | **25.5** | 291.6 | MLX |
| max_axis_256x512 | 154.2 | 54.5 | 48.7 | **45.3** | 47.3 | 204.6 | JAX |
| min_axis_256x512 | 154.2 | 53.4 | **48.3** | 48.3 | 49.4 | 330.5 | TF |
| var_256x512 | 238.4 | 277.8 | 221.8 | 82.1 | **60.9** | - | MLX |
| prod_axis_256x512 | 149.2 | 38.9 | 48.8 | 54.1 | **26.3** | - | MLX |
| logsumexp_256x512 | 387.9 | 196.8 | 330.7 | 281.4 | **119.9** | - | MLX |
| cumsum_256x512 | 122.2 | **77.3** | 196.2 | 202.3 | 144.5 | 612.4 | PT |
| argmax_axis_256x512 | 154.5 | 95.9 | **75.8** | 173.0 | 184.3 | 1335.3 | TF |
| sum_axis_1024x1024 | **941.8** | - | - | - | - | - | PG |
| var_1024x1024 | **1936.0** | - | - | - | - | - | PG |

### Shape Ops

| Op | Peregrine | PyTorch | TF | JAX | MLX | TinyGrad | Winner |
|----|-----------|---------|-----|-----|-----|----------|--------|
| tril_256x256 | **34.7** | 39.2 | 54.2 | 38.8 | 63.6 | 1850.2 | PG |
| triu_256x256 | **34.0** | 40.4 | 51.7 | 36.9 | 59.8 | 1854.2 | PG |
| repeat_64x128_2x3 | 124.6 | 45.2 | 78.8 | **29.0** | 32.4 | - | JAX |
| pad_64x128 | 16.8 | **4.4** | 86.8 | 18.2 | 21.8 | 89.6 | PT |
| stack_8x64x128 | 14.4 | **8.7** | 55.1 | 162.3 | 47.1 | 931.9 | PT |
| diagonal_512x512 | 0.8 | **0.6** | 12.5 | 10.1 | 28.4 | - | PT |

### Losses

| Op | Peregrine | PyTorch | TF | JAX | MLX | TinyGrad | Winner |
|----|-----------|---------|-----|-----|-----|----------|--------|
| cross_entropy_64x10 | **2.6** | 35.9 | 638.0 | 54.7 | 24.0 | 3387.7 | PG |
| l1_loss_64x10 | **1.0** | 5.3 | 43.3 | 12.4 | 19.2 | 1129.7 | PG |
| mse_loss_64x10 | **3.8** | 4.9 | 39.5 | 24.0 | 20.7 | 455.5 | PG |
| huber_loss_64x10 | 5.3 | **4.8** | 237.7 | 49.1 | 39.6 | - | PT |
| smooth_l1_loss_64x10 | 5.2 | **5.1** | 236.2 | 49.5 | 40.5 | - | PT |
| kl_div_loss_64x10 | **2.5** | 6.4 | 383.4 | 63.4 | 23.5 | - | PG |
| cosine_sim_loss_64x64 | 13.8 | **10.2** | 245.3 | 71.4 | 125.2 | - | PT |

### Layers

| Op | Peregrine | PyTorch | TF | JAX | MLX | TinyGrad | Winner |
|----|-----------|---------|-----|-----|-----|----------|--------|
| rmsnorm_64x512 | 58.8 | 68.4 | 438.7 | 75.5 | **39.0** | - | MLX |
| conv1d_1x32x128_k3 | **20.9** | 55.0 | 523.6 | 74.2 | 29.9 | - | PG |
| avgpool2d_1x16x32x32 | **25.3** | 41.8 | 64.6 | 44.7 | 283.6 | - | PG |
| groupnorm_4x64x16x16 | 72.5 | **53.6** | 787.3 | 267.8 | 235.1 | - | PT |

### RNN

| Op | Peregrine | PyTorch | TF | JAX | MLX | TinyGrad | Winner |
|----|-----------|---------|-----|-----|-----|----------|--------|
| rnn_seq32_128_256 | **194.9** | 270.3 | - | - | - | - | PG |
| lstm_seq32_128_256 | 1136.0 | **807.9** | - | - | - | - | PT |
| gru_seq32_128_256 | 807.3 | **782.5** | - | - | - | - | PT |

### Optimizers

| Op | Peregrine | PyTorch | TF | JAX | MLX | TinyGrad | Winner |
|----|-----------|---------|-----|-----|-----|----------|--------|
| optim_adam_64 | **795.4** | 1240.6 | - | - | - | - | PG |
| optim_rmsprop_64 | **933.3** | 1175.8 | - | - | - | - | PG |
| optim_lion_64 | **917.8** | - | - | - | - | - | PG |
| optim_adafactor_64 | **1305.8** | - | - | - | - | - | PG |

### Random

| Op | Peregrine | PyTorch | TF | JAX | MLX | TinyGrad | Winner |
|----|-----------|---------|-----|-----|-----|----------|--------|
| rand_uniform_100k | **109.6** | 257.5 | 121.1 | 553.9 | 520.5 | 2453.0 | PG |
| rand_normal_100k | 783.1 | 974.1 | **354.0** | 623.3 | 748.1 | 3331.2 | TF |
| rand_bernoulli_100k | 312.8 | 250.1 | **221.1** | 560.4 | 490.7 | - | TF |
| rand_uniform_1M | 1073.5 | 2610.0 | **419.6** | 2274.8 | 4873.0 | 2439.7 | TF |
| rand_normal_1M | 7686.7 | 9877.7 | **2088.5** | 2951.7 | 7038.8 | 3306.2 | TF |

### FFT

| Op | Peregrine | PyTorch | TF | JAX | MLX | TinyGrad | Winner |
|----|-----------|---------|-----|-----|-----|----------|--------|
| rfft_1k | **2.2** | 4.5 | 44.7 | 50.0 | 20.6 | - | PG |
| rfft_4k | **7.5** | 15.1 | 55.5 | 65.5 | 29.7 | - | PG |
| rfft_16k | **30.3** | 66.2 | 107.1 | 117.0 | 83.8 | - | PG |
| fft_1k | **3.3** | 6.8 | 9.1 | 45.1 | 24.4 | - | PG |
| fft_4k | **12.0** | 26.9 | 18.7 | 61.7 | 44.4 | - | PG |

### Linear Algebra

| Op | Peregrine | PyTorch | TF | JAX | MLX | TinyGrad | Winner |
|----|-----------|---------|-----|-----|-----|----------|--------|
| norm_l2_1k | **1.1** | 1.3 | 70.1 | 4.0 | 20.8 | - | PG |
| solve_64x64 | **12.0** | 23.5 | 25.6 | 36.1 | 100.3 | - | PG |
| inv_64x64 | 37.5 | **25.2** | 33.8 | 44.3 | 50.5 | - | PT |
| cholesky_64x64 | **9.7** | 41.6 | 20.3 | 20.5 | 22.0 | - | PG |
| svd_64x64 | **277.4** | 279.7 | 494.8 | 311.4 | 305.3 | - | PG |
| qr_64x64 | **41.3** | 84.5 | 85.9 | 67.2 | 62.4 | - | PG |
| eigh_64x64 | 380.3 | 215.4 | **144.5** | 253.3 | 236.4 | - | TF |
| det_64x64 | 23.2 | **19.9** | 23.4 | 28.4 | - | - | PT |
| solve_128x128 | 50.2 | **45.3** | 78.2 | 84.5 | 209.5 | - | PT |
| inv_128x128 | 94.4 | **60.8** | 141.4 | 91.7 | 97.1 | - | PT |
| cholesky_128x128 | 50.8 | 53.2 | 59.6 | 36.2 | **31.5** | - | MLX |
| svd_128x128 | **994.4** | 994.6 | 1890.5 | 1025.0 | 1025.6 | - | PG |
| qr_128x128 | **187.0** | 221.2 | 328.3 | 194.2 | 205.6 | - | PG |
| eigh_128x128 | 1844.6 | **712.9** | 725.2 | 750.9 | 747.8 | - | PT |
| det_128x128 | 52.3 | **49.6** | 84.7 | 77.5 | - | - | PT |
| solve_256x256 | 189.5 | **182.9** | 384.9 | 287.4 | 747.4 | - | PT |
| inv_256x256 | 460.7 | 285.7 | 859.9 | 333.7 | **250.6** | - | MLX |
| cholesky_256x256 | 226.2 | 84.3 | 287.0 | 116.9 | **53.3** | - | MLX |
| svd_256x256 | 5930.1 | **5881.8** | 8208.2 | 5913.0 | 6162.9 | - | PT |
| qr_256x256 | 1026.4 | **986.2** | 1736.7 | 989.0 | 1065.7 | - | PT |
| eigh_256x256 | 6082.2 | **3441.4** | 4629.2 | 3619.1 | 3615.8 | - | PT |
| det_256x256 | 212.9 | **205.7** | 440.5 | 207.5 | - | - | PT |

### Pipeline Ops

| Op | Peregrine | PyTorch | TF | JAX | MLX | TinyGrad | Winner |
|----|-----------|---------|-----|-----|-----|----------|--------|
| matmul_bias_gelu_196x768x3072 | 1070.7 | **921.8** | 2377.5 | 2172.8 | - | 1253.7 | PT |
| matmul_bias_gelu_196x1024x4096 | 2205.9 | 2023.6 | 3737.7 | 3465.6 | - | **1293.4** | tinygrad |
| add_layernorm_196x768 | 111.5 | **102.8** | 1211.4 | 231.8 | - | 1147.2 | PT |
| add_layernorm_196x1024 | 139.7 | **104.1** | 1316.7 | 294.9 | - | 1139.2 | PT |

### Int8 Quantized Matmul

Peregrine-only benchmark comparing f32 (Apple Accelerate cblas_sgemm) vs int8 quantized matmul (NEON vmull+vpadalq, per-column weight + per-row activation quantization).

| Op | f32 (us) | i8 (us) | Ratio |
|----|----------:|--------:|------:|
| matmul_196x768x3072 | 616 | 14,702 | 23.8x slower |
| matmul_196x1024x4096 | 1,515 | 26,874 | 17.7x slower |

The int8 path is currently slower than f32 because: (a) NEON `vmull_s8`+`vpadalq_s16` (stable) vs hardware `sdot` (unstable), (b) competing against Apple Accelerate's heavily optimized cblas_sgemm. The primary benefit is **4× memory reduction** for weight storage, enabling larger models to fit in memory. The Metal GPU dequant path loads 4× less data from device memory.

## Analysis

### Where Peregrine Wins (vs PyTorch)
- **Elementwise (small tensors)**: add, mul, relu, clip — near-zero dispatch overhead, 3-5x faster
- **Softmax**: 7-27x faster (fused implementation, 1.2µs vs 32µs at 8x128)
- **Losses**: cross_entropy 14x, l1 5x, kl_div 2.6x — minimal dispatch overhead
- **FFT**: rfft/fft 2-4x faster across all sizes
- **Train step**: 1.6x faster end-to-end
- **Optimizers**: Adam 1.6x, RMSProp 1.3x
- **Linalg (small)**: solve/cholesky/qr/svd at 64x64 — 1.0-4.3x faster
- **Trig**: sinh 2.6x, cosh 2.8x, arctan 1.7x

### Where PyTorch Wins (vs Peregrine)
- **Matmul (medium)**: 256x256 2.2x, 512x512 1.5x — Apple Accelerate advantage
- **GELU**: 1.1x faster
- **Reductions**: sum/mean 3x, max/min 2.8x
- **Log/erf**: log1p 1.6x, erf 1.3x (JAX now wins erf)
- **Shape ops**: pad 3.8x, stack 1.7x
- **Linalg (large)**: eigh_256 1.8x, cholesky_256 2.7x
- **LSTM**: 1.4x

### Changes from Previous Run
- **matmul_128**: Peregrine now wins (5.7µs vs 5.9µs, was tied)
- **softmax_8x128**: Peregrine improved from 3.79µs to 1.2µs (3.2x faster)
- **softmax_8x512**: Peregrine improved from 14.5µs to 4.3µs and now wins (was TF)
- **tril/triu**: Peregrine now wins both (was PyTorch)
- **arcsin**: Peregrine now wins (53.1µs, was TF at 51.1µs)
- **svd_128**: Peregrine now wins (994.4µs vs 994.6µs, was PT)
- **JAX gains**: JAX now wins more ops (16 vs 12), especially activations and math

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
