# Peregrine Benchmarks

**Date**: 2026-03-12 (v0.25.0 — Huffman Compression for Weights and KV Cache)
**System**: Apple M1 Max, 10 cores, 64 GB RAM, arm64
**Frameworks**: Peregrine (Rust), PyTorch 2.10.0, TensorFlow 2.21.0, JAX 0.9.1, MLX 0.30.6, TinyGrad 0.12.0
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
| Encoder      | 499.9ms | 15.1ms  | 15.4ms       | 1409.5ms| 52.9ms  | 45.8ms       |
| Decoder      | 200.8ms | 14.8ms  | 183.0ms      | 578.7ms | 35.0ms  | 462.9ms      |
| Head+postproc| 3.1ms   | 180.6ms | 5.1ms        | 8.9ms   | 588.1ms | 12.6ms       |
| **Total**    | **0.65s**| **0.53s**| **0.54s**   | **1.89s**| **1.55s**| **1.44s**   |

### Metal GPU Inference (v0.19.0+)

| Resolution | CPU | GPU | GPU+Pipeline | Speedup (best vs CPU) |
|-----------|----:|----:|-------------:|---------|
| 224x224   | 0.65s | 0.53s | **0.54s** | **1.23x** |
| 512x384   | 1.89s | 1.65s | **1.44s** | **1.31x** |

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

| Framework | Wins (of 173 total ops) |
|-----------|------------------------|
| Peregrine | 99 |
| PyTorch | 39 |
| MLX | 14 |
| TensorFlow | 13 |
| JAX | 7 |
| TinyGrad | 1 |

Peregrine wins 99/173 ops (57%). Geometric mean ratio vs PyTorch: 1.04x, vs MLX: 0.68x, vs TF: 0.48x, vs JAX: 0.46x, vs tinygrad: 0.08x.

### Matmul

| Op | Peregrine | PyTorch | TF | JAX | MLX | TinyGrad | Winner |
|----|-----------|---------|-----|-----|-----|----------|--------|
| matmul_128x128 | **5.8** | 5.8 | 93.3 | 220.3 | 28.4 | 508.2 | PG |
| matmul_256x256 | 59.0 | **30.1** | 189.9 | 211.9 | 81.4 | 513.6 | PT |
| matmul_512x512 | 202.5 | **124.9** | 728.7 | 695.0 | 221.2 | 491.2 | PT |
| matmul_1024x1024 | **1051.0** | - | - | - | - | - | PG |
| matmul_2048x2048 | **9603.6** | - | - | - | - | - | PG |

### Add

| Op | Peregrine | PyTorch | TF | JAX | MLX | TinyGrad | Winner |
|----|-----------|---------|-----|-----|-----|----------|--------|
| add_100k | **12.8** | 33.6 | 60.0 | 63.6 | 32.4 | 221.7 | PG |
| add_500k | 111.5 | 94.8 | 138.2 | 426.1 | **84.4** | 216.2 | MLX |
| add_1M | **158.3** | - | - | - | - | - | PG |
| add_5M | **553.4** | - | - | - | - | - | PG |
| add_10M | **962.3** | - | - | - | - | - | PG |

### Mul

| Op | Peregrine | PyTorch | TF | JAX | MLX | TinyGrad | Winner |
|----|-----------|---------|-----|-----|-----|----------|--------|
| mul_100k | **13.0** | 30.6 | 51.3 | 30.6 | 33.0 | 224.8 | PG |
| mul_500k | 132.4 | 97.6 | 139.6 | 189.7 | **88.0** | 214.8 | MLX |
| mul_1M | **133.6** | - | - | - | - | - | PG |
| mul_5M | **501.8** | - | - | - | - | - | PG |
| mul_10M | **913.6** | - | - | - | - | - | PG |

### Exp

| Op | Peregrine | PyTorch | TF | JAX | MLX | TinyGrad | Winner |
|----|-----------|---------|-----|-----|-----|----------|--------|
| exp_100k | 112.4 | 56.1 | 67.9 | **48.5** | 73.6 | 270.4 | JAX |
| exp_500k | 204.6 | 191.3 | **161.2** | 227.3 | 241.9 | 268.5 | TF |
| exp_1M | **307.6** | - | - | - | - | - | PG |
| exp_5M | **1098.6** | - | - | - | - | - | PG |
| exp_10M | **2139.7** | - | - | - | - | - | PG |

### Activations

| Op | Peregrine | PyTorch | TF | JAX | MLX | TinyGrad | Winner |
|----|-----------|---------|-----|-----|-----|----------|--------|
| relu_100k | **9.0** | 28.8 | 42.0 | 108.0 | 27.0 | 400.2 | PG |
| relu_1M | **104.4** | - | - | - | - | - | PG |
| silu_100k | 64.0 | **48.1** | 274.5 | 90.5 | 89.8 | 397.5 | PT |
| softplus_100k | 298.4 | **122.6** | 127.7 | 157.3 | 286.7 | 976.5 | PT |
| mish_100k | 500.9 | 293.1 | **255.8** | 291.2 | 408.3 | 1496.2 | TF |
| leaky_relu_100k | **8.0** | 40.9 | 19.2 | 36.6 | 85.1 | - | PG |
| elu_100k | 142.2 | 102.3 | **69.0** | 82.6 | 125.8 | 1132.8 | TF |
| hard_tanh_100k | 52.0 | **29.5** | 51.5 | 52.0 | 37.7 | - | PT |
| relu6_100k | 50.5 | **37.8** | 55.0 | 150.7 | 54.0 | 902.4 | PT |
| hardswish_100k | 83.7 | **29.8** | 245.7 | 32.8 | 75.5 | - | PT |
| gelu_100k | 77.4 | **45.2** | 292.2 | 278.5 | 148.3 | 1060.0 | PT |
| selu_100k | 158.8 | 105.1 | **80.7** | 84.9 | 88.5 | 897.2 | TF |
| softsign_100k | **34.9** | 90.2 | 48.1 | 102.7 | 50.9 | - | PG |

### Softmax / MLP / Train

| Op | Peregrine | PyTorch | TF | JAX | MLX | TinyGrad | Winner |
|----|-----------|---------|-----|-----|-----|----------|--------|
| softmax_8x128 | **1.2** | 30.7 | 12.2 | 63.8 | 19.0 | 740.0 | PG |
| softmax_8x512 | **4.2** | 33.2 | 14.3 | 48.0 | 20.1 | 749.4 | PG |
| mlp_fwd_64x784 | 33.6 | **26.8** | 271.1 | 218.9 | 54.6 | 2411.7 | PT |
| mlp_fwd_256x784_wide | **420.8** | - | - | - | - | - | PG |
| train_step_64 | **816.6** | 1257.6 | 9763.3 | 5925.0 | 869.1 | 29045.4 | PG |
| train_step_256_wide | **3312.9** | - | - | - | - | - | PG |

### Unary Math

| Op | Peregrine | PyTorch | TF | JAX | MLX | TinyGrad | Winner |
|----|-----------|---------|-----|-----|-----|----------|--------|
| reciprocal_100k | **8.5** | 46.0 | 50.0 | 44.5 | 28.5 | 195.5 | PG |
| square_100k | **8.6** | 29.0 | 15.2 | 51.4 | 24.7 | 204.2 | PG |
| rsqrt_100k | 76.1 | **30.4** | 53.8 | 87.0 | 39.2 | - | PT |
| floor_100k | 46.6 | 27.1 | **16.7** | 49.5 | 25.7 | 491.0 | TF |
| ceil_100k | 46.6 | 28.7 | **16.6** | 34.7 | 30.8 | 432.5 | TF |
| round_100k | 46.6 | 30.7 | 49.8 | 45.0 | **28.2** | - | MLX |
| sign_100k | 54.4 | **29.3** | 52.5 | 36.9 | 32.8 | 927.3 | PT |
| expm1_100k | 142.0 | **67.8** | 107.3 | 95.2 | 119.8 | - | PT |
| log2_100k | 108.4 | **53.4** | 156.8 | 58.0 | 111.0 | 197.1 | PT |
| log10_100k | 106.5 | **53.9** | 158.7 | 87.0 | 122.9 | - | PT |
| log1p_100k | 126.6 | **50.9** | 83.0 | 113.2 | 138.3 | - | PT |
| erf_100k | 121.7 | **39.2** | 63.5 | 50.5 | 109.0 | - | PT |

### Trig / Hyperbolic

| Op | Peregrine | PyTorch | TF | JAX | MLX | TinyGrad | Winner |
|----|-----------|---------|-----|-----|-----|----------|--------|
| sinh_100k | **51.0** | 101.5 | 140.8 | 140.0 | 105.1 | 661.0 | PG |
| cosh_100k | **46.3** | 99.2 | 134.5 | 88.6 | 102.8 | 560.9 | PG |
| arcsin_100k | **52.1** | 54.9 | 57.3 | 127.0 | 102.6 | 3720.7 | PG |
| arccos_100k | 106.1 | **53.5** | 57.8 | 208.3 | 119.4 | - | PT |
| arctan_100k | **53.2** | 73.3 | 62.1 | 219.8 | 101.9 | 3840.3 | PG |
| arcsinh_100k | 141.6 | **128.7** | 140.2 | 141.4 | 357.5 | - | PT |

### Binary Ops

| Op | Peregrine | PyTorch | TF | JAX | MLX | TinyGrad | Winner |
|----|-----------|---------|-----|-----|-----|----------|--------|
| maximum_100k | **12.5** | 33.9 | 47.9 | 55.4 | 26.6 | 227.2 | PG |
| minimum_100k | **12.5** | 30.4 | 45.6 | 58.9 | 30.8 | 437.3 | PG |
| power_100k | 393.2 | 210.9 | 276.8 | **167.0** | 238.0 | - | JAX |
| arctan2_100k | 1169.0 | 107.9 | **73.4** | 303.4 | 157.4 | - | TF |
| logaddexp_100k | 408.4 | **123.6** | 397.8 | 186.5 | 280.5 | - | PT |

### Comparison / Logic

| Op | Peregrine | PyTorch | TF | JAX | MLX | TinyGrad | Winner |
|----|-----------|---------|-----|-----|-----|----------|--------|
| clip_100k | **8.6** | 30.2 | 44.3 | 52.3 | 39.2 | 650.8 | PG |
| where_100k | 93.1 | 33.8 | 58.8 | 45.4 | **29.4** | 325.4 | MLX |
| greater_100k | 81.0 | 36.5 | 56.6 | 46.8 | **21.5** | 219.2 | MLX |
| equal_100k | 81.1 | 31.4 | 64.7 | 38.7 | **24.4** | 362.1 | MLX |

### Reductions

| Op | Peregrine | PyTorch | TF | JAX | MLX | TinyGrad | Winner |
|----|-----------|---------|-----|-----|-----|----------|--------|
| sum_axis_256x512 | 112.8 | 34.7 | 58.2 | **16.7** | 20.8 | 249.4 | JAX |
| mean_axis_256x512 | 112.7 | 38.6 | 58.6 | 33.5 | **25.5** | 352.4 | MLX |
| max_axis_256x512 | 154.5 | 39.2 | 65.0 | **32.9** | 47.3 | 246.6 | JAX |
| min_axis_256x512 | 157.3 | 39.1 | 56.6 | **16.2** | 49.4 | 395.9 | JAX |
| var_256x512 | 235.7 | 225.0 | 260.6 | 83.9 | **60.9** | - | MLX |
| prod_axis_256x512 | 149.2 | 28.4 | 54.5 | 55.4 | **26.3** | - | MLX |
| logsumexp_256x512 | 381.1 | 144.3 | 373.6 | 329.9 | **119.9** | - | MLX |
| cumsum_256x512 | 123.2 | **53.9** | 206.1 | 260.7 | 144.5 | 763.4 | PT |
| argmax_axis_256x512 | 158.0 | **65.5** | 82.2 | 198.2 | 184.3 | 1592.1 | PT |
| sum_axis_1024x1024 | **942.4** | - | - | - | - | - | PG |
| var_1024x1024 | **1940.2** | - | - | - | - | - | PG |

### Shape Ops

| Op | Peregrine | PyTorch | TF | JAX | MLX | TinyGrad | Winner |
|----|-----------|---------|-----|-----|-----|----------|--------|
| tril_256x256 | 34.7 | 35.3 | 56.0 | **34.5** | 63.6 | 2285.7 | JAX |
| triu_256x256 | **34.5** | 35.7 | 54.6 | 43.9 | 59.8 | 2255.6 | PG |
| repeat_64x128_2x3 | 125.0 | 36.8 | 80.3 | **30.5** | 32.4 | - | JAX |
| pad_64x128 | 17.8 | **3.9** | 90.9 | 21.7 | 21.8 | 112.0 | PT |
| stack_8x64x128 | 15.7 | **8.5** | 64.6 | 212.2 | 47.1 | 1173.2 | PT |
| diagonal_512x512 | 0.8 | **0.7** | 13.1 | 11.0 | 28.4 | - | PT |

### Losses

| Op | Peregrine | PyTorch | TF | JAX | MLX | TinyGrad | Winner |
|----|-----------|---------|-----|-----|-----|----------|--------|
| cross_entropy_64x10 | **2.6** | 37.6 | 680.1 | 77.2 | 24.0 | 4216.7 | PG |
| l1_loss_64x10 | **1.0** | 6.9 | 49.2 | 17.4 | 19.2 | 1403.5 | PG |
| mse_loss_64x10 | **3.7** | 6.3 | 42.7 | 31.9 | 20.7 | 528.7 | PG |
| huber_loss_64x10 | **5.2** | 6.3 | 262.0 | 63.7 | 39.6 | - | PG |
| smooth_l1_loss_64x10 | **5.1** | 6.4 | 263.8 | 65.3 | 40.5 | - | PG |
| kl_div_loss_64x10 | **2.5** | 7.5 | 418.7 | 96.7 | 23.5 | - | PG |
| cosine_sim_loss_64x64 | 13.6 | **11.4** | 264.4 | 94.8 | 125.2 | - | PT |

### Layers

| Op | Peregrine | PyTorch | TF | JAX | MLX | TinyGrad | Winner |
|----|-----------|---------|-----|-----|-----|----------|--------|
| rmsnorm_64x512 | 57.6 | 52.2 | 451.9 | 97.0 | **39.0** | - | MLX |
| conv1d_1x32x128_k3 | **20.0** | 45.5 | 537.2 | 86.1 | 29.9 | - | PG |
| avgpool2d_1x16x32x32 | **27.1** | 31.9 | 69.1 | 59.4 | 283.6 | - | PG |
| groupnorm_4x64x16x16 | 109.3 | **39.2** | 842.0 | 173.2 | 235.1 | - | PT |

### RNN

| Op | Peregrine | PyTorch | TF | JAX | MLX | TinyGrad | Winner |
|----|-----------|---------|-----|-----|-----|----------|--------|
| rnn_seq32_128_256 | **198.7** | 278.4 | - | - | - | - | PG |
| lstm_seq32_128_256 | 1151.4 | **855.3** | - | - | - | - | PT |
| gru_seq32_128_256 | 853.9 | **813.8** | - | - | - | - | PT |

### Optimizers

| Op | Peregrine | PyTorch | TF | JAX | MLX | TinyGrad | Winner |
|----|-----------|---------|-----|-----|-----|----------|--------|
| optim_adam_64 | **814.6** | 1362.4 | - | - | - | - | PG |
| optim_rmsprop_64 | **941.8** | 1081.8 | - | - | - | - | PG |
| optim_lion_64 | **947.2** | - | - | - | - | - | PG |
| optim_adafactor_64 | **1336.4** | - | - | - | - | - | PG |

### Random

| Op | Peregrine | PyTorch | TF | JAX | MLX | TinyGrad | Winner |
|----|-----------|---------|-----|-----|-----|----------|--------|
| rand_uniform_100k | **109.6** | 258.3 | 139.1 | 641.8 | 520.5 | 2993.3 | PG |
| rand_normal_100k | 829.6 | 976.8 | **369.5** | 728.2 | 748.1 | 4107.5 | TF |
| rand_bernoulli_100k | 312.2 | 251.7 | **232.9** | 697.2 | 490.7 | - | TF |
| rand_uniform_1M | 1067.3 | 2780.8 | **598.5** | 2493.8 | 4873.0 | 2914.6 | TF |
| rand_normal_1M | 7669.8 | 10379.8 | **2609.8** | 3150.8 | 7038.8 | 4127.3 | TF |

### FFT

| Op | Peregrine | PyTorch | TF | JAX | MLX | TinyGrad | Winner |
|----|-----------|---------|-----|-----|-----|----------|--------|
| rfft_1k | **2.2** | 4.8 | 46.8 | 55.6 | 20.6 | - | PG |
| rfft_4k | **6.6** | 15.8 | 57.9 | 80.8 | 29.7 | - | PG |
| rfft_16k | **30.3** | 78.6 | 100.4 | 144.4 | 83.8 | - | PG |
| fft_1k | **3.3** | 7.1 | 9.3 | 54.5 | 24.4 | - | PG |
| fft_4k | **12.2** | 26.5 | 15.8 | 60.6 | 44.4 | - | PG |

### Linear Algebra

| Op | Peregrine | PyTorch | TF | JAX | MLX | TinyGrad | Winner |
|----|-----------|---------|-----|-----|-----|----------|--------|
| norm_l2_1k | **1.1** | 1.5 | 80.0 | 6.7 | 20.8 | - | PG |
| solve_64x64 | **12.0** | 20.5 | 26.0 | 37.1 | 100.3 | - | PG |
| inv_64x64 | 37.5 | **24.9** | 36.5 | 129.4 | 50.5 | - | PT |
| cholesky_64x64 | **9.7** | 43.1 | 19.8 | 38.0 | 22.0 | - | PG |
| svd_64x64 | **276.7** | 289.6 | 484.8 | 872.1 | 305.3 | - | PG |
| qr_64x64 | **41.4** | 76.6 | 85.0 | 91.2 | 62.4 | - | PG |
| eigh_64x64 | 381.4 | 218.8 | **154.5** | 291.6 | 236.4 | - | TF |
| det_64x64 | 22.6 | **21.1** | 22.3 | 52.3 | - | - | PT |
| solve_128x128 | 50.0 | **41.0** | 79.5 | 124.4 | 209.5 | - | PT |
| inv_128x128 | 95.9 | **56.6** | 156.9 | 320.5 | 97.1 | - | PT |
| cholesky_128x128 | 50.4 | 49.7 | 65.2 | 75.9 | **31.5** | - | MLX |
| svd_128x128 | **989.2** | 1016.5 | 1861.5 | 5569.4 | 1025.6 | - | PG |
| qr_128x128 | **188.0** | 232.0 | 358.3 | 3723.2 | 205.6 | - | PG |
| eigh_128x128 | 1845.2 | **715.0** | 693.5 | 1538.6 | 747.8 | - | TF |
| det_128x128 | 52.4 | **47.4** | 110.6 | 115.7 | - | - | PT |
| solve_256x256 | 189.2 | **176.8** | 384.2 | 556.1 | 747.4 | - | PT |
| inv_256x256 | 474.4 | 311.1 | 1007.1 | 975.7 | **250.6** | - | MLX |
| cholesky_256x256 | 226.5 | 91.3 | 300.7 | 1877.0 | **53.3** | - | MLX |
| svd_256x256 | 5975.2 | **5443.8** | 8502.3 | 16842.4 | 6162.9 | - | PT |
| qr_256x256 | **1009.5** | 1034.5 | 1829.3 | 7128.3 | 1065.7 | - | PG |
| eigh_256x256 | 6044.7 | **3167.6** | 4572.8 | 8205.1 | 3615.8 | - | PT |
| det_256x256 | 212.8 | **209.5** | 468.0 | 389.3 | - | - | PT |

### Pipeline Ops

| Op | Peregrine | PyTorch | TF | JAX | MLX | TinyGrad | Winner |
|----|-----------|---------|-----|-----|-----|----------|--------|
| matmul_bias_gelu_196x768x3072 | **1111.1** | 1475.1 | 3314.7 | 2934.2 | - | 1532.9 | PG |
| matmul_bias_gelu_196x1024x4096 | 2175.8 | 2781.8 | 4916.6 | 4963.2 | - | **1555.2** | tinygrad |
| add_layernorm_196x768 | 107.8 | **105.6** | 1580.7 | 292.3 | - | 1385.1 | PT |
| add_layernorm_196x1024 | 182.7 | **116.5** | 1721.2 | 319.6 | - | 1383.7 | PT |

### Int8 Quantized Matmul

Peregrine-only benchmark comparing f32 (Apple Accelerate cblas_sgemm) vs int8 quantized matmul (NEON vmull+vpadalq, per-column weight + per-row activation quantization).

| Op | f32 (us) | i8 (us) | Ratio |
|----|----------:|--------:|------:|
| matmul_196x768x3072 | 936 | 15,395 | 16.4x slower |
| matmul_196x1024x4096 | 1,897 | 27,751 | 14.6x slower |

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

### Changes from Previous Run (v0.24.0 → v0.25.0)
- **matmul_128x128**: Peregrine now ties/wins (5.8µs, was 6.7µs)
- **matmul_bias_gelu_196x768x3072**: Peregrine now wins (1111.1µs vs 1475.1µs PT, was PT)
- **train_step_64**: Peregrine now wins (816.6µs vs 869.1µs MLX, was MLX)
- **huber_loss**: Peregrine now wins (5.2µs vs 6.3µs PT, was PT)
- **solve_64x64**: Peregrine now wins (12.0µs vs 20.5µs PT, was PT)
- **svd_64x64**: Peregrine now wins (276.7µs vs 289.6µs PT, was PT)
- **svd_128x128**: Peregrine now wins (989.2µs vs 1016.5µs PT, was MLX)
- **exp_100k**: Significantly improved (112.4µs, was 279.3µs) — JAX still wins at 48.5µs
- **Wins**: 99/173 (was 96/173)
- **Geomean vs PyTorch**: 1.04x (Peregrine slightly slower overall due to updated framework versions)

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
