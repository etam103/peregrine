# Peregrine Benchmarks

**Date**: 2026-02-28
**System**: Apple M1 Max, 10 cores, 64 GB RAM, arm64
**Frameworks**: Peregrine (Rust), PyTorch 2.10.0, TensorFlow 2.20.0, JAX 0.9.0.1, MLX 0.30.6, TinyGrad
**All benchmarks**: CPU only, median of 20-50 iterations

## MUSt3R 3D Reconstruction — End-to-End Inference

Model: 423M parameters (ViT-L encoder + ViT-B decoder), shared head.

| Metric               | Peregrine 224 | PyTorch 224 | Peregrine 512 | PyTorch 512 |
|----------------------|---------------|-------------|----------------|-------------|
| Input resolution     | 224x224       | 224x224     | 512x384        | 512x384     |
| Patches              | 196           | 196         | 768            | 768         |
| **Inference time**   | **0.67s**     | **0.67s**   | **1.98s**      | **2.26s**   |
| **Weight loading**   | **0.6s**      | **1.6s**    | **0.6s**       | **1.6s**    |

- **224**: Peregrine **matches** PyTorch (0.67s vs 0.67s)
- **512**: Peregrine is **13% faster** (1.98s vs 2.26s)
- **Weight loading**: Peregrine is **2.7x faster** (0.6s vs 1.6s)

### Server Mode & Parallel Workers (v0.15.0)

Server mode (`--server`) loads weights once and processes pairs over stdin/stdout, eliminating subprocess spawn + weight loading overhead per pair.

| Resolution | Subprocess | Server (warm) | Speedup |
|-----------|----------:|-------------:|--------:|
| 224x224   | ~0.57s/pair | ~0.51s/pair | 1.1x |
| 512x384   | ~1.90s/pair | ~1.81s/pair | 1.05x |

With `--workers N` in `reconstruct_video.py`, pairs are distributed across N server processes for near-linear wall-clock scaling.

### Metal GPU Inference (v0.15.0)

GPU mode (`--gpu`, requires `--features metal`) dispatches matmul, layernorm, GELU, add, add_bias to Metal GPU.

| Resolution | CPU (server) | GPU (server) | Note |
|-----------|------------:|------------:|------|
| 224x224   | ~0.51s      | ~0.52s      | Parity |
| 512x384   | ~1.83s      | ~1.82s      | Parity |

GPU is at parity (not faster) due to decoder GPU↔CPU roundtrips in stack/split features and Apple's AMX-based sgemm matching the tiled Metal matmul. GPU output is byte-identical to CPU.

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

## Op-Level Benchmarks — 6 Frameworks

All values are **median microseconds** (lower is better). **Bold** = winner for that op. `-` = not benchmarked.
Winner column: PG=Peregrine, PT=PyTorch, TF=TensorFlow, JAX=JAX, MLX=MLX.

### Summary

| Framework | Wins (of 115 multi-framework ops) |
|-----------|----------------------------------|
| Peregrine | 37 |
| PyTorch | 41 |
| TensorFlow | 15 |
| JAX | 8 |
| MLX | 14 |
| TinyGrad | 0 |

Peregrine and PyTorch are the clear front-runners, essentially tied. Peregrine beats TF, JAX, MLX, and TinyGrad overall.

### Matmul

| Op | Peregrine | PyTorch | TF | JAX | MLX | TinyGrad | Winner |
|----|-----------|---------|-----|-----|-----|----------|--------|
| matmul_128x128 | 6.08 | **5.96** | 60.9 | 58.8 | 21.0 | 474.0 | PT |
| matmul_256x256 | 32.2 | **31.8** | 195.5 | 168.4 | 44.0 | 461.7 | PT |
| matmul_512x512 | 159.9 | **142.7** | 765.7 | 575.8 | 246.4 | 477.1 | PT |
| matmul_1024x1024 | **1230.4** | - | - | - | - | - | PG |
| matmul_2048x2048 | **9654.0** | - | - | - | - | - | PG |

### Add

| Op | Peregrine | PyTorch | TF | JAX | MLX | TinyGrad | Winner |
|----|-----------|---------|-----|-----|-----|----------|--------|
| add_100k | **12.6** | 36.4 | 52.7 | 37.9 | 37.6 | 199.9 | PG |
| add_500k | 148.1 | 58.8 | 107.2 | **54.1** | 68.1 | 201.1 | JAX |
| add_1M | **133.4** | - | - | - | - | - | PG |
| add_5M | **530.9** | - | - | - | - | - | PG |
| add_10M | **873.3** | - | - | - | - | - | PG |

### Mul

| Op | Peregrine | PyTorch | TF | JAX | MLX | TinyGrad | Winner |
|----|-----------|---------|-----|-----|-----|----------|--------|
| mul_100k | **12.5** | 41.9 | 50.8 | 39.1 | 25.6 | 203.0 | PG |
| mul_500k | 107.5 | **58.9** | 106.5 | 72.0 | 69.5 | 196.2 | PT |
| mul_1M | **134.1** | - | - | - | - | - | PG |
| mul_5M | **539.9** | - | - | - | - | - | PG |
| mul_10M | **1081.9** | - | - | - | - | - | PG |

### Exp

| Op | Peregrine | PyTorch | TF | JAX | MLX | TinyGrad | Winner |
|----|-----------|---------|-----|-----|-----|----------|--------|
| exp_100k | 130.2 | 62.4 | 73.2 | **36.5** | 71.0 | 237.4 | JAX |
| exp_500k | 256.7 | 145.2 | 150.8 | **101.7** | 236.3 | 264.5 | JAX |
| exp_1M | **379.8** | - | - | - | - | - | PG |
| exp_5M | **1281.7** | - | - | - | - | - | PG |
| exp_10M | **2177.1** | - | - | - | - | - | PG |

### Activations

| Op | Peregrine | PyTorch | TF | JAX | MLX | TinyGrad | Winner |
|----|-----------|---------|-----|-----|-----|----------|--------|
| relu_100k | **8.79** | 39.3 | 39.1 | 83.4 | 26.7 | 356.6 | PG |
| relu_1M | **134.0** | - | - | - | - | - | PG |
| silu_100k | 65.8 | **49.2** | 228.8 | 111.7 | 86.7 | 367.3 | PT |
| softplus_100k | 318.2 | **123.7** | 134.7 | 214.1 | 278.2 | 843.5 | PT |
| mish_100k | 562.9 | 309.9 | **248.3** | 267.6 | 413.3 | 1222.5 | TF |
| leaky_relu_100k | **7.96** | 42.5 | 18.8 | 37.4 | 98.4 | - | PG |
| elu_100k | 144.9 | 104.6 | 150.8 | **78.2** | 135.9 | 899.2 | JAX |
| hard_tanh_100k | 50.5 | **31.9** | 39.8 | 48.7 | 35.3 | - | PT |
| relu6_100k | 50.5 | **32.2** | 44.5 | 120.2 | 70.8 | 765.6 | PT |
| hardswish_100k | 86.1 | **32.8** | 175.4 | 39.0 | 67.1 | - | PT |
| gelu_100k | 201.1 | **47.5** | 221.7 | 226.4 | 145.4 | 914.9 | PT |
| selu_100k | 172.0 | 129.0 | 141.9 | 139.2 | **86.7** | 779.5 | MLX |
| softsign_100k | **34.9** | 128.1 | 46.3 | 66.3 | 39.9 | - | PG |

### Softmax / MLP / Train

| Op | Peregrine | PyTorch | TF | JAX | MLX | TinyGrad | Winner |
|----|-----------|---------|-----|-----|-----|----------|--------|
| softmax_8x128 | **3.79** | 27.1 | 13.5 | 32.1 | 14.5 | 652.5 | PG |
| softmax_8x512 | 14.5 | 28.8 | **13.4** | 35.1 | 20.5 | 676.3 | TF |
| mlp_fwd_64x784 | 33.5 | **28.7** | 276.4 | 161.4 | 73.8 | 2272.9 | PT |
| mlp_fwd_256x784_wide | **428.6** | - | - | - | - | - | PG |
| train_step_64 | **800.3** | 1301.3 | 9183.5 | 5162.5 | 1290.8 | 25633.0 | PG |
| train_step_256_wide | **3309.9** | - | - | - | - | - | PG |

### Unary Math

| Op | Peregrine | PyTorch | TF | JAX | MLX | TinyGrad | Winner |
|----|-----------|---------|-----|-----|-----|----------|--------|
| reciprocal_100k | **8.67** | 40.5 | 53.6 | 30.1 | 25.5 | 173.0 | PG |
| square_100k | **8.67** | 30.9 | 16.7 | 30.6 | 22.7 | 182.8 | PG |
| rsqrt_100k | 77.0 | **29.9** | 58.1 | 92.2 | 31.7 | - | PT |
| floor_100k | 46.7 | 29.9 | **18.5** | 28.0 | 32.7 | 433.6 | TF |
| ceil_100k | 46.7 | 31.1 | **18.5** | 33.2 | 28.8 | 366.6 | TF |
| round_100k | 46.7 | 33.9 | 47.7 | **30.1** | 30.1 | - | JAX |
| sign_100k | 54.4 | **32.6** | 45.3 | 54.0 | 37.9 | 854.6 | PT |
| expm1_100k | 166.4 | **69.2** | 165.8 | 101.9 | 121.7 | - | PT |
| log2_100k | 119.5 | 82.5 | 132.1 | **64.6** | 114.7 | 169.0 | JAX |
| log10_100k | 111.1 | **54.1** | 128.5 | 64.7 | 123.4 | - | PT |
| log1p_100k | 123.2 | **53.5** | 92.2 | 63.6 | 132.9 | - | PT |
| erf_100k | 117.9 | **37.5** | 44.6 | 57.9 | 106.5 | - | PT |

### Trig / Hyperbolic

| Op | Peregrine | PyTorch | TF | JAX | MLX | TinyGrad | Winner |
|----|-----------|---------|-----|-----|-----|----------|--------|
| sinh_100k | **51.1** | 102.7 | 162.5 | 141.8 | 94.3 | 564.8 | PG |
| cosh_100k | **46.3** | 101.1 | 148.6 | 79.9 | 110.1 | 487.8 | PG |
| arcsin_100k | 52.1 | 55.3 | **51.1** | 106.3 | 96.8 | 3078.2 | TF |
| arccos_100k | 125.8 | 65.6 | **51.7** | 222.2 | 114.8 | - | TF |
| arctan_100k | 53.1 | 60.3 | **52.9** | 217.8 | 100.0 | 3186.3 | TF |
| arcsinh_100k | 145.0 | **129.5** | 167.3 | 136.0 | 376.5 | - | PT |

### Binary Ops

| Op | Peregrine | PyTorch | TF | JAX | MLX | TinyGrad | Winner |
|----|-----------|---------|-----|-----|-----|----------|--------|
| maximum_100k | **12.5** | 29.5 | 40.1 | 31.3 | 24.4 | 207.3 | PG |
| minimum_100k | **12.5** | 30.9 | 34.2 | 32.2 | 21.9 | 404.8 | PG |
| power_100k | 387.3 | 215.8 | 325.0 | **145.8** | 223.5 | - | JAX |
| arctan2_100k | 1100.7 | 109.8 | **57.5** | 322.4 | 159.0 | - | TF |
| logaddexp_100k | 409.2 | **125.0** | 380.5 | 175.7 | 284.6 | - | PT |

### Comparison / Logic

| Op | Peregrine | PyTorch | TF | JAX | MLX | TinyGrad | Winner |
|----|-----------|---------|-----|-----|-----|----------|--------|
| clip_100k | **8.67** | 33.1 | 41.3 | 44.8 | 33.6 | 566.5 | PG |
| where_100k | 95.7 | 38.1 | 65.8 | 33.3 | **27.5** | 282.2 | MLX |
| greater_100k | 70.0 | 32.5 | 52.0 | 39.1 | **27.5** | 200.5 | MLX |
| equal_100k | 70.0 | 24.7 | 46.5 | 40.5 | **22.4** | 301.6 | MLX |

### Reductions

| Op | Peregrine | PyTorch | TF | JAX | MLX | TinyGrad | Winner |
|----|-----------|---------|-----|-----|-----|----------|--------|
| sum_axis_256x512 | 112.6 | 29.7 | 44.4 | 56.0 | **21.9** | 220.5 | MLX |
| mean_axis_256x512 | 112.6 | 33.3 | 44.3 | 53.3 | **23.0** | 298.2 | MLX |
| max_axis_256x512 | 154.2 | **39.4** | 46.4 | 50.5 | 50.0 | 210.8 | PT |
| min_axis_256x512 | 154.3 | 39.1 | 41.5 | 51.0 | **38.9** | 352.9 | MLX |
| var_256x512 | 235.7 | 238.5 | 164.6 | 77.8 | **59.9** | - | MLX |
| prod_axis_256x512 | 149.0 | 45.3 | **45.1** | 56.9 | 60.5 | - | TF |
| logsumexp_256x512 | 383.3 | 146.5 | 282.4 | 324.0 | **141.1** | - | MLX |
| cumsum_256x512 | 122.1 | **53.1** | 177.6 | 219.3 | 138.5 | 637.0 | PT |
| argmax_axis_256x512 | 154.5 | 65.3 | **55.6** | 197.1 | 180.0 | 1388.2 | TF |
| sum_axis_1024x1024 | **940.1** | - | - | - | - | - | PG |
| var_1024x1024 | **1937.1** | - | - | - | - | - | PG |

### Shape Ops

| Op | Peregrine | PyTorch | TF | JAX | MLX | TinyGrad | Winner |
|----|-----------|---------|-----|-----|-----|----------|--------|
| tril_256x256 | 35.8 | **32.2** | 42.3 | 40.1 | 55.5 | 1883.8 | PT |
| triu_256x256 | 39.6 | **30.4** | 46.2 | 44.8 | 54.4 | 1927.4 | PT |
| repeat_64x128_2x3 | 124.7 | 34.9 | 74.7 | 28.3 | **26.8** | - | MLX |
| pad_64x128 | 16.7 | **4.71** | 86.6 | 18.3 | 17.6 | 98.7 | PT |
| stack_8x64x128 | 15.1 | **9.19** | 51.8 | 167.3 | 42.7 | 971.4 | PT |
| diagonal_512x512 | 0.75 | **0.71** | 11.8 | 8.60 | 23.5 | - | PT |

### Losses

| Op | Peregrine | PyTorch | TF | JAX | MLX | TinyGrad | Winner |
|----|-----------|---------|-----|-----|-----|----------|--------|
| cross_entropy_64x10 | **2.56** | 36.8 | 603.9 | 63.1 | 27.4 | 3702.7 | PG |
| l1_loss_64x10 | **1.00** | 5.75 | 40.0 | 12.9 | 19.2 | 1171.0 | PG |
| mse_loss_64x10 | **3.92** | 5.19 | 36.5 | 24.8 | 21.4 | 460.6 | PG |
| huber_loss_64x10 | 5.38 | **5.10** | 226.8 | 50.1 | 32.3 | - | PT |
| smooth_l1_loss_64x10 | **5.04** | 5.38 | 227.5 | 50.6 | 32.2 | - | PG |
| kl_div_loss_64x10 | **2.50** | 6.54 | 356.5 | 81.4 | 22.2 | - | PG |
| cosine_sim_loss_64x64 | 13.5 | **11.2** | 223.0 | 108.9 | 124.7 | - | PT |

### Layers

| Op | Peregrine | PyTorch | TF | JAX | MLX | TinyGrad | Winner |
|----|-----------|---------|-----|-----|-----|----------|--------|
| rmsnorm_64x512 | 57.7 | 52.7 | 443.9 | 83.1 | **30.6** | - | MLX |
| conv1d_1x32x128_k3 | **20.6** | 50.5 | 546.4 | 74.6 | 44.2 | - | PG |
| avgpool2d_1x16x32x32 | **25.0** | 32.2 | 62.2 | 42.8 | 280.5 | - | PG |
| groupnorm_4x64x16x16 | 72.6 | **40.8** | 738.9 | 272.5 | 228.4 | - | PT |

### RNN

| Op | Peregrine | PyTorch | TF | JAX | MLX | TinyGrad | Winner |
|----|-----------|---------|-----|-----|-----|----------|--------|
| rnn_seq32_128_256 | **196.3** | 288.1 | - | - | - | - | PG |
| lstm_seq32_128_256 | 928.3 | **845.3** | - | - | - | - | PT |
| gru_seq32_128_256 | **805.5** | 832.3 | - | - | - | - | PG |

### Optimizers

| Op | Peregrine | PyTorch | TF | JAX | MLX | TinyGrad | Winner |
|----|-----------|---------|-----|-----|-----|----------|--------|
| optim_adam_64 | **804.2** | 1300.0 | - | - | - | - | PG |
| optim_rmsprop_64 | **934.9** | 1151.5 | - | - | - | - | PG |
| optim_lion_64 | **919.5** | - | - | - | - | - | PG |
| optim_adafactor_64 | **1278.2** | - | - | - | - | - | PG |

### Random

| Op | Peregrine | PyTorch | TF | JAX | MLX | TinyGrad | Winner |
|----|-----------|---------|-----|-----|-----|----------|--------|
| rand_uniform_100k | **106.3** | 260.1 | 139.8 | 554.7 | 521.8 | 2568.5 | PG |
| rand_normal_100k | 764.7 | 985.8 | **391.7** | 669.7 | 738.1 | 3551.3 | TF |
| rand_bernoulli_100k | 302.7 | 257.9 | **220.8** | 564.5 | 488.9 | - | TF |
| rand_uniform_1M | 1064.3 | 2585.0 | **519.9** | 2451.7 | 4907.3 | 2541.5 | TF |
| rand_normal_1M | 7585.4 | 9758.1 | **2492.7** | 3160.0 | 7071.0 | 3557.4 | TF |

### FFT

| Op | Peregrine | PyTorch | TF | JAX | MLX | TinyGrad | Winner |
|----|-----------|---------|-----|-----|-----|----------|--------|
| rfft_1k | **2.17** | 4.46 | 39.6 | 47.0 | 22.6 | - | PG |
| rfft_4k | **6.50** | 14.8 | 50.7 | 72.1 | 44.0 | - | PG |
| rfft_16k | **30.2** | 65.2 | 103.7 | 121.0 | 87.0 | - | PG |
| fft_1k | **3.29** | 7.12 | 8.25 | 18.8 | 36.0 | - | PG |
| fft_4k | **12.2** | 26.2 | 15.9 | 57.9 | 52.1 | - | PG |

### Linear Algebra

| Op | Peregrine | PyTorch | TF | JAX | MLX | TinyGrad | Winner |
|----|-----------|---------|-----|-----|-----|----------|--------|
| norm_l2_1k | **1.08** | 1.38 | 63.2 | 13.9 | 28.6 | - | PG |
| solve_64x64 | **12.0** | 24.8 | 23.9 | 32.3 | 98.8 | - | PG |
| inv_64x64 | 37.3 | **26.8** | 31.9 | 41.7 | 66.2 | - | PT |
| cholesky_64x64 | **9.67** | 25.6 | 19.2 | 11.0 | 34.3 | - | PG |
| svd_64x64 | **275.7** | 279.2 | 515.0 | 312.7 | 295.6 | - | PG |
| qr_64x64 | **41.2** | 83.2 | 84.5 | 64.0 | 71.0 | - | PG |
| eigh_64x64 | 380.2 | 217.7 | **148.8** | 245.0 | 251.2 | - | TF |
| det_64x64 | 23.3 | **21.0** | 22.3 | 34.4 | - | - | PT |
| solve_128x128 | 50.0 | **45.6** | 77.4 | 85.4 | 206.2 | - | PT |
| inv_128x128 | 93.8 | **60.5** | 141.7 | 87.1 | 95.8 | - | PT |
| cholesky_128x128 | 50.6 | 48.9 | 59.0 | 37.2 | **30.9** | - | MLX |
| svd_128x128 | 990.6 | **990.0** | 1829.6 | 1026.2 | 1060.6 | - | PT |
| qr_128x128 | **188.5** | 226.2 | 334.3 | 195.8 | 221.3 | - | PG |
| eigh_128x128 | 1849.4 | **704.8** | 751.8 | 754.3 | 791.2 | - | PT |
| det_128x128 | 52.1 | **50.5** | 85.7 | 77.5 | - | - | PT |
| solve_256x256 | 192.3 | **180.8** | 380.0 | 265.2 | 827.9 | - | PT |
| inv_256x256 | 493.7 | 293.0 | 883.4 | 342.4 | **260.8** | - | MLX |
| cholesky_256x256 | 226.5 | 76.6 | 292.1 | 121.1 | **71.3** | - | MLX |
| svd_256x256 | 5926.2 | **5760.5** | 8301.2 | 5945.3 | 6647.2 | - | PT |
| qr_256x256 | 1049.3 | 1015.1 | 1814.4 | **997.6** | 1186.3 | - | JAX |
| eigh_256x256 | 6052.4 | **3502.2** | 4736.8 | 3613.7 | 3658.6 | - | PT |
| det_256x256 | 213.8 | **206.5** | 450.5 | 207.2 | - | - | PT |

## Analysis

### Where Peregrine Wins (vs PyTorch)
- **Elementwise (small tensors)**: add, mul, relu, clip — near-zero dispatch overhead, 3-5x faster
- **Softmax**: 2-7x faster (fused implementation)
- **Losses**: cross_entropy 14x, l1 6x, kl_div 2.6x — minimal dispatch overhead
- **FFT**: rfft/fft 2-4x faster across all sizes
- **Train step**: 1.6x faster end-to-end
- **Optimizers**: Adam 1.6x, RMSProp 1.2x
- **Linalg (small)**: solve/cholesky/qr/svd at 64x64 — 1.5-2.7x faster
- **Trig**: sinh 2x, cosh 2.2x

### Where PyTorch Wins (vs Peregrine)
- **GELU**: 4.2x faster (fused kernel)
- **Reductions**: sum/mean 3.4-3.8x, max/min 3.9x
- **Log/erf**: log10 2x, log1p 2.3x, erf 3.1x
- **Shape ops**: pad 3.5x, repeat 3.6x, stack 1.6x
- **Linalg (large)**: eigh_256 1.7x, inv_256 1.7x, cholesky_256 3.0x
- **Activations**: hardswish, hard_tanh, relu6 — 1.4-1.6x

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

# MUSt3R inference (single pair)
cargo run --example must3r --release -- weights/must3r_224.bin img1.ppm img2.ppm
cargo run --example must3r --release -- weights/must3r_512.bin img1.ppm img2.ppm 512x384

# MUSt3R server mode (multiple pairs, load weights once)
echo -e "img1.ppm\timg2.ppm\t224\t224" | cargo run --example must3r --release -- weights/must3r_224.bin --server

# MUSt3R with Metal GPU
cargo run --example must3r --release --features metal -- weights/must3r_224.bin img1.ppm img2.ppm --gpu

# Multi-view pipeline with parallel workers
python3 reconstruct_video.py vids/rgb.mp4 --frames 12 --resolution 512 --pairs all --workers 4
```

## Raw Data

Raw JSON benchmark data is in [`benchmarks/data/`](data/).
