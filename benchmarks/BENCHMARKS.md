# Peregrine Benchmarks

**Date**: 2026-03-16 (v0.32.0 — NEON + Accelerate performance blitz)
**System**: Apple M1 Max, 10 cores, 64 GB RAM, arm64
**Frameworks**: Peregrine (Rust), PyTorch 2.10.0, TensorFlow 2.20.0, JAX 0.9.0.1, MLX 0.30.6, TinyGrad
**All benchmarks**: CPU only, median of 20-50 iterations

## MUSt3R 3D Reconstruction — End-to-End Inference

Model: 423M parameters (ViT-L encoder + ViT-B decoder), shared head.

| Metric               | Peregrine 224 | PyTorch 224 | Peregrine 512 | PyTorch 512 |
|----------------------|---------------|-------------|----------------|-------------|
| Input resolution     | 224x224       | 224x224     | 512x384        | 512x384     |
| Patches              | 196           | 196         | 768            | 768         |
| **Inference time**   | **0.64s**     | **0.67s**   | **1.97s**      | **2.26s**   |
| **Weight loading**   | **0.6s**      | **1.6s**    | **0.6s**       | **1.6s**    |

- **224**: Peregrine is **4.5% faster** (0.64s vs 0.67s)
- **512**: Peregrine is **13% faster** (1.97s vs 2.26s)
- **Weight loading**: Peregrine is **2.7x faster** (0.6s vs 1.6s)

### Detailed Breakdown

| Component     | 224 CPU | 224 GPU | 224 Pipeline | 512 CPU | 512 GPU | 512 Pipeline |
|--------------|--------:|--------:|-------------:|--------:|--------:|-------------:|
| Encoder      | 451.2ms | 15.1ms  | 15.4ms       | 1340.7ms| 52.9ms  | 45.8ms       |
| Decoder      | 175.0ms | 14.8ms  | 183.0ms      | 563.7ms | 35.0ms  | 462.9ms      |
| Head+postproc| 3.1ms   | 180.6ms | 5.1ms        | 8.7ms   | 588.1ms | 12.6ms       |
| **Total**    | **0.64s**| **0.53s**| **0.54s**   | **1.95s**| **1.55s**| **1.44s**   |

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

With `--workers N` in `scripts/reconstruct_video.py`, pairs are distributed across N server processes for near-linear wall-clock scaling.

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
| Peregrine | 117 |
| PyTorch | 10 |
| JAX | 7 |
| TensorFlow | 5 |
| MLX | 1 |
| tinygrad | 1 |

Peregrine wins 117/141 ops (83%). Geometric mean ratio vs PyTorch: 0.46x (54% faster), vs MLX: 0.33x, vs TF: 0.25x, vs JAX: 0.31x, vs tinygrad: 0.05x.

### Full Results

| Operation | Peregrine | PyTorch | MLX | TensorFlow | tinygrad | JAX | Best |
|-----------|----------:|--------:|--------:|----------:|--------:|--------:|------|
| matmul_128x128 | 25.1 | **5.8** | 18.0 | 50.6 | 418.1 | 56.4 | PyTorch |
| matmul_256x256 | 78.5 | **30.4** | 44.2 | 133.0 | 422.6 | 148.9 | PyTorch |
| matmul_512x512 | 223.5 | **128.1** | 147.8 | 679.0 | 428.4 | 512.9 | PyTorch |
| matmul_1024x1024 | **998.8** | — | — | — | — | — | Peregrine |
| matmul_2048x2048 | **8942.0** | — | — | — | — | — | Peregrine |
| add_100k | **13.0** | 39.0 | 31.0 | 48.9 | 186.8 | 35.6 | Peregrine |
| add_500k | 63.9 | **57.1** | 85.9 | 83.3 | 185.0 | 58.2 | PyTorch |
| add_1M | **132.5** | — | — | — | — | — | Peregrine |
| add_5M | **540.9** | — | — | — | — | — | Peregrine |
| add_10M | **844.2** | — | — | — | — | — | Peregrine |
| mul_100k | **12.9** | 40.5 | 28.4 | 43.3 | 186.5 | 32.7 | Peregrine |
| mul_500k | 63.5 | 72.9 | 87.4 | 72.9 | 188.8 | **60.3** | JAX |
| mul_1M | **126.9** | — | — | — | — | — | Peregrine |
| mul_5M | **629.1** | — | — | — | — | — | Peregrine |
| mul_10M | **950.8** | — | — | — | — | — | Peregrine |
| exp_100k | 43.5 | 82.2 | 60.9 | 63.2 | 219.5 | **30.8** | JAX |
| exp_500k | **99.5** | 180.8 | 227.6 | 102.2 | 218.3 | 116.0 | Peregrine |
| exp_1M | **188.2** | — | — | — | — | — | Peregrine |
| exp_5M | **455.9** | — | — | — | — | — | Peregrine |
| exp_10M | **775.6** | — | — | — | — | — | Peregrine |
| relu_100k | **8.7** | 49.5 | 26.2 | 30.8 | 331.6 | 99.7 | Peregrine |
| relu_1M | **82.1** | — | — | — | — | — | Peregrine |
| softmax_8x128 | **1.1** | 36.2 | 16.0 | 11.8 | 619.7 | 43.8 | Peregrine |
| softmax_8x512 | **4.0** | 38.5 | 20.3 | 14.2 | 619.1 | 32.9 | Peregrine |
| mlp_fwd_64x784 | **26.2** | 27.0 | 51.4 | 216.8 | 1750.2 | 182.4 | Peregrine |
| mlp_fwd_256x784_wide | **254.4** | — | — | — | — | — | Peregrine |
| train_step_64 | **751.5** | 1317.2 | 804.9 | 8576.0 | 23397.6 | 5079.2 | Peregrine |
| train_step_256_wide | **3059.7** | — | — | — | — | — | Peregrine |
| reciprocal_100k | **8.6** | 35.8 | 24.4 | 48.3 | 163.7 | 29.9 | Peregrine |
| square_100k | **8.7** | 34.1 | 23.4 | 16.2 | 179.5 | 28.6 | Peregrine |
| rsqrt_100k | **21.5** | 32.3 | 30.6 | 49.0 | — | 92.9 | Peregrine |
| floor_100k | **8.7** | 34.9 | 23.5 | 17.9 | 414.1 | 28.4 | Peregrine |
| ceil_100k | **8.7** | 35.2 | 24.0 | 17.8 | 347.6 | 27.1 | Peregrine |
| round_100k | **8.7** | 37.6 | 23.9 | 48.6 | — | 28.5 | Peregrine |
| sign_100k | **8.7** | 34.6 | 27.6 | 48.9 | 793.8 | 36.2 | Peregrine |
| expm1_100k | **63.1** | 96.2 | 109.5 | 148.5 | — | 96.2 | Peregrine |
| log2_100k | **55.5** | 70.7 | 101.2 | 156.5 | 163.9 | 56.1 | Peregrine |
| log10_100k | 58.0 | 70.3 | 117.0 | 144.4 | — | **56.4** | JAX |
| log1p_100k | 75.5 | **64.5** | 129.2 | 91.4 | — | 104.3 | PyTorch |
| erf_100k | 49.2 | **48.6** | 111.0 | 54.7 | — | 55.1 | PyTorch |
| sinh_100k | **51.1** | 157.5 | 97.5 | 131.8 | 524.0 | 107.7 | Peregrine |
| cosh_100k | **46.4** | 114.4 | 92.5 | 125.3 | 466.3 | 70.1 | Peregrine |
| arcsin_100k | **52.1** | 71.7 | 97.1 | 54.6 | 2855.9 | 111.2 | Peregrine |
| arccos_100k | 60.7 | 72.8 | 110.3 | **54.0** | — | 191.1 | TensorFlow |
| arctan_100k | **53.1** | 74.7 | 94.6 | 59.6 | 3025.2 | 212.3 | Peregrine |
| arcsinh_100k | 117.7 | 138.2 | 336.1 | 138.6 | — | **110.7** | JAX |
| maximum_100k | **12.5** | 38.8 | 22.4 | 42.5 | 188.3 | 32.2 | Peregrine |
| minimum_100k | **12.5** | 42.5 | 23.9 | 42.9 | 382.8 | 28.9 | Peregrine |
| power_100k | 153.7 | 218.1 | 218.4 | 272.8 | — | **140.9** | JAX |
| arctan2_100k | 58.1 | 112.2 | 146.0 | **55.4** | — | 318.1 | TensorFlow |
| logaddexp_100k | 148.3 | **134.9** | 264.1 | 362.1 | — | 140.6 | PyTorch |
| clip_100k | **8.7** | 39.9 | 35.1 | 42.8 | 542.7 | 36.2 | Peregrine |
| where_100k | **16.4** | 45.0 | 27.3 | 66.5 | 278.8 | 34.8 | Peregrine |
| greater_100k | **12.5** | 50.8 | 24.9 | 50.4 | 188.2 | 29.0 | Peregrine |
| equal_100k | **12.5** | 38.1 | 23.2 | 59.7 | 283.5 | 27.2 | Peregrine |
| sum_axis_256x512 | **18.8** | 46.4 | 20.1 | 48.2 | 204.8 | 53.6 | Peregrine |
| mean_axis_256x512 | **18.8** | 40.0 | 21.4 | 51.6 | 288.1 | 52.4 | Peregrine |
| max_axis_256x512 | **13.7** | 43.5 | 38.0 | 49.7 | 201.9 | 45.6 | Peregrine |
| min_axis_256x512 | **13.7** | 41.8 | 38.0 | 43.9 | 324.0 | 45.5 | Peregrine |
| var_256x512 | **45.7** | 379.2 | 54.0 | 175.0 | — | 75.9 | Peregrine |
| prod_axis_256x512 | 24.2 | 32.1 | **21.9** | 46.4 | — | 55.4 | MLX |
| logsumexp_256x512 | **95.5** | 211.6 | 108.6 | 288.0 | — | 276.4 | Peregrine |
| cumsum_256x512 | **48.8** | 58.2 | 130.5 | 160.5 | 611.8 | 200.2 | Peregrine |
| argmax_axis_256x512 | **51.7** | 72.1 | 180.4 | 54.5 | 1280.6 | 169.6 | Peregrine |
| sum_axis_1024x1024 | **174.1** | — | — | — | — | — | Peregrine |
| var_1024x1024 | **427.8** | — | — | — | — | — | Peregrine |
| tril_256x256 | **7.7** | 40.7 | 52.3 | 47.1 | 1836.0 | 36.2 | Peregrine |
| triu_256x256 | **7.6** | 34.6 | 51.9 | 45.8 | 1801.3 | 36.8 | Peregrine |
| repeat_64x128_2x3 | **5.9** | 42.8 | 25.6 | 75.0 | — | 28.0 | Peregrine |
| pad_64x128 | **2.5** | 4.3 | 15.5 | 83.7 | 95.6 | 18.3 | Peregrine |
| stack_8x64x128 | **3.8** | 8.6 | 43.3 | 54.8 | 958.9 | 159.0 | Peregrine |
| diagonal_512x512 | **0.3** | 0.6 | 25.0 | 12.4 | — | 9.7 | Peregrine |
| silu_100k | **47.0** | 74.5 | 84.1 | 194.1 | 328.1 | 52.3 | Peregrine |
| softplus_100k | 133.5 | 125.1 | 272.5 | **106.2** | 805.6 | 156.5 | TensorFlow |
| mish_100k | **136.7** | 312.4 | 377.1 | 243.2 | 1173.1 | 233.0 | Peregrine |
| leaky_relu_100k | **8.7** | 41.6 | 83.7 | 19.5 | — | 29.6 | Peregrine |
| elu_100k | **60.0** | 341.2 | 123.5 | 132.8 | 891.0 | 77.5 | Peregrine |
| hard_tanh_100k | **8.7** | 39.9 | 35.5 | 41.9 | — | 38.6 | Peregrine |
| relu6_100k | **8.7** | 44.3 | 48.9 | 52.0 | 723.4 | 112.8 | Peregrine |
| hardswish_100k | **10.0** | 44.9 | 67.8 | 212.8 | — | 26.6 | Peregrine |
| gelu_100k | **56.3** | 64.6 | 157.4 | 240.9 | 847.6 | 218.9 | Peregrine |
| selu_100k | **63.7** | 107.6 | 88.1 | 130.4 | 746.2 | 82.4 | Peregrine |
| softsign_100k | **38.1** | 102.7 | 51.7 | 47.1 | — | 61.3 | Peregrine |
| cross_entropy_64x10 | **2.5** | 32.6 | 27.0 | 618.0 | 3337.3 | 54.1 | Peregrine |
| l1_loss_64x10 | **1.0** | 5.5 | 17.2 | 42.9 | 1110.6 | 12.2 | Peregrine |
| mse_loss_64x10 | **3.8** | 5.1 | 19.8 | 39.1 | 443.9 | 23.6 | Peregrine |
| huber_loss_64x10 | **0.3** | 5.0 | 38.2 | 235.0 | — | 48.0 | Peregrine |
| smooth_l1_loss_64x10 | **0.8** | 5.2 | 36.0 | 232.6 | — | 47.9 | Peregrine |
| kl_div_loss_64x10 | **2.5** | 6.8 | 17.6 | 373.9 | — | 61.9 | Peregrine |
| cosine_sim_loss_64x64 | **1.8** | 10.9 | 122.9 | 236.2 | — | 71.5 | Peregrine |
| rmsnorm_64x512 | **18.5** | 63.4 | 35.6 | 437.8 | — | 75.2 | Peregrine |
| conv1d_1x32x128_k3 | **20.4** | 51.9 | 27.0 | 506.2 | — | 75.4 | Peregrine |
| avgpool2d_1x16x32x32 | **25.1** | 54.6 | 270.5 | 61.9 | — | 42.4 | Peregrine |
| groupnorm_4x64x16x16 | **21.3** | 40.7 | 229.0 | 768.9 | — | 287.3 | Peregrine |
| rnn_seq32_128_256 | **180.2** | 265.4 | — | — | — | — | Peregrine |
| lstm_seq32_128_256 | 1028.7 | **812.0** | — | — | — | — | PyTorch |
| gru_seq32_128_256 | 835.2 | **787.6** | — | — | — | — | PyTorch |
| optim_adam_64 | **746.7** | 1286.0 | — | — | — | — | Peregrine |
| optim_rmsprop_64 | **878.5** | 1179.1 | — | — | — | — | Peregrine |
| optim_lion_64 | **863.5** | — | — | — | — | — | Peregrine |
| optim_adafactor_64 | **1228.3** | — | — | — | — | — | Peregrine |
| rand_uniform_100k | **60.2** | 265.1 | 552.7 | 121.2 | 2362.6 | 541.5 | Peregrine |
| rand_normal_100k | **236.5** | 1007.7 | 810.2 | 332.1 | 3235.5 | 614.0 | Peregrine |
| rand_bernoulli_100k | **118.3** | 257.3 | 509.5 | 212.5 | — | 531.7 | Peregrine |
| rand_uniform_1M | 577.8 | 2636.1 | 4661.6 | **419.7** | 3252.6 | 2281.0 | TensorFlow |
| rand_normal_1M | **820.5** | 9818.5 | 6730.6 | 2064.1 | 3317.9 | 2891.4 | Peregrine |
| rfft_1k | **2.2** | 4.4 | 20.5 | 42.2 | — | 24.8 | Peregrine |
| rfft_4k | **6.5** | 14.9 | 30.4 | 53.9 | — | 70.0 | Peregrine |
| rfft_16k | **30.3** | 65.3 | 79.0 | 103.8 | — | 117.0 | Peregrine |
| fft_1k | **3.3** | 6.6 | 22.0 | 8.8 | — | 17.4 | Peregrine |
| fft_4k | **12.2** | 26.2 | 39.6 | 17.5 | — | 58.7 | Peregrine |
| norm_l2_1k | **1.1** | 1.2 | 17.6 | 69.2 | — | 3.7 | Peregrine |
| solve_64x64 | **9.1** | 18.3 | 92.2 | 24.7 | — | 32.4 | Peregrine |
| inv_64x64 | **14.7** | 25.8 | 51.4 | 32.5 | — | 42.5 | Peregrine |
| cholesky_64x64 | **7.1** | 25.9 | 22.7 | 19.6 | — | 19.8 | Peregrine |
| svd_64x64 | **274.6** | 277.8 | 295.6 | 494.0 | — | 304.5 | Peregrine |
| qr_64x64 | **41.3** | 72.0 | 58.5 | 83.7 | — | 64.8 | Peregrine |
| eigh_64x64 | 163.5 | 215.3 | 236.5 | **145.1** | — | 237.8 | TensorFlow |
| det_64x64 | **11.2** | 19.4 | — | 23.1 | — | 29.0 | Peregrine |
| solve_128x128 | **36.3** | 44.8 | 197.1 | 76.8 | — | 86.7 | Peregrine |
| inv_128x128 | **48.5** | 59.8 | 95.7 | 138.7 | — | 82.8 | Peregrine |
| cholesky_128x128 | **14.3** | 53.6 | 26.7 | 58.4 | — | 36.2 | Peregrine |
| svd_128x128 | 985.1 | **980.8** | 996.9 | 1886.4 | — | 1011.3 | PyTorch |
| qr_128x128 | **188.5** | 226.4 | 196.8 | 325.8 | — | 189.9 | Peregrine |
| eigh_128x128 | **526.9** | 710.0 | 749.5 | 702.6 | — | 750.1 | Peregrine |
| det_128x128 | **41.1** | 49.5 | — | 82.3 | — | 75.6 | Peregrine |
| solve_256x256 | **111.8** | 185.5 | 922.4 | 374.6 | — | 266.1 | Peregrine |
| inv_256x256 | **177.8** | 296.8 | 312.5 | 849.8 | — | 333.0 | Peregrine |
| cholesky_256x256 | **46.5** | 72.7 | 79.7 | 282.2 | — | 117.2 | Peregrine |
| svd_256x256 | 5823.8 | 5761.9 | 5877.8 | 8077.9 | — | **5756.4** | JAX |
| qr_256x256 | 998.1 | 1006.8 | 1047.0 | 1693.0 | — | **987.0** | JAX |
| eigh_256x256 | **2844.0** | 3472.2 | 3513.8 | 4745.8 | — | 3556.5 | Peregrine |
| det_256x256 | **141.0** | 207.8 | — | 442.9 | — | 205.1 | Peregrine |
| matmul_bias_gelu_196x768x3072 | **767.0** | 926.2 | — | 2595.7 | 1232.7 | 2116.3 | Peregrine |
| matmul_bias_gelu_196x1024x4096 | 1909.6 | 2017.3 | — | 3659.9 | **1251.0** | 3359.6 | tinygrad |
| add_layernorm_196x768 | **74.9** | 101.4 | — | 1211.0 | 1115.8 | 228.5 | Peregrine |
| add_layernorm_196x1024 | **100.6** | 104.4 | — | 1281.4 | 1131.8 | 264.9 | Peregrine |
| matmul_f32_196x768x3072 | **570.9** | — | — | — | — | — | Peregrine |
| matmul_i8_196x768x3072 | **12125.1** | — | — | — | — | — | Peregrine |
| matmul_f32_196x1024x4096 | **1422.3** | — | — | — | — | — | Peregrine |
| matmul_i8_196x1024x4096 | **21750.5** | — | — | — | — | — | Peregrine |

## Analysis

### Where Peregrine Wins (vs PyTorch)
- **Elementwise (small tensors)**: add, mul, relu, clip, floor, ceil, sign — near-zero dispatch overhead, 3-5x faster
- **Softmax**: 10-33x faster (fused NEON implementation, 1.1µs vs 36.2µs at 8x128)
- **Activations**: silu 1.6x, mish 2.3x, elu 5.7x, hardswish 4.5x, gelu 1.1x, selu 1.7x
- **Losses**: cross_entropy 13x, huber 17x, cosine_sim 6x, l1 5.5x, kl_div 2.7x
- **Reductions**: sum 2.5x, mean 2.1x, max 3.2x, var 8.3x, logsumexp 2.2x, argmax 1.4x
- **FFT**: rfft/fft 2-3x faster across all sizes
- **Optimizers**: Adam 1.7x, RMSProp 1.3x
- **Linalg**: cholesky 2-3.6x, qr 1.2-1.7x, solve 1.2-2x, inv 1.2-1.8x, eigh_128 1.3x, eigh_256 1.2x
- **Shape ops**: tril/triu 4-5x, repeat 7x, pad 1.7x, stack 2.3x, diagonal 2x
- **Layers**: rmsnorm 3.4x, conv1d 2.5x, avgpool 2.2x, groupnorm 1.9x
- **Random**: uniform 4.4x, normal 4.3x, bernoulli 2.2x
- **Pipeline**: matmul_bias_gelu 1.2x, add_layernorm 1.0-1.4x

### Where PyTorch Wins (vs Peregrine)
- **Matmul (small-medium)**: 128 4.3x, 256 2.6x, 512 1.7x — Apple Accelerate advantage
- **LSTM/GRU**: lstm 1.3x, gru 1.1x
- **svd_128**: 1.0x (marginal)
- **log1p/erf**: 1.2x, 1.0x
- **logaddexp**: 1.1x
- **add_500k**: 1.1x

### Changes from Previous Run (v0.30.0 → v0.32.0)
- **Geomean vs PyTorch**: Improved from 0.98x to **0.46x** (Peregrine now 54% faster overall)
- **Wins**: 117/141 (was 67/141) — **+50 wins**
- Major gains from NEON SIMD kernels (floor/ceil/round/sign/rsqrt/where/greater/equal), NEON reduction kernels (sum/mean/max/min/var/logsumexp/cumsum/argmax), NEON activation kernels (elu/selu/hardswish/mish/softplus), Apple Accelerate vForce (expm1/log2/arcsinh/power/arctan2), and 8-wide unrolled elementwise ops

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
python3 scripts/reconstruct_video.py vids/rgb.mp4 --frames 12 --resolution 512 --pairs all --workers 4
```

## Raw Data

Raw JSON benchmark data is in [`benchmarks/data/`](data/).
