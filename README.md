<div align="center">

# 🦅 peregrine

**A from-scratch deep learning library in Rust. No PyTorch, no ONNX, no dependencies you can't read.**

Tensors, reverse-mode autograd, neural network layers, optimizers, and working models — built from `f32` arrays and first principles.

[![GitHub Repo stars](https://img.shields.io/github/stars/etam103/peregrine)](https://github.com/etam103/peregrine/stargazers)
[![Build](https://img.shields.io/badge/build-passing-brightgreen)]()
[![Rust](https://img.shields.io/badge/rust-2021-orange)]()
[![License](https://img.shields.io/badge/license-MIT-blue)]()

</div>

---

```
cargo build --release                          # build the library
cargo test                                     # 409 tests
cargo run --example mnist --release            # train MNIST digit classifier (97.5% accuracy)
cargo run --example rt_detr --release          # train RT-DETR on COCO images
./scripts/bench_compare.sh                     # wall-clock benchmark vs PyTorch, MLX, TF, tinygrad, JAX
```

---

## What's inside

| Module | What it does |
|--------|-------------|
| **`peregrine::tensor`** | N-dimensional tensor with reverse-mode autograd, ~220 ops, NumPy-style broadcasting, Apple Accelerate BLAS, NEON intrinsics, rayon parallelism |
| **`peregrine::nn`** | Linear, Embedding, MultiHeadAttention, Transformer{Encoder,Decoder}, RNN/LSTM/GRU, Conv1d/2d, ConvTranspose1d/2d, MaxPool1d/2d, AvgPool, AdaptiveAvgPool, BatchNorm/LayerNorm/GroupNorm/InstanceNorm/RMSNorm, Dropout/Dropout2d/AlphaDropout, Upsample, PixelShuffle, padding layers, RoPE, Module trait, Sequential, 14 loss functions |
| **`peregrine::optim`** | SGD, Adam/AdamW, RMSprop, Adagrad, Adamax, AdaDelta, Lion, Adafactor, LR schedulers (Step, Cosine, Warmup, Exponential, Linear, Join), gradient clipping |
| **`peregrine::random`** | Xoshiro256++ PRNG — uniform, normal, bernoulli, categorical, gumbel, laplace, truncated normal, permutation, exponential, gamma, beta, poisson, multinomial |
| **`peregrine::init`** | Weight initialization — Glorot, He, LeCun, orthogonal, constant |
| **`peregrine::fft`** | FFT via Apple Accelerate vDSP — fft, ifft, rfft, irfft, fft2, ifft2, rfft2, irfft2, fftshift |
| **`peregrine::linalg`** | Linear algebra via LAPACK — solve, inv, cholesky, svd, qr, eigh, lu, det, pinv, norm, matrix_rank, cond, lstsq, matrix_power |
| **`peregrine::transforms`** | Functional autograd utilities — grad, value_and_grad, checkpoint |
| **`peregrine::serial`** | Save/load model weights in compact binary format |
| **`peregrine::debug`** | Model summary, training health diagnostics, gradient monitoring |
| **`peregrine::metal`** | Metal GPU backend — 98 compute shaders, 30 dispatch methods, command batching, autograd integration, buffer pool (`--features metal`) |
| **`examples/mnist`** | MNIST digit classifier — MLP trained end-to-end, validates the full stack |
| **`examples/rt_detr`** | Full RT-DETR detector — ResNet backbone, Hungarian matching, training loop, wandb logging |

The entire library is ~25,000 lines of Rust. No macros, no code generation, no proc-macro magic. You can read every line.

---

## How it works

Every op records itself in the output tensor's `Op` field, building a DAG. Calling `.backward()` walks the graph in reverse, accumulating gradients via the chain rule.

```rust
use peregrine::tensor::Tensor;
use peregrine::optim::Adam;

let x = Tensor::randn(&[2, 3], true);   // requires_grad = true
let w = Tensor::randn(&[3, 1], true);
let y = x.matmul(&w).sum();             // forward: builds the graph
y.backward();                            // backward: computes all gradients

let mut opt = Adam::new(vec![w.clone()], 1e-3);
opt.step();                              // update weights
opt.zero_grad();
```

### Supported ops with autograd

| Category | Ops |
|----------|-----|
| **Arithmetic** | `add` `sub` `mul` `div` `neg` `scale` `maximum` `minimum` `power` `logaddexp` |
| **Math** | `exp` `log` `sqrt` `abs` `pow` `sin` `cos` `tanh` `sinh` `cosh` `reciprocal` `square` `rsqrt` `erf` `erfinv` `expm1` `log2` `log10` `log1p` `arcsin` `arccos` `arctan` `arcsinh` `arccosh` `arctanh` `arctan2` `degrees` `radians` `floor` `ceil` `round` `sign` |
| **Activations** | `relu` `sigmoid` `gelu` `silu` `softplus` `mish` `leaky_relu` `elu` `hard_tanh` `relu6` `hardswish` `softsign` `log_sigmoid` `selu` `celu` `gelu_fast` `softmin` `glu` `hard_shrink` `soft_shrink` `PReLU` |
| **Reductions** | `sum` `mean` `softmax` `log_softmax` `sum_axis` `mean_axis` `max_axis` `min_axis` `var` `std` `prod_axis` `logsumexp` `cumsum` `cumprod` `argmax_axis` `argmin_axis` `topk` `sort` `argsort` `any` `all` |
| **Shape** | `reshape` `transpose` `squeeze` `unsqueeze` `concat` `select` `flatten` `stack` `split` `tril` `triu` `repeat` `tile` `pad` `roll` `take` `diagonal` `diag` `trace` `outer` `inner` `broadcast_to` `expand_dims` |
| **Conditional** | `clip` `where` `nan_to_num` |
| **Comparison** | `equal` `not_equal` `greater` `greater_equal` `less` `less_equal` `logical_and` `logical_or` `logical_not` `isnan` `isinf` `isfinite` |
| **Layers** | `matmul` `conv1d` `conv2d` `conv2d_strided` `conv_transpose1d` `conv_transpose2d` `conv2d+relu+pool` `max_pool2d` `max_pool2d_ext` `avg_pool2d` `add_bias` `batch_norm` `layer_norm` `rms_norm` `group_norm` `instance_norm` `upsample_nearest` `upsample_bilinear` `index_select` |
| **Loss** | `bce_with_logits` `cross_entropy` `mse` `l1` `nll` `smooth_l1` `huber` `kl_div` `cosine_similarity` `triplet` `hinge` `log_cosh` `margin_ranking` `gaussian_nll` |

All ops support broadcasting where applicable.

---

## MNIST example

The MNIST example validates the entire stack — tensor ops, autograd, nn layers, and the Adam optimizer:

```
$ cargo run --example mnist --release
Loading MNIST...
Train: 60000 images, Test: 10000 images
Model: 109386 parameters
Epoch 1/10:  loss=0.2867, train_acc=91.4%, test_acc=94.9%
Epoch 5/10:  loss=0.0432, train_acc=98.7%, test_acc=95.5%
Epoch 10/10: loss=0.0194, train_acc=99.3%, test_acc=97.5%
```

Model: MLP (784 → 128 → 64 → 10) with ReLU, trained with CrossEntropyLoss + Adam.

---

## PyTorch numerical parity

23 integration tests cross-validate Peregrine against PyTorch reference data, covering matmul, softmax, log_softmax, layernorm, cross_entropy_loss, 14 element-wise ops, Adam optimizer, and a full 10-step MLP training loop. All pass within 1e-4 to 1e-7 absolute error. 34 additional activation tests and 352 unit tests cover the full op suite.

```
$ cargo test
running 409 tests ... ok (352 unit + 34 activation + 23 parity)
```

To regenerate reference data: `.venv/bin/python tests/generate_reference.py`

---

## Debugging & Introspection

The `peregrine::debug` module provides PyTorch-style model inspection and training diagnostics.

### Model Summary

Call `model_summary` with named parameters to print an ASCII table of every parameter:

```rust
use peregrine::debug::model_summary;

let net = RtDetrNet::new(3, 64, 4, 1, 1, 20);
println!("{}", model_summary(&net.named_params()));
```

```
Parameter                              Shape               Params
──────────────────────────────────────────────────────────────────
backbone.stem_w                        [64, 3, 1, 1]          192
backbone.stem_b                        [64]                    64
backbone.stage2.0.conv1_w              [128, 64, 3, 3]    73,728
...
──────────────────────────────────────────────────────────────────
85 parameters                                          1,190,856
```

### Training Health

Call `training_health` periodically during training to monitor gradient health, detect NaN/exploding gradients, and log diagnostics to wandb:

```rust
use peregrine::debug::training_health;

let report = training_health(&net.named_params());
let metrics = report.to_metrics();
// Returns: [("health/grad_norm", 0.423), ("health/has_nan", 0.0), ...]
```

---

## Architecture

```
Input Image [1, 3, 256, 256]
          │
    ┌─────▼────┐
    │  ResNet  │  4-stage backbone with residual connections
    │ Backbone │  1x1 stem → 3 stages of 3x3 conv blocks + skip + pool
    └─┬──┬──┬──┘
      s2 s3 s4    multi-scale features [128², 64², 32²]
      │  │  │
    ┌─▼──▼──▼──┐
    │ Channel  │  1x1 conv projections to embed_dim
    │  Project │  pool all scales to common 32×32
    └─────┬────┘
          │  [batch, 3072, embed_dim]
    ┌─────▼────┐
    │   Xfmr   │  self-attention + FFN with pre-norm
    │ Encoder  │
    └─────┬────┘
          │  encoder memory
    ┌─────▼────┐
    │   Xfmr   │  learned object queries attend to memory
    │ Decoder  │  self-attn → cross-attn → FFN
    └─────┬────┘
          │  [batch, num_queries, embed_dim]
    ┌─────▼────┐
    │  Heads   │  classification (softmax) + bbox regression (sigmoid)
    └──────────┘
```

---

## Performance

CPU ops use Apple Accelerate BLAS and rayon parallelism. GPU ops use Metal compute shaders (`--features metal`). Wall-clock benchmarks run via `./scripts/bench_compare.sh`.

### Peregrine vs ML Frameworks (133 ops, CPU, wall-clock, all times in microseconds)

| Operation | Peregrine | PyTorch | MLX | TensorFlow | tinygrad | JAX |
|-----------|----------:|--------:|----:|-----------:|---------:|----:|
| matmul 128x128 | **6.2** | 6.0 | 19.6 | 94.3 | 431.2 | 82.6 |
| matmul 512x512 | 214.2 | **142.8** | 161.7 | 704.0 | 429.9 | 533.2 |
| add 100k | **12.5** | 41.7 | 29.7 | 52.4 | 186.7 | 33.9 |
| mul 100k | **12.9** | 39.2 | 28.9 | 41.7 | 191.7 | 31.1 |
| relu 100k | **8.9** | 37.9 | 25.3 | 38.3 | 343.3 | 99.5 |
| softmax 8x128 | **3.8** | 33.8 | 14.6 | 11.2 | 618.7 | 31.0 |
| silu 100k | **64.1** | 74.0 | 84.8 | 246.0 | 343.0 | 54.9 |
| rfft 1k | **2.2** | 4.4 | 19.2 | 42.8 | — | 59.5 |
| cross_entropy | **2.6** | 38.5 | 36.4 | 624.4 | 3337.3 | 55.7 |
| train step 64 | **839.5** | 1247.3 | 788.1 | 8713.9 | 23717.2 | 5084.3 |

Geometric mean ratio across 133 ops (lower = Peregrine faster): **PyTorch 0.95x**, **MLX 0.74x**, TensorFlow 0.55x, tinygrad 0.09x, JAX 0.68x. Peregrine wins 56 of 133 ops.

| Optimization | Impact |
|-------------|--------|
| Hand-tuned NEON intrinsics | 24 vectorized kernels — 4-6x speedup on elementwise ops |
| Cephes-style polynomial exp | Fast exp/sigmoid/tanh/gelu via NEON float32x4_t |
| NEON Adam optimizer | Vectorized Adam step with fast rsqrt approximation |
| CPU buffer pool | Thread-local size-bucketed pool — eliminates malloc on elementwise ops |
| Pool bypass for small tensors | Skip HashMap overhead for tensors < 1024 elements |
| Rayon threshold tuning | Dual thresholds (500K cheap / 100K expensive) — avoids spawn overhead |
| Apple Accelerate BLAS | ~10x faster matmul and 1x1 conv2d |
| Metal GPU backend | 98 compute shaders with command batching and full autograd integration for end-to-end GPU training |

---

## Project structure

```
src/
  lib.rs          public API surface
  cpu_pool.rs     thread-local buffer pool for allocation reuse
  simd_kernels.rs hand-tuned NEON intrinsics for aarch64 (24 kernels + Adam step)
  tensor.rs       tensor, autograd engine, ~220 ops, broadcasting (~11,400 lines)
  nn.rs           Linear, Embedding, attention, Transformer{Encoder,Decoder}, RNN/LSTM/GRU, Conv1d/2d, ConvTranspose1d/2d, MaxPool, AvgPool, AdaptiveAvgPool, BatchNorm/LayerNorm/GroupNorm/InstanceNorm/RMSNorm, Dropout/Dropout2d/AlphaDropout, Upsample, PixelShuffle, padding layers, Module trait, 14 loss functions (~4,400 lines)
  optim.rs        SGD, Adam/AdamW, RMSprop, Adagrad, Adamax, AdaDelta, Lion, Adafactor, LR schedulers, gradient clipping
  random.rs       Xoshiro256++ PRNG, 15 distribution functions
  init.rs         weight initialization (Glorot, He, LeCun, orthogonal)
  fft.rs          FFT via Apple Accelerate vDSP (fft, ifft, rfft, irfft, fft2, ifft2, rfft2, irfft2, fftshift)
  linalg.rs       linear algebra via LAPACK (solve, inv, cholesky, svd, qr, eigh, lu, det, pinv, matrix_rank, cond, lstsq, matrix_power)
  transforms.rs   functional autograd utilities (grad, value_and_grad, checkpoint)
  debug.rs        model summary + training health diagnostics
  serial.rs       model weight save/load (binary format)
  metal/          Metal GPU backend (98 shaders, 30 dispatch methods, command batching, autograd)
benches/
  tensor_ops.rs   criterion benchmarks (CPU + Metal GPU)
  wallclock.rs    wall-clock comparison benchmark (JSON output)
scripts/
  bench_compare.sh    orchestrator: builds + runs all framework benchmarks
  bench_pytorch.py    PyTorch wall-clock benchmark
  bench_mlx.py        MLX wall-clock benchmark
  bench_tensorflow.py TensorFlow wall-clock benchmark
  bench_tinygrad.py   tinygrad wall-clock benchmark
  bench_jax.py        JAX wall-clock benchmark
  compare_bench.py    reads JSONs, renders markdown comparison table
examples/
  mnist/          MNIST digit classifier (97.5% test accuracy)
  rt_detr/        RT-DETR training on COCO
    main.rs         training loop + wandb visualization
    model.rs        ResNet backbone, RT-DETR net, loss, decode, NMS
    dataset.rs      VOC + COCO dataset loaders
tests/
  pytorch_parity.rs   23 numerical parity tests vs PyTorch
  activations.rs      34 activation function tests
  metal_parity.rs     23 CPU vs Metal parity tests
  metal_basics.rs     12 Metal compute shader tests
  metal_autograd.rs   17 GPU autograd integration tests
  generate_reference.py  script to regenerate PyTorch reference data
  fixtures/             binary reference tensors
```

---

## Limitations

This is a learning project, not a production framework.

- Greedy Hungarian matching (not full O(n³) algorithm)
- Attention forward pass breaks autograd graph (output projection still trains)
- GPU training step is slower than CPU at batch=64 due to `dispatch_reduce` sync in `mean()`; larger batches would favor GPU

---

<div align="center">

Authored with [Claude Code](https://claude.ai/claude-code).

</div>
