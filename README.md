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
cargo test                                     # 633 tests
cargo run --example mnist --release            # train MNIST digit classifier (97.5% accuracy)
cargo run --example rt_detr --release          # train RT-DETR on COCO images
./scripts/bench_compare.sh                     # wall-clock benchmark vs PyTorch, MLX, TF, tinygrad, JAX
```

---

## What's inside

| Module | What it does |
|--------|-------------|
| **`peregrine::tensor`** | N-dimensional tensor with reverse-mode autograd, ~220 ops, NumPy-style broadcasting, Apple Accelerate BLAS, NEON intrinsics, rayon parallelism |
| **`peregrine::attention`** | Core GQA attention — `StandardKVCache` (append/rollback), `gqa_attention_cpu`, `AttentionMask` (None/Causal/SlidingWindow/LocalGlobal), `PostScoreTransform` (None/LogitCap) |
| **`peregrine::speculative`** | Speculative decoding — `CausalLM` trait, draft-propose/target-verify with stochastic acceptance (Leviathan 2023) |
| **`peregrine::nn`** | Linear, Embedding, MultiHeadAttention, Transformer{Encoder,Decoder}, RNN/LSTM/GRU, Conv1d/2d, ConvTranspose1d/2d, MaxPool1d/2d, AvgPool, AdaptiveAvgPool, BatchNorm/LayerNorm/GroupNorm/InstanceNorm/RMSNorm, Dropout/Dropout2d/AlphaDropout, Upsample, PixelShuffle, padding layers, RoPE, Module trait, Sequential, 14 loss functions |
| **`peregrine::optim`** | SGD, Adam/AdamW, RMSprop, Adagrad, Adamax, AdaDelta, Lion, Adafactor, LR schedulers (Step, Cosine, Warmup, Exponential, Linear, Join), gradient clipping |
| **`peregrine::random`** | Xoshiro256++ PRNG — uniform, normal, bernoulli, categorical, gumbel, laplace, truncated normal, permutation, exponential, gamma, beta, poisson, multinomial |
| **`peregrine::init`** | Weight initialization — Glorot, He, LeCun, orthogonal, constant |
| **`peregrine::fft`** | FFT via Apple Accelerate vDSP — fft, ifft, rfft, irfft, fft2, ifft2, rfft2, irfft2, fftshift |
| **`peregrine::linalg`** | Linear algebra via LAPACK — solve, inv, cholesky, svd, qr, eigh, lu, det, pinv, norm, matrix_rank, cond, lstsq, matrix_power |
| **`peregrine::transforms`** | Functional autograd utilities — grad, value_and_grad, checkpoint |
| **`peregrine::quant`** | Int8 quantized inference — per-column symmetric weight quantization, per-row dynamic activation quantization, NEON i8 GEMM, Metal dequant matmul |
| **`peregrine::sparse`** | 2:4 structured sparsity — prune to 2:4 pattern, nibble-packed indices, NEON sparse GEMM, Metal sparse matmul (scalar 16x16 + simdgroup 32x32), 1.78x bandwidth reduction |
| **`peregrine::serial`** | Save/load model weights in compact binary format (f32, int8 quantized, and 2:4 sparse) |
| **`peregrine::debug`** | Model summary, training health diagnostics, gradient monitoring |
| **`peregrine::rl`** | RL algorithms and infrastructure — PPO, DQN, REINFORCE, replay buffers, rollout buffers, Environment/ReasoningEnv traits, action spaces |
| **`peregrine::envs`** | 10 RL environments — CartPole, MountainCar, GridWorld, FrozenLake, BasicArithmetic, ChainArithmetic, NumberSorting, SequenceCompletion, PropositionalLogic, TicTacToe |
| **`peregrine::metal`** | Metal GPU backend — 108 compute shaders, 41 dispatch methods, fused op pipelines, causal masked SDPA with GQA, 2:4 sparse matmul, command batching, autograd integration, buffer pool, heterogeneous GPU+CPU scheduling (`--features metal`) |
| **`examples/mnist`** | MNIST digit classifier — MLP trained end-to-end, validates the full stack |
| **`examples/rt_detr`** | Full RT-DETR detector — ResNet backbone, Hungarian matching, training loop, wandb logging |
| **`examples/must3r`** | MUSt3R 3D reconstruction — 423M param ViT-L/B, matches PyTorch speed (0.67s at 224, 13% faster at 512). Server mode (`--server`) for persistent weight loading, parallel workers (`--workers N`), Metal GPU (`--gpu`) with full GPU-resident attention (27% faster than CPU at 512), heterogeneous GPU+CPU pipeline (`--pipeline`) overlaps decoder views. Multi-view pipeline with global pose optimization and point fusion (`reconstruct_video.py`) |
| **`examples/grok1`** | Grok-1 (314B MoE) inference — 64-layer transformer with GQA (48/8 heads), 8 experts top-2, SwiGLU FFN, RoPE, RMSNorm, KV cache, SentencePiece tokenizer. `--small` mode for testing without checkpoint, `--speculative N` for speculative decoding |
| **`examples/deepseek`** | DeepSeek-V3/R1 (671B MoE) inference — 61-layer transformer with MLA (Multi-head Latent Attention), compressed KV cache (512-dim latent), 256 routed experts top-8 with shared expert, YaRN RoPE, sigmoid routing with group-limited selection. `--small` mode for testing without checkpoint, `--speculative N` for speculative decoding |
| **`examples/rl_demo`** | RL training demos with interactive HTML visualizations — PPO on CartPole, DQN on GridWorld, REINFORCE on BasicArithmetic. Generates learning curve charts and canvas animations |
| **`examples/moba`** | MOBA 3v3 with LSTM-based PPO and self-play — single-lane map (32x16), heroes, towers, creeps, bases. Train, selfplay, watch (HTML replay), video (MP4 export via FFmpeg) |

The entire library is ~35,400 lines of Rust. No macros, no code generation, no proc-macro magic. You can read every line.

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
| **Layers** | `matmul` `matmul_quantized` `matmul_sparse_24` `conv1d` `conv2d` `conv2d_strided` `conv_transpose1d` `conv_transpose2d` `conv2d+relu+pool` `max_pool2d` `max_pool2d_ext` `avg_pool2d` `add_bias` `batch_norm` `layer_norm` `rms_norm` `group_norm` `instance_norm` `upsample_nearest` `upsample_bilinear` `index_select` `matmul_bias_gelu` `add_layer_norm` |
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

23 integration tests cross-validate Peregrine against PyTorch reference data, covering matmul, softmax, log_softmax, layernorm, cross_entropy_loss, 14 element-wise ops, Adam optimizer, and a full 10-step MLP training loop. All pass within 1e-4 to 1e-7 absolute error. 34 additional activation tests, 8 GQA attention tests, 4 speculative decoding tests, 4 sparse parity tests, and 352 unit tests cover the full op suite.

```
$ cargo test
running 633 tests ... ok (356 unit + 34 activation + 23 parity + 4 quantization + 4 sparse + 8 attention + 4 speculative + 35 metal parity + 12 metal basics + 17 metal autograd + ...)
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

### Peregrine vs ML Frameworks (173 ops, CPU, wall-clock, all times in microseconds)

| Operation | Peregrine | PyTorch | MLX | TensorFlow | tinygrad | JAX |
|-----------|----------:|--------:|----:|-----------:|---------:|----:|
| matmul 128x128 | 6.7 | **6.2** | 28.4 | 69.1 | 457.9 | 71.4 |
| matmul 512x512 | 216.8 | **142.0** | 221.2 | 724.8 | 462.0 | 621.9 |
| add 100k | **13.5** | 41.0 | 32.4 | 46.1 | 195.2 | 45.0 |
| mul 100k | **13.6** | 39.5 | 33.0 | 43.6 | 205.2 | 47.4 |
| relu 100k | **9.2** | 39.2 | 27.0 | 41.2 | 375.3 | 109.0 |
| softmax 8x128 | **1.3** | 33.2 | 19.0 | 10.4 | 795.8 | 33.3 |
| gelu 100k | **66.2** | 77.7 | 148.3 | 331.8 | 903.6 | 248.9 |
| rfft 1k | **2.1** | 4.7 | 20.6 | 41.2 | — | 51.1 |
| cross_entropy | **2.8** | 44.3 | 24.0 | 627.2 | 3812.0 | 66.7 |
| train step 64 | 874.0 | 1316.3 | **869.1** | 8918.2 | 27402.5 | 5820.6 |
| matmul+bias+gelu 196x768x3072 | 1680.2 | **1089.0** | — | 3659.8 | 1481.7 | 2375.9 |
| add+layernorm 196x768 | **117.6** | 140.8 | — | 1315.4 | 1182.5 | 217.9 |

Geometric mean ratio across 173 ops (lower = Peregrine faster): **PyTorch 0.99x**, **MLX 0.75x**, TensorFlow 0.54x, tinygrad 0.10x, JAX 0.60x. Peregrine wins 96 of 173 ops.

### MUSt3R 3D Reconstruction (423M params, Apple Silicon)

| Resolution | CPU | GPU | GPU+Pipeline | PyTorch CPU |
|-----------|----:|----:|-------------:|------------:|
| 224x224 | 0.65s | 0.53s | **0.54s** | 0.67s |
| 512x384 | 1.89s | 1.55s | **1.44s** | 2.26s |
| Weight load | **0.6s** | 0.6s | 0.6s | 1.6s |

- **224**: Peregrine is **3% faster** on CPU (0.65s vs 0.67s), **23% faster** with GPU (0.53s)
- **512**: Peregrine is **16% faster** on CPU (1.89s vs 2.26s), **31% faster** with GPU+Pipeline (1.44s)
- **Weight loading**: Peregrine is **2.7x faster** (0.6s vs 1.6s)

GPU mode (`--gpu`) keeps the entire attention pipeline on Metal — QKV reshape, 2D RoPE, scaled dot-product attention, and output reshape all run as GPU kernels with no CPU round-trips. Pipeline mode (`--pipeline`) overlaps feat1 (GPU) and feat2 (CPU/AMX) decoder processing via `MTLSharedEvent` signaling — single-threaded, no `Send`/`Sync` needed.

### Multi-View Reconstruction Pipeline

`reconstruct_video.py` extracts frames from video, runs all-pairs MUSt3R inference via Peregrine, then jointly optimizes camera poses and fuses pointmaps into a coherent 3D reconstruction. Supports server mode for persistent weight loading, parallel workers for multi-process inference, and optional Metal GPU acceleration.

```
python3 reconstruct_video.py vids/rgb.mp4 --frames 12 --resolution 512 --pairs all --workers 4
```

| Mode | Pairs | Inference (12 frames) |
|------|------:|----------------------:|
| consecutive | 11 | ~7s (224) |
| dense | 31 | ~21s (224) |
| all | 67 | ~45s (224), ~3min (512) |

Server mode (`--server` flag on the Rust binary) loads weights once and processes pairs over stdin/stdout, eliminating ~0.5s overhead per pair. Parallel workers (`--workers N`) distribute pairs across N server processes for near-linear wall-clock scaling.

| Optimization | Impact |
|-------------|--------|
| Hand-tuned NEON intrinsics | 24 vectorized kernels — 4-6x speedup on elementwise ops |
| Cephes-style polynomial exp | Fast exp/sigmoid/tanh/gelu via NEON float32x4_t |
| NEON Adam optimizer | Vectorized Adam step with fast rsqrt approximation |
| CPU buffer pool | Thread-local size-bucketed pool — eliminates malloc on elementwise ops |
| Pool bypass for small tensors | Skip HashMap overhead for tensors < 1024 elements |
| Rayon threshold tuning | Dual thresholds (500K cheap / 100K expensive) — avoids spawn overhead |
| Apple Accelerate BLAS | ~10x faster matmul and 1x1 conv2d |
| Int8 quantized inference | Per-column symmetric weight quantization, per-row dynamic activation quantization, NEON i8 GEMM (vmull+vpadalq, 16 MACs/iter, 4-row blocking), Metal dequant matmul (scalar 16×16 + simdgroup 32×32) |
| 2:4 structured sparsity | Prune to 2:4 pattern (keep 2 largest-abs per group of 4), nibble-packed indices, NEON sparse GEMM, Metal sparse matmul (scalar + simdgroup 32×32 with shared memory expansion), 1.78x bandwidth reduction |
| Metal GPU backend | 108 compute shaders with fused op pipelines (matmul+bias+gelu, add+layernorm, double-buffered matmul, int8 dequant matmul, 2:4 sparse matmul), causal masked SDPA with GQA support, command batching, full autograd integration, GPU-resident attention (QKV reshape, RoPE2D, SDPA), and heterogeneous GPU+CPU scheduling via MTLSharedEvent |

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
  quant.rs        int8 quantized inference — per-column weight quantization, per-row activation quantization, NEON i8 GEMM, Metal dequant matmul
  sparse.rs       2:4 structured sparsity — prune/densify, nibble-packed indices, NEON sparse GEMM, Metal sparse matmul
  serial.rs       model weight save/load (binary format, f32 + int8 + 2:4 sparse)
  attention.rs    core GQA attention — StandardKVCache, gqa_attention_cpu, AttentionMask, PostScoreTransform
  speculative.rs  speculative decoding — CausalLM trait, draft-propose/target-verify with stochastic acceptance
  rl.rs           RL algorithms — PPO, DQN, REINFORCE, replay/rollout buffers, Environment trait (~1,750 lines)
  envs.rs         10 RL environments — CartPole, MountainCar, GridWorld, FrozenLake, BasicArithmetic, ChainArithmetic, NumberSorting, SequenceCompletion, PropositionalLogic, TicTacToe (~2,150 lines)
  metal/          Metal GPU backend (108 shaders, 41 dispatch methods, fused pipelines, causal masked SDPA with GQA, 2:4 sparse matmul, command batching, autograd, het scheduling)
benches/
  tensor_ops.rs   criterion benchmarks (CPU + Metal GPU)
  wallclock.rs    wall-clock comparison benchmark (JSON output)
reconstruct_video.py  multi-view 3D reconstruction pipeline (global pose optimization + point fusion)
visualize_must3r.py   single-pair MUSt3R visualization utility
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
  grok1/          Grok-1 (314B MoE) inference
    main.rs         CLI, tokenizer integration, greedy/temperature generation
    model.rs        Grok1Config (full/small), top-level model, tied embeddings
    decoder.rs      DecoderLayer — pre/post-norm attention + MoE with residuals
    attention.rs    GroupedQueryAttention — GQA via core attention module, RoPE, logit capping, KV cache
    moe.rs          MoELayer — top-k router, DenseBlock (SwiGLU FFN) experts
    tokenizer.rs    SentencePiece BPE tokenizer (pure Rust protobuf parser)
  deepseek/       DeepSeek-V3/R1 (671B MoE) inference
    main.rs         CLI, tokenizer integration, greedy/temperature generation
    model.rs        DeepSeekConfig (full/small), top-level Transformer
    decoder.rs      Block — pre-norm MLA + FFN/MoE with residuals
    attention.rs    MLA — Multi-head Latent Attention, compressed KV cache, YaRN RoPE
    moe.rs          MoE — sigmoid gate, group-limited top-k, shared experts
    tokenizer.rs    HuggingFace tokenizer.json BPE parser (pure Rust)
  rl_demo/        RL training demos with HTML animations
    main.rs         PPO CartPole, DQN GridWorld, REINFORCE Arithmetic
  moba/           MOBA 3v3 with LSTM PPO and self-play
    main.rs         train, selfplay, watch, video subcommands
    game.rs         MobaGame — 32x16 grid, heroes, towers, creeps, bases
    entities.rs     Hero, Tower, Creep, Base entity types
    env.rs          MobaEnv — RL environment wrapper, opponent policies
    policy.rs       RecurrentActorCritic (LSTM), SelfPlayManager
    render.rs       HTML replay animation + MP4 video export (software rasterizer → FFmpeg)
scripts/
  convert_grok1.py  Grok-1 JAX checkpoint → Peregrine binary format (dequantize 8-bit, transpose, rename)
  convert_deepseek.py  DeepSeek HuggingFace SafeTensors → Peregrine binary format (transpose, rename)
  convert_weights_int8.py  convert f32 Peregrine checkpoint to int8 quantized format
tests/
  pytorch_parity.rs   23 numerical parity tests vs PyTorch
  activations.rs      34 activation function tests
  attention.rs        8 GQA attention tests (MHA, GQA, MQA, causal, sliding window, logit cap, rollback)
  speculative.rs      4 speculative decoding tests (greedy, acceptance, EOS, rollback)
  metal_parity.rs     35 CPU vs Metal parity tests (incl. het_execute, causal SDPA, GQA, 2:4 sparse)
  metal_basics.rs     12 Metal compute shader tests
  metal_autograd.rs   17 GPU autograd integration tests
  quant_parity.rs     4 int8 quantization tests (roundtrip, matmul parity, Metal parity, serialization)
  sparse_parity.rs    4 structured sparsity tests (CPU parity, GPU parity, K%4 assertion, serialization roundtrip)
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
