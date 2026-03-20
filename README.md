<div align="center">

# 🦅 peregrine

**A from-scratch deep learning library in Rust. No PyTorch, no ONNX, no dependencies you can't read.**

Tensors, reverse-mode autograd, neural network layers, optimizers, and working models — built from `f32` arrays and first principles.

[![crates.io](https://img.shields.io/crates/v/peregrine-ml)](https://crates.io/crates/peregrine-ml)
[![GitHub Repo stars](https://img.shields.io/github/stars/etam103/peregrine)](https://github.com/etam103/peregrine/stargazers)
[![Build](https://github.com/etam103/peregrine/actions/workflows/ci.yml/badge.svg)](https://github.com/etam103/peregrine/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

**Platform:** macOS on Apple Silicon (M1/M2/M3/M4). Uses Apple Accelerate for BLAS/FFT/vForce, NEON SIMD intrinsics, and optional Metal GPU compute.

</div>

---

```
cargo build --release                          # build the library
cargo test                                     # 600+ tests
cargo run --example mnist --release            # train MNIST digit classifier (97.5% accuracy)
cargo run --example rt_detr --release          # train RT-DETR on COCO images
./scripts/bench_compare.sh                     # wall-clock benchmark vs PyTorch, MLX, TF, tinygrad, JAX
cargo run -p peregrine-bench --release         # reproducible benchmark suite (141 ops, JSON output)
pip install ./peregrine-py && python -c "import peregrine"  # Python bindings
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
| **`peregrine::rl`** | RL algorithms and infrastructure — PPO, DQN, REINFORCE, GRPO, RLOO, Dr. GRPO, DAPO, CISPO, DPPO, MaxRL, ScaleRL, replay buffers, rollout buffers, Environment/ReasoningEnv traits, action spaces |
| **`peregrine::rlm`** | Recursive Language Models — `GenerativeLM` trait, `RlmOrchestrator` recursive execution engine, REPL context with peek/grep/partition-map/summarize actions, depth and token budget control |
| **`peregrine::envs`** | 10 RL environments — CartPole, MountainCar, GridWorld, FrozenLake, BasicArithmetic, ChainArithmetic, NumberSorting, SequenceCompletion, PropositionalLogic, TicTacToe |
| **`peregrine::gguf`** | GGUF binary format parser — load quantized model weights (Q8_0, Q4_0, Q4_1, F16, F32), metadata extraction, dequantization |
| **`peregrine::safetensors`** | Safetensors binary format parser — mmap-based loading on Unix, F32/F16/BF16 dequantization, handwritten JSON header parser |
| **`peregrine::hf_config`** | HuggingFace config.json parser — `ModelConfig` with Llama/Mistral fields, JSON value extraction with scientific notation |
| **`peregrine::hf_hub`** | HuggingFace Hub integration — download and cache safetensors weights, config.json, tokenizer.json. Auth via `HF_TOKEN`, multi-shard support (`--features hf`) |
| **`peregrine::models::llama`** | Library-level Llama model — `Llama`, `LlamaConfig`, `LlamaBlock`, `LlamaAttention`, `KVCache`, `Tokenizer`. Loadable from GGUF or safetensors. Used by both the CLI example and Python bindings |
| **`peregrine::metal`** | Metal GPU backend — 108 compute shaders, 41 dispatch methods, fused op pipelines, causal masked SDPA with GQA, 2:4 sparse matmul, command batching, autograd integration, buffer pool, heterogeneous GPU+CPU scheduling, thermal-aware scheduling (`--features metal`) |
| **`examples/mnist`** | MNIST digit classifier — MLP trained end-to-end (97.5% accuracy) |
| **`examples/rt_detr`** | RT-DETR object detector — ResNet backbone, transformer encoder/decoder, Hungarian matching, wandb logging |
| **`examples/must3r`** | MUSt3R 3D reconstruction — 423M param ViT-L/B, server mode, parallel workers, Metal GPU, heterogeneous pipeline |
| **`examples/grok1`** | Grok-1 (314B MoE) inference — GQA, 8 experts top-2, SentencePiece tokenizer, speculative decoding |
| **`examples/deepseek`** | DeepSeek-V3/R1 (671B MoE) inference — MLA, compressed KV cache, 256 routed experts, speculative decoding |
| **`examples/llama`** | Llama 3.2 inference — GGUF/safetensors/HF Hub, streaming decode, sustained profiling, chunked prefill |
| **`examples/rl_demo`** | RL training demos — PPO on CartPole, DQN on GridWorld, REINFORCE on Arithmetic, HTML visualizations |
| **`examples/rlm`** | Recursive Language Model inference — Llama-backed `GenerativeLM`, recursive context decomposition with depth/token budget control |
| **`examples/moba`** | MOBA 3v3 with LSTM PPO and self-play — train, selfplay, watch (HTML replay), video (MP4 export) |
| **`peregrine::sched`** | Request scheduler — priority-based prefill/decode aggregation, chunked prefill, EMA latency tracking |
| **`peregrine::thermal`** | Thermal monitoring via Darwin notifications — thermal-aware GPU/CPU scheduling |
| **`peregrine-py`** | Python bindings via PyO3 — `peregrine.Tensor` with NumPy interop, `peregrine.load_model()` for LLM inference |
| **`peregrine-bench`** | Reproducible benchmark CLI — 141 ops, hardware auto-detection, JSON output, GitHub Pages dashboard |

The entire library is ~37,900 lines of Rust. No macros, no code generation, no proc-macro magic. You can read every line.

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

## Performance

CPU ops use Apple Accelerate BLAS and rayon parallelism. GPU ops use Metal compute shaders (`--features metal`). Wall-clock benchmarks run via `./scripts/bench_compare.sh`.

### Peregrine vs ML Frameworks (141 ops, CPU, wall-clock, all times in microseconds)

| Operation | Peregrine | PyTorch | MLX | TensorFlow | tinygrad | JAX |
|-----------|----------:|--------:|----:|-----------:|---------:|----:|
| matmul 128x128 | **7.6** | 6.0 | 24.0 | 54.6 | 435.9 | 59.1 |
| matmul 512x512 | **220.2** | 136.7 | 172.1 | 791.4 | 451.2 | 569.7 |
| add 100k | **12.9** | 35.7 | 32.3 | 54.1 | 190.9 | 35.2 |
| mul 100k | **12.8** | 44.0 | 32.0 | 45.1 | 192.8 | 36.6 |
| relu 100k | **8.3** | 40.4 | 30.5 | 37.7 | 340.6 | 106.6 |
| softmax 8x128 | **1.2** | 33.3 | 18.4 | 12.4 | 629.3 | 33.4 |
| gelu 100k | **58.0** | 62.0 | 146.2 | 278.3 | 858.0 | 223.0 |
| silu 100k | 54.8 | **51.6** | 89.8 | 255.9 | 334.1 | 59.8 |
| erf 100k | 50.7 | **40.4** | 101.0 | 64.3 | — | 57.8 |
| rfft 1k | **2.2** | 4.4 | 19.0 | 44.6 | — | 49.5 |
| cross_entropy | **2.7** | 47.8 | 24.8 | 629.4 | 3436.6 | 58.2 |
| train step 64 | **771** | 1334 | 810 | 10103 | 25538 | 5310 |
| add+layernorm 196x768 | **83.9** | 109.7 | — | 1238.5 | 1207.3 | 252.4 |
| cholesky 128x128 | **17.3** | 62.5 | 26.7 | 58.6 | — | 36.0 |
| inv 64x64 | **17.1** | 26.5 | 47.4 | 32.5 | — | 44.0 |
| solve 128x128 | **36.8** | 45.0 | 187.2 | 76.8 | — | 87.3 |

Geometric mean ratio across 141 ops (lower = Peregrine faster): **PyTorch 0.47x** (53% faster), **MLX 0.34x**, TensorFlow 0.24x, tinygrad 0.05x, JAX 0.31x. Peregrine wins 116 of 141 ops.

### MUSt3R 3D Reconstruction (423M params, Apple Silicon)

| Resolution | Peregrine CPU | Peregrine GPU+Pipeline | PyTorch CPU |
|-----------|----:|-------------:|------------:|
| 224x224 | 0.59s | **0.54s** | 0.67s |
| 512x384 | 1.95s | **1.44s** | 2.26s |

12-36% faster than PyTorch on CPU, up to 36% faster with Metal GPU pipeline. See [`examples/must3r/`](examples/must3r/) for details.

See [`benchmarks/BENCHMARKS.md`](benchmarks/BENCHMARKS.md) for detailed results and methodology.

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
  gguf.rs         GGUF binary format parser — Q8_0/Q4_0/Q4_1/F16/F32 dequantization, metadata extraction
  safetensors.rs  safetensors binary parser — mmap loading, F32/F16/BF16 dequant, JSON header parser
  hf_config.rs    HuggingFace config.json parser — ModelConfig, JSON value extraction
  hf_hub.rs       HuggingFace Hub download & cache — safetensors + config + tokenizer (--features hf)
  serial.rs       model weight save/load (binary format, f32 + int8 + 2:4 sparse)
  attention.rs    core GQA attention — StandardKVCache, gqa_attention_cpu, AttentionMask, PostScoreTransform
  speculative.rs  speculative decoding — CausalLM trait, draft-propose/target-verify with stochastic acceptance
  sched.rs        request scheduler — priority-based prefill/decode aggregation, chunked prefill, dynamic chunk-size tuning, EMA latency tracking
  thermal.rs      thermal monitoring via Darwin notifications — ThermalState, rate-limited polling, thread-local singleton
  models/
    llama/        Library-level Llama model (attention, decoder, model, tokenizer)
  rl.rs           RL algorithms — PPO, DQN, REINFORCE, GRPO, RLOO, Dr. GRPO, DAPO, CISPO, DPPO, MaxRL, ScaleRL, replay/rollout buffers, Environment trait
  rlm.rs          Recursive Language Models — GenerativeLM trait, RlmOrchestrator, REPL context, 7 action types, recursive execution engine
  envs.rs         10 RL environments — CartPole, MountainCar, GridWorld, FrozenLake, BasicArithmetic, ChainArithmetic, NumberSorting, SequenceCompletion, PropositionalLogic, TicTacToe (~2,150 lines)
  metal/          Metal GPU backend (108 shaders, 41 dispatch methods, fused pipelines, causal masked SDPA with GQA, 2:4 sparse matmul, command batching, autograd, het scheduling, thermal-aware scheduling)
peregrine-py/     Python bindings (PyO3 + maturin)
  src/            py_tensor.rs, py_nn.rs, py_inference.rs
  python/         peregrine/__init__.py
peregrine-bench/  Reproducible benchmark CLI
  src/            main.rs, suite.rs (141 ops), hardware.rs (sysctl detection)
docs/             GitHub Pages benchmark dashboard
  index.html      Interactive results viewer
  results/        JSON benchmark submissions
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
  reconstruct_video.py  multi-view 3D reconstruction pipeline (global pose optimization + point fusion)
  visualize_must3r.py   single-pair MUSt3R visualization utility
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
  llama/          Llama 3.2 inference (GGUF, safetensors, or HF Hub)
    main.rs         CLI, auto-detect model format, greedy/temperature/top-p sampling, streaming decode, sustained profiling, chunked prefill, multi-request scheduling
    wandb.rs        W&B logging for sustained profiling metrics
    model.rs        LlamaConfig (from GGUF or config.json), weight loading from GGUF or safetensors with transpose
    decoder.rs      LlamaBlock — pre-norm GQA attention + SwiGLU FFN
    attention.rs    GQA with RoPE (theta=500000), causal masking, KV cache
    tokenizer.rs    BPE tokenizer from GGUF embedded vocabulary or HF tokenizer.json (byte-level BPE)
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
  gguf_parity.rs      5 GGUF parser tests (Q8_0/Q4_0/Q4_1 dequant, invalid magic, minimal valid parse)
  generate_reference.py  script to regenerate PyTorch reference data
  fixtures/             binary reference tensors
```

---

## Limitations

Known limitations:

- Greedy Hungarian matching in RT-DETR (not full O(n³) Kuhn-Munkres)
- Attention forward pass breaks autograd graph (output projection still trains)
- GPU training step is slower than CPU at batch=64 due to `dispatch_reduce` sync in `mean()`; larger batches would favor GPU

---

<div align="center">

Authored with [Claude Code](https://claude.ai/claude-code).

</div>
