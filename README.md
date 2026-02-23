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
cargo test                                     # 63 tests, all pass
cargo run --example mnist --release            # train MNIST digit classifier (97.5% accuracy)
cargo run --example rt_detr --release          # train RT-DETR on COCO images
```

---

## What's inside

| Module | What it does |
|--------|-------------|
| **`peregrine::tensor`** | N-dimensional tensor with reverse-mode autograd, NumPy-style broadcasting, Apple Accelerate BLAS, rayon parallelism |
| **`peregrine::nn`** | Linear, Embedding, MultiHeadAttention, Transformer encoder/decoder, CrossEntropyLoss, MSELoss |
| **`peregrine::optim`** | SGD (with momentum, Nesterov, weight decay), Adam, AdamW, LR schedulers (StepLR, CosineAnnealing, Warmup), gradient clipping |
| **`peregrine::debug`** | Model summary, training health diagnostics, gradient monitoring |
| **`examples/mnist`** | MNIST digit classifier — MLP trained end-to-end, validates the full stack |
| **`examples/rt_detr`** | Full RT-DETR detector — ResNet backbone, Hungarian matching, training loop, wandb logging |

The entire library is ~3,900 lines of Rust. No macros, no code generation, no proc-macro magic. You can read every line.

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

**Arithmetic:** add, sub, mul, div, neg, scale
**Math:** exp, log, sqrt, abs, pow, sin, cos, tanh
**Activations:** relu, sigmoid, gelu
**Reductions:** sum, mean, softmax, log_softmax
**Shape:** reshape, transpose, squeeze, unsqueeze, concat, select, flatten
**Layers:** matmul, conv2d, conv2d+relu+pool (fused), max_pool2d, add_bias, batch_norm, layer_norm
**Loss:** bce_with_logits, cross_entropy_loss, mse_loss

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

23 integration tests cross-validate Peregrine against PyTorch reference data, covering matmul, softmax, log_softmax, layernorm, cross_entropy_loss, 14 element-wise ops, Adam optimizer, and a full 10-step MLP training loop. All pass within 1e-4 to 1e-7 absolute error.

```
$ cargo test --test pytorch_parity
running 23 tests ... ok
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
    ┌────▼────┐
    │  ResNet  │  4-stage backbone with residual connections
    │ Backbone │  1x1 stem → 3 stages of 3x3 conv blocks + skip + pool
    └─┬──┬──┬─┘
      s2 s3 s4   multi-scale features [128², 64², 32²]
      │  │  │
    ┌─▼──▼──▼─┐
    │ Channel  │  1x1 conv projections to embed_dim
    │  Project │  pool all scales to common 32×32
    └────┬─────┘
         │  [batch, 3072, embed_dim]
    ┌────▼────┐
    │  Xfmr   │  self-attention + FFN with pre-norm
    │ Encoder  │
    └────┬─────┘
         │  encoder memory
    ┌────▼────┐
    │  Xfmr   │  learned object queries attend to memory
    │ Decoder  │  self-attn → cross-attn → FFN
    └────┬─────┘
         │  [batch, num_queries, embed_dim]
    ┌────▼────┐
    │  Heads   │  classification (softmax) + bbox regression (sigmoid)
    └─────────┘
```

---

## Performance

All computation runs on CPU. On Apple Silicon (M-series), matmul and conv2d are accelerated via the Accelerate framework (BLAS). Element-wise ops parallelize via rayon above 10k elements.

| Optimization | Impact |
|-------------|--------|
| Apple Accelerate BLAS | ~10× faster matmul and 1×1 conv2d |
| Rayon parallelism | Parallel add/mul/relu/sigmoid/sum and their backward passes |
| Clone elimination | `std::mem::replace` for op ownership, direct `RefCell` borrows |
| Xavier init | `std = sqrt(1/fan_in)` — prevents NaN loss in deep networks |
| Multi-scale pooling | 3,072 encoder tokens instead of 56,784 — attention fits in memory |

---

## Project structure

```
src/
  lib.rs          public API surface
  tensor.rs       tensor, autograd engine, ops, broadcasting (~2,400 lines)
  nn.rs           Linear, Embedding, attention, transformer, loss functions
  optim.rs        SGD, Adam, AdamW, LR schedulers, gradient clipping
  debug.rs        model summary + training health diagnostics
examples/
  mnist/          MNIST digit classifier (97.5% test accuracy)
  rt_detr/        RT-DETR training on COCO
    main.rs         training loop + wandb visualization
    model.rs        ResNet backbone, RT-DETR net, loss, decode, NMS
    dataset.rs      VOC + COCO dataset loaders
tests/
  pytorch_parity.rs   23 numerical parity tests vs PyTorch
  generate_reference.py  script to regenerate PyTorch reference data
  fixtures/             binary reference tensors
```

---

## Limitations

This is a learning project, not a production framework.

- CPU only — no GPU acceleration yet (Metal backend planned)
- Greedy Hungarian matching (not full O(n³) algorithm)
- Attention forward pass breaks autograd graph (output projection still trains)
- No model save/load

---

<div align="center">

Authored with [Claude Code](https://claude.ai/claude-code).

</div>
