<div align="center">

# 🦅 peregrine

**A from-scratch deep learning library in Rust. No PyTorch, no ONNX, no dependencies you can't read.**

Tensors, reverse-mode autograd, transformer layers, and a working object detector — built from `f32` arrays and first principles.

[![GitHub Repo stars](https://img.shields.io/github/stars/etam103/peregrine)](https://github.com/etam103/peregrine/stargazers)
[![Build](https://img.shields.io/badge/build-passing-brightgreen)]()
[![Rust](https://img.shields.io/badge/rust-2021-orange)]()
[![License](https://img.shields.io/badge/license-MIT-blue)]()

</div>

---

```
cargo build --release     # build the library
cargo test                # 13 tests, all pass
cargo run --example rt_detr --release   # train RT-DETR on COCO images
```

---

## What's inside

| Layer | What it does |
|-------|-------------|
| **`peregrine::tensor`** | N-dimensional tensor with reverse-mode autograd, Apple Accelerate BLAS, rayon parallelism |
| **`peregrine::nn`** | Multi-head attention, transformer encoder/decoder layers |
| **`examples/rt_detr`** | Full RT-DETR detector — ResNet backbone, Hungarian matching, training loop, TensorBoard logging |

The entire library is ~2,200 lines of Rust. No macros, no code generation, no proc-macro magic. You can read every line.

---

## How it works

Every op records itself in the output tensor's `Op` field, building a DAG. Calling `.backward()` walks the graph in reverse, accumulating gradients via the chain rule.

```rust
use peregrine::tensor::Tensor;

let x = Tensor::randn(&[2, 3], true);   // requires_grad = true
let w = Tensor::randn(&[3, 1], true);
let y = x.matmul(&w).sum();             // forward: builds the graph
y.backward();                            // backward: computes all gradients
x.sgd_step(0.01);                       // update weights
```

The backward pass supports: `matmul`, `conv2d`, `add`, `mul`, `relu`, `sigmoid`, `gelu`, `softmax`, `sum`, `scale`, `batch_norm`, `layer_norm`, `transpose`, `reshape`, `concat`, `select`, `bce_with_logits`.

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
  tensor.rs       tensor, autograd engine, ops, SGD (~1,900 lines)
  nn.rs           multi-head attention, transformer encoder/decoder
examples/
  rt_detr/        RT-DETR training on COCO
    main.rs         training loop + TensorBoard visualization
    model.rs        ResNet backbone, RT-DETR net, loss, decode, NMS
    dataset.rs      VOC + COCO dataset loaders
```

---

## Limitations

This is a learning project, not a production framework.

- CPU only — no GPU acceleration yet
- No Adam optimizer (SGD only)
- Greedy Hungarian matching (not full O(n³) algorithm)
- Attention forward pass breaks autograd graph (output projection still trains)
- No model save/load

---

<div align="center">

Authored with [Claude Code](https://claude.ai/claude-code).

</div>
