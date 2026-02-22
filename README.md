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
| **`peregrine::debug`** | Model summary, training health diagnostics, gradient monitoring |
| **`examples/rt_detr`** | Full RT-DETR detector — ResNet backbone, Hungarian matching, training loop, wandb logging |

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
backbone.stage2.0.conv1_b              [128]                  128
backbone.stage2.0.conv2_w              [128, 128, 3, 3]  147,456
backbone.stage2.0.conv2_b              [128]                  128
backbone.stage2.0.skip_w               [128, 64, 1, 1]     8,192
backbone.stage2.0.skip_b               [128]                  128
...
encoder.0.mha.wq                       [64, 64]             4,096
encoder.0.mha.bq                       [1, 64]                 64
...
object_queries                         [20, 64]             1,280
decoder.0.self_attn.wq                 [64, 64]             4,096
...
cls_w                                  [64, 4]                256
cls_b                                  [1, 4]                   4
bbox_w                                 [64, 4]                256
bbox_b                                 [1, 4]                   4
──────────────────────────────────────────────────────────────────
85 parameters                                          1,190,856
```

### Training Health

Call `training_health` periodically during training to monitor gradient health, detect NaN/exploding gradients, and log diagnostics to wandb:

```rust
use peregrine::debug::training_health;

// Every 50 steps during training:
let report = training_health(&net.named_params());

// Log to wandb
let metrics = report.to_metrics();
// Returns: [("health/grad_norm", 0.423), ("health/has_nan", 0.0),
//           ("health/zero_grad_params", 0.0), ("health/max_grad_weight_ratio", 0.12)]
wandb_run.log_metrics(step, &metrics);

// Print warnings to terminal (only when issues are detected)
if !report.warnings.is_empty() {
    print!("{}", report.display());
}
```

When issues are detected, `report.display()` prints a diagnostic table:

```
Training Health  (grad_norm = 14.238710)
  param                                   w_mean       w_std      g_mean       g_std   g/w ratio
  backbone.stem_w                       0.001523    0.124838   -0.000842    0.193841    1.5528
  backbone.stage2.0.conv1_w             0.000012    0.038215    0.000000    0.000000    0.0000 ZERO
  encoder.0.ffn_w1                      0.000183    0.017421    0.042158    0.087312    5.0119
  ...
  Warnings:
    - backbone.stem_w: grad/weight ratio 1.5528 — possible exploding gradients
    - backbone.stage2.0.conv1_w: gradient is all zeros — parameter may be stalled
    - encoder.0.ffn_w1: grad/weight ratio 5.0119 — possible exploding gradients
```

The RT-DETR example integrates both automatically — model summary prints at startup, and training health runs every 50 steps with metrics logged to wandb.

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
  debug.rs        model summary + training health diagnostics
examples/
  rt_detr/        RT-DETR training on COCO
    main.rs         training loop + wandb visualization
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
