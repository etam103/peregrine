# 🦅 peregrine

A from-scratch tensor and autograd library in Rust, with an RT-DETR object detector as an example application.

## Library

The `peregrine` crate provides reusable building blocks for deep learning:

| Module | Contents |
|--------|----------|
| `peregrine::tensor` | N-dimensional tensor with reverse-mode autograd, BLAS-accelerated matmul/conv2d, and rayon parallelism |
| `peregrine::nn` | Multi-head attention, transformer encoder/decoder layers |

## Quick start

Build the library:

```bash
cargo build --release
```

Run the RT-DETR example (trains a real-time detection transformer on COCO images):

```bash
cargo run --example rt_detr --release
```

Run tests:

```bash
cargo test
```

## How autograd works

Every operation records itself in the output tensor's `Op` field, forming a DAG. Calling `backward()` on a scalar loss walks this graph in reverse, applying the chain rule to accumulate gradients on all `requires_grad` tensors.

```
loss.backward()
  -> sum backward      // broadcasts grad to all elements
  -> mul backward      // applies product rule
  -> matmul backward   // dA = dC @ B^T, dB = A^T @ dC
  -> relu backward     // masks gradient where input <= 0
  -> sigmoid backward  // g * sig * (1 - sig)
```

## Project structure

```
src/
  lib.rs       -- pub mod tensor; pub mod nn;
  tensor.rs    -- Tensor type, forward ops, backward (autograd), SGD
  nn.rs        -- Multi-head attention, transformer encoder/decoder layers
examples/
  rt_detr/     -- RT-DETR object detection training example
Journal.md     -- Performance journal
```

## Built with

Authored with Claude Code.
