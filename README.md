# rustorch

A from-scratch implementation of PyTorch's core algorithms in Rust: tensors, reverse-mode automatic differentiation (autograd), and gradient-based optimization.

## What's implemented

| Component | Description |
|-----------|-------------|
| **Tensor** | N-dimensional tensor with shared ownership (`Rc<RefCell>`) and shape tracking |
| **Autograd** | Reverse-mode autodiff via a computational graph stored as an `Op` enum on each tensor |
| **Operations** | `add`, `mul`, `matmul`, `relu`, `sigmoid`, `sum`, `scale`, `add_bias` — each with correct forward and backward rules |
| **Optimizer** | SGD (`param -= lr * grad`) |
| **Object Detection** | A YOLO-style fully-connected detector that predicts `(x, y, w, h, confidence, class)` on synthetic 8x8 images |

## Quick start

```bash
cargo run --release
```

This trains a 2-layer object detection network on synthetic data (car/person/dog classes) and prints detection results with ASCII-art bounding box visualizations.

## How autograd works

Every operation (matmul, relu, etc.) records itself in the output tensor's `Op` field, forming a DAG. Calling `backward()` on a scalar loss walks this graph in reverse, applying the chain rule to accumulate gradients on all `requires_grad` tensors.

```
loss.backward()    // seeds grad=1.0, then recurses through:
  -> sum backward  // broadcasts grad to all elements
  -> mul backward  // applies product rule
  -> matmul backward  // dA = dC @ B^T, dB = A^T @ dC
  -> relu backward // masks gradient where input <= 0
  -> sigmoid backward // g * sig * (1 - sig)
```

## Project structure

```
src/
  tensor.rs   -- Tensor type, forward ops, backward (autograd), SGD
  main.rs     -- Object detection demo: data generation, training loop, inference
```

## Limitations

This is a proof-of-concept, not a production framework. Notable gaps:

- No Conv2d (uses fully-connected layers only)
- No GPU acceleration
- No broadcasting (except `add_bias`)
- No dynamic batching or reshape
- Simple recursive backward (no topological sort)
- Toy PRNG for weight initialization

## Built with

Authored with Claude Code.
