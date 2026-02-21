# Changelog

## [0.2.0] - 2026-02-20

### Performance
- BLAS acceleration via Apple Accelerate framework (macOS) for conv2d (1x1 fast path) and matmul, both forward and backward
- Rayon parallelism for element-wise ops (add, mul, relu, sigmoid, scale, sum, add_bias) and their backward passes, with 10k-element threshold
- Clone elimination in backward pass: `std::mem::replace` to take op ownership instead of cloning, direct `RefCell` borrows instead of `.data()`/`.shape()` which cloned entire Vecs

### Added
- Per-epoch timing in training loop
- Non-macOS fallback for all BLAS-accelerated paths

## [0.1.0] - 2026-02-18

### Added
- Tensor type with N-dimensional storage, shape tracking, and shared ownership
- Reverse-mode autograd engine with computational graph
- Forward and backward for: `add`, `mul`, `matmul`, `relu`, `sigmoid`, `sum`, `scale`, `add_bias`
- SGD optimizer
- Object detection demo with synthetic 8x8 image data
- Three-class detection (car, person, dog) with bounding box + confidence + class prediction
- ASCII bounding box visualization and IoU scoring
- Learning rate scheduling (warmup → decay)

## Roadmap

- [ ] Conv2d forward and backward
- [ ] Batch normalization
- [ ] Proper topological sort for backward pass
- [ ] Broadcasting for all ops
- [ ] Tensor reshape / view
- [ ] Adam optimizer
- [ ] Load real image data (PNG/JPEG)
- [ ] GPU acceleration via wgpu or Metal
