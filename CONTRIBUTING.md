# Contributing to Peregrine

Thanks for your interest in contributing! Peregrine is a from-scratch deep learning library in Rust, and we welcome contributions of all kinds.

## Getting Started

### Prerequisites

- **Rust 2021 edition** (stable)
- **macOS with Apple Silicon** (M1/M2/M3/M4) — required for NEON SIMD and Apple Accelerate
- **Python 3.10+** with a virtual environment (for benchmark comparisons)

### Building

```bash
cargo build --release                    # CPU only
cargo build --release --features metal   # with Metal GPU support
```

### Testing

```bash
cargo test                               # all tests (~530+)
cargo test --release --lib               # lib tests only (faster)
cargo test --release --features metal    # include Metal GPU tests
```

### Benchmarking

```bash
cargo bench --bench wallclock            # Peregrine wall-clock benchmarks
./scripts/bench_compare.sh              # full comparison vs PyTorch/MLX/TF/JAX/tinygrad
./autoresearch/bench_quick.sh silu       # quick single-op benchmark
```

## Project Structure

- `src/tensor.rs` — core tensor with autograd (~11K lines)
- `src/simd_kernels.rs` — hand-tuned NEON SIMD kernels
- `src/nn.rs` — neural network layers
- `src/metal/` — Metal GPU backend
- `src/linalg.rs` — linear algebra via LAPACK
- `examples/` — working models (MNIST, MUSt3R, Llama, Grok-1, DeepSeek)
- `benches/wallclock.rs` — benchmark harness
- `tests/` — parity tests against PyTorch

## Making Changes

1. **Fork and branch** from `main`
2. **Write tests** for any new functionality
3. **Run the full test suite** before submitting
4. **Keep changes focused** — one feature or fix per PR
5. **Follow existing code style** — no macros, no proc-macros, readable Rust

## Performance Optimization

If you're optimizing performance:

- **Benchmark before and after** using `./autoresearch/bench_quick.sh`
- **Check for regressions** across all ops, not just the one you changed
- **Test on cold CPU** — thermal throttling can mask real performance
- **Read `autoresearch/program.md`** for optimization techniques and known dead ends

### Known Dead Ends (don't retry)

- `fast_recip_f32x4` — Apple's `vdivq_f32` is faster (3-4 cycles vs 5-6)
- Polynomial sigmoid on [-8,8] — doesn't converge for f32 precision
- `vvexpf` for silu — multi-pass overhead exceeds benefit
- NEON `vec_arccos_f32` — slower than Apple's `vvacosf`

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
